// ============================================================
// MC 史莱姆区块种子搜索器 — CUDA GPU 实现（bitpack 寄存器版）
// 目标: (-200,-200)~(200,200) 的 401×401 区块网格内,
//       寻找 5×5 窗口 ≥19 个史莱姆区块的密集种子（找到即停）
// 算法: isSlimeChunk（模拟 Java 32 位溢出）
//       + 行数据位打包进寄存器（5 行 × 7×64bit，零 local memory）
//       + 窗口检查 = 5 行 popcount（__popcll 单指令）
// 对比: slime_search.cu 的 rows[5][401]+col_win[401] 是 2808B
//       local memory（VRAM），本版全部移入寄存器（片上）
// 编译: nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gpu/slime_search_bp.cu -o gpu/slime_search_bp.exe
// 运行: gpu\slime_search_bp.exe --start 0 --end 10000000 [--checkpoint cp] [--results res]
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GRID 401
#define OFFSET 200
#define WINDOW 5
#define DEFAULT_THRESHOLD 19   // 默认阈值，可用 --threshold N 覆盖
#define DEFAULT_BATCH 1500000ULL   // 每批种子数：150 万 → ~1.1s/batch，checkpoint ~0.9 次/s
#define ROW_WORDS 7          // 7×64 = 448 bit ≥ 401
#define ROW_BITS 64

// ------------------------------------------------------------
// 史莱姆区块判定（与 test_slime.c 逐行一致，已验证 seed=12345 得 16111 个）
// ------------------------------------------------------------
__device__ __forceinline__ bool isSlimeChunk(long long worldSeed, int chunkX, int chunkZ) {
    int t1 = chunkX * chunkX * 0x4c1906;
    int t2 = chunkX * 0x5ac0db;
    int t3_z = chunkZ * chunkZ;
    long long t3 = (long long)t3_z * 0x4307a7LL;
    int t4 = chunkZ * 0x5f24f;

    long long seed = (worldSeed + t1 + t2 + t3 + t4) ^ 0x3ad8025fLL;
    seed = (seed ^ 0x5DEECE66DLL) & 0xFFFFFFFFFFFFLL;
    // Java Random.next(31)
    // （注：26.2 实测 MC 用拒绝采样 nextInt(10)，差异仅 8/2^31≈3.7e-9/判定，
    //   统计无感且实现后性能 -25%，决定不实现——详见 DEVELOPMENT_LOG.md 2026-08-14）
    seed = (seed * 0x5DEECE66DLL + 0xBLL) & 0xFFFFFFFFFFFFLL;
    int bits = (int)(seed >> 17);
    return (bits % 10) == 0;
}

// ------------------------------------------------------------
// 搜索内核（bitpack 版）:
//   每线程一串种子（grid-stride）；行数据存 35 个 uint64 寄存器
//   窗口检查: 每 z 行对 cx 0..396 做 5 行 popcount（全寄存器）
// ------------------------------------------------------------
__global__ void search_kernel(long long seed_start, long long seed_end,
                              long long step, int threshold, int neg, int cont,
                              volatile int* d_found,
                              unsigned long long* d_result_seed,
                              int* d_result_x, int* d_result_z, int* d_result_count,
                              int* d_found_count,          // 仅 cont 模式
                              unsigned long long* d_seeds, // 仅 cont 模式
                              int* d_wins_x, int* d_wins_z, int* d_wins_c,
                              int max_results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (!cont && *d_found) return;

    unsigned long long row_bits[WINDOW][ROW_WORDS];  // 35 个 uint64 寄存器
    long long seed;
    for (long long s = 0; ; s += step) {
        if (neg) {
            seed = seed_start - tid - s;          // 负方向：从 start 往下
            if (seed <= seed_end) break;
        } else {
            seed = seed_start + tid + s;          // 正方向：从 start 往上
            if (seed >= seed_end) break;
        }
        if (!cont && *d_found) return;

        int filled = 0;
        int hit = 0;
        int hit_sum = 0, hit_x = 0, hit_z = 0;

        for (int z = -OFFSET; z <= OFFSET && !hit; z++) {
            int slot = filled % WINDOW;

            // 打包当前行: 7 个 64 位块, 每块 64 个 x
            #pragma unroll
            for (int b = 0; b < ROW_WORDS; b++) {
                unsigned long long bits = 0;
                int x0 = b * ROW_BITS - OFFSET;
                // 块内 64 个 x；第 7 块只到 x=200
                int x_end = x0 + ROW_BITS;
                if (x_end > OFFSET + 1) x_end = OFFSET + 1;
                for (int i = 0; i < 64; i++) {
                    int x = x0 + i;
                    if (x < x_end) {
                        if (isSlimeChunk((long long)seed, x, z)) bits |= 1ULL << i;
                    }
                }
                row_bits[slot][b] = bits;
            }
            filled++;

            if (filled >= WINDOW) {
                int z_win_start = filled - WINDOW;  // 窗口最老一行
                // 水平滑窗: cx 0..GRID-WINDOW (0..396)
                for (int cx = 0; cx <= GRID - WINDOW; cx++) {
                    int sum = 0;
                    int byte_idx = cx >> 6;
                    int bit_off = cx & 63;
                    #pragma unroll
                    for (int r = 0; r < WINDOW; r++) {
                        int sr = (z_win_start + r) % WINDOW;
                        unsigned long long v;
                        if (bit_off <= 59) {
                            v = (row_bits[sr][byte_idx] >> bit_off) & 0x1FULL;
                        } else {
                            // 跨 64 位边界: 取两段
                            v = ((row_bits[sr][byte_idx] >> bit_off) |
                                 (row_bits[sr][byte_idx + 1] << (64 - bit_off))) & 0x1FULL;
                        }
                        sum += __popcll(v);
                    }
                    if (sum >= threshold) {
                        hit = 1; hit_sum = sum; hit_x = cx - OFFSET; hit_z = z - WINDOW + 1;
                        break;
                    }
                }
            }
        }

        if (hit) {
            if (cont) {
                // 找到继续：写入结果数组，不退出，继续下一个种子
                int slot = atomicAdd(d_found_count, 1);
                if (slot < max_results) {
                    d_seeds[slot] = (unsigned long long)seed;
                    d_wins_x[slot] = hit_x;
                    d_wins_z[slot] = hit_z;
                    d_wins_c[slot] = hit_sum;
                }
            } else {
                int old = atomicCAS((int*)d_found, 0, 1);
                if (old == 0) {
                    *d_result_seed = seed;
                    *d_result_x = hit_x;
                    *d_result_z = hit_z;
                    *d_result_count = hit_sum;
                }
                return;
            }
        }
    }
}

// ------------------------------------------------------------
// 验证内核: 单种子完整网格统计（与 CPU 参考 verify_known_seed 对比用）
// ------------------------------------------------------------
__global__ void verify_kernel(long long seed, int* d_total, int* d_best,
                              int* d_best_x, int* d_best_z) {
    unsigned long long row_bits[WINDOW][ROW_WORDS];
    int total = 0, best = 0, best_x = 0, best_z = 0;
    int filled = 0;

    for (int z = -OFFSET; z <= OFFSET; z++) {
        int slot = filled % WINDOW;
        #pragma unroll
        for (int b = 0; b < ROW_WORDS; b++) {
            unsigned long long bits = 0;
            int x0 = b * ROW_BITS - OFFSET;
            int x_end = x0 + ROW_BITS;
            if (x_end > OFFSET + 1) x_end = OFFSET + 1;
            for (int i = 0; i < 64; i++) {
                int x = x0 + i;
                if (x < x_end) {
                    if (isSlimeChunk(seed, x, z)) { bits |= 1ULL << i; total++; }
                }
            }
            row_bits[slot][b] = bits;
        }
        filled++;

        if (filled >= WINDOW) {
            int z_win_start = filled - WINDOW;
            for (int cx = 0; cx <= GRID - WINDOW; cx++) {
                int sum = 0;
                int byte_idx = cx >> 6;
                int bit_off = cx & 63;
                #pragma unroll
                for (int r = 0; r < WINDOW; r++) {
                    int sr = (z_win_start + r) % WINDOW;
                    unsigned long long v;
                    if (bit_off <= 59) {
                        v = (row_bits[sr][byte_idx] >> bit_off) & 0x1FULL;
                    } else {
                        v = ((row_bits[sr][byte_idx] >> bit_off) |
                             (row_bits[sr][byte_idx + 1] << (64 - bit_off))) & 0x1FULL;
                    }
                    sum += __popcll(v);
                }
                if (sum > best) { best = sum; best_x = cx - OFFSET; best_z = z - WINDOW + 1; }
            }
        }
    }
    *d_total = total;
    *d_best = best;
    *d_best_x = best_x;
    *d_best_z = best_z;
}

// ------------------------------------------------------------
// 主机端
// ------------------------------------------------------------
struct Options {
    long long start = 0;        // 起始种子（默认 0）
    long long end = 0;          // 终点（has_end=false 时无限搜索）
    bool has_end = false;
    bool neg = false;           // 负方向：从 start 往下搜索
    bool cont = false;          // 找到继续（不暂停）：收集所有命中种子
    int threshold = DEFAULT_THRESHOLD;
    std::string checkpoint;
    std::string results;
    bool verify = false;
    long long verify_seed = 0;
};

static bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value: %s\n", name); return nullptr; }
            return argv[++i];
        };
        if (a == "--start") { const char* v = need("--start"); if (!v) return false; opt.start = strtoll(v, nullptr, 10); }
        else if (a == "--end") { const char* v = need("--end"); if (!v) return false; opt.end = strtoll(v, nullptr, 10); opt.has_end = true; }
        else if (a == "--neg") { opt.neg = true; }
        else if (a == "--continue") { opt.cont = true; }
        else if (a == "--threshold") { const char* v = need("--threshold"); if (!v) return false; opt.threshold = atoi(v); }
        else if (a == "--checkpoint") { const char* v = need("--checkpoint"); if (!v) return false; opt.checkpoint = v; }
        else if (a == "--results") { const char* v = need("--results"); if (!v) return false; opt.results = v; }
        else if (a == "--verify") { const char* v = need("--verify"); if (!v) return false; opt.verify = true; opt.verify_seed = strtoll(v, nullptr, 10); }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return false; }
    }
    if (opt.has_end && (opt.neg ? (opt.end >= opt.start) : (opt.end <= opt.start))) {
        fprintf(stderr, "--end 与方向不匹配（正方向 end 必须 > start，负方向 end 必须 < start）\n"); return false;
    }
    if (opt.threshold < 1 || opt.threshold > 25) { fprintf(stderr, "--threshold must be in [1, 25]\n"); return false; }
    return true;
}

static bool load_checkpoint(const std::string& path, long long& start) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    if (fscanf(f, "%lld", &start) != 1) { fclose(f); return false; }
    fclose(f);
    return true;
}

static void save_checkpoint(const std::string& path, long long next) {
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "w");
    if (f) { fprintf(f, "%lld\n", next); fclose(f); }
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        fprintf(stderr, "usage: %s --start N [--end N] [--neg] [--continue] [--threshold N] [--checkpoint FILE] [--results FILE] | --verify SEED\n", argv[0]);
        return 1;
    }

    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) { fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    if (opt.verify) {
        int* d_total; int* d_best; int* d_best_x; int* d_best_z;
        cudaMalloc(&d_total, sizeof(int)); cudaMalloc(&d_best, sizeof(int));
        cudaMalloc(&d_best_x, sizeof(int)); cudaMalloc(&d_best_z, sizeof(int));
        verify_kernel<<<1, 1>>>((long long)opt.verify_seed, d_total, d_best, d_best_x, d_best_z);
        cudaDeviceSynchronize();
        int total, best, bx, bz;
        cudaMemcpy(&total, d_total, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&best, d_best, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bx, d_best_x, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&bz, d_best_z, sizeof(int), cudaMemcpyDeviceToHost);
        printf("verify seed=%lld\n", opt.verify_seed);
        printf("total_slime=%d ratio=%.4f\n", total, (double)total / (GRID * GRID));
        printf("best_window=%d best_x=%d best_z=%d\n", best, bx, bz);
        cudaFree(d_total); cudaFree(d_best); cudaFree(d_best_x); cudaFree(d_best_z);
        return 0;
    }

    if (!opt.checkpoint.empty() && load_checkpoint(opt.checkpoint, opt.start)) {
        printf("[checkpoint] resume from %lld\n", opt.start);
        if (opt.has_end && (opt.neg ? (opt.start <= opt.end) : (opt.start >= opt.end))) {
            printf("checkpoint already at end\n"); return 0;
        }
    }

    printf("GPU: %s (CC %d.%d, %d SM)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    int block = 256;
    int sm_count = prop.multiProcessorCount;
    int grid = sm_count * 4;
    long long step = (long long)block * grid;

    int* d_found = nullptr;
    unsigned long long* d_result_seed = nullptr;
    int* d_result_x = nullptr;
    int* d_result_z = nullptr;
    int* d_result_count = nullptr;
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_result_seed, sizeof(unsigned long long));
    cudaMalloc(&d_result_x, sizeof(int));
    cudaMalloc(&d_result_z, sizeof(int));
    cudaMalloc(&d_result_count, sizeof(int));

    // --continue 模式：结果收集数组（大小 = 一批种子数，2MB 级，足够）
    int* d_found_count = nullptr;
    unsigned long long* d_seeds = nullptr;
    int* d_wins_x = nullptr, *d_wins_z = nullptr, *d_wins_c = nullptr;
    if (opt.cont) {
        cudaMalloc(&d_found_count, sizeof(int));
        cudaMalloc(&d_seeds, DEFAULT_BATCH * sizeof(unsigned long long));
        cudaMalloc(&d_wins_x, DEFAULT_BATCH * sizeof(int));
        cudaMalloc(&d_wins_z, DEFAULT_BATCH * sizeof(int));
        cudaMalloc(&d_wins_c, DEFAULT_BATCH * sizeof(int));
    }

    int h_found = 0;
    unsigned long long h_seed = 0;
    int h_x = 0, h_z = 0, h_count = 0;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    long long next = opt.start;
    long long found_at = 0;
    double elapsed = 0.0;
    bool found = false;
    long long batch_count = 0;

    printf("search range: start=%lld%s  direction=%s  step=%lld  batch=%lld  target: %dx%d window >=%d\n",
           opt.start, opt.has_end ? "" : " (infinite)", opt.neg ? "neg" : "pos",
           step, DEFAULT_BATCH, WINDOW, WINDOW, opt.threshold);
    printf("start...\n");

    cudaEventRecord(t0);
    while (true) {
        if (opt.has_end && (opt.neg ? (next <= opt.end) : (next >= opt.end))) break;

        long long batch_end;
        if (opt.neg) {
            batch_end = next - DEFAULT_BATCH;
            if (opt.has_end && batch_end < opt.end) batch_end = opt.end;
        } else {
            batch_end = next + DEFAULT_BATCH;
            if (opt.has_end && batch_end > opt.end) batch_end = opt.end;
        }

        cudaMemset(d_found, 0, sizeof(int));
        if (opt.cont) cudaMemset(d_found_count, 0, sizeof(int));
        search_kernel<<<grid, block>>>(next, batch_end, step, opt.threshold, opt.neg ? 1 : 0,
                                       opt.cont ? 1 : 0,
                                       d_found, d_result_seed, d_result_x, d_result_z, d_result_count,
                                       d_found_count, d_seeds, d_wins_x, d_wins_z, d_wins_c,
                                       (int)DEFAULT_BATCH);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "kernel failed: %s\n", cudaGetErrorString(err)); return 1; }

        if (opt.cont) {
            // 找到继续：收集本批所有命中并打印，不暂停
            int n = 0;
            cudaMemcpy(&n, d_found_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (n > 0) {
                std::vector<unsigned long long> seeds(n);
                std::vector<int> xs(n), zs(n), cs(n);
                cudaMemcpy(seeds.data(), d_seeds, (size_t)n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                cudaMemcpy(xs.data(), d_wins_x, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(zs.data(), d_wins_z, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(cs.data(), d_wins_c, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
                FILE* rf = opt.results.empty() ? nullptr : fopen(opt.results.c_str(), "a");
                for (int i = 0; i < n; i++) {
                    printf("\n=== FOUND dense seed ===\n");
                    printf("seed: %lld\n", (long long)seeds[i]);
                    printf("window top-left chunk: (%d, %d)\n", xs[i], zs[i]);
                    printf("5x5 slime count: %d (threshold %d)\n---\n", cs[i], opt.threshold);
                    found = true;
                    if (rf) fprintf(rf, "seed=%lld win_x=%d win_z=%d count=%d\n",
                                    (long long)seeds[i], xs[i], zs[i], cs[i]);
                }
                if (rf) fclose(rf);
            }
        } else {
            cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_found) {
                cudaMemcpy(&h_seed, d_result_seed, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_x, d_result_x, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_z, d_result_z, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
                found = true;
                found_at = next;
                next = batch_end;
                break;
            }
        }

        next = batch_end;
        save_checkpoint(opt.checkpoint, next);   // 每批（10 万）写一次 → ~12Hz 平滑更新
        batch_count++;
        if (batch_count % 100 == 0) {
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, t0, t1);
            double sec = ms / 1000.0;
            double rate = sec > 0 ? (double)next / sec : 0.0;
            printf("  searched %lld seeds (%.2fM seeds/s, %.1fs)\n", next, rate / 1e6, sec);
        }
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    elapsed = ms / 1000.0;

    if (found && !opt.cont) {
        printf("\n=== FOUND dense seed ===\n");
        printf("seed: %lld\n", (long long)h_seed);
        printf("window top-left chunk: (%d, %d)\n", h_x, h_z);
        printf("5x5 slime count: %d (threshold %d)\n", h_count, opt.threshold);
        printf("seeds checked: %lld\n", found_at - opt.start);
        printf("elapsed: %.2fs\n", elapsed);
        if (!opt.results.empty()) {
            FILE* f = fopen(opt.results.c_str(), "w");
            if (f) {
                fprintf(f, "seed=%lld\nwin_x=%d\nwin_z=%d\ncount=%d\n", (long long)h_seed, h_x, h_z, h_count);
                fclose(f);
                printf("results written: %s\n", opt.results.c_str());
            }
        }
    } else if (!found) {
        printf("[%lld, %lld) no dense seed found\n", opt.start, opt.end);
        save_checkpoint(opt.checkpoint, opt.end);
    }

    double total_seeds = (double)(next - opt.start);
    if (total_seeds < 0) total_seeds = -total_seeds;
    printf("throughput: %.2f seeds/s (%.2fM seeds/s)\n", total_seeds / elapsed, total_seeds / elapsed / 1e6);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_found);
    if (opt.cont) {
        cudaFree(d_found_count);
        cudaFree(d_seeds);
        cudaFree(d_wins_x);
        cudaFree(d_wins_z);
        cudaFree(d_wins_c);
    }
    cudaFree(d_result_seed);
    cudaFree(d_result_x);
    cudaFree(d_result_z);
    cudaFree(d_result_count);
    return found ? 0 : 2;
}
