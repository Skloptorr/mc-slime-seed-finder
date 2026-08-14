// ============================================================
// MC 史莱姆区块种子搜索器 — CUDA GPU 实现
// 目标: 在 (-200,-200)~(200,200) 的 401×401 区块网格内,
//       寻找 5×5 窗口 ≥19 个史莱姆区块的密集种子（找到即停）
// 算法: isSlimeChunk（模拟 Java 32 位溢出）+ 滑动窗口行前缀和
// 编译: nvcc -O3 -arch=sm_89 gpu/slime_search.cu -o gpu/slime_search.exe
// 运行: gpu\slime_search.exe --start 0 --end 100000000
//                            [--checkpoint cp.dat] [--results res.txt]
// ============================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GRID 401
#define OFFSET 200
#define WINDOW 5
#define THRESHOLD 19
#define DEFAULT_BATCH 1000000ULL      // 每批种子数（一次内核 launch）

// ------------------------------------------------------------
// 史莱姆区块判定（与 test_slime.c 逐行一致，已验证 seed=12345 得 16111 个）
// 注意: 32 位 int 溢出截断是必须的；LCG 乘法溢出只影响高位，低 48 位保留
// ------------------------------------------------------------
__device__ __forceinline__ bool isSlimeChunk(long long worldSeed, int chunkX, int chunkZ) {
    int t1 = chunkX * chunkX * 0x4c1906;        // 32 位 int 自动溢出
    int t2 = chunkX * 0x5ac0db;
    int t3_z = chunkZ * chunkZ;
    long long t3 = (long long)t3_z * 0x4307a7LL; // 截断后提升 long 再乘
    int t4 = chunkZ * 0x5f24f;

    long long seed = (worldSeed + t1 + t2 + t3 + t4) ^ 0x3ad8025fLL;

    // Java Random.setSeed()
    seed = (seed ^ 0x5DEECE66DLL) & 0xFFFFFFFFFFFFLL;
    // Java Random.next(31)
    // （注：26.2 实测 MC 用拒绝采样 nextInt(10)，差异仅 8/2^31≈3.7e-9/判定，
    //   统计无感且实现后性能 -25%，决定不实现——详见 DEVELOPMENT_LOG.md 2026-08-14）
    seed = (seed * 0x5DEECE66DLL + 0xBLL) & 0xFFFFFFFFFFFFLL;
    int bits = (int)(seed >> 17);
    return (bits % 10) == 0;
}

// ------------------------------------------------------------
// 搜索内核: 每个线程负责一串种子（grid-stride）
// 内存: 每线程 ~2.8KB 局部数组（uint8 5 行历史 + int16 列和）
// 算法: 增量维护 5 行窗口列和 col_win，再做 5 宽滑动窗口检查
//       相比逐窗口 10 次读，访问量降 ~5 倍
// 提前退出: volatile 全局 found 标志 + atomicCAS 抢占结果槽
// ------------------------------------------------------------
__global__ void search_kernel(unsigned long long seed_start, unsigned long long seed_end,
                              unsigned long long step,
                              volatile int* d_found,
                              unsigned long long* d_result_seed,
                              int* d_result_x, int* d_result_z, int* d_result_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (*d_found) return;

    unsigned char rows[WINDOW][GRID];  // 5 行历史 slime 值（0/1）
    short col_win[GRID];               // 最近 5 行的逐列和（最大 5）

    for (unsigned long long seed = seed_start + tid; seed < seed_end; seed += step) {
        if (*d_found) return;

        for (int i = 0; i < GRID; i++) col_win[i] = 0;  // 每种子重置列和
        int filled = 0;
        int hit = 0;
        int hit_sum = 0, hit_x = 0, hit_z = 0;

        for (int z = -OFFSET; z <= OFFSET && !hit; z++) {
            int slot = filled % WINDOW;
            for (int c = 0; c < GRID; c++) {
                int x = c - OFFSET;
                unsigned char s = isSlimeChunk((long long)seed, x, z) ? 1u : 0u;
                unsigned char old = rows[slot][c];
                rows[slot][c] = s;
                if (filled < WINDOW) {
                    col_win[c] = (short)(col_win[c] + s);          // 前 5 行只加
                } else {
                    col_win[c] = (short)(col_win[c] + s - old);    // 加新行减最旧行
                }
            }
            filled++;

            if (filled >= WINDOW) {
                // 5 宽滑动窗口检查（col_win 的连续 5 列和）
                int win_sum = 0;
                for (int c = 0; c < WINDOW; c++) win_sum += col_win[c];
                if (win_sum >= THRESHOLD) {
                    hit = 1; hit_sum = win_sum; hit_x = 0 - OFFSET; hit_z = z - WINDOW + 1;
                    break;
                }
                for (int cx = 1; cx <= GRID - WINDOW; cx++) {
                    win_sum += col_win[cx + WINDOW - 1] - col_win[cx - 1];
                    if (win_sum >= THRESHOLD) {
                        hit = 1; hit_sum = win_sum; hit_x = cx - OFFSET; hit_z = z - WINDOW + 1;
                        break;
                    }
                }
            }
        }

        if (hit) {
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

// ------------------------------------------------------------
// 验证内核: 单种子完整网格统计（与 CPU 参考 verify_known_seed 对比用）
// 输出: 史莱姆区块总数 / 最佳 5×5 窗口 / 最佳窗口左上角坐标
// ------------------------------------------------------------
__global__ void verify_kernel(long long seed, int* d_total, int* d_best,
                              int* d_best_x, int* d_best_z) {
    unsigned char rows[WINDOW][GRID];
    short col_win[GRID];
    for (int i = 0; i < GRID; i++) col_win[i] = 0;
    int total = 0, best = 0, best_x = 0, best_z = 0;
    int filled = 0;
    for (int z = -OFFSET; z <= OFFSET; z++) {
        int slot = filled % WINDOW;
        for (int c = 0; c < GRID; c++) {
            int x = c - OFFSET;
            unsigned char s = isSlimeChunk(seed, x, z) ? 1u : 0u;
            unsigned char old = rows[slot][c];
            rows[slot][c] = s;
            total += s;
            col_win[c] = (short)(col_win[c] + s - (filled >= WINDOW ? old : 0));
        }
        filled++;
        if (filled >= WINDOW) {
            int win_sum = 0;
            for (int c = 0; c < WINDOW; c++) win_sum += col_win[c];
            if (win_sum > best) { best = win_sum; best_x = 0 - OFFSET; best_z = z - WINDOW + 1; }
            for (int cx = 1; cx <= GRID - WINDOW; cx++) {
                win_sum += col_win[cx + WINDOW - 1] - col_win[cx - 1];
                if (win_sum > best) { best = win_sum; best_x = cx - OFFSET; best_z = z - WINDOW + 1; }
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
    unsigned long long start = 0;
    unsigned long long end = 100000000ULL;
    std::string checkpoint;
    std::string results;
    bool verify = false;
    long long verify_seed = 0;
};

static bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "缺少参数值: %s\n", name); return nullptr; }
            return argv[++i];
        };
        if (a == "--start") { const char* v = need("--start"); if (!v) return false; opt.start = strtoull(v, nullptr, 10); }
        else if (a == "--end") { const char* v = need("--end"); if (!v) return false; opt.end = strtoull(v, nullptr, 10); }
        else if (a == "--checkpoint") { const char* v = need("--checkpoint"); if (!v) return false; opt.checkpoint = v; }
        else if (a == "--results") { const char* v = need("--results"); if (!v) return false; opt.results = v; }
        else if (a == "--verify") { const char* v = need("--verify"); if (!v) return false; opt.verify = true; opt.verify_seed = strtoll(v, nullptr, 10); }
        else { fprintf(stderr, "未知参数: %s\n", a.c_str()); return false; }
    }
    if (opt.end <= opt.start) { fprintf(stderr, "--end 必须大于 --start\n"); return false; }
    return true;
}

static bool load_checkpoint(const std::string& path, unsigned long long& start) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return false;
    if (fscanf(f, "%llu", &start) != 1) { fclose(f); return false; }
    fclose(f);
    return true;
}

static void save_checkpoint(const std::string& path, unsigned long long next) {
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "w");
    if (f) { fprintf(f, "%llu\n", next); fclose(f); }
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        fprintf(stderr, "用法: %s --start N --end N [--checkpoint FILE] [--results FILE] | --verify SEED\n", argv[0]);
        return 1;
    }

    int device = 0;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) { fprintf(stderr, "cudaSetDevice 失败: %s\n", cudaGetErrorString(err)); return 1; }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // ---- verify 模式：单种子网格统计 ----
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

    // 断点续传：checkpoint 存在则覆盖 --start
    if (!opt.checkpoint.empty() && load_checkpoint(opt.checkpoint, opt.start)) {
        printf("[checkpoint] 从 %llu 继续\n", opt.start);
        if (opt.start >= opt.end) {
            printf("checkpoint 已到终点，无需继续\n");
            return 0;
        }
    }

    printf("GPU: %s (CC %d.%d, %d SM)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // 线程布局: 每个 SM 若干 block，覆盖全部核心
    int block = 256;
    int sm_count = prop.multiProcessorCount;
    int grid = sm_count * 4;  // 8.9 每 SM 最多 2048 线程 → 4 block/SM 稳
    unsigned long long step = (unsigned long long)block * grid;

    // 设备内存
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

    int h_found = 0;
    unsigned long long h_seed = 0;
    int h_x = 0, h_z = 0, h_count = 0;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    unsigned long long next = opt.start;
    unsigned long long found_at = 0;
    double elapsed = 0.0;
    bool found = false;
    long long batch_count = 0;

    printf("搜索范围: [%llu, %llu)  step=%llu  batch=%llu  目标: %dx%d 窗口 ≥%d\n",
           opt.start, opt.end, step, DEFAULT_BATCH, WINDOW, WINDOW, THRESHOLD);
    printf("开始搜索...\n");

    cudaEventRecord(t0);
    while (next < opt.end) {
        unsigned long long batch_end = next + DEFAULT_BATCH < opt.end ? next + DEFAULT_BATCH : opt.end;

        cudaMemset(d_found, 0, sizeof(int));
        search_kernel<<<grid, block>>>(next, batch_end, step, d_found,
                                       d_result_seed, d_result_x, d_result_z, d_result_count);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { fprintf(stderr, "内核执行失败: %s\n", cudaGetErrorString(err)); return 1; }

        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_found) {
            cudaMemcpy(&h_seed, d_result_seed, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_x, d_result_x, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_z, d_result_z, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_count, d_result_count, sizeof(int), cudaMemcpyDeviceToHost);
            found = true;
            found_at = next;
            next = batch_end;  // 标记进度
            break;
        }

        next = batch_end;
        save_checkpoint(opt.checkpoint, next);
        batch_count++;
        if (batch_count % 10 == 0) {
            cudaEventRecord(t1);
            cudaEventSynchronize(t1);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, t0, t1);
            double sec = ms / 1000.0;
            double rate = sec > 0 ? (double)next / sec : 0.0;
            printf("  已搜索 %llu 种子 (%.2fM seeds/s, %.1fs)\n", next, rate / 1e6, sec);
        }
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    elapsed = ms / 1000.0;

    if (found) {
        printf("\n=== 找到密集种子 ===\n");
        printf("seed: %llu\n", h_seed);
        printf("窗口左上角区块: (%d, %d)\n", h_x, h_z);
        printf("窗口右下角区块: (%d, %d)\n", h_x + WINDOW - 1, h_z + WINDOW - 1);
        printf("5×5 窗口内史莱姆区块: %d (阈值 %d)\n", h_count, THRESHOLD);
        printf("检查种子数: %llu\n", found_at - opt.start);
        printf("耗时: %.2fs\n", elapsed);
        if (!opt.results.empty()) {
            FILE* f = fopen(opt.results.c_str(), "w");
            if (f) {
                fprintf(f, "seed=%llu\nwin_x=%d\nwin_z=%d\ncount=%d\n", h_seed, h_x, h_z, h_count);
                fclose(f);
                printf("结果已写入: %s\n", opt.results.c_str());
            }
        }
    } else {
        printf("\n[%llu, %llu) 未找到满足条件的种子\n", opt.start, opt.end);
        save_checkpoint(opt.checkpoint, opt.end);
    }

    double total_seeds = (double)(next - opt.start);
    printf("总吞吐: %.2f seeds/s (%.2fM seeds/s)\n", total_seeds / elapsed, total_seeds / elapsed / 1e6);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_found);
    cudaFree(d_result_seed);
    cudaFree(d_result_x);
    cudaFree(d_result_z);
    cudaFree(d_result_count);
    return found ? 0 : 2;
}
