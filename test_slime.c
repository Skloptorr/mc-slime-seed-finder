#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

// ============================================================
// 史莱姆区块判定函数
// 完全模拟Minecraft Java版算法（含32位int溢出截断）
// 参考：Minecraft Wiki - Slime chunk (Java Edition)
// ============================================================
bool isSlimeChunk(int64_t worldSeed, int chunkX, int chunkZ) {
    // 第一步：构建Random种子（所有int运算是32位溢出截断）
    int t1   = chunkX * chunkX * 0x4c1906;      // 32位int自动溢出
    int t2   = chunkX * 0x5ac0db;                // 32位int自动溢出
    int t3_z = chunkZ * chunkZ;                   // chunkZ² 32位溢出截断
    int64_t t3 = (int64_t)t3_z * 0x4307a7L;      // 截断后转64位再乘
    int t4   = chunkZ * 0x5f24f;                  // 32位int自动溢出

    int64_t seed = (worldSeed + t1 + t2 + t3 + t4) ^ 0x3ad8025fL;

    // 第二步：Java Random.setSeed()
    seed = (seed ^ 0x5DEECE66DL) & 0xFFFFFFFFFFFFL;

    // 第三步：Java Random.next(31)
    // （注：26.2 实测 MC 用拒绝采样 nextInt(10)，差异仅 3.7e-9/判定，决定不实现——见 DEVELOPMENT_LOG.md 2026-08-14）
    seed = (seed * 0x5DEECE66DL + 0xBL) & 0xFFFFFFFFFFFFL;
    int bits = (int)(seed >> 17);

    return (bits % 10) == 0;
}

int main(int argc, char *argv[]) {
    int64_t seed;
    int minX, maxX, minZ, maxZ;

    if (argc >= 5) {
        seed = atoll(argv[1]);
        minX = atoi(argv[2]);
        maxX = atoi(argv[3]);
        minZ = atoi(argv[4]);
        maxZ = atoi(argv[5]);
    } else {
        // 默认参数
        seed = 12345L;
        minX = -200; maxX = 200;
        minZ = -200; maxZ = 200;
    }

    printf("=== MC 史莱姆区块查找 ===\n");
    printf("种子: %lld\n", (long long)seed);
    printf("范围: chunkX [%d, %d], chunkZ [%d, %d]\n", minX, maxX, minZ, maxZ);
    printf("转换: 方块坐标 = 区块坐标 × 16\n\n");

    int total = (maxX - minX + 1) * (maxZ - minZ + 1);
    int count = 0;

    // 扫描所有区块
    for (int z = minZ; z <= maxZ; z++) {
        for (int x = minX; x <= maxX; x++) {
            if (isSlimeChunk(seed, x, z)) {
                if (count < 50) {  // 先显示前50个
                    printf("区块 (%4d, %4d)  → 方块 (%4d, %4d)\n",
                           x, z, x * 16, z * 16);
                } else if (count == 50) {
                    printf("... (后续结果省略，共找到超过50个)\n");
                }
                count++;
            }
        }
    }

    printf("\n=== 统计 ===\n");
    printf("总区块数: %d\n", total);
    printf("史莱姆区块: %d (%.2f%%)\n", count, 100.0 * count / total);
    printf("理论期望: ~10%%\n");

    // 额外：检查5x5窗口密集度
    printf("\n=== 5x5窗口密集度检查 (≥19个) ===\n");
    int dense_count = 0;
    int best_sum = 0;
    int best_cx = 0, best_cz = 0;

    // 预计算所有区块的判定结果到数组
    int grid[401][401];
    for (int z = -200; z <= 200; z++) {
        for (int x = -200; x <= 200; x++) {
            grid[z + 200][x + 200] = isSlimeChunk(seed, x, z) ? 1 : 0;
        }
    }

    // 扫描所有5x5窗口
    for (int cz = -200; cz <= 196; cz++) {
        for (int cx = -200; cx <= 196; cx++) {
            int sum = 0;
            for (int dz = 0; dz < 5; dz++) {
                for (int dx = 0; dx < 5; dx++) {
                    sum += grid[cz + dz + 200][cx + dx + 200];
                }
            }
            if (sum >= 19) {
                if (dense_count < 5) {
                    printf("密集窗口: 区块 (%d,%d) 到 (%d,%d)  →  5x5内 %d个史莱姆区块 (≥19 ✓)\n",
                           cx, cz, cx + 4, cz + 4, sum);
                }
                dense_count++;
            }
            if (sum > best_sum) {
                best_sum = sum;
                best_cx = cx;
                best_cz = cz;
            }
        }
    }

    printf("\n总计 %d 个密集窗口 (5x5内≥19)\n", dense_count);
    printf("最佳窗口: 区块 (%d,%d) 到 (%d,%d)  →  %d个史莱姆区块\n",
           best_cx, best_cz, best_cx + 4, best_cz + 4, best_sum);

    return 0;
}
