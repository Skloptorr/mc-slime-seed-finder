#!/usr/bin/env python3
"""MC 史莱姆区块种子搜索器 — CPU 参考实现（numpy 向量化）

完全模拟 Minecraft Java 版史莱姆区块判定算法（含 32 位 int 溢出截断），
用二维前缀和在 O(1) 内求任意 5×5 窗口的史莱姆区块数。

用法:
    py slime_search.py --start 0 --end 100000
    py slime_search.py --start 0 --end 100000 --window 5 --threshold 19 --grid 401

依赖: numpy (>= 1.20)
"""
import argparse
import time

import numpy as np

# ---- 常量（与 Java 版源码一致） ----
A_MASK48 = np.uint64(0xFFFFFFFFFFFF)          # (1L << 48) - 1
MULT = np.uint64(0x5DEECE66D)                 # LCG 乘数
ADD = np.uint64(0xB)                          # LCG 加数
XOR_SEED = np.uint64(0x3ad8025f)              # 种子异或


def is_slime_chunk_vec(world_seed: int, chunk_x: np.ndarray, chunk_z: np.ndarray) -> np.ndarray:
    """向量化史莱姆区块判定。

    关键语义（与 Java 版一致）:
      * 所有 int 运算是 32 位有符号溢出截断（wrap-around）
      * t3 = (int)(chunkZ^2) * 0x4307a7L —— 先截断成 32 位再提升为 long 乘
      * 第三步 LCG 乘法用 uint64 承载：乘积超 64 位只影响高位，
        低 48 位完整保留（64 > 48），与 Java 的 (mod 2^48) 等价
    """
    x = chunk_x.astype(np.int32)
    z = chunk_z.astype(np.int32)

    t1 = np.int32(np.int32(x * x) * np.int32(0x4c1906))          # chunkX^2 * 0x4c1906 (int32 溢出)
    t2 = np.int32(x * np.int32(0x5ac0db))                         # chunkX * 0x5ac0db
    t3_z = np.int32(z * z)                                       # chunkZ^2 (int32 溢出截断)
    t3 = t3_z.astype(np.int64) * np.int64(0x4307a7)              # (int)截断后提升 long 再乘
    t4 = np.int32(z * np.int32(0x5f24f))                          # chunkZ * 0x5f24f

    seed = (np.int64(world_seed) + t1.astype(np.int64) + t2.astype(np.int64)
            + t3 + t4.astype(np.int64)) ^ np.int64(0x3ad8025f)

    # Java Random.setSeed()
    seed_u = (seed.astype(np.uint64) ^ np.uint64(0x5DEECE66D)) & A_MASK48
    # Java Random.next(31)
    # （注：26.2 实测 MC 用拒绝采样 nextInt(10)，差异仅 3.7e-9/判定，
    #   统计无感且实现后 GPU 性能 -25%，决定不实现——详见 DEVELOPMENT_LOG.md 2026-08-14）
    seed_u = (seed_u * MULT + ADD) & A_MASK48
    bits = seed_u >> np.uint64(17)                                # 48-31=17

    return (bits % np.uint64(10)) == np.uint64(0)


def build_prefix(world_seed: int, grid: int) -> np.ndarray:
    """构建 401×401 网格的二维前缀和（含 +1 边框）。

    返回 shape (grid+1, grid+1)：prefix[i+1][j+1] = 以 (0,0)-(i,j) 为对角
    的矩形内史莱姆区块总数。区块坐标 (cx, cz) 映射到网格 (cz+offset, cx+offset)。
    """
    offset = grid // 2  # 200
    coords = np.arange(-offset, offset + 1, dtype=np.int64)
    cx, cz = np.meshgrid(coords, coords)
    slime = is_slime_chunk_vec(world_seed, cx, cz)

    prefix = np.zeros((grid + 1, grid + 1), dtype=np.int64)
    np.cumsum(slime.astype(np.int64), axis=0, out=prefix[1:, 1:])
    np.cumsum(prefix[1:, 1:], axis=1, out=prefix[1:, 1:])
    return prefix


def window_sums(prefix: np.ndarray, window: int) -> np.ndarray:
    """所有 window×window 窗口和（O(1) 前缀和查询）。

    返回 shape (G-w+1, G-w+1)，元素 = 以 (i,j) 为左上角的窗口内史莱姆区块数。
    """
    return (prefix[window:, window:] - prefix[:-window, window:]
            - prefix[window:, :-window] + prefix[:-window, :-window])


def find_dense_seed(start: int, end: int, window: int = 5,
                    threshold: int = 19, grid: int = 401,
                    stats: bool = False) -> dict | None:
    """顺序枚举 [start, end) 种子，返回第一个满足条件的种子信息。

    返回: {"seed", "win_x", "win_z", "count", "windows_checked"} 或 None
    win_x/win_z 为窗口左上角区块坐标；count 为窗口内史莱姆区块数。
    """
    t0 = time.perf_counter()
    checked = 0
    for seed in range(start, end):
        checked += 1
        prefix = build_prefix(seed, grid)
        wins = window_sums(prefix, window)
        hit = np.argwhere(wins >= threshold)
        if hit.size:
            zi, xi = hit[0]  # 行优先 = z 外层升序、x 内层升序
            offset = grid // 2
            result = {
                "seed": seed,
                "win_x": int(xi) - offset,
                "win_z": int(zi) - offset,
                "count": int(wins[zi, xi]),
                "windows_checked": checked,
            }
            if stats:
                result["elapsed_s"] = round(time.perf_counter() - t0, 3)
            return result
    if stats:
        return {"windows_checked": checked, "elapsed_s": round(time.perf_counter() - t0, 3)}
    return None


def verify_known_seed(seed: int = 12345, grid: int = 401) -> dict:
    """回归验证：已知种子 12345 的完整统计（开发日志 v0.1 已确认 16,111 个 / 10.02%）。"""
    prefix = build_prefix(seed, grid)
    total = int(prefix[-1, -1])
    wins = window_sums(prefix, 5)
    best = int(wins.max())
    zi, xi = np.unravel_index(np.argmax(wins), wins.shape)
    offset = grid // 2
    return {
        "seed": seed,
        "total_slime": total,
        "ratio": round(total / (grid * grid), 4),
        "best_window": best,
        "best_x": int(xi) - offset,
        "best_z": int(zi) - offset,
    }


def main():
    ap = argparse.ArgumentParser(description="MC 史莱姆区块种子搜索（CPU 参考实现）")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--threshold", type=int, default=19)
    ap.add_argument("--grid", type=int, default=401)
    ap.add_argument("--stats", action="store_true", help="输出耗时统计")
    args = ap.parse_args()

    result = find_dense_seed(args.start, args.end, args.window,
                             args.threshold, args.grid, args.stats)
    if result and "seed" in result:
        print(f"找到密集种子: seed={result['seed']}")
        print(f"  窗口左上角区块: ({result['win_x']}, {result['win_z']})")
        print(f"  窗口右下角区块: ({result['win_x'] + args.window - 1}, {result['win_z'] + args.window - 1})")
        print(f"  5×5 窗口内史莱姆区块: {result['count']} (阈值 {args.threshold})")
        print(f"  已检查种子数: {result['windows_checked']}")
        if "elapsed_s" in result:
            print(f"  耗时: {result['elapsed_s']}s")
    else:
        print(f"[{args.start}, {args.end}) 范围内未找到满足条件的种子")
        if result:
            print(f"  已检查种子数: {result['windows_checked']}，耗时 {result['elapsed_s']}s")


if __name__ == "__main__":
    main()
