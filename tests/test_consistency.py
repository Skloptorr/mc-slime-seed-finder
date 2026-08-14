#!/usr/bin/env python3
"""CPU 参考 vs GPU 实现一致性测试。

验证 GPU (slime_search.cu) 与 CPU 参考 (slime_search.py) 完全一致：
  1. 已知种子回归（seed=12345）：网格统计 + 最佳窗口位置
  2. 随机种子抽查（固定随机种子，可复现）：5 个种子的完整网格统计
  3. 搜索一致性：小范围 [0, N) 内 GPU 与 CPU 找到的结果一致（含都未找到）

用法: py tests/test_consistency.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import slime_search as ref  # noqa: E402

GPU_EXE = ROOT / "gpu" / "slime_search.exe"


def gpu_verify(seed: int) -> dict:
    out = subprocess.run([str(GPU_EXE), "--verify", str(seed)],
                         capture_output=True, text=True, timeout=60).stdout
    d = {}
    for line in out.splitlines():
        for tok in line.split():
            if tok.startswith("total_slime="):
                d["total_slime"] = int(tok.split("=")[1])
            elif tok.startswith("best_window="):
                d["best_window"] = int(tok.split("=")[1])
            elif tok.startswith("best_x="):
                d["best_x"] = int(tok.split("=")[1])
            elif tok.startswith("best_z="):
                d["best_z"] = int(tok.split("=")[1])
    return d


def test_known_seed():
    cpu = ref.verify_known_seed(12345)
    gpu = gpu_verify(12345)
    assert cpu["total_slime"] == gpu["total_slime"], f"total: CPU={cpu} GPU={gpu}"
    assert cpu["best_window"] == gpu["best_window"], f"best: CPU={cpu} GPU={gpu}"
    assert cpu["best_x"] == gpu["best_x"] and cpu["best_z"] == gpu["best_z"], \
        f"pos: CPU=({cpu['best_x']},{cpu['best_z']}) GPU=({gpu['best_x']},{gpu['best_z']})"
    print(f"[PASS] 已知种子 12345: total={cpu['total_slime']} best={cpu['best_window']} "
          f"pos=({cpu['best_x']},{cpu['best_z']})")


def test_random_seeds():
    rng = np.random.default_rng(20260814)
    seeds = rng.integers(0, 2**32, size=5).tolist()
    for seed in seeds:
        cpu = ref.verify_known_seed(int(seed))
        gpu = gpu_verify(int(seed))
        assert cpu["total_slime"] == gpu["total_slime"], \
            f"seed={seed} total: CPU={cpu['total_slime']} GPU={gpu['total_slime']}"
        assert cpu["best_window"] == gpu["best_window"], \
            f"seed={seed} best: CPU={cpu['best_window']} GPU={gpu['best_window']}"
        assert cpu["best_x"] == gpu["best_x"] and cpu["best_z"] == gpu["best_z"], \
            f"seed={seed} pos: CPU=({cpu['best_x']},{cpu['best_z']}) GPU=({gpu['best_x']},{gpu['best_z']})"
    print(f"[PASS] 随机 5 种子网格统计完全一致: {seeds}")


def test_search_consistency():
    n = 2000  # 小范围（CPU 参考可承受）
    cpu = ref.find_dense_seed(0, n)
    out = subprocess.run([str(GPU_EXE), "--start", "0", "--end", str(n)],
                         capture_output=True, text=True, timeout=120).stdout
    gpu_found = "找到密集种子" in out or "seed:" in out
    assert (cpu is None) == (not gpu_found), \
        f"搜索不一致: CPU={'找到' if cpu else '未找到'} GPU={'找到' if gpu_found else '未找到'}"
    if cpu:
        gpu_seed = int([l for l in out.splitlines() if l.startswith("seed:")][0].split(":")[1])
        assert cpu["seed"] == gpu_seed, f"种子不一致: CPU={cpu['seed']} GPU={gpu_seed}"
        assert cpu["win_x"] == int([l for l in out.splitlines() if "窗口左上角" in l][0].split("(")[1].split(",")[0])
    print(f"[PASS] 搜索一致性 [{0}, {n}): CPU/GPU 一致（{'找到 seed=' + str(cpu['seed']) if cpu else '均未找到'}）")


def main():
    print("=== MC 史莱姆区块搜索器 一致性测试 ===")
    test_known_seed()
    test_random_seeds()
    test_search_consistency()
    print("\n全部通过 ✓")


if __name__ == "__main__":
    main()
