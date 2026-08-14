# Spec: MC 史莱姆区块种子搜索器 — 初步完善

> 版本：v0.2（2026-08-14）
> 前置：项目企划及对话记录.md（v0.1，算法已验证）

## Objective

在 Minecraft Java 版 2^48 种子空间中，寻找满足以下条件的种子：
- **5×5 区块窗口**（80×80 方块）内存在 **≥19 个史莱姆区块**（概率 76%，正常期望仅 2.5 个，极端稀有）
- 搜索范围：以 (0,0) 为中心，(-200,-200) ~ (200,200) 的 401×401 区块网格

本次「初步完善」的目标：从「算法验证完成但只有 CPU 小验证」推进到**可实际搜索**——
提供 CPU 参考实现（可验证的正确性基准）+ CUDA GPU 搜索实现（实际算力）+ 一致性验证 + 性能基线。

成功标准：GPU 版与 CPU 参考版在小范围种子空间内结果完全一致；GPU 版能批量搜索并断点续传。

## Tech Stack

- GPU：NVIDIA RTX 4070 SUPER（compute capability 8.9，12GB）
- CUDA Toolkit 13.3（nvcc V13.3.73，`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`）
- Host 编译器：MSVC 14.44（VS 2022 Build Tools）+ Windows SDK 10.0.26100
- CPU 参考：Python 3.14 + numpy 2.4.3（向量化 isSlimeChunk 判定）
- 版本控制：Gitea `root/mc-slime-seed-finder`（自建 Gitea 仓库）

## Commands

```
# Python 参考实现（CPU，向量化）：搜索 [start, end) 内第一个满足条件的种子
py slime_search.py --start 0 --end 100000
# 可选参数：--window 5 --threshold 19 --grid 401 --stats（输出统计）

# CUDA 搜索（GPU）：
# 编译（需先进入 VS 开发者环境或指定 -ccbin）
nvcc -O3 -arch=sm_89 gpu/slime_search.cu -o gpu/slime_search.exe
# 运行
gpu\slime_search.exe --start 0 --end 100000000 --checkpoint checkpoint.dat --results results.txt

# 一致性测试（CPU 参考 vs GPU）
py tests/test_consistency.py

# 回归测试（已知种子 12345，验证算法正确性）
py -c "import slime_search as s; print(s.verify_known_seed())"
```

## Project Structure

```
MC史莱姆区块计算器/
├── README.md                 → 项目概览、构建/运行说明
├── SPEC.md                   → 本文档
├── 项目企划及对话记录.md      → v0.1 企划（算法原理、硬件方案、原始对话）
├── 执行清单.md               → 任务清单（勾选完成项）
├── DEVELOPMENT_LOG.md        → 开发日志（本次完善记录）
├── check_slime.py            → v0.1 小范围验证脚本（保留）
├── test_slime.c              → v0.1 CPU C 验证程序（保留）
├── slime_search.py           → CPU 参考实现（numpy 向量化，CLI 入口）
├── gpu/
│   └── slime_search.cu       → CUDA 内核 + 主机调度（搜索主程序）
└── tests/
    └── test_consistency.py   → CPU vs GPU 一致性测试
```

## Code Style

- Python：PEP8；函数级 docstring；类型标注（int/long 语义与 Java 一致处写注释）
- CUDA：`isSlimeChunk` 设备函数与 C 参考版逐行一致（32 位溢出语义）；内核名 `search_kernel`；
  主机端参数解析用简单 `--key value` 格式，不引第三方库
- 关键算法语义（32 位 int 溢出截断）必须在实现处注释标注

## Testing Strategy

- **回归**：seed=12345，401×401 网格 = 16,111 个史莱姆区块（10.02%）；最佳 5×5 窗口 11 个（不达标）——验证算法正确性
- **一致性**：`tests/test_consistency.py` 对比 CPU 参考与 GPU 输出——小范围 [0, N) 找到的第一个密集种子（种子号、窗口坐标、窗口计数）必须完全一致
- **性能基线**：GPU 版报告 seeds/s 吞吐（分批计时），记录于开发日志

## Boundaries

- **Always**：测试通过才提交；32 位溢出语义不被优化破坏；提交前跑一致性测试
- **Ask first**：修改目标条件（窗口/阈值/网格大小）；换搜索策略（随机抽样/多 GPU）；改 Gitea 仓库可见性
- **Never**：向记忆文件写入 token/密码；删除 v0.1 验证代码（保留回归价值）；在 GPU 长任务时锁屏导致断连无续传

## Success Criteria

- [ ] `nvcc -O3 -arch=sm_89` 编译通过
- [ ] seed=12345 回归：GPU/CPU 均输出 16,111 个史莱姆区块（±0 容差）
- [ ] 一致性测试：GPU 与 CPU 参考在 [0, 10000) 范围内找到的第一个密集种子完全一致
- [ ] GPU 吞吐基线 ≥ 100 万 seeds/s（记录实际值）
- [ ] 断点续传：中断后从 checkpoint 继续，无重复无遗漏
- [ ] 全部文件推送 Gitea（master/main 分支干净）

## Open Questions

- 首个搜索目标范围（首轮搜多少种子）——待一致性验证后与项目负责人确认
- 找到第一个密集种子后是否继续搜多个候选——默认只找第一个，找到即停
