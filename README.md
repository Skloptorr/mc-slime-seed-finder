# MC Slime Seed Finder — 史莱姆区块密集种子搜索器

在 Minecraft Java 版 2^48 种子空间中，寻找满足「401×401 区块网格内存在 5×5 窗口 ≥19 个史莱姆区块」的密集种子。

- **搜索范围**：以 (0,0) 为中心，(-200,-200) ~ (200,200) 共 401×401 区块
- **目标**：5×5 窗口（25 区块）内 ≥19 个史莱姆区块（概率 76%；正常期望仅 2.5 个，极端稀有）
- **实测成果**：seed = `157140467`（窗口 (194,48) 恰好 19 个，CPU 独立复核通过）

## 特性

- ✅ 完全模拟 Java 版史莱姆区块判定算法（含 **32 位 int 溢出截断**，已对 26.2 版本字节码实测确认）
- ✅ **bitpack 寄存器优化**：行数据位打包进寄存器（5 行 × 7×64bit），零 local memory 访问，吞吐 **103k → 1.26M seeds/s（12 倍提升）**
- ✅ GPU 搜索（CUDA，sm_89+）、CPU 参考实现（numpy 向量化）、C 验证程序
- ✅ 无限搜索 / 指定范围、正负方向搜索、断点续传（checkpoint）
- ✅ 阈值 n 可调（1-25，运行时参数）
- ✅ Python 管理界面（tkinter）：参数/开始停止/实时进度/日志文件
- ✅ 找种子即停（默认）或找到继续（`--continue` 实时输出全部命中）

## 算法原理

```c
// Java 版史莱姆区块判定（所有 int 运算是 32 位有符号溢出截断！）
int t1   = chunkX * chunkX * 0x4c1906;         // 32 位 int 自动溢出
int t2   = chunkX * 0x5ac0db;
int t3_z = chunkZ * chunkZ;
long t3  = (long)t3_z * 0x4307a7L;             // 截断后提升 long 再乘
int t4   = chunkZ * 0x5f24f;
long seed = (worldSeed + t1 + t2 + t3 + t4) ^ 0x3ad8025fL;
// Java Random.setSeed() + next(31)，判定 next(31) % 10 == 0
```

**关键点**（最容易错的地方）：
- 32 位溢出截断是必须的——全程 64 位计算会得到完全不同的结果
- `chunkZ²` 是**先截断成 32 位再提升 long 乘**，不是全程 64 位
- 随机数本质是 `java.util.Random` 的 LCG（0x5DEECE66D + 0xB，48 位掩码）
- **26.2 兼容性**：已对 26.2-server.jar 字节码实测——种子公式与 LCG 逐位一致；唯一差异是 MC 的 `nextInt(10)` 带拒绝采样（差异率 ~3.7e-9，统计无感，决定不实现，详见 `docs/DEVELOPMENT_LOG.md`）

**性能优化（bitpack 寄存器版）**：
- 朴素实现把行数据放 local memory（物理在 VRAM），每种子 ~90 万次内存访问（ptxas: 2808B 栈帧）
- bitpack 版把 5 行 × 401 位打包进 35 个 uint64 寄存器，窗口检查用 `__popcll` 单指令，零内存访问（ptxas: 280B 栈帧）
- 实测：RTX 4070 SUPER 上 **1.26-1.39M seeds/s**

## 构建

### CUDA GPU 版（需要 nvcc + MSVC/VS Build Tools）

```bash
# slime_search_bp.cu 是主力优化版（bitpack）
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gpu/slime_search_bp.cu -o gpu/slime_search_bp.exe
# 原版（local memory 版，参考）
nvcc -O3 -arch=sm_89 -Xcompiler /utf-8 gpu/slime_search.cu -o gpu/slime_search.exe
```

### C 验证程序（gcc/clang）

```bash
gcc -O2 test_slime.c -o test_slime
```

### Python（CPU 参考实现）

```bash
pip install -r requirements.txt   # numpy
```

## 运行

### 命令行（GPU）

```bash
# 指定范围搜索 [0, 1亿)，找到即停
gpu\slime_search_bp.exe --start 0 --end 100000000 --threshold 19 --checkpoint cp.dat

# 无限搜索（不传 --end）
gpu\slime_search_bp.exe --start 0 --threshold 19 --checkpoint cp.dat

# 负方向（从 0 往 -1,-2,...）
gpu\slime_search_bp.exe --start 0 --neg --threshold 19

# 找到继续：输出全部命中，不暂停
gpu\slime_search_bp.exe --start 0 --end 100000000 --threshold 19 --continue --results res.txt

# 回归验证（seed=12345 应得 16,111 个史莱姆区块 / 10.02% / 最佳窗口 11）
gpu\slime_search_bp.exe --verify 12345
```

### Python CPU 参考

```bash
python slime_search.py --start 0 --end 100000 --stats
python -c "import slime_search as s; print(s.verify_known_seed())"
```

### 管理界面（GUI）

```bash
python gui.py
```

参数（start/end/方向/阈值 n/找到即暂停开关）可视化操作，实时进度，会话日志写入 `logs/`（文件名含时间戳/范围/阈值/结果）。

## 测试

```bash
# 一致性：CPU 参考 vs GPU（小范围）
python tests/test_consistency.py

# 回归：seed=12345（C / Python / GPU 三路一致：16,111 / 10.02% / 最佳 11）
```

## 目录结构

```
├── README.md
├── LICENSE
├── requirements.txt          # numpy
├── slime_search.py           # CPU 参考实现（numpy 向量化）
├── check_slime.py            # v0.1 小范围验证
├── gui.py                    # tkinter 管理界面
├── test_slime.c              # C 验证程序
├── gpu/
│   ├── slime_search.cu       # CUDA 搜索（local memory 版，参考）
│   └── slime_search_bp.cu    # CUDA 搜索（bitpack 寄存器优化版，主力）
├── tests/
│   └── test_consistency.py   # CPU vs GPU 一致性
└── docs/
    ├── SPEC.md               # 规格书
    ├── DEVELOPMENT_LOG.md    # 开发日志（含 26.2 验证、性能优化记录）
    ├── 执行清单.md            # 任务清单
    └── 项目企划及对话记录.md  # 原始企划
```

## 许可

[MIT](LICENSE)
