# 🎮 MC Slime Seed Finder - 开发日志

> 项目：Minecraft 史莱姆区块种子搜索器
> 目标：GPU加速枚举，寻找5×5窗口内≥19个史莱姆区块的密集种子

---

## 2026-08-14 找到密集种子（项目目标达成）🎯

- **seed = 157140467**（正方向搜索 [0, 1.57亿)，约 110 秒，1.42M seeds/s）
- 5×5 窗口 (194, 48)：**19 个史莱姆区块**（阈值 19）
- **CPU 参考实现独立复核通过**：总史莱姆区块 16,260（10.11%）、最佳窗口 19 @ (194,48)、全网格 ≥19 窗口**仅此 1 个**
- 后续优化：GUI 管理界面（参数/开始停止/日志文件）、无限搜索 + 正负方向、进度平滑（batch 10 万）

---

## 2026-08-14 26.2 版本兼容性验证 + GPU 性能优化（bitpack 寄存器版）

### 26.2 算法验证结论（基于生产服 26.2-server.jar 字节码实锤）

对 26.2 版服务端 jar 反编译验证（WorldgenRandom / RandomSource / SingleThreadedRandomSource / BitRandomSource）：

1. **种子公式完全一致** ✅：`chunkX²×4987142 + chunkX×5947611 + (int)(chunkZ²)×4392871L + chunkZ×389711 + seed ^ 987234911`，int32 溢出顺序、「先截断再乘 long」全部与当前实现相同
2. **随机数完全一致** ✅：`RandomSource.createThreadLocalInstance(seed)` → SingleThreadedRandomSource = java.util.Random LCG（0x5DEECE66D + 0xB，48 位掩码），`setSeed`/`next(bits)` 逐位相同
3. **唯一差异：nextInt(10) 拒绝采样** ⚠️（见下）

### 拒绝采样差异（决定不实现）

- MC 的 `BitRandomSource.nextInt(int)` 对非 2 的幂走**标准拒绝采样**：`do { bits = next(31); val = bits % 10 } while (bits - val + 9 < 0)`，拒绝区 = `next(31) ∈ [2147483640, 2147483647]`（8/2^31 ≈ 3.7e-9）
- 我们的实现简化为 `next(31) % 10 == 0`（无拒绝采样）
- **影响评估**：差异率 ~3.7e-9/判定，统计上无感知（seed=12345 回归：16,111 / 10.02% / 最佳 11 不变）；严格等价需补拒绝采样
- **实测代价**：实现拒绝采样后 GPU 吞吐 **1.26M → 0.93M seeds/s（-25%）**——每窗口多 1 条 setp+branch（每种子 16 万次），SIMT 下分支指令占发射槽、打断 LCG 关键路径与 popcount 的 ILP 重叠
- **决定：不实现拒绝采样**，保持 `next(31) % 10 == 0` 简化版（差异 3.7e-9 且统计无感，性能收益优先）；本记录留档，如未来需要严格等价可参考「快速版+种子级复核」方案（无分支累积拒绝标志 + 仅 1/1700 种子整种子复核，可恢复 1.24M 且严格等价，未实施）

### GPU 性能优化：bitpack 寄存器版（slime_search_bp.cu）

- **原版瓶颈**：rows[5][401] + col_win[401] = 2808B/线程，编译为 local memory 栈帧（物理在 VRAM），每种子 ~90 万次 local 读写（ptxas: 2808B stack frame, 30 regs）
- **优化**：行数据位打包进寄存器（5 行 × 7×64bit = 35 个 uint64），窗口检查改用 `__popcll` popcount，**零 local memory 访问**（ptxas: 280B stack frame, 43 regs）
- **实测**：103k → **1.26M seeds/s（12.2 倍）**，超过 SPEC 目标（100 万/s）；verify seed=12345 与 C 版完全一致
- 期望搜索量：6.5 亿种子（P(单窗口≥19)=9.7e-15），1.26M/s 下约 **8.6 分钟**找到第一个

---

## 2026-06-12 算法修正与验证

### 背景

从 DeepSeek Chat 导出对话记录后，对照 [Minecraft Wiki 官方源码](https://minecraft.wiki/w/Slime#Slime_chunks) 发现初始计划文档中的史莱姆区块判定算法存在严重错误。

### ❌ 发现的问题

初始算法用了全程64位整数运算（`(long)` 强制转换），但 **Minecraft Java 版使用的是32位 `int` 运算，会发生溢出截断**。两者计算出的种子完全不同。

| 项 | 官方（正确） | 计划文档（错误） |
|:--:|:----------:|:---------------:|
| chunkX²×0x4c1906 | `(int)` 32位溢出截断 | `(long)` 64位无溢出 |
| chunkX×0x5ac0db | `(int)` 32位溢出截断 | `(long)` 64位无溢出 |
| chunkZ²×0x4307a7 | 先 `(int)` 截断再乘 | `(long)` 全程64位 |
| chunkZ×0x5f24f | `(int)` 32位溢出截断 | `(long)` 64位无溢出 |

### ✅ 修正后的算法

```c
// 第一步：构建Random种子（32位int溢出截断，模拟Java行为）
int t1   = chunkX * chunkX * 0x4c1906;         // 32位int自动溢出
int t2   = chunkX * 0x5ac0db;                   // 32位int自动溢出
int t3_z = chunkZ * chunkZ;                      // 32位int溢出截断
int64_t t3 = (int64_t)t3_z * 0x4307a7L;         // 截断后的值转long再乘
int t4   = chunkZ * 0x5f24f;                     // 32位int自动溢出

int64_t seed = (worldSeed + t1 + t2 + t3 + t4) ^ 0x3ad8025fL;

// 第二步：Java Random.setSeed()
seed = (seed ^ 0x5DEECE66DL) & 0xFFFFFFFFFFFFL;

// 第三步：Java Random.next(31)
seed = (seed * 0x5DEECE66DL + 0xBL) & 0xFFFFFFFFFFFFL;
int bits = (int)(seed >> 17);

return (bits % 10) == 0;
```

### 🔬 双重验证

#### 验证1：C语言测试（`test_slime.c`）

- 种子：12345
- 范围：(-200,-200) ~ (200,200)，共 160,801 个区块
- **结果：16,111 个史莱姆区块（10.02%）✅**
- 理论期望：10%，**完全吻合**

| 指标 | 实测值 | 理论值 |
|:----|:-----:|:------:|
| 总区块数 | 160,801 | — |
| 史莱姆区块 | 16,111 | ~16,080 |
| 占比 | **10.02%** | **10%** |
| 最佳5×5窗口 | 11个/25 | 未达标（目标≥19） |

#### 验证2：Python测试（`check_slime.py`）

- 种子：12345
- 范围：(-5,-5) ~ (5,5)，共121个区块
- **结果：13个史莱姆区块（10.7%）✅**
- 与C程序结果一致 ✅

### 📁 本次新增/修改文件

| 文件 | 说明 |
|:----|:------|
| `项目企划及对话记录.md` | 🔄 修正算法公式，添加32位溢出说明 |
| `执行清单.md` | 🔄 修正算法公式，更新任务描述 |
| `test_slime.c` | 🆕 CPU验证程序（C语言，可直接编译运行） |
| `check_slime.py` | 🆕 Python验证脚本（用于快速验证） |
| `DEVELOPMENT_LOG.md` | 🆕 本开发日志 |

### ⚠️ 后续注意事项

1. CUDA实现时同样要注意 **32位int溢出截断** 的模拟
2. 不同MC版本的算法可能有差异，需以目标版本源码为准
3. 种子12345的最佳窗口仅11个，距目标19很远，必须GPU大规模搜索
