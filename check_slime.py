def is_slime_chunk(world_seed, chunk_x, chunk_z):
    # 32位int溢出截断（Python模拟）
    def to_int32(x):
        x = x & 0xFFFFFFFF
        if x >= 0x80000000:
            x -= 0x100000000
        return x

    # 第一步：构建Random种子（32位int溢出截断）
    t1 = to_int32(to_int32(chunk_x * chunk_x) * 0x4c1906)
    t2 = to_int32(chunk_x * 0x5ac0db)
    t3_z = to_int32(chunk_z * chunk_z)
    t3 = t3_z * 0x4307a7       # 截断后的32位直接乘（Python大整数不会溢出）
    t4 = to_int32(chunk_z * 0x5f24f)

    seed = (world_seed + t1 + t2 + t3 + t4) ^ 0x3ad8025f

    # 第二步：Java Random.setSeed()
    seed = (seed ^ 0x5DEECE66D) & 0xFFFFFFFFFFFF

    # 第三步：Java Random.next(31)
    # （注：26.2 实测 MC 用拒绝采样 nextInt(10)，差异仅 3.7e-9/判定，决定不实现——见 DEVELOPMENT_LOG.md 2026-08-14）
    seed = (seed * 0x5DEECE66D + 0xB) & 0xFFFFFFFFFFFF
    bits = seed >> 17

    return (bits % 10) == 0


seed = 12345
print(f"种子: {seed}")
print(f"范围: chunkX [-5, 5], chunkZ [-5, 5]")
print()
header = f"{'区块X':>6}  {'区块Z':>6}  {'方块X':>8}  {'方块Z':>8}  {'史莱姆？':>6}"
print(header)
print("-" * len(header))

count = 0
total = 0
for z in range(5, -6, -1):
    for x in range(-5, 6):
        total += 1
        slime = is_slime_chunk(seed, x, z)
        if slime:
            count += 1
        mark = "✅ 是" if slime else "❌ 否"
        print(f"{x:>6}  {z:>6}  {x*16:>8}  {z*16:>8}  {mark:>6}")
    print()

print(f"总计: {total} 个区块, {count} 个是史莱姆区块 ({100*count/total:.1f}%)")
