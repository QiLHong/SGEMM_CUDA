# 【学习补充】共享内存 bank 映射的纸面计算辅助脚本，不是 GPU 性能分析器。
# 对以 float 为单位的线性索引，模 32 得到本例的 bank 编号。
# 行跨度从 32 变成 33 会改变行起点的 bank，从而改变具体访问模式的冲突情况。
# 不能只统计 bank 编号就忽略“同地址广播”等硬件行为。
banks_naive = lambda r, c: (r * 32 + c) % 32
banks_one_extra = lambda r, c: (r * 33 + c) % 32

ITEMS_PER_WARP = 8


def printBankConflicts(bank_fun):
    for c in range(1):
        banks = []
        for i in range(32):
            row = (i * ITEMS_PER_WARP) // 16 
            col = (i * ITEMS_PER_WARP + c) % 16
            banks.append((i, row, col, bank_fun(row, col)))
        print("Step", c, "\n", "\n".join(["(" + ",".join(str(x) for x in i) + ")" for i in banks]))
        d = {k: 0 for k in range(32)}
        for i in banks:
            d[i[-1]] += 1

        count = 0
        for key, val in d.items():
            if val > 0:
                count += 1

        print(
            f"Bank conflicts (Step {c}): {sorted(d.items(), key=lambda item: item[1], reverse=True)[0][1]}, banks accessed: {count}/32\n"
        )


# 【上游限制】函数定义只有一个参数，下面原调用却传入两个，直接运行会触发 TypeError。
# 中文学习分支保留这个调用，不把代码修复混入翻译；详见 docs/zh/03-原版限制与验证.md。
print("---NAIVE---")
printBankConflicts(banks_naive, 32)

print("\n---EXTRA COL---")
printBankConflicts(banks_one_extra, 33)
