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


# 后续实验提交修复了上游多传一个参数的 TypeError；纯翻译验证仍固定在 aadcd8e。
print("---NAIVE---")
printBankConflicts(banks_naive)

print("\n---EXTRA COL---")
printBankConflicts(banks_one_extra)
