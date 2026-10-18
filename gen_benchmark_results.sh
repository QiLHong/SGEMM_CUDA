#!/usr/bin/env bash
# 【文件导读】上游批量 benchmark 与绘图入口
# 给初学者：这是保留实验历史的工具，不是推荐的一键运行入口；先看下面的改写/覆盖风险。
# 运行：在仓库根目录 bash gen_benchmark_results.sh；先构建 build/sgemm。
# 依次运行编号 0..10，写 benchmark_results/<编号>_output.txt，最后调用 plot_benchmark_results.py。
# 此入口会覆盖同名日志，并由绘图脚本更新 README.md 与 benchmark_results.png。
# 保留它便于复现上游流程；K30/K32 的公平研究记录应走独立 audit，勿把这份图当当前最佳研究基线。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail

# 运行 sgemm，并把输出保存到 benchmark_results/，最后调用绘图脚本。
# 【学习补充】原注释称“全部已有 kernel”，但实际循环只包含 0–10，不含 11/12。
# 请从仓库根目录运行；绘图脚本会改写 README.md 结果表及 benchmark_results.png。
# 为保留上游历史表，建议在单独实验分支运行，而不是直接覆盖中文学习分支。

mkdir -p benchmark_results

for kernel in {0..10}; do
    echo ""
    ./build/sgemm $kernel | tee "benchmark_results/${kernel}_output.txt"
    sleep 2
done

python3 plot_benchmark_results.py
