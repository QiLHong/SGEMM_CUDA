#!/usr/bin/env bash

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
