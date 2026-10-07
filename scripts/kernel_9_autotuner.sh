#!/usr/bin/env bash

set -u

# 【学习补充】本脚本会原地改写 runner 和 kernel 参数，并清空同名输出文件。
# 请在独立实验分支/副本运行；本中文分支仅翻译注释，不自动执行调优。
# 原脚本只有 set -u，没有 set -e/pipefail；不能仅看 tee 的成功就认为编译/测试成功。

# 为各参数定义候选值。
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
NUM_THREADS_VALUES=(256)

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/runner.cu"
KERNEL="../src/kernels/9_kernel_autotuned.cuh"
OUTPUT="../benchmark_results/kernel_9_autotune_results.txt"

# 清空结果文件；父目录必须已经存在。
echo "" > $OUTPUT

# 原脚本固定选择 GPU 2，会覆盖外部传入的 DEVICE。
# 单卡 RTX 3080 通常只有设备 0；不要原样在单卡机上直接启动此调优脚本。
export DEVICE="2"

TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# 遍历候选组合；不满足覆盖/整除条件的组合在编译前跳过。
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          for nt in ${NUM_THREADS_VALUES[@]}; do
            echo ""
            CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

            # 约束含义与 runner.cu 中 static_assert 对照。
            # QUANTIZATION 在此指 tile 取整问题，不是 INT8/INT4 数值量化。
            config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"
            if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK = $(( ($nt * 4) % bk )) != 0))"
              continue
            fi
            if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN = $(( ($nt * 4) % bn )) != 0))"
              continue
            fi
            if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
              continue
            fi
            if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
              continue
            fi
            if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi
            if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
              continue
            fi

            # 原地更新源码中的模板实参/线程常量；这不是只读 benchmark。
            sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $RUNNER
            sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $RUNNER
            sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $RUNNER
            sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $RUNNER
            sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $RUNNER
            sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $nt;/" $KERNEL
            
            # 重编译；原脚本要求 build/ 使用 Makefile 生成器，不是 Ninja。
            make 

            echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NUM_THREADS=$nt" |& tee -a $OUTPUT
            # 执行 9 号 benchmark；4 秒内未结束则由 timeout 终止。
            # 超时不等同于 kernel 数值错误，也不能把未完成输出当有效性能结果。
            timeout -v 4 ./sgemm 9 | tee -a $OUTPUT
          done
        done
      done
    done
  done
done
