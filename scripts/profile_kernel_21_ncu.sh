#!/usr/bin/env bash
# 【文件导读】采集 K21 iterator 的 Nsight 指标
# 给初学者：profiler 是 GPU 的性能体检工具，会干扰正常计时；先证明结果正确，再用它解释为什么快或慢。
# 运行：bash scripts/profile_kernel_21_ncu.sh；需要已经构建的 sgemm、ncu、GPU 及性能计数器访问权限。
# 输入：各 mode 对应的 binary/kernel 名称与 METRICS 清单；下面循环列出实际采样集合。
# 参数：OUTPUT_DIR 保存报告，RUN_TAG 区分批次，DEVICE 选择设备，SEED 固定输入。
# 输出：ncu_*.ncu-rep；同名报告已存在会退出，后续用对应 summarize 脚本导出 CSV。
# 采集：4096³、beta=3、C 每次恢复、warmup=0、repeat=1，按函数过滤第一个匹配 launch。
# 这代脚本未使用 NVTX measure/ 过滤，命中的可能是计时前数值检查 launch；以报告为准。
# METRICS 包含资源、issue/FP32、等待或存储服务量；不同列的单位与归一化方式不同。
# profiler 可能 replay kernel；报告 duration 与 fair 脚本的正常 event 计时应分开比较。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
# 下面是原始硬件计数器名字：sum 是总量，pct 是百分比，per_cycle 是按周期归一化。
# 采样会改变执行环境；这些指标解释正常 benchmark 的快慢，不应用采样耗时代替正常计时。
METRICS="gpu__time_duration.sum,launch__registers_per_thread,launch__shared_mem_per_block,sm__warps_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__issue_inst0.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,smsp__warps_issue_stalled_dispatch_stall.avg.per_cycle_active,smsp__warps_issue_stalled_barrier.avg.per_cycle_active,smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active,smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active,dram__bytes_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__inst_executed_pipe_fma.sum,smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_adu.sum,smsp__inst_executed_op_ldgsts.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
mkdir -p "${OUTPUT_DIR}"

for mode in ${MODES:-strict k10 v1 v2}; do
  if [[ "${mode}" == strict ]]; then variant=1; kernel=13; regex=''
  elif [[ "${mode}" == k10 ]]; then variant=1; kernel=10; regex='sgemmWarptiling'
  elif [[ "${mode}" == v1 ]]; then variant=1; kernel=21; regex='sgemmK21Iterator'
  else variant=2; kernel=21; regex='sgemmK21Pipeline'
  fi
  report="${OUTPUT_DIR}/ncu_k21_${mode}_${RUN_TAG}"
  [[ ! -e "${report}.ncu-rep" ]] || { printf 'Refusing to overwrite %s.ncu-rep\n' "${report}" >&2; exit 1; }
  ncu_args=(--target-processes all --launch-count 1 --metrics "${METRICS}" -o "${report}")
  if [[ -n "${regex}" ]]; then ncu_args+=(--kernel-name "regex:${regex}"); fi
  env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" SGEMM_BENCH_SIZE=4096 \
    SGEMM_WARMUP=0 SGEMM_REPEAT=1 SGEMM_SEED="${SEED:-20261012}" \
    SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
    ncu "${ncu_args[@]}" "${ROOT_DIR}/build-k21-v${variant}/sgemm" "${kernel}"
done
