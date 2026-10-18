#!/usr/bin/env bash
# 【文件导读】K23 的构建、正确性、访存与性能入口
# 给初学者：一个脚本可能有多个动作；先选 build，再 correctness，最后 fair/profile，不要把它们混成一步。
# 运行：bash scripts/kernel_23_audit.sh {build|correctness|memcheck|fair}；一次只选择一个动作。
# build 生成 build-k23-v1..v4；V1 复用 K21 iterator，V2/V3 单独改一种 shared 布局，V4 合并。
# VARIANTS 可限制变体；build 默认 1..4，correctness/memcheck 默认 2..4，两者默认集合不同。
# correctness：每变体六尺寸 × beta=0/1/3；memcheck：128/256、beta=3，不包含其他工具。
# fair：默认 strict/k10/v1..v4，4096³ 循环移位测量；MODES/ROUNDS/REPEAT/WARMUP 可配置。
# OUTPUT_DIR/RUN_TAG/SEED 决定新结果和输入；每类输出独立命名，不覆盖已有主 CSV。
# Nsight 另用 profile_kernel_23_ncu.sh；SASS 另用 audit_kernel_23_sass.sh，名称 audit 不代表已完成所有阶段。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

# Reproducible K23 correctness, fair timing, memcheck and Nsight runs.
# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# OUTPUT_DIR 是结果存放处，RUN_TAG（若有）是批次名字；两者不是 kernel 参数，换标签不会重新选 tile。
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark_results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
SEED="${SEED:-20261013}"
mkdir -p "${OUTPUT_DIR}"
mode="${1:-}"
case "${mode}" in
  build)
    for variant in ${VARIANTS:-1 2 3 4}; do
      build="${ROOT_DIR}/build-k23-v${variant}"
      cmake -S "${ROOT_DIR}" -B "${build}" -DCMAKE_BUILD_TYPE=Release \
        -DSGEMM_K23_VARIANT="${variant}" -DCMAKE_CUDA_FLAGS='--ptxas-options=-v'
      cmake --build "${build}" --target sgemm --parallel "${JOBS:-2}"
    done
    ;;
  correctness)
    csv="${OUTPUT_DIR}/kernel_23_correctness_${RUN_TAG}.csv"
    [[ ! -e "${csv}" ]] || { printf 'Refusing to overwrite %s\n' "${csv}" >&2; exit 1; }
    printf 'variant,beta,size,result\n' >"${csv}"
    pass_count=0
    expected=0
    for variant in ${VARIANTS:-2 3 4}; do
      for beta in 0 1 3; do
        output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
          SGEMM_WARMUP=1 SGEMM_REPEAT=1 SGEMM_SEED="${SEED}" \
          SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA="${beta}" \
          "${ROOT_DIR}/build-k23-v${variant}/sgemm" 23)"
        awk -v v="${variant}" -v b="${beta}" '
          /dimensions\(m=n=k\)/ {gsub(/,/, "", $2); size=$2}
          /Correctness vs strict FP32 cuBLAS:/ {print v "," b "," size "," $NF}
        ' <<<"${output}" >>"${csv}"
        passed="$(grep -c 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}")"
        pass_count=$((pass_count + passed))
        expected=$((expected + 6))
        printf 'V%s beta=%s: %s\n' "${variant}" "${beta}" \
          "${passed}"
      done
    done
    [[ "${pass_count}" == "${expected}" ]]
    printf 'Correctness %s/%s PASS: %s\n' "${pass_count}" "${expected}" "${csv}"
    ;;
  memcheck)
    csv="${OUTPUT_DIR}/kernel_23_memcheck_${RUN_TAG}.csv"
    [[ ! -e "${csv}" ]] || { printf 'Refusing to overwrite %s\n' "${csv}" >&2; exit 1; }
    printf 'variant,size,result,log\n' >"${csv}"
    pass_count=0
    expected=0
    for variant in ${VARIANTS:-2 3 4}; do
      for size in 128 256; do
        log="${OUTPUT_DIR}/kernel_23_memcheck_v${variant}_${size}_${RUN_TAG}.log"
        if env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
          SGEMM_BENCH_SIZE="${size}" SGEMM_REPEAT=1 SGEMM_WARMUP=0 \
          SGEMM_SEED="${SEED}" SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 \
          compute-sanitizer --tool memcheck --error-exitcode=99 \
          "${ROOT_DIR}/build-k23-v${variant}/sgemm" 23 >"${log}" 2>&1; then
          result=PASS
        else
          result=FAIL
        fi
        printf '%s,%s,%s,%s\n' "${variant}" "${size}" "${result}" "${log#${ROOT_DIR}/}" >>"${csv}"
        expected=$((expected + 1))
        [[ "${result}" != PASS ]] || pass_count=$((pass_count + 1))
      done
    done
    [[ "${pass_count}" == "${expected}" ]]
    printf 'Memcheck %s/%s PASS: %s\n' "${pass_count}" "${expected}" "${csv}"
    ;;
  fair)
    csv="${OUTPUT_DIR}/kernel_23_fair_${RUN_TAG}.csv"
    [[ ! -e "${csv}" ]] || { printf 'Refusing to overwrite %s\n' "${csv}" >&2; exit 1; }
    printf '%s\n' 'round,position,mode,kernel,variant,gflops,kernel_time_ms,correctness,temp_before_c,temp_after_c,sm_clock_before_mhz,sm_clock_after_mhz,mem_clock_before_mhz,mem_clock_after_mhz,beta,c_policy,seed' >"${csv}"
    read -r -a modes <<<"${MODES:-strict k10 v1 v2 v3 v4}"
    snapshot() {
      nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory \
        --format=csv,noheader,nounits | head -n 1 | tr -d ' '
    }
    for ((round=1;round<=${ROUNDS:-3};round++)); do
      start=$(((round - 1) % ${#modes[@]}))
      for ((position=0;position<${#modes[@]};position++)); do
        name="${modes[$(((start + position) % ${#modes[@]}))]}"
        variant=na
        if [[ "${name}" == strict ]]; then kernel=13; binary="${ROOT_DIR}/build-k23-v1/sgemm"
        elif [[ "${name}" == k10 ]]; then kernel=10; binary="${ROOT_DIR}/build-k23-v1/sgemm"
        else kernel=23; variant="${name#v}"; binary="${ROOT_DIR}/build-k23-v${variant}/sgemm"
        fi
        before="$(snapshot)"
        output="$(env -u NVIDIA_TF32_OVERRIDE DEVICE="${DEVICE:-0}" \
          SGEMM_BENCH_SIZE=4096 SGEMM_WARMUP="${WARMUP:-10}" \
          SGEMM_REPEAT="${REPEAT:-50}" SGEMM_SEED="${SEED}" \
          SGEMM_RESET_C_EACH_REPEAT=1 SGEMM_BETA=3 "${binary}" "${kernel}")"
        after="$(snapshot)"
        seconds="$(awk -F'[()]' '/Average elapsed time:/ {v=$2} END {gsub(/[^0-9.]/,"",v); print v}' <<<"${output}")"
        gflops="$(awk -F'[()]' '/performance:/ {v=$4} END {gsub(/[^0-9.]/,"",v); print v}' <<<"${output}")"
        milliseconds="$(awk -v s="${seconds}" 'BEGIN {printf "%.4f",s*1000}')"
        correctness=reference
        if [[ "${kernel}" != 13 ]]; then
          grep -q 'Correctness vs strict FP32 cuBLAS: PASS' <<<"${output}"
          correctness=PASS
        fi
        IFS=',' read -r tb sb mb <<<"${before}"; IFS=',' read -r ta sa ma <<<"${after}"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,3,reset-each-repeat,%s\n' \
          "${round}" "$((position+1))" "${name}" "${kernel}" "${variant}" \
          "${gflops}" "${milliseconds}" "${correctness}" "${tb}" "${ta}" \
          "${sb}" "${sa}" "${mb}" "${ma}" "${SEED}" >>"${csv}"
        printf 'Round %s %s: %s GFLOPS, %s ms\n' "${round}" "${name}" "${gflops}" "${milliseconds}"
      done
    done
    printf 'Fair CSV: %s\n' "${csv}"
    ;;
  *) printf 'Usage: %s {build|correctness|memcheck|fair}\n' "$0" >&2; exit 2;;
esac
