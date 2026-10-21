# RTX5090 严格 FP32 五实现 host 审计

这是独立派生入口，尚未通过本目录创建动作产生任何 GPU 实测。放到主仓库 `experiments/rtx5090/strict-fp32/`，保留已有 `scripts/windows_benchmark.cu` 与两个 device header。

本入口增加显式 `--alpha 0.5|1`、五实现 `paired` 组、GPU开始前关闭的顺序表，以及候选输出直接对CPU double的校验。原 `all`/`sync` 的集合和顺序表保留，默认alpha仍0.5。新增CPU校验在event外，因此这应标为“统一审计host配置”，不能冒充未经改动的旧入口。

`provenance.json` 保存冻结源码提交/原始字节SHA256、派生字节SHA256和未改device头SHA256；`host-harness.diff` 是全部host差异。生成器先断言原始文件哈希并拒绝覆盖不同内容。

## 构建

```bash
cd /root/autodl-tmp/SGEMM_CUDA-rtx5090-20261022
cmake -S experiments/rtx5090/strict-fp32 -B /root/autodl-tmp/build-rtx5090-strict-paired \
  -G Ninja -DCMAKE_MAKE_PROGRAM=/root/autodl-tmp/rtx5090-tools/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_TRY_COMPILE_CONFIGURATION=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build /root/autodl-tmp/build-rtx5090-strict-paired --parallel 2 --verbose
```

只有一个target。也可原生编译并保留完整命令：

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo -Xptxas=-v \
  -DSGEMM_K10_MIN_BLOCKS=2 experiments/rtx5090/strict-fp32/benchmark.cu -lcublas \
  -o /root/autodl-tmp/build-rtx5090-strict-paired/rtx5090_strict_fp32
```

检查实际sm120编译命令及cuobjdump目标；不要使用fast_math或忽略compiler检查。PTXAS/占用率需依据本机产物，不套用旧GPU。

## 运行（每条完成后单独闭合和发布）

```bash
# 创建新批次目录，以下变量必须是互不重复的实际目录。
BIN=/root/autodl-tmp/build-rtx5090-strict-paired/rtx5090_strict_fp32

# 两数学问题各自先做128方阵全量CPU double验证。
NVIDIA_TF32_OVERRIDE=0 "$BIN" 128 3 1 1 1 "$SMOKE_ORIGINAL/events.csv" paired --alpha 0.5
NVIDIA_TF32_OVERRIDE=0 "$BIN" 128 0 1 1 1 "$SMOKE_AB/events.csv" paired --alpha 1

# 1024、2048、3072、4096逐尺寸独立批次，alpha=.5,beta=3保留历史数学问题。
NVIDIA_TF32_OVERRIDE=0 "$BIN" 1024 3 7 30 10 "$BATCH_ORIGINAL/events.csv" paired --alpha 0.5
# 另开目录，C=AB，不混表。
NVIDIA_TF32_OVERRIDE=0 "$BIN" 1024 0 7 30 10 "$BATCH_AB/events.csv" paired --alpha 1
```

命令前检查目录新建、命令后立即保存退出码与stdout/stderr。不得将失败输出覆盖后重测。标准完整批次为5实现×7轮×30正式event=1050，预热为每轮每实现10次；小验证为5个event，独立标记非性能结果。

`events.csv.order.csv` 在GPU初始化前写完并关闭；PAD4/PAD8、V1/V2始终相邻，轮间交换先后；cuBLAS轮换到首、中、末。每次恢复相同C0，恢复位于start event前；同一默认stream，descriptor/量化不涉及此严格FP32组。所有分配、CPU校验、D2H都不计时。

正确性保留原绝对≤0.01及NaN/Inf失败。GPU参考使用PEDANTIC FP32全输出；CPU double直接核验每个实现，n128全16384点，其余固定256点。报告max abs/max rel；相对误差分母为max(abs(reference),1e-12)，不改变原阈值。CSV的错误项属于每轮最后一次输出，不宣称30次逐次输出全验。

输入发生器与旧入口完全相同：seed20261020，三流xorshift32，离散有界FP32，行主序、stride=n；全部同轮实现接收相同实际输入。FNV1a64是实际float字节指纹，生成器及源SHA256另外保存；不能将其称为SHA256输入哈希。

## 独立统计与恢复

```bash
/root/miniconda3/bin/python experiments/rtx5090/strict-fp32/summarize_batch.py \
  --csv "$BATCH_ORIGINAL/events.csv" --out "$BATCH_ORIGINAL/analysis" \
  --selection paired --rounds 7 --repeats 30
```

脚本检查每个唯一event键、精确计数、预生成顺序、输入指纹/数学问题一致性、CPU覆盖范围及数值检查；从event重新算轮均值，核对原记录，再取跨轮中位延迟换算2N³/time。逐轮配对增益另表，不能拿两个列中位数比值冒充配对中位数。

成功后才能写 `BATCH_COMPLETE.json`（closed=true/status=COMPLETE/independent_summary_verified=true），调用任务checkpoint/publish脚本。失败或超时的关闭输出标PARTIAL并保留日志。原始CSV、顺序CSV与该统计脚本可从固定GitHub提交下载到全新目录再运行，作为早期恢复检查；不需要GPU。

原文件 `scripts/windows_benchmark.cu` 的 `all` 和 `sync` 可另行直接运行以保留完全相同host流程；此派生统一组以及C=AB必须各自命名，不能替代旧alpha=.5,beta3表。

## 单批闭合包装器

`run_closed_batch.py` 只运行一批，不编译、不提交、不推送。它以非阻塞独占 `flock` 持有实际 Git 目录的 `rtx5090-gpu.lock`，要求目标分支与全新的 `results/rtx5090/<batch>` 目录；保存source HEAD、相关源码/设备头SHA256、二进制SHA256及可用构建文件指纹。

```bash
# Smoke: 1轮 / 1次正式 / 1次预热，n128完整CPU double。
/root/miniconda3/bin/python experiments/rtx5090/strict-fp32/run_closed_batch.py \
  --binary /root/autodl-tmp/build-rtx5090-strict-paired/rtx5090_strict_fp32 \
  --output results/rtx5090/20261022-strict-smoke-original \
  --mode smoke --alpha 0.5 --beta 3

# 正式: 7轮 / 30次正式 / 10次预热；此调用只测一个尺寸、一个数学问题。
/root/miniconda3/bin/python experiments/rtx5090/strict-fp32/run_closed_batch.py \
  --binary /root/autodl-tmp/build-rtx5090-strict-paired/rtx5090_strict_fp32 \
  --output results/rtx5090/20261022-strict-paired-1024-original \
  --mode events --n 1024 --alpha 0.5 --beta 3
```

C=AB另用新的输出目录和 `--alpha 1 --beta 0`。包装器保存确切命令、benchmark实际退出码和日志，180秒超时只结束自己创建的benchmark进程。250ms NVML遥测只查询GPU层面字段，不保存其他进程名称/PID；无样本明确记MISSING。统计脚本独立重算并核对5/1050条实际event；只有通过独立统计且源/二进制指纹稳定的关闭批次标COMPLETE。数值失败不改成PASS：保留程序退出码，统计验证失败时标PARTIAL。调用结束后立即使用任务发布流程，不能等全部尺寸执行完。
