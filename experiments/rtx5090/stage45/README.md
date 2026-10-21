# 阶段4/5 Linux 复跑准备（尚未运行GPU）

来源仓库HEAD：`a2b23dabadc79440bf9cbf63f0b566f7eec429ba`；vitamin提交：`520f97093a2abc239977be83d244b327d0e2e2ef`。只读审阅真实源码、既有Windows脚本及第26—33章。本目录提供host适配生成器与单批runner；不修改冻结源码，不替代GPU结果。

## 必要的Linux/SM120适配

原CMake已用CUDA17、Release参数、条件MSVC选项，整体可在Linux使用，但默认架构89，必须显式120。五个Tensor实验host有架构门禁：base/grid/ktile/bn64只允许major8，robustness只允许8.9；不改门禁就会在5090上立即退出。robustness还无条件使用`process.h/_getpid`，需POSIX适配。`prepare_linux.py`只生成新host文件：显式追加12.0接受范围、PID shim、include路径重定位，并保留每个diff和所有device头SHA。所有kernel头直接包含原文件，原算法与模板不变。

生成器创建tensor、grid、ktile、bn64、robustness、simt_original、simt_strict、dbf_default、dbf_maxr128九个独立target。原数学选项保留：SIMT original保留fast_math和普通cuBLAS；strict关闭fast_math、使用原严格模式；其余Tensor实验无fast_math。dbf_maxr128唯一资源实验选项为原`--maxrregcount=128`。

```bash
python prepare_linux.py --repo /actual/SGEMM_CUDA --output /actual/stage45-linux-src
cmake -S /actual/stage45-linux-src -B /actual/build-stage45-sm120 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build /actual/build-stage45-sm120 --target simt_original simt_strict tensor grid ktile bn64 robustness dbf_default dbf_maxr128 -j 2
```

所有`/actual`由主执行代理替换为真实服务器路径。不能上传本地generation-check中的Windows绝对路径产物作为Linux构建源；必须在服务器重新运行生成器。先查编译日志/compile_commands确为sm120，再小尺寸和Sanitizer。

## 真正的选择器及event数

下表每个正式比较默认7轮×10预热×30单调用event；预热不计正式event。

| target/选择器 | 实际覆盖 | 每尺寸正式event |
|---|---|---:|
| simt_original/strict `all` | 五SIMT+K10+K30V2+该配置cuBLAS；两个配置分表 | 1680/配置 |
| tensor `all` | K30 strictFP32锚点、PEDANTIC cuBLAS、TF32 single/dbf/cuBLAS、FP16 single/dbf_rw/cuBLAS |1680|
| tensor `tensor_pairs` | TF32 dbf/cuBLAS及FP16 dbf_rw/cuBLAS交换先后确认 |840|
| grid `all` | TF32与FP16分别W8原版/W1/W2/W4/cuBLAS |2100|
| grid `grid_w1/2/4` | 两精度分别原版/指定W/cuBLAS的相邻配对确认 |1260|
| ktile `all` | TF32原版、动态BK16、动态BK32、TF32 cuBLAS |840|
| ktile `confirm` | TF32原版、BK32、TF32 cuBLAS |630|
| bn64 `all` | TF32原版、BN128/BK32、BN64/BK32、TF32 cuBLAS |840|
| bn64 `confirm` | TF32原版、BN64/BK32、TF32 cuBLAS |630|
| robustness `--case` | 同上三实现；自定义seed；一个独立进程 |630|
| dbf_pair | 原DBF default vs maxr128；每轮两独立进程、轮换先后 |420|

重要：`all`不总是所有enum。ktile/bn64包含其他实现仅供旧host共用，默认仅上述四项。grid的W8就是原作者版，没有另一个`tf32_w8`字符串。逐个实现名受真实chosen逻辑限制，不能仅凭enum存在就认为所有选择都有event；建议以上已验证组选择器。

Tensor共同合法小形状128×1024×32/64/96，M%128、N%1024、K%32，当前host最大4096；3072方阵合法。SIMT/DBF小形状128方阵。正式先4096（历史主尺寸）再1024/2048；阶段5按请求先seed20261020的1024/2048/3072/4096，再多seed/多进程。所有输入保留历史有界xorshift；普通randn Python入口另表。

```bash
python run_one_batch.py --build /actual/build-stage45-sm120 --family tensor --m 128 --n 1024 --k 32 --smoke --output /actual/results/smoke-tensor-k32
python run_one_batch.py --build /actual/build-stage45-sm120 --family simt_original --size 128 --smoke --output /actual/results/smoke-simt-original
python run_one_batch.py --build /actual/build-stage45-sm120 --family simt_strict --size 4096 --output /actual/results/simt-strict-4096 --publish-hook /actual/publish-closed-directory
python run_one_batch.py --build /actual/build-stage45-sm120 --family grid --size 2048 --selection all --output /actual/results/grid-2048 --publish-hook /actual/publish-closed-directory
python run_one_batch.py --build /actual/build-stage45-sm120 --family robustness_case --size 3072 --seed 20261020 --output /actual/results/bn64-first-3072 --publish-hook /actual/publish-closed-directory
```

runner仅运行一个完整尺寸批次，闭合日志与CSV，检查退出码/event数、独立复算轮均值及跨轮中位数并生成SHA。`--publish-hook`是主执行代理既有发布器的可执行适配，参数为该已关闭目录；未指定就只保存本地，不能宣称已推送。异常保留FAILURE/PARTIAL及原始文件。源码compile manifest、GPU遥测、逐轮配对提升和远端核验仍由主发布器完整记录。

## 第33章22680的准确范围与增量发布

`4尺寸×3seed×3独立进程×3实现×7轮×30=22680`，不含smoke、转换、验证或profiler。原`--batch`每进程12案例、7560events，三个进程依次启动；每案例前六轮覆盖全部六种实现顺序，第七轮按batch轮换；尺寸/seed循环也按batch轮换。生成的`robustness-plan.csv`忠实复用此规则，共756行group schedule。

不能用36次`--case`（36个进程）冒充这个三进程实验，也不能把先行seed20261020四尺寸的2520events回填完整矩阵。要严格22680，应在同一冻结二进制上新跑全部三批。

生成的robustness在每个`runCase`返回（各case输出均关闭）后调用`RTX5090_CASE_CHECKPOINT`，以**绝对CSV路径**作为唯一参数；shell不参与参数拼接。设置为已存在的绝对可执行脚本。该脚本必须只归档该CSV及其同前缀sidecars，生成manifest、commit/push/远端回读，返回0才继续下一case。**不要提交仍打开的batch process.csv或持续写入的batch stdout/遥测**。其回执在下一检查点保存，进程结束后再发布最终batch日志/process.csv/BATCH_COMPLETE。

```bash
export RTX5090_CASE_CHECKPOINT=/actual/publish-closed-case
/actual/build-stage45-sm120/robustness --smoke /actual/results/bn64-smoke
/actual/build-stage45-sm120/robustness --batch 1 /actual/stage45-linux-src/robustness-plan.csv /actual/results/bn64-full/batch-1
# 核验batch-1完成并发布最终文件，再依次batch2、batch3。
```

若checkpoint失败，host停止后续case，保留已关闭证据。此host回调会在case之间增加离机保存停顿，属于本次自动备份要求引入的协议差异，应披露；七轮内部测量及三进程边界保持。

## 阶段4仍需上游入口的缺项

以上历史native harness只覆盖TF32 `sgemm_tf32_swizzle_bcf`（single）、`sgemm_tf32_swizzle_bcf_dbf`，普通HGEMM覆盖FP16的bcf及dbf_rw。完整上游清单还要明确运行：TF32的`sgemm_tf32_bt`、`sgemm_tf32_bt_swizzle`、`sgemm_tf32_bt_swizzle_dbf`；HGEMM的naive/bcf/dbf/dbf_rw各FP16/BF16，及同dtype cuBLAS。这些不能借SM120版同名结果代替。普通HGEMM四核也固定BM128/BN128/BK32、SWIZZLE_W8；公共小矩形128×1024×32合法，先检查dtype、CPUdouble与Sanitizer。

上游`kernels/sgemm/test.py`载入sgemm.cu+cublas.cu+sgemm_tf32.cu；`kernels/hgemm/test.py`载入hgemm.cu+cublas.cu。可用现有torch2.8/cu128与`TORCH_CUDA_ARCH_LIST=12.0`直接import编译，保留源数学标志、作者10warm/200repeat墙钟，额外统一event另表。`scripts/vitamin_author_source.py`首尝试原load，失败后fallback硬编码Windows路径；Linux只允许首尝试或另写Linux包装，不走其Windows fallback。

FP16/BF16输入输出同dtype，FP32 accumulator；TF32输入为FP32存储，native历史入口显式转换TF32位模式，原Python入口未必同契约。BN64/BK32派生只属于TF32，不可宣称FP16也有此变体。
