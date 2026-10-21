# vitamin SM120 源码与教程适用性审计

审计日期：2026-10-22（Asia/Shanghai）。本文件是源码审计，不是 GPU 测试结果。

来源：`QiLHong/vitamin-cuda`，分支 `zh-cn-beginner`，提交 `520f97093a2abc239977be83d244b327d0e2e2ef`。本地只读工作区检查干净，未发现仓库内 AGENTS.md。本审计不修改上游源文件。

## 真实入口、共同合法域

`kernels/hgemm_sm120/hgemm_sm120.cu` 的 `PYBIND11_MODULE` 只导出下表五个入口。`hgemm_tma_rw_k_stages_kernel` 是空函数且未导出，不能计作已实现实验。

| 入口 | BM/BN/BK、stage | shared 用户字节 | 输入/输出、累加 |
|---|---|---:|---|
| hgemm_bcf_dbf_rw | 128/128/32，2 | 32768 静态 | FP16 或 BF16、同型输出、FP32累加 |
| hgemm_k_stages | 128/128/32，默认3 | 49152 静态 | 同上 |
| hgemm_tma_r_k_stages_64 | 128/128/64，3 | 98328 动态（含24B barrier） | 同上 |
| hgemm_tma_r_k_stages_32 | 128/128/32，3 | 49176 动态（含24B barrier） | 同上 |
| hgemm_cublas | cuBLAS 自选 | 运行时算法决定 | A/B/C同型16F或16BF，CUBLAS_COMPUTE_32F |

四个作者核均为256线程、8warp。每warp负责C的64×32，共同组成128×128；每线程有64个FP32累加数。最终舍入使用 `__float22half2_rn` 或 `__float22bfloat162_rn`，写入复用的shared，再以16B向量写回global。表内为源码推导的用户shared，实际寄存器、spill、驱动额外shared及驻留数必须由本机工具确认。

所有作者核硬编码 `SWIZZLE_W=8`，没有尾块保护。公共安全域为 M 正整数倍128、N 正整数倍1024、K 正整数倍64且K>=128；A=M×K、B=K×N、C=M×N，三者同GPU、同FP16/BF16类型、行主序连续。最小公共冒烟形状是 **M=128,N=1024,K=128**，而不是128方阵。此形状覆盖TMA64恰好两个prologue tile；另做K=256或384才能覆盖主循环及阶段复用/phase翻转。

更细的K约束：DBF为32倍数且至少32；cp.async三stage、TMA32为32倍数且至少64；TMA64为64倍数且至少128。三stage代码无条件预取2个tile，因此较小K不合法。仅源码合法不等于已通过Sanitizer；运行应使用外部超时。

原binding只检查CUDA和contiguous，既不验证矩阵形状也不验证共同dtype/device。非half类型直接走BF16 reinterpret。审计包装必须限制输入，不能把非法形状造成的失败归为GPU不支持。

## TMA descriptor、同步与计时

A tensor map：globalDim={K,M}、globalStride={K*2字节}、box={BK,128}，A按BK32/64使用64B/128B swizzle。B tensor map：globalDim={N,K}、stride={N*2字节}、box={64,BK}，128B swizzle；每个stage对B左右两个64列半块分别发起一次TMA，因此每stage有1次A搬运与2次B搬运。

各stage独立8B mbarrier；thread0初始化arrival count=1，并调用 arrive.expect_tx，预计字节为(128*BK+BK*128)*2，即BK32的16384B或BK64的32768B。全CTA等待对应phase，ldmatrix将shared片段取至寄存器，mma.sync.m16n8k16进行FP32累加。主循环读完后全CTA同步再轮转stage；compute_stage绕回0时phase翻转。TMA核另外使用两个寄存器片段buffer。它不是producer专用warp模型，也没有跨CTA collective MMA。

上游host每次调用TMA函数都重新执行两个 `cuTensorMapEncodeTiled` 和一次 `cudaFuncSetAttribute`。原Python `benchmark` 是10次预热、200次调用的perf_counter墙钟，显式GPU同步后计算均值；计时包括Python/C++派发和descriptor准备，但不包括输入生成/分配。围绕Python调用放CUDA event也可能计入主机准备期间GPU空等，必须命名 **API-dispatch CUDA-event**，不能称缓存descriptor的纯kernel时间。

原cuBLAS入口使用PyTorch当前handle，alpha=1、beta=0，行主序通过交换A/B调用N×M的column-major GEMM；输出同dtype。原入口没有单独冻结handle math mode，也不检查cuBLAS status；正常运行同步和数值校验仍需记录。不能把其COMPUTE_32F基线改称PEDANTIC strict FP32。

## 保留上游编译与原入口

`test.py` 使用 `torch.utils.cpp_extension.load`，源文件为hgemm_sm120.cu+cublas.cu，并依赖 `../common/pack.cuh`。保留 `-O3 -std=c++17 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda`、half/BF16 undef宏、`-Xptxas -v`。构建目标由 `TORCH_CUDA_ARCH_LIST=12.0` 确定，必须核验实际命令与产物。原链接选项为 `-L/usr/local/cuda/lib64/stubs -lcuda`；stub只用于链接，禁止加入运行时库路径。

上游默认 `test_4096()` 为BF16、4096方阵，输入在CPU以FP32 randn生成再CPU舍入BF16、传GPU；默认函数自己没有manual_seed。固定顺序为torch、cuBLAS、DBF、stage3、TMA64、TMA32。包装器在调用前固定seed42，并明确披露这一可复现性设置。

作者判据：`torch.allclose(reference,candidate,atol=.016,rtol=.016)`，相对项使用第二个参数candidate。项目abs<=.01单独记录，不用它替代作者低精度判据，也不隐藏它的失败。

## 包装器交付与边界

`sm120_upstream_wrapper.py`提供smoke、author、events三个模式。smoke全输出CPU double（实际低精度输入再同类型输出舍入）及同dtype cuBLAS参考；author直接调用未改的test_4096，日志包装保留原benchmark/diff_check；events逐尺寸逐dtype产生一个1050正式event完整批次（5实现×7轮×30），每轮10预热，保存预生成轮换顺序、实际输入SHA256、C0在start event外恢复、每轮全输出cuBLAS校验及固定256位置CPU double抽查。每批独立目录，闭合CSV后回读统计。

events是明确命名的API-dispatch测量；它不代替native cached-descriptor kernel-only测量，也尚未提供PEDANTIC大尺寸全输出GPU参考。原Python日志与统一event表分开。所有结果中的NaN/Inf、作者阈值和abs.01阈值分别报告。包装器仅通过Python AST语法检查，本审计未运行GPU；真正测试状态由主执行记录决定。

Linux调用示例（环境变量沿用当前隔离工具链设置）：

```bash
/root/miniconda3/bin/python sm120_upstream_wrapper.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --mode smoke --dtype both --k 128 --output smoke-k128
/root/miniconda3/bin/python sm120_upstream_wrapper.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --mode smoke --dtype both --k 384 --output smoke-k384
/root/miniconda3/bin/python sm120_upstream_wrapper.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --mode author --dtype bf16 --seed 42 --output author-bf16-4096
/root/miniconda3/bin/python sm120_upstream_wrapper.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --mode events --dtype bf16 --size 1024 --seed 42 --output events-bf16-1024
```

## 教程机制与5090适用性

以下链接均于2026-10-22读取。官方GPU表将RTX5090列为12.0，B200/GB200为10.0。[NVIDIA GPU计算能力](https://developer.nvidia.com/cuda/gpus)。CUTLASS明确区分SM100与GeForce SM120，并说明SM100a条件特性编译的核不能用于RTX50系列。[CUTLASS overview](https://docs.nvidia.com/cutlass/latest/overview.html)

| 教程机制 | 5090适用性 | 本项目实现/处理 |
|---|---|---|
| 基础核tcgen05 MMA、TMEM accumulator | 教程路线硬件不兼容 | 标记unsupported，不安装/升级JAX试图改变硬件限制 |
| collective 2CTA MMA | 教程路线硬件不兼容 | 不执行；README one/two blocks不能解释为此路线 |
| TMA搬运 | 可由SM120支持的指令实现 | 上述cp.async.bulk.tensor.2d + mbarrier |
| 寄存器Tensor Core MMA | 当前上游实现 | ldmatrix + mma.sync，累加保存在普通FP32寄存器 |
| 多stage、warp分工、分块epilogue | 可借鉴方法，须独立实现验证 | 现有多stage和shared写回只按其真实源码记名 |
| persistent/grid tiling | 算法思路可借鉴 | 旧实验复现后才允许至多一个独立新实验 |
| 正态输入、矩形4096×8192×4096 | 测量建议可复用 | 公共合法域包含该矩形，但需先小矩形验证；输入SHA共享 |

PTX `tcgen05.mma` 的Target ISA列表为SM100a/相关100或110家族，未含SM120；其累加目标是Tensor Memory。TMA则是global/shared数据搬运机制，不能与TMEM储存空间混称。[PTX tcgen05.mma](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma)

当前官方教程与中文译文都在基础核使用tcgen05/TMEM，随后加入2CTA、persistent与grid tiling。其109.6%是原文章特定输入与平台结果，cuBLAS标尺来自jax.dot，不是本项目固定cuBLAS算法的验收值。[JAX官方教程](https://docs.jax.dev/en/latest/pallas/gpu/blackwell_matmul.html)；[用户指定中文教程](https://pengchengneo.github.io/posts/pallas-jax-kernel-language-cn/pallas_gpu_blackwell_matmul_cn/)。本轮不宣称完整复现Pallas教程。

