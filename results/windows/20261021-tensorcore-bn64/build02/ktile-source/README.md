# TF32 K tile：BK16 / BK32 受控实验

来源：QiLHong / vitamin-cuda，提交 `520f97093a2abc239977be83d244b327d0e2e2ef`，
`kernels/sgemm/sgemm_tf32.cu` 的 `sgemm_tf32_swizzle_bcf_dbf_kernel`。
原函数名、署名、许可证保留，许可证见 [vendor/LICENSE](../tensorcore/vendor/LICENSE)。
本目录是基于原核的优化实验，**不是作者未经改动的源码**。
原核仍从 `../tensorcore/vendor/vitamin_tf32.cuh` 编译作为独立对照。

## 本次只研究什么

- 原版：BM=BN=128，BK=16，W8，256 线程，静态 shared 32 KiB。
- 控制：动态 shared BK16；隔离动态分配适配本身。
- 候选：动态 shared BK32，以两个 16 宽 slab 保持原有 swizzle/bank 地址映射。
- cuBLAS：相同量化输入，`CUBLAS_COMPUTE_32F_FAST_TF32`，FP32 累加和输出。

目的：减少 K 循环、commit/wait 和 CTA barrier 次数。输出 tile、warp/fragment、
W8 网格排列、输入、精度、K 方向 MMA 累加顺序不变。
BK32 的资源、编译器调度变化是该设计的代价，不能当成已隔离的“纯 barrier”实验。

## 文件与再生成验证

- `tf32_ktile.cuh`：派生 device 实现；完整改动见 `device.diff` / `provenance.json`。
- `benchmark.cu`：由上一轮 grid harness 派生，保留统一输入、C0 恢复、数值检查；
  `host-harness.diff` 记录完整差异。
- `derive_benchmark.py` / `derive_support.py`：生成主机入口和构建/运行/统计脚本。
- `../../scripts/prepare_tensor_ktile.py`：由已核验的 vendor device region 生成候选。

已有文件只验证、不覆盖：

```powershell
& 'D:\Anaconda\Anaconda\python.exe' scripts\prepare_tensor_ktile.py --verify
& 'D:\Anaconda\Anaconda\python.exe' experiments\tensorcore-ktile\derive_benchmark.py --verify
& 'D:\Anaconda\Anaconda\python.exe' experiments\tensorcore-ktile\derive_support.py --verify
```

可执行文件主模式为 `M N K rounds repeats warmup new.csv all` 或 `confirm`。
`all` 比较四个实现，动态 BK16/BK32 相邻并交换先后；`confirm` 比较原版/BK32/cuBLAS，
原版/BK32 相邻并交换先后。每轮轮换组的位置。
主机只接受 M=128..4096 且整除128，N=1024..4096 且整除1024，
K=32..4096 且整除32；没有尾块保护，不得绕过入口限制。
继承 harness 内仍有 FP16/K30 参考工具代码及注释；本轮正式计时仅选择上述 TF32 四项。

## RTX 4090 D 实测

独立七轮确认：BK32 相对原版的逐轮配对吞吐提升中位数，
1024 为 **+2.09%（7/7）**，2048 为 **−4.75%（0/7）**，4096 为 **−6.45%（0/7）**。
三个尺寸均未超过本轮 TF32 cuBLAS。BK32 寄存器127→170，用户 shared 32→64 KiB，
理论常驻 CTA 2→1；不作为通用默认版本。
156 次全量原版逐位比较零差异；这不表示通过严格 FP32 精度标准。

完整讲解、NCU/SASS、数值局限和准确复跑命令见
[第30章](../../docs/zh/30-TensorCore-K分块实验与资源代价.md)。
原始数据仅保存在 [4090D Windows 独立归档](../../results/windows/20261021-tensorcore-ktile/README.md)，
未混入 RTX 3080、其他 GPU 或 Linux 历史结果。
