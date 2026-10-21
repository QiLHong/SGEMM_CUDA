# BM128 / BN64 / BK32：TF32 tile 实验

这是对上一轮 BK32 的进一步改写，不是作者未修改源码。
来源：QiLHong / vitamin-cuda，提交 `520f97093a2abc239977be83d244b327d0e2e2ef`，
`kernels/sgemm/sgemm_tf32.cu` 的 `sgemm_tf32_swizzle_bcf_dbf_kernel`。
保留函数名、署名和[上游许可证](../tensorcore/vendor/LICENSE)。

父版本为 `../tensorcore-ktile/tf32_ktile.cuh`，它和原版均保留并在同一二进制中重测。
`provenance.json` 记录来源和SHA256，`device.diff`记录完整设备改动，
`host-harness.diff`记录公平计时入口改动。

- 输出tile：128×128 → 128×64，256线程仍按2×4 warp排列。
- 每warp输出64×16；N方向fragment 4→2，每线程累加器64→32。
- B搬运：每个16宽K slab由两次16B改为一次，`row=tid/16, col=(tid%16)*4`。
- 双shared buffer总计48KiB；A布局、K累加顺序、TF32精度、同步协议不变。
- 预期获得更好的驻留和任务分配，但A重复加载增加，不能称纯occupancy单因素实验。

运行入口仍严格限制 M为128倍数、N为1024倍数、K为32倍数，范围见第31章。
默认四实现：作者BK16、父BK32、BN64/BK32、同输入TF32 cuBLAS。
`all`让父BK32/BN64相邻并交换顺序；`confirm`让原版/BN64相邻，cuBLAS轮换位置。

```powershell
Set-Location 'D:\CUDA-Lab\cuda-learning\SGEMM_CUDA'
& 'D:\Anaconda\Anaconda\python.exe' scripts\prepare_tensor_bn64.py --verify
```

RTX4090D独立确认：BN64为32.599 / 57.998 / 71.156 TFLOPS（1024/2048/4096）。
相对作者BK16配对中位数 +4.11% / +19.56% / −0.43%；相对同精度cuBLAS −8.45% / +11.57% / +2.79%。
102寄存器、无spill、48KiB用户shared，已验证2CTA/SM。
这是TF32输入、FP32累加及输出，未满足原始严格FP32误差标准。

[第31章：代码、复跑命令与解释](../../docs/zh/31-TensorCore-BN64优化与尺寸效应.md)
· [4090D Windows独立归档](../../results/windows/20261021-tensorcore-bn64/README.md)
