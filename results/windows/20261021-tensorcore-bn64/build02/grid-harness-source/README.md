# Tensor Core CTA grid-width experiment

中文教程：[第29章](../../docs/zh/29-TensorCore网格分组优化与反例.md)。

基线为 QiLHong / vitamin-cuda 的
`520f97093a2abc239977be83d244b327d0e2e2ef`：

- `kernels/sgemm/sgemm_tf32.cu::sgemm_tf32_swizzle_bcf_dbf_kernel`
- `kernels/hgemm/hgemm.cu::hgemm_bcf_dbf_rw_kernel`

原基线与许可证保留在 [../tensorcore/vendor](../tensorcore/vendor)。
本目录是实验派生版本，不声称是未修改的上游。
只将 `SWIZZLE_W=8` 参数化为 `GRID_W`，实例化 1、2、4；
保留原函数名、tile、shared 布局、MMA、累加精度和同步算法。
两个 `*-device.diff` 与 `provenance.json` 记录全部 device 差异。
`prepare_tensor_grid.py --verify` 从原始提取结果重新验证差异范围；
`derive_benchmark.py --verify` 验证 benchmark 派生过程。

调用接口：

```text
windows_tensor_grid_benchmark.exe M N K rounds repeats warmup new.csv [all|grid_w1|grid_w2|grid_w4]
windows_tensor_grid_benchmark.exe --profile implementation N
```

主机拒绝不满足 M%128=0、N%1024=0、K%32=0 的形状，
且限制在本次验证过的范围 M=128..4096、N=1024..4096、K=32..4096。
原核没有边界保护，不可任意修改 grid 或尺寸。

`all`：TF32 与 FP16 分组分别扫描 W8、W1、W2、W4、同精度 cuBLAS；
每轮轮换并反转顺序。`grid_w1/2/4`：原版与一个候选保持相邻，
奇偶轮交换先后，cuBLAS 在配对之外轮换位置。
每次调用前同 stream 的 D2D C0 恢复在 events 之外。
原版和候选逐位比较；另保留原 FP32、量化输入、同精度 cuBLAS、
CPU double 的所有数值判据。新版本没有改变计算精度。

2026-10-21 RTX 4090 D 实测：W1 对 TF32 1024 有约 1% 的小收益，
没有证据支持它作为所有尺寸、两种精度的通用替换。原项目默认入口不变。
本目录新增 README 不影响已归档的编译源；实际 build01 的完整源另存于结果目录。
