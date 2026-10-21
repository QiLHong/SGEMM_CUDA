# vitamin-cuda SIMT 原生移植来源

来源：QiLHong/vitamin-cuda 的 zh-cn-beginner，commit `520f97093a2abc239977be83d244b327d0e2e2ef`。
原作者 wingedge，MIT 完整许可证见 `vitamin-LICENSE`；原项目上游 [WingEdge777/vitamin-cuda](https://github.com/WingEdge777/vitamin-cuda)。

`vitamin_sgemm.cuh` 包含 `kernels/sgemm/sgemm.cu` 的五个原始 SIMT device 函数，函数名、模板、映射和同步未改变。
`extract_vitamin_simt.py --verify` 对完整提取段（统一换行及尾部空行）检查相等；`provenance.json` 保存来源和 SHA256。
只移除未研究的 naive、PyTorch 包装及不必要头文件，保留原 FLOAT4、WARP_SIZE、SWIZZLE_A；增加命名空间防止冲突。

主机调用包装位于 `scripts/windows_vitamin_benchmark.cu`，只接受128..4096且128整倍数的方阵；cudaMalloc提供对齐且A/B/C互不重叠。
一般安全域需要M/N为128倍数、K为正16倍数和连续FP32的16字节对齐；本入口故意只暴露已审计的方阵子集。
原函数内部没有边界检查，不能用grid向上取整支持任意形状。
