# CUTLASS 对照发现

- 起点 a2b23dabadc79440bf9cbf63f0b566f7eec429ba，分支 results/rtx4090d-k30-stage2-20261021，开始时工作区干净。
- 已有 scripts/cutlass_simt_reference.cu 和第16章是历史 CUTLASS SIMT 入口，需要审计后复用，不能当作当前4090D数据。
- 上一轮 host harness 提供逐次 C0 恢复、显式 TF32 输入、CPU double、完整 event/检查/指纹输出，可作为新独立入口基础。
- nvidia-smi 当前为 RTX4090D / 驱动610.60 / WDDM / 24564MiB；显示的 UMD CUDA13.3 不是编译 Toolkit。
- 未发现适用 AGENTS.md，也没有既有根目录计划文件。
- 已下载官方 CUTLASS v3.5.1 到 ../deps/cutlass-v3.5.1，固定 f7b19de32c5d1f3cedfc735c2849f12b537522ee；未使用旧 WSL 路径。
- 当前 Windows 实测工具链：CUDA12.4.131、MSVC19.37.32826.1、CMake3.31.6、Ninja1.12.1，复用原有开发环境初始化。
- 测量前选定两份官方 unit-test TF32 配置：128x128x32/warp64x64x32/stages3，与128x64x16/warp64x32x16/stages4。前者shared约96KiB，后者约48KiB，避免只覆盖大shared配置；不是单因素对比、不是自动调优最佳。
- CUTLASS 1xTF32 使用 tfloat32_t + OpMultiplyAdd；FastF32/TF32x3不混入。官方参考： https://github.com/NVIDIA/cutlass/blob/v3.5.1/test/unit/gemm/device/gemm_tf32t_tf32t_f32t_tensor_op_f32_sm80.cu 。
- build03最终二进制F971E74DA275F5787583ABD9AA81535AC175001991AD6D77F2495131E93E1D8C；所有正式有效批次使用同一文件。CUTLASS配置的实际shared分别98304、49152、49152字节，threads128、128、256；模板之外没有修改上游库。
- run04为最终候选：仅复用run02中重叠之前结束的三个完整进程，其余重新执行。复用逐文件SHA、原始PID/UTC/exit0保留。增加运行和profile共享文件锁，已实测另一打开请求被拒绝。
- 最终四尺寸CUT小TF32中位数35.851/62.946/64.203/74.780 TFLOPS，BN64为32.647/57.939/59.447/69.300。2048配对领先BN64 8.095%～8.803%，4096领先6.810%～8.725%；不能当作严格FP32收益。
- strict FP32 CUT SIMT四尺寸14.093/27.350/37.241/43.897 TFLOPS，均未超过K30或cuBLAS。FP32 3072有720条SIMT allclose失败记录，abs<=0.01通过，两个标准不能合并为PASS。
- NCU十次成功，无local load/store及ptxas spill。CUT小配置相对BN64 occupancy低、bank conflict多，但shared-load wavefront少、barrier stall低、Tensor elapsed利用率高；这是多因素相关证据，不是单因素因果证明。
