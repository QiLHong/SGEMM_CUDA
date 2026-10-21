# RTX 5090 GEMM 分阶段复现

日期：2026-10-22（Asia/Shanghai）。这是启动检查点，**尚无 RTX 5090 GEMM 实测结果**。

唯一发布分支：`results/rtx5090-gemm-reproduction-20261022`。
启动 GitHub HEAD：`a2b23dabadc79440bf9cbf63f0b566f7eec429ba`。参考 vitamin-cuda：`520f97093a2abc239977be83d244b327d0e2e2ef`，只读。
已确认 Ubuntu22.04.5 / RTX5090 / SM120 / 170SM / CUDA12.8.93 / PyTorch2.8.0+cu128。完整环境、最小编译测试、Sanitizer和NCU仍待完成。

## 阶段顺序

0. 实际推送与回读、环境、sm_120 smoke、完整历史清单。
1. SM120 HGEMM：先BF16作者入口及TMA32/TMA64/cp.async/cuBLAS，再FP16。
2. 严格FP32：K10 PAD4/PAD8、K30 V1/V2、PEDANTIC cuBLAS；alpha0.5/beta3与C=AB分表。
3. K1–K32历史系列，包含K31/K32独立入口、失败及资源限制配置。
4. vitamin SIMT/TF32/FP16/BF16、寄存器上限及原Python入口。
5. grid/BK/BN64及多seed独立进程稳健性。
6. 主线备份后补充正态输入、合法矩形及至多一个新实验。

每个闭合批次保存全部event、输入指纹和来源，生成manifest，commit、显式push、核验SHA并回读。GPU串行。未完成配置始终保留。SM100的tcgen05/TMEM/2CTA collective路线标记硬件不兼容，不强行运行。

## 备份通道

服务器Git网络操作仅在当前shell加载 `/etc/network_turbo`。服务器无GitHub登录，采用已有本机Git认证中转：SSH读取远端相同提交到独立发布仓库，然后只推送指定5090分支。不复制凭据、不改全局代理。不改已有4090D工作区。
启动证据位于 `results/rtx5090/20261022-00-bootstrap/`。已核验提交回执写入后续检查点。
