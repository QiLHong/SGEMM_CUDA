# RTX 5090 GEMM 分阶段复现

日期：2026-10-22（Asia/Shanghai）。阶段0已完成：环境、sm_120构建、最小GEMM、SM120双精度小形状校验及244项历史配置清单。作者BF16/4096原墙钟入口已完成；七轮统一CUDA event正式对照待完成。

唯一发布分支：`results/rtx5090-gemm-reproduction-20261022`。
启动 GitHub HEAD：`a2b23dabadc79440bf9cbf63f0b566f7eec429ba`。参考 vitamin-cuda：`520f97093a2abc239977be83d244b327d0e2e2ef`，只读。
已确认 Ubuntu22.04.5 / RTX5090 / SM120 / 170SM / CUDA12.8.93 / PyTorch2.8.0+cu128。最小编译测试和BF16小形状三类Sanitizer已通过；NCU受宿主机权限限制。

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

## 当前证据导航

- [环境与构建](stage0-environment.md)
- [SM120源码与教程适用性](sm120-source-audit.md)
- [244项复现清单](reproduction-matrix.csv)
- [机器可读进度](progress.json)
- [历史入口说明](../../../results/rtx5090/20261022-03-inventory/entrypoint-notes.md)

- [BF16/4096作者原协议](../../../results/rtx5090/20261022-04-author-bf16-4096/original-wallclock.csv)：单批10warmup/200repeat，含host包装和descriptor准备，非纯kernel计时。全部五项作者低精度检查通过。

- [阶段1：SM120 TMA、cp.async、cuBLAS实测与失败证据](stage1-results.md)：六个正式批次共7560个event；原作者墙钟结果另表。

- [阶段2：严格 FP32 核心对照](stage2-results.md)：K10 PAD4/PAD8、K30 V1/V2、PEDANTIC cuBLAS，两协议八批。
