# RTX 4090 D：作者原入口复跑与寄存器上限实验

本目录为 Windows 原生、RTX 4090 D 24GB、CUDA 12.4.131 / MSVC 19.37 的独立补充实验。
第三阶段主归档保持冻结；来源为 vitamin-cuda 提交
`520f97093a2abc239977be83d244b327d0e2e2ef`，没有修改其工作区。

两条路线分别记录：

- `author-source/`：直接执行作者原 `test_4096()` 的 SIMT 部分，原函数体、
  benchmark 和 diff_check 不变。通过库代理在末尾首次 TF32 预热调用前停止。
  七批固定顺序，每批原预热10次、墙钟200次，外层固定seed42。
  Windows load适配属于包装差异，不能称默认main原封完整执行。
- `register-limit/`：严格FP32、C=AB、独立DBF默认编译与寄存器上限128对照。
  设备源码、tile、线程映射与同步协议不变；逐轮交换两个版本先后。
  正常event计时与NCU采样分开。
- `preflight/`：本轮开始时Git状态、源码快照及旧归档冻结清单。

作者原文使用RTX 5060移动版，并公开过两组不同时间的摘要/详细日志。
本机实验只检验该源码在4090 D上是否领先本机cuBLAS，不能用跨GPU时延否定作者原始测量。
原始Python墙钟结果、严格原生events和作者历史成绩分开呈现。

实际完成：

- 作者原入口7批、49项墙钟记录；每批6次原校验通过由原控制流到达TF32停止点确认。
  4096中位时延：普通cuBLAS 2.975850 ms、DBF 3.229833 ms；
  7个逐批DBF/cuBLAS吞吐比均小于1。
- 寄存器实验保存1260个正式event、42轮、132条校验，两种误差标准全部通过。
  限制版相对默认版的逐轮配对提升中位数：1024 −7.398953%、2048 −2.497720%、4096 +2.610080%。
  成功统计目录为 `register-limit/run01/analysis02`；初次统计的MHz单位解析错误保留在 `analysis/`，
  仅修正CPU统计后重算，没有重跑或删除GPU样本。
- 完成8次独立NCU采样。寄存器143→128、零spill/local流量；4096 occupancy约16.66%→31.97%。
  实际shared配置也由65536→102400字节/SM，SASS调度改变，不能只归因于occupancy。
- 正常计时、NCU与原入口墙钟分目录保存；Sanitizer仍未验证。

`author-source/independent-statistical-audit.json` 和
`register-limit/run01/independent-statistical-audit.json` 为独立原始数据复算。
`register-limit/static/` 保存默认版/上限版的SASS/PTX分析；
`register-limit/profile-analysis/` 保存CPU源码一致性审计与NCU指标汇总。

完整说明见 [第27章](../../../docs/zh/27-vitamin原入口复跑与寄存器上限实验.md)。

最终交付：`delivery-source.zip` 与 `delivery-sha256.csv` 保存当前源码；
`verification/` 记录新增/追加文件、旧源码与第三阶段归档保护检查；
`archive-sha256.csv` 覆盖本目录文件，清单自身除外。
