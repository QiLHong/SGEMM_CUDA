# CUTLASS 对照执行记录

2026-10-22：启动。读取现有 Windows 初始化、BN64 robustness 构建运行和统计代码；将 CUTLASS adapter 与新 host harness 分给独立协作者，主执行负责工具链、实验协议、Windows 脚本和实际GPU测试。

已完成独立wrapper/harness/Build-Run-Profile脚本；选择两份官方TF32模板与原SIMT模板。build01在原有CUDA12.4/MSVC19.37上成功；因为编译同时发生host输出字段补充，主动把build01仅作为开发记录，全新build02固定最终快照。预运行计划排列缺陷已在冻结与GPU运行前修正。未修改CUTLASS头文件。

build02/正式run01启动；小矩阵两组各K32/64/96全量CPU double验算通过，TF32首批通过。FP32在3072 seed20261020发现K10的独立SIMT allclose标准失败4点，尽管绝对误差max0.005615<0.01；因原host把两标准合并为硬门限而中止。保留run01 partial和日志。修host门限/统计为两标准分别报告，不改阈值和device代码，下一次从全新build03/run02重新开始。

最终使用 build03/run04。run02 恢复时误启重叠进程，明确停止，保留全部 partial，按事先时间截止只复用三个已结束完整进程、重跑其余三批。共享文件锁已加入脚本并测试互斥。全部六进程不重叠、68040 events。analysis01 与独立 verification02 通过，复核恢复的258个批内文件、日志、遥测和执行记录。

NCU 十次定点采集成功；static01 确认 sm89、TF32 MMA、cp.async、无 TMA、无 spill。公平计时与 profiler 分开。CUT 小 TF32 配置四尺寸快于 BN64；strict FP32 CUT SIMT 不超过 K30/cuBLAS。FP32 3072 SIMT allclose 失败保留；TF32 严格绝对误差失败保留。

已写第34章、独立归档说明及复跑命令，更新 README/阅读路线/第32章后续链接/脚本和结果索引。实际执行 PowerShell 解析、作者 device 提取一致性、BN64 派生一致性；最终归档哈希与交付源码快照单独保存。不自动提交推送。
