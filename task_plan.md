# CUTLASS RTX 4090 D 公平对照

目标：在 Windows 原生 CUDA 12.4 / MSVC 19.37 / sm_89 上实际运行公开 CUTLASS，与已有 BN64、作者 TF32、K10/K30 和同精度 cuBLAS 比较。保留既有结果，不自动提交推送。

## 阶段
1. [完成] 核验工具链、CUTLASS 来源、Git 状态及精度协议。
2. [完成] 新建独立 CUTLASS wrapper、benchmark 和 Windows 构建运行脚本。
3. [完成] 编译命令/指令核验，小尺寸正确性与非法尺寸拒绝；数值标准各自判定，保留 FAIL。
4. [完成] run04 最终 68040 events，所有误差、输入指纹和遥测保留；完整进程恢复链核验。
5. [完成] analysis01 + verification02 独立复算，10 次 NCU、静态证据、第34章与复跑命令；交付哈希清单。

## 约束
- TF32 与 strict FP32 分组，C=AB；恢复 C0 在同 stream、event 外。
- 每轮 warmup=10、repeat=30，7 轮；以轮均值延迟的中位数汇总。
- 不改旧 device kernel、工具链、驱动、代理、Conda 或 Codex 配置。
- CUTLASS 模板参数、版本、输入量化与计时范围必须显式记录。
- 不能把有限模板配置的结果称为 CUTLASS 最佳性能。

## 错误记录
- 预运行审阅发现初始排列公式对4项FP32只生成4种排列；冻结前改成 floor(round/2) 偏移，生成7种不同排列。没有产生无效正式样本。
- build01编译期间host meta字段补充造成源码/二进制关联歧义，保留开发记录；用全新构建目录生成build02作为唯一正式候选，未使用build01跑GPU。
- run01烟测通过，但正式FP32第一批到3072/seed20261020时，K10相对strict cuBLAS的abs<=0.01通过（max0.005615），SIMT allclose eps0.001 有4点失败，原脚本因此停止。原始partial证据保留。两种历史标准本应分别判定，后续host验证保留两套原阈值及FAIL计数，用abs门限决定是否完成测量；报告会明确SIMT标准失败。需新建build03/run02，绝不把run01失败样本并入新正式表。
- 静态脚本第一次在组装loop_positions时遇到重复pc关键字的TypeError，未创建STATIC_AUDIT_PASS；已修正，待公平基准结束后重跑避免并行CPU干扰。
- 用户连续中断后，用Get-Process未能发现旧工具会话仍运行，启动run03恢复导致2026-10-21T17:31:12Z左右开始短暂双进程重叠。已通过write_stdin Ctrl+C明确终止旧session48782及恢复session87301。run02的TF32批2/3与run03新跑数据全部判为无效；仅复用17:30:25Z之前已完成的三个批次（TF32批1、FP32批1/2），在run04补齐其余三批。新增脚本共享文件锁，并记录恢复源和复用文件SHA；不依赖Get-Process作为唯一运行判断。
- recovery verifier 最初用 JSON 时间字符串比较，遇到 PowerShell 将 +00:00 重序列化成 +08:00 的同一时刻而失败；已改为带时区 datetime 比较，非时间字段仍精确比较。verification02 实际通过，未修改原始 JSON。
- 最终交付不提交、不推送；sanitizer 未验证，未做 CUTLASS 全库调优或新 kernel 优化。
