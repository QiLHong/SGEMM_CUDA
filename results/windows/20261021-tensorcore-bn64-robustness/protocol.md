# 运行前固定的实验协议

冻结对象：vitamin_tf32_dbf、tf32_bn64_bk32、cublas_tf32。device 源文件与旧实验不变。
正式尺寸：1024、2048、3072、4096；seed：20261020、42、12345。
3 个顺序启动的独立 benchmark 进程，各自创建 CUDA 上下文；每批 12 案例。
各案例每实现 7 轮、每轮 10 预热、30 个单次 GEMM CUDA event。
预计正式 event 总数 22680；冒烟、数值参考、profiler 不计入。

具体执行顺序由 Run-TensorBN64Robustness.ps1 在任何 GPU 测试前生成 plan.csv 并计算 SHA256。
六种实现排列各出现一次；第七排列随批次变化。尺寸/seed 案例列表跨批次轮转。
不因已观察的计时改变顺序、seed、次数或阈值。

计算 C=AB、alpha=1 beta=0。沿用 xorshift 输入算法、有界原分布；准备阶段显式 __float_to_tf32。
所有实现复用 ta/tb 相同指针/字节；存储、累加和输出为 FP32。
输入转换不计入 GEMM event。C0 全零；每个 warmup 和正式调用前同一stream D2D恢复，event之外。
C0恢复会影响缓存，这不是完全冷缓存或端到端耗时。
cuBLAS COMPUTE_32F_FAST_TF32、MATH_DISALLOW_REDUCED_PRECISION_REDUCTION、ATOMICS_NOT_ALLOWED、GEMM_DEFAULT。

每例预检和每轮末检查完整输出；原作者baseline逐位比较为硬门槛。
另外记录同TF32 cuBLAS、量化输入PEDANTIC、原始FP32 PEDANTIC、独立CPU double的误差。
小矩形128x1024,K32/64/96 CPU全量；方阵CPU固定256点，与全量GPU检查分开。
abs<=0.01、SIMT allclose(.001)、TF32 allclose(.1)分别记录；allclose相对项为abs(got)。
保留原始FP32失败计数，不放宽阈值。

每组合：30 events均值 -> 7轮均值中位数 -> TFLOPS；逐轮配对提升单独计算。
列全36组组合，不把22680 events当独立统计样本；不剔除慢样本，不报告最佳轮作为主结果。
本轮不计算跨机器/跨天置信区间。温度/频率/功耗稀疏遥测，不锁频、不改功率。

Sanitizer先单个小案例memcheck：attach timeout20秒、进程总时限60秒，失败/超时即停止该项。
正常benchmark全部结束后固定seed20261020、2048/3072/4096×作者/BN64共6次NCU。
NCU kernel replay、clock-control none、cache-control none；每次最多180秒。权限/超时保留日志停止。
NCU时间不进入性能表；新计数器与旧证据作相关性解释，不把所有收益归为单因素。

完成统计、独立复算、报告和导航修复即停止；不新增算法/参数搜索/分派，不自动commit/push。
