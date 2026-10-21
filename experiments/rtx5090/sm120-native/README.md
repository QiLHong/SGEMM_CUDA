# SM120 原device源码 + 原生host计时

本目录仅新增host包装与审计脚本，不改上游device实现。`build_native.py`从固定vitamin-cuda工作树读取`hgemm_sm120.cu`和`cublas.cu`，把common include改为实际绝对路径，在PYBIND之前插入`native_host.inc`。原device区域从`#define SWIZZLE_A`至`#define CHECK_T`逐字节断言相同，并写SHA256清单。上游原始文件不写回。

此目录交付时只做Python AST检查，编译与GPU结果由服务器的独立结果目录记录。原上游Python入口仍是另一条实验路线；本目录命名为**原device源码、原生host审计**。

## 计时边界

每次native `audit_measure`先取PyTorch当前stream和cuBLAS handle，核对同stream/host pointer mode，记录math mode和实际cuBLAS版本，创建TMA descriptors、设置dynamic shared并查询资源/occupancy；全部在计时外。TMA32/64每组只编码两次tensor map，重复调用共用descriptor。每次warmup和正式调用前以同stream D2D copy恢复C0；正式start event位于copy之后。每次只发起一个GEMM，stop event后同步读取毫秒。

native CUDA event含设备执行/库内部所需kernel；分配、输入转换、descriptor创建、C0恢复、资源查询与验证均不在event中。cuBLAS调用的算法和内部kernel数由当前库决定。本计时不是端到端延迟；`descriptor_encode_cpu_us`只单列descriptor编码墙钟，不能加总成完整端到端。与Python API-dispatch CUDA event分表。

## 协议与校验

正式批次固定5实现×7轮×30次=1050个正式event，每组10预热；预生成轮换并反转顺序。每次同输入A/B及C0；保存raw FP32和实际FP16/BF16字节指纹。默认normal是CPU FP32 randn后CPU舍入，另有uniform[-1,1]选项且必须单独目录归档。

所有轮末输出对完整同dtype cuBLAS、实际输入FP32 PEDANTIC参考再同dtype舍入、原始FP32输入PEDANTIC参考分别报告。smoke另做CPU double全输出；大尺寸256个固定坐标、全K归约CPU double。保留作者`.016/.016`allclose（相对项第二参数candidate）及strict abs.01诊断；原dtype allclose和FP64复算的阈值差异分开记名，不调阈值。任何一类失败均保留原始结果。

`resources_and_setup.json`含每轮各核registers、local字节、静态/动态shared、binary/PTX目标、occupancy API的active blocks/SM，以及实际device资源。它不是NCU动态occupancy，也不把local字节自动等同动态spill流量。

## 命令

先用服务器既有隔离环境、`TORCH_CUDA_ARCH_LIST=12.0`、已有ninja。`--build-root`是新的专用Release编译目录，不能指向历史构建。`--output`必须不存在。

```bash
python build_native.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --build-root /root/autodl-tmp/build-sm120-native-audit
python run_native.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --build-root /root/autodl-tmp/build-sm120-native-audit --output smoke-bf16-k128 --mode smoke --dtype bf16 --k 128
python run_native.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --build-root /root/autodl-tmp/build-sm120-native-audit --output smoke-bf16-k384 --mode smoke --dtype bf16 --k 384
python run_native.py --upstream /root/autodl-tmp/vitamin-cuda-rtx5090/kernels/hgemm_sm120 --build-root /root/autodl-tmp/build-sm120-native-audit --output bf16-1024-seed42 --mode events --dtype bf16 --size 1024 --seed 42
```

smoke形状M128/N1024、K128或K384；一次5个event仅验证而非性能结论。计时主表逐尺寸/seed完成后立即发布；只存在PARTIAL.json且无COMPLETED.json的目录不得进入完整统计。
