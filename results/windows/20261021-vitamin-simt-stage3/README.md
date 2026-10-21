# RTX 4090 D · Windows · vitamin SIMT 跨库复现

本目录只归档 RTX 4090 D（24GB）在 Windows 原生 CUDA 12.4.131、
MSVC 19.37、Release / sm_89 下的第三阶段实验，不能混作其他 GPU 或 Linux 的成绩。
第一、二阶段目录及上游历史日志保留。

来源与实验口径：

- SGEMM_CUDA 基点 `545fb01a576ae38d84d00df6359e084eb14c94c2`，工作区叠加了尚未提交的前两阶段代码。
- vitamin-cuda `zh-cn-beginner` 精确基点 `520f97093a2abc239977be83d244b327d0e2e2ef`，工作区未修改。
- 五个 SIMT kernel 的 device 函数不改算法，只分离 PyTorch 包装；许可证及来源见 [vendor](../../../src/kernels/vendor)。
- A：上游 fast-math、普通 `CUBLAS_COMPUTE_32F`；B：无 fast-math、PEDANTIC FP32。
- 共同 `C=AB`、seed=20261020、相同输入字节、逐次恢复 C0；恢复在同一 stream 且在 event 外。
- 主表是原生移植复现。Anaconda base 无 torch，补查发现 `envs/pytorch` 已有兼容的 PyTorch 2.5.1 / CUDA 12.4；适配后的 Python 路线已在 `python04/` 成功完成，不安装依赖，也不混入 event 表。
- 所有性能采用未挂 profiler 的逐次 CUDA events；NCU 数据另存，不能并入性能表。
- Sanitizer 本轮未完成，不能将普通正确性通过写成无竞争/无越界的动态证明。

证据入口：

| 路径 | 内容 |
|---|---|
| `preflight/` | 两仓库状态、之前源码快照、工具链、原测试口径、历史文件指纹 |
| `mapping/` | 五 kernel 完整 CTA 输出、shared 生产/消费、bank 模型和 lane 示例 |
| `mapping-k30/` | 本轮复跑已有 K10/K30 完整 CTA 映射检查 |
| `run01/` | A/B 小尺寸校验与七轮 event、GPU 遥测及独立统计 |
| `profile01/` | B 配置的 NCU 报告、原始指标、命令及退出码 |
| `profile-cublas-B/` | 严格 cuBLAS 的2048/4096附加采集，各一次 |
| `static/` | A/B 实际二进制 PTX、SASS、资源与指令排列分析 |
| `python01/`～`python03/` | 保留编码、参数分词与新包装器解析错误的失败/部分运行记录；python03包含成功扩展编译日志 |
| `python04/` | Windows 包装器复用原 benchmark/diff_check，3尺寸×7项的单批墙钟结果、输入SHA与扩展产物 |
| `verification/` | 独立 event 审计、历史文件/旧源码保护记录和最终交付检查 |
| `delivery-source.zip`、`delivery-sha256.csv` | 最终源码快照及文件指纹；包含原有源码与本轮新增内容 |
| `archive-sha256.csv` | 本阶段归档文件的 SHA256 清单（不包含清单自身） |

原生数据共10080个正式event、336轮和492条校验，两种误差标准均通过。
另行Python测试共21行，其中18个候选校验也全部通过；复用原函数并做Windows包装适配，
没有运行原 `test.py` 默认TF32入口，不能称为原脚本不加修改的完整复现。

`python04/source-as-run/` 保存与实际命令中 SHA256 一致的执行版包装器；
当前脚本之后仅增加扩展产物自动归档逻辑，做过语法检查，没有再次运行 GPU。
差异及成功后补存产物的说明见该目录的 `provenance.json`。

`run01/environment.json` 在只检查 base Python 时记录了“torch absent”；
这仅说明当时该原生 run 没有调用 Python。完整环境盘点和兼容环境运行时检查见
`preflight/conda-environments-audit.json`、`pytorch-env-runtime-audit.json`，
不把该字段误读成整台电脑没有 PyTorch。

完整中文解释及结论见 [第26章](../../../docs/zh/26-vitamin-SIMT复现与K30公平对照.md)。
