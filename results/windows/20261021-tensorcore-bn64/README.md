# RTX 4090 D / Windows：BM128 BN64 BK32

本归档是2026-10-21本机4090D独立实验，没有混入3080、其他GPU或Linux历史成绩。
[第31章中文教程及复跑命令](../../../docs/zh/31-TensorCore-BN64优化与尺寸效应.md)
· [实验源码与来源](../../../experiments/tensorcore-bn64/README.md)

## 实际完成

| 路径 | 内容 |
|---|---|
| `preflight/` | Git起点及未提交patch、426个既有源码文件和1583个历史文件SHA256 |
| `build01/` | 成功但未运行；生成器误改计时轮换顺序，运行前已发现并留档 |
| `build02/` | 正式测量版本，实际sm89命令、二进制、PTX、SASS、资源及源码快照 |
| `mapping01/` | 输出、A/B复制地址、operand包、grid完整CPU枚举 |
| `run01/` | 四实现2520正式events；父BK32/新BN64相邻交换 |
| `analysis01/` | 初测统计与逐轮配对，未删除event |
| `confirm01/` | 三实现1890正式events；作者原版/BN64相邻交换 |
| `confirm-analysis01/` | 独立确认统计 |
| `profile01/` | 3尺寸×4实现，12次NCU完整rep、CSV和命令 |
| `confirm-analysis02/` | 同一批确认统计，加主NCU指标长表 |
| `profile-elapsed01/` | 2048/4096三自定义版本，6次补充NCU active/elapsed分母分析 |
| `elapsed-analysis01/` | 仍是同一批确认events，加补充NCU；不是第三批性能测试 |
| `metric-query.txt` | 本机NCU查询，确认elapsed指标支持 |
| `evidence01/` | 静态、保护、顺序、输入指纹和主NCU审计 |
| `evidence02/` | 最终审计，另包含补充6次NCU，共18次 |
| `source-as-delivered.zip` | 项目源码快照，排除历史results和build目录 |
| `source-as-delivered-sha256.csv` | UTF-8源码清单，与zip逐项核对 |
| `archive-sha256.csv` | UTF-8归档清单，覆盖生成时本目录除自身以外的文件 |

两批共4410正式events，小矩形另24events；156次全量原版逐位比较为0差异。
全部NaN/Inf为0；相对原始严格FP32的误差失败仍保留，不以TF32同精度相同冒充严格FP32通过。
CPU double：128×1024全量，方阵固定256位置；Sanitizer未执行。

初始生成器Python引号错误在生成实验前修复；build01轮换顺序修复仅影响host，
build01/02的所有device SASS相同。所有旧kernel的SASS与上一轮相同。
没有删除失败记录或慢样本，没有覆盖用户未提交内容，没有提交或推送。

## 独立七轮确认

| N | 原版TFLOPS | BN64 TFLOPS | TF32 cuBLAS TFLOPS | BN64/原版配对中位数 | BN64/cuBLAS |
|---:|---:|---:|---:|---:|---:|
| 1024 | 31.363 | 32.599 | 35.646 | +4.11%，7/7 | −8.45%，0/7 |
| 2048 | 48.503 | 57.998 | 51.973 | +19.56%，7/7 | +11.57%，7/7 |
| 4096 | 71.504 | 71.156 | 69.222 | −0.43%，2/7 | +2.79%，7/7 |

C=AB，seed20261020，相同显式TF32输入；7轮×10预热×30单调用CUDA event。
C0恢复在同stream、event外；转换和参考计算不计时。
TFLOPS由七个轮均值延迟的中位数换算，配对比值独立计算。
GPU未锁频、WDDM；43..53°C，SM2520..2715MHz。

102regs、48KiB用户shared、2CTA/SM、零spill。
2048平均SM活跃/经过周期75.17%→89.49%，Tensor elapsed33.26%→39.82%；
4096的SM活跃已约99%，新增读取导致DRAM364.27→493.35MB，没有超过原版中位数。
支持任务分布与访存代价的解释，不把整个tile改写包装成单因素因果实验。

归档SHA按原始字节计算；未来发布需避免Git换行转换改变CSV/日志哈希。
