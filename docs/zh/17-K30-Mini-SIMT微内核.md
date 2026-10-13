# K30：从局部技巧进入分层 SIMT 微内核

本实验针对同一台 RTX 3080 / SM86 / CUDA 12.4。K30 是独立研究入口，
不修改 K10，也不替换 `sgemm` 的默认实现。不使用 CUTLASS 库、Tensor Core、
WMMA、MMA、cp.async 或 warp specialization。

## 1. 实验边界与因果关系

|版本|CTA / threads|BK|shared stages|与谁比较、改变什么|
|---|---|---|---|---|
|K10|128×128 / 128|16|1|原实现，保留不动|
|K30 V1|128×128 / 128|16|1|建立分层基础；不是仅一处代码改动的 K10 消融|
|K30 V2|同 V1|16|2|相对 V1 只改变 staging / 同步协议|
|K30 V3|同 V1|16|4|相对 V2 只改变环形 shared stage 数|
|K30 V4|256×128 / 256|16|4|相对 V3 扩展 CTA，保持 warp、fragment、pipeline|

V1 先通过六尺寸 × beta=0/1/3 的 18 项验证，随后才实现 V2–V4。
4×8 lane lattice 的依据来自[上一轮公开 CUTLASS 审计](16-公开CUTLASS对照与主循环成本审计.md)
中的 SIMT policy/iterator 坐标推导；K30 自行实现这些映射，不链接或包含 CUTLASS。
不把旧 K15–K23 失败推广为“这一整类优化永远无效”。尤其 K18 没有覆盖
这里的 `WMITER=4, WNITER=2` 完整 ownership。

## 2. 分层映射与寄存器生命周期

```text
128×128 CTA：4 warps（2×2）；256×128 CTA：8 warps（4×2）
  └─ 每 warp 64×64
       └─ 32 lanes：4×8 lane lattice，M 方向 interleave=2
            ├─ GlobalIterator：float4 全局读 → 临时 fragment → shared store
            ├─ SharedLayoutA：offset(m,k) = k*(BM+4)+m
            ├─ SharedLayoutB：offset(k,n) = k*128+n
            ├─ WarpIteratorA/B：基指针 + 编译期 fragment 偏移；advance 一个 k
            ├─ FragmentIterator：逻辑坐标 ↔ A16/B8 ↔ accumulator128
            └─ 每个 k：16×8 次 FP32 FFMA → 最终 float4 epilogue
```

令 `w=threadIdx.x/32, l=threadIdx.x%32`：

- warp 行列：`wr=w/2, wc=w%2`。
- lane 格点：`lm=2*(l/16)+(l%2), ln=(l%16)/2`。
- `A[i]` 的 CTA 行：`64*wr+4*lm+16*(i/4)+i%4`，`i=0..15`。
- `B[j]` 的 CTA 列：`64*wc+4*ln+32*(j/4)+j%4`，`j=0..7`。
- `acc[i*8+j]` 对应两者笛卡尔积。每 lane 128 个独立输出，每个输出按 k 递增累加。

例如 warp 0 的 lane 0：A 行 `0..3,16..19,32..35,48..51`；
B 列 `0..3,32..35`。lane 1 的 A 行整体 +4，B 列不变；lane 2 的
A 行回到 lane 0，B 列整体 +4。完整 32 lane × 128 accumulator CSV
由 `analyze_k30_mapping.py` 生成，并验证两种 CTA 精确覆盖、无重复、无越界。

shared 物理布局没有 XOR，也没有伪造“更少 bank conflict”。变化发生在
lane/fragment 如何使用该布局。所有 shared 写入和读取都经过对应 layout 的契约。

寄存器逻辑生命期：128 个 accumulator 贯穿整个 K；A16+B8 在一个 k 的外积中复用；
global 临时 fragment 在一次 load/store 之间存活。128-thread 版本每次 global tile
临时数据为 A16+B16 共 32 个 float；256-thread 版本为 A16+B8 共 24 个 float。
这些逻辑数目不能直接相加充当 ptxas 物理寄存器数，编译器还要安排指针、谓词和临时值。

## 3. shared stage 生命周期与真实指令调度

V1 每 tile：`global load → shared store → barrier → compute → barrier`。

V2/V3 预填 `S-1` 个 shared tile，保留一个空 slot：

```text
future = tile + S - 1
source: load future into registers → compute current → store future → barrier
        current slot = tile % S       future slot = future % S
```

这里的 S 指 shared 容量，不是 S 组异步 copy；没有 cp.async，因此 `LDGSTS/DEPBAR`
预期为零。前一轮的 barrier 已保证要重用的 slot 没有旧读者；本轮 current 与 future
slot 不同，末尾 barrier 同时发布 future、结束 current 的读取。尾部不加载越界 tile，
但所有线程仍共同执行 barrier。没有按 lane 条件退出或跨 warp 的隐含同步。

重要的 SASS 反证：V2/V3/V4 的主循环 global LDG 都出现在 2046 条 FFMA 之后，
只剩 2 条 FFMA 在其后；V1/K10 的 LDG 在 2048 条 FFMA 之前。
**源码的 register lookahead 被编译器下沉了，不能声称实现了大范围 load/compute overlap。**
该静态位置由 `analyze_k30_schedule.py` 自动验证；它不是延迟周期或动态 stall 证明。
仍然保留下来的结构改变，是双缓冲允许把两次 block barrier 合为一次。

## 4. 验证与测量口径

- strict cuBLAS：`CUBLAS_PEDANTIC_MATH + CUBLAS_COMPUTE_32F_PEDANTIC + CUBLAS_GEMM_DEFAULT`。
- driver 脚本清除 `NVIDIA_TF32_OVERRIDE`；K30/K10 全部 FP32 SIMT。
- 相同输入，seed=20261013，alpha=0.5；六尺寸 128–4096，beta=0/1/3，绝对误差不超过 0.01，拒绝非有限误差。
- 公平计时使用 4096³、beta=3，每次调用前从相同 C0 复位 C；复位不计入 kernel event。
  这不是旧 benchmark 的 `C←alpha*AB+3*C` 无复位连续增长口径。
- 全局预热后每项 warmup=10 / repeat=50，七轮轮换顺序。各模式独立进程，GPU 串行执行；
  CSV 保留每项测量前后的温度、SM/memory clock。均值和波动均来自未插桩计时。
- Nsight 的冷缓存采集：仅 NVTX `measure/` 中一个 launch，kernel replay、cache=all、clock=base。
  profiler duration 不当作公平 benchmark 时间；不和旧文档的不同采样口径拼成同轮对照。
- `issue_inst0` 记作 no-issue，并非单独证明“无 eligible warp”；同时报告 eligible-warps、scoreboard、barrier。
- Compute Sanitizer 覆盖 memcheck、racecheck、synccheck；精确尺寸与通过情况见结果 CSV。

## 5. 结果

数据目录：[k30-20261013](../results/k30-20261013/)。原始日志和 `.ncu-rep`
保存在本机 `benchmark_results/k30-20261013/`，没有覆盖任何旧实验。

### 未插桩七轮性能

|版本|平均 GFLOPS|最小–最大|标准差|平均 ms|同轮 strict cuBLAS 比例|
|---|---:|---:|---:|---:|---:|
|strict cuBLAS|21592.73|21468.75–21617.44|54.76|6.3651|100%|
|K10|19652.21|19549.66–19706.22|61.07|6.9936|91.01%|
|K30 V1|18575.74|18465.13–18662.27|69.52|7.3989|86.03%|
|K30 V2|20801.16|20675.02–20860.63|85.55|6.6074|96.33%|
|K30 V3|15705.58|15544.37–15900.59|119.49|8.7514|72.74%|
|K30 V4|18375.95|18283.35–18460.66|65.52|7.4794|85.10%|

V2 相对 K10 平均 **+5.85%**；逐轮配对增益 **+5.52% 到 +6.27%**，
不是从多个版本中挑一次最高分。V2 相对自身无流水 V1 为 +11.98%。
V1/V3/V4 不满足性能成功标准，作为反例保留，不继续堆叠变体。

独立追加三轮确认（不合并进上面的七轮）：K10 **19561.30**、V2 **20660.52**、
strict **21465.31 GFLOPS**；V2 仍提升 **5.62%**，达到 **96.25%** strict。
均值下降随同轮基线一起出现，不能用前一轮较高值去除后一轮较低的 cuBLAS。
原始值见 `confirm-fair.csv`，汇总见 `confirm-performance.csv`。

### 资源与 Nsight：同轮冷缓存口径

|版本|regs/thread|shared 用户字节 / NCU 字节|occupancy %|issue %|FP32 pipe %|eligible warps/scheduler|no-issue %|
|---|---:|---:|---:|---:|---:|---:|---:|
|strict|202|49152 / 50176|16.66|87.32|78.55|1.579|12.68|
|K10|184|16640 / 17664|16.60|79.16|73.71|1.356|20.84|
|V1|199|16640 / 17664|16.54|73.08|68.34|1.254|26.92|
|V2|235|33280 / 34304|16.54|83.30|77.49|1.389|16.70|
|V3|233|66560 / 67584|8.33|59.89|55.66|0.599|40.11|
|V4|225|99328 / 100352|16.65|70.72|65.82|1.347|29.28|

NCU shared 值包含 1024 B 平台开销。所有 K30 ptxas spill-load/store=0，
stack frame=0；所有被测版本 NCU local-load/store sectors=0。
V2 的 235 寄存器高于 K22 阶段曾使用的 220 预算；本轮 K30 请求没有指定该上限，
因此如实保留资源代价，并维持独立研究入口，不把它悄悄写回 K10。
没有用 `maxrregcount` 强行把资源数字压低。

235 registers/thread 并未让 V2 比 K10 少驻留 warp：两个 128-thread CTA
仍能落在 SM 的寄存器/shared 资源容量内。V3 则因为 65 KiB shared/CTA
只能驻留一个 128-thread CTA，occupancy 减半是测量事实，不是“寄存器越多必然越慢”。
V4 一个 256-thread CTA 恢复 8 warp，但并未恢复 issue/FP32 利用率。
`launch_bounds` 是编译器约束，不是运行时驻留保证。

|版本|shared-load waves (M)|shared-load conflicts|DRAM read (GB)|L2 hit %|long scoreboard|short scoreboard|barrier stall|
|---|---:|---:|---:|---:|---:|---:|---:|
|strict|201.900|0|0.516|87.83|0.02345|0.08274|0.09522|
|K10|335.546|0|0.972|76.50|0.08478|0.02953|0.11210|
|V1|201.327|0|1.499|58.92|0.20368|0.11594|0.08437|
|V2|201.349|22480|0.984|76.04|0.18100|0.05148|0.07109|
|V3|201.327|0|1.179|73.39|0.23695|0.02958|0.03059|
|V4|201.327|0|0.701|79.13|0.30187|0.05449|0.07500|

scoreboard/barrier 单位为 warps per active cycle，不是时间百分比；
V2 的少量硬件 conflict 计数完整保留，不四舍五入伪装成零。
逐 PC SourceCounters 的理想地址模型中 K30 LDS excessive=0；不能将它与另一次
采集的硬件 conflict 计数混为同一统计量。shared-store wave/conflict、dispatch 等完整字段见 CSV。
cuBLAS 使用 LDGSTS，普通 LSU store counter 并不包含其全部 shared 写入成本，不能据此说 cuBLAS 写入完全无冲突。

另采 steady 口径（application replay、cache=none、clock=none、相同 warmup）：
K10→V2 的 issue **78.71→83.08%**、FP32 **73.29→77.05%**、
eligible warps **1.339→1.382**、barrier **0.11199→0.07088**，改善方向一致。
同时 long-scoreboard **0.10047→0.19519**，并没有变好；不能用局部收益掩盖反向指标。
该组只采 strict/K10/V2，没有用 cold 结果补其他版本的 steady 数据。

### SASS：同数量的 LDS，实际服务成本不同

|版本|静态总指令|FFMA|LDS|LDGSTS|DEPBAR|IMAD|IADD3|MOV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|strict|4496|1152|310|48|4|212|341|19|
|K10|2768|2176|96|0|0|20|26|29|
|V1|3408|2304|96|0|0|50|104|39|
|V2|3360|2304|96|0|0|58|120|21|
|V3|3496|2304|96|0|0|72|139|21|
|V4|3472|2304|96|0|0|64|139|21|

这是整个函数静态指令数，不直接比较工作量。K30 额外 128 条静态 FFMA 属于
beta=0 的另一条 epilogue 分支；在 beta=3 的动态执行中，所有版本有效 thread-FFMA
均为 **68,736,253,952**。没有更换精度，也没有少算乘加。HMMA/MMA 都为零。

逐 PC 实测（不是从总 bank-conflict 反推）：

- K10 96 个 hot LDS.128：32 个每次 2 wavefront、64 个每次 4 wavefront。
- K30 V1–V4 全部 96 个 hot LDS.128：每次均为 2 wavefront。
- K10/K30 动态 LDS 次数都为 **100,663,296**，但主循环模型 wavefront 从
  **335,544,320 降为 201,326,592（−40%）**。即每 warp 每 k 从 20 waves 降到 12。
- 动态 BAR 从 K10/V1 的 **2,097,152** 降到 V2/V3/V4 的 **1,052,672**：
  `4096 warps × (256 K tiles + 1 prefill barrier)`，与源码同步生命周期一致。
- V2 动态 IMAD=16,826,368、IADD3=10,706,944、MOV=2,166,784；K10 分别
  为 15,749,120 / 3,239,936 / 4,247,552。不能说 iterator 把所有整数指令都减少了。

这说明新 ownership 确实降低 shared 服务成本，但 V1 的负收益同样真实。
V2 才让“较低 fragment 服务成本 + 单次 barrier 的双缓冲组织 + 保持 8 resident warps”
共同转化为吞吐收益。不能把 +5.85% 全部数值归因于其中一个因素：本轮没有完成完整的
ownership × barrier 协议因子消融，而且源码 lookahead 没有变成大范围硬件重叠。

### 正确性和工具检查

最终 binary：K30 四版 72/72，加 K10 对照 18/18，共 **90/90 PASS**；
最大绝对误差 0.000610351562，阈值仍为 0.01。
memcheck：四版 × 128/256/4096，共 **12/12 PASS**；
racecheck 与 synccheck：四版 × 128/256 × 两工具，共 **16/16 PASS**。
这些是指定尺寸的验证，不宣称对任意尺寸做了形式化证明。

## 6. 对核心问题的回答

**为什么 K10 只有约 91%？** 本轮找到可测量、可改进的成本：相同 LDS 指令数量下，
K10 的 fragment ownership 需要更多 shared wavefront；其单 shared tile 协议每 BK
需要两次 block barrier。两者与调度相互影响，K30 V2 在保持 occupancy 的情况下
提高了 issue/FP32 利用率，已从 91.01% 提升到 96.33%。这不是“证明所有差距都来自 bank conflict”。

**为什么 `256x128_8x4` 不是简单扩大 CTA？** public CUTLASS 命名中的 8/4 是 BK/stages，
不是一个线程只有 8×4 accumulator。相同名称并不足以证明 cuBLAS 私有实现的每层映射。
cuBLAS 本轮只有约 49 KiB 用户 shared、202 regs；我们的 V4 是 97 KiB / 225 regs，
BK 也不同，虽然 CTA 外形相同，资源与时间组织并不相同。V4 occupancy 接近 cuBLAS
仍慢，不能归咎于“warp 不够”，更不能据此声称私有实现一定用了 warp specialization。

**哪个 K30 设计有效？** 是 V2 的组合，不是独立 iterator 抽象，也不是 stage 越多越好。
V1 只改善服务量却不改善吞吐；V3 的额外 stage 削弱 resident warp；V4 恢复 warp 数但
long-scoreboard 高、issue 低，显示 CTA、数据复用与搬运/计算节奏仍未协调好。
V2 同时满足 GFLOPS、issue、FP32 pipe 提升且无 spill 的本轮接受标准，但不替换默认 K10。

**剩余差距如何解释？** V2 距同轮 strict cuBLAS 还差 3.67%，并非已经追平。
fragment-load wavefront 已接近 cuBLAS，但冷缓存 DRAM read 仍约 1.91 倍，
long-scoreboard 仍更高，issue 仍差约 4.02 个百分点。它们是后续定位线索，不是
可相加的耗时份额；不能仅凭这些 counter 把剩余差距唯一分配给 CTA/layout/pipeline。
当前证据支持 **E：多因素联合**，而非某个缺失的小技巧。

下一步只值得做窄范围因果验证：冻结 V2，显式把 global load 放到 compute 后，
比较 ptxas 是否生成相同/更短的 lifetime，确定收益是否主要来自同步协议；随后才考虑
固定此协议做 ownership 对照。先解释已经获得的收益，不扩展 tile/stage 搜索。

## 7. 可复现入口

```bash
OUT=benchmark_results/k30-reproduce-unique
bash scripts/build_k30_audit.sh "$OUT"
python3 scripts/analyze_k30_mapping.py --output "$OUT/lane-mapping.csv"
python3 scripts/analyze_k30_schedule.py "$OUT/audit" --output "$OUT/schedule.csv"
python3 scripts/run_k30_audit.py correctness --binary "$OUT/audit" --output "$OUT/tests"
python3 scripts/run_k30_audit.py fair --binary "$OUT/audit" --output "$OUT/timing" --rounds 7
python3 scripts/run_k30_audit.py profile --binary "$OUT/audit" --output "$OUT/profiles"
python3 scripts/run_k30_audit.py memcheck --binary "$OUT/audit" --output "$OUT/sanitizer"
python3 scripts/run_k30_audit.py sync --binary "$OUT/audit" --output "$OUT/sanitizer"
python3 scripts/run_k30_audit.py source --binary "$OUT/audit" --output "$OUT/source"
python3 scripts/run_k30_audit.py fair --binary "$OUT/audit" --output "$OUT/confirm" --rounds 3 --modes strict k10 v2
python3 scripts/run_k30_audit.py profile --binary "$OUT/audit" --output "$OUT/profiles" --policies steady --modes strict k10 v2
python3 scripts/summarize_k30_audit.py "$OUT" --output docs/results/k30-reproduce-unique
python3 scripts/analyze_cutlass_reference_source.py "$OUT/source" \
  --modes strict k10 v1 v2 v3 v4 --skip-mapping --output docs/results/k30-reproduce-unique/source
```

所有脚本拒绝覆盖已有同名输出。构建为 `-O3 -DNDEBUG -std=c++17 -arch=sm_86 -lineinfo`，
并保存 ptxas 统计、binary/source SHA256。也可用 CMake Release 加
`-DSGEMM_BUILD_K30_AUDIT=ON` 构建 `k30_audit`；实际结果使用脚本构建的 binary。
支持方阵 N=128 的整数倍、范围 128–4096；不声称支持任意 stride、尾 K 或矩阵别名。
所有 kernel 数值实验使用脚本构建产物；另外已验证 CMake Release 目标编译成功，
Python 语法、shell 语法与 Git whitespace 检查通过。纯翻译验证仍只针对 `aadcd8e`，
35 个原文件检查失败数为零，不把 K30 算法改动伪装为翻译。

版本 diff 不靠复制四份主体：同一 `sgemmK30<Shape,Stages>` 的 `STAGES==1` 与环形
分支直接呈现 V1/V2 的差异；V2→V3 只有 `Stages:2→4`，V3→V4 只有 `BM/Threads:128→256`。
`SharedLayout`、`FragmentIterator`、`compute`、`epilogue` 对所有 K30 版本共用。

编译期版本差异（完整主体见独立 header）：

```diff
// V1 → V2: 同一 Small shape，单缓冲改为双缓冲同步协议
-runK30<Small,1>(...);
+runK30<Small,2>(...);
// V2 → V3: 只扩 shared ring
-runK30<Small,2>(...);
+runK30<Small,4>(...);
// V3 → V4: 保留 4-stage 协议，128×128/128 threads → 256×128/256 threads
-runK30<Small,4>(...);
+runK30<Large,4>(...);
```

相对 K10 的结构差异是独立分层 iterator、A16/B8 的 ownership、允许 beta=0 跳过
C 读取的 epilogue 与 M 尾行屏蔽。浮点外积及每个输出的 K 累加顺序不改变。
K10 文件的 Git Diff 为零；未伪装成只有一行参数变化的优化。

## 8. 后续：K31 没有把精修包装成新突破

本页及 K30 CSV 保留原始轮次，不用新结果覆盖。后续
[K31 审计](18-K31-微内核精修与因果审计.md)已实际执行上述 late-load 因果对照，
并测试逐 float4 临时量、编译器寄存器预算、3/4-stage 与 256-thread 双阶段。
K30 七轮再次达到 96.30% strict；晚加载/逐向量虽然降到 219/209 regs，却分别慢
3.39%/7.98%。r240 独立五轮约 +0.24%，但 issue 未改善，未接受为新最佳。
新的结论与完整验证见 K31，不回写成 K30 历史实验当时已经知道的事实。
