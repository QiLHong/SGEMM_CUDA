# 从矩阵乘法到 CUDA 微内核

一本以本仓库 RTX 3080 SGEMM 实验为线索的中文微教材。

你已经有了优化故事和面试题，但“知道结果”与“能够推导”之间还有一段路。
这本小书不按问答罗列知识，而是带你完成一次思考过程：先写对一个输出，再把
输出交给线程；发现访存问题后学习分块；最后用真实实验理解资源、同步与调度。

前置要求只有基本 C/C++ 数组、循环、函数和指针。无需先懂 CUTLASS、PTX 或
Nsight。前六章的推导可以用纸笔和 Python 完成；第七章才介绍 GPU 验证入口。
CPU 练习不是 CUDA 正确性证明，也不产生 GPU 性能结果。

## 这本书怎样读

每章包含一个主问题、完整例子、工程源码精读和三道带答案的练习。先遮住答案做题，
再核对；不要把“看懂了答案”当作“自己会推导”。可以每天读一章，遇到代码就停下来
画坐标，而不是连续看完全部内容。

|章节|本章主问题|读完应能做的事|
|---|---|---|
|[第 1 章 从点积到内存地址](01-从点积到内存地址.md)|GPU 究竟在算什么|手算乘积并写出 A/B/C 地址|
|[第 2 章 把输出分给线程](02-把输出分给线程.md)|一个输出归谁负责|区分 grid、block、warp、lane、SM|
|[第 3 章 访存与数据复用](03-访存与数据复用.md)|为什么算得对却慢|列出同一 warp 的访问地址|
|[第 4 章 分块与寄存器小矩阵](04-分块与寄存器小矩阵.md)|怎样把一次读取用得更充分|推导 block tile 和线程外积|
|[第 5 章 Warp 与 Shared 映射](05-Warp与Shared映射.md)|同样的输出可以怎样分配|区分 ownership、物理布局和 fragment|
|[第 6 章 同步与双缓冲](06-同步与双缓冲.md)|什么时候才能读或覆盖|画出两个槽位的生命周期|
|[第 7 章 验证与性能分析](07-验证与性能分析.md)|怎样证明一个改动有效|区分精度、计时、资源和调度证据|
|[第 8 章 用 K30 串起整个项目](08-用K30串起整个项目.md)|这些知识如何形成一次研究|解释成功与失败，不夸大因果|

## 源码伴读路线：每章真正打开哪些文件

本次扩写按代码快照 `99d0d623870b2741a9c0d1a4e09b3cdd8087aefd` 核对实现；
教材修改不改变 kernel。相对源码链接默认打开你当前浏览的分支版本，`#L` 帮助
定位；以后行号移动时，以表中的函数名搜索。需要永久复核时可从
[该代码快照](https://github.com/QiLHong/SGEMM_CUDA/tree/99d0d623870b2741a9c0d1a4e09b3cdd8087aefd)
进入同名文件。

|章|工程入口与搜索词|这一章要亲手完成的追踪|
|---|---|---|
|1|[1_naive.cuh](../../../src/kernels/1_naive.cuh)：tmp、K 循环|将 C[1,0] 的三次累加展开成地址和值|
|2|[runner.cu](../../../src/runner.cu)：run_sgemm_naive/coalesce|从 launch 配置推导 K1/K2 的第一个 warp|
|3|[3_kernel_shared_mem_blocking.cuh](../../../src/kernels/3_kernel_shared_mem_blocking.cuh)：As/Bs|说明 tid=33 依赖哪两个线程搬来的数据|
|4|[5_kernel_2D_blocktiling.cuh](../../../src/kernels/5_kernel_2D_blocktiling.cuh)：regM/regN/threadResults|区分 tid=17 的搬运职责与 8×8 输出职责|
|5|[30_kernel_mini_simt.cuh](../../../src/kernels/30_kernel_mini_simt.cuh)：FragmentIterator/WarpIterator|从 acc[46] 反查 shared 坐标、global 搬运者和 C|
|6|[同一 K30 文件](../../../src/kernels/30_kernel_mini_simt.cuh)：sgemmK30 的 STAGES 分支|画出 prologue、三轮循环与最后一次 barrier|
|7|[k32_audit.cu](../../../scripts/k32_audit.cu)、[run_k30_audit.py](../../../scripts/run_k30_audit.py)|跟踪 C0 恢复、校验、event、RESULT 与 CSV|
|8|[K30 归档](../../results/k30-20261013/)、[K30 报告](../17-K30-Mini-SIMT微内核.md)|从同批原始数据重新计算 +5.85% 和 96.33%|

推荐把教材与源码并排打开，每次只追一个线程或一个 accumulator。
每段先回答“这行做了什么”，再回答“为什么需要”，最后才问“会不会更快”。
不要跳过 host runner：只有知道实际模板参数和 launch 配置，设备端公式才有意义。

终端里可这样定位（只读，不编译、不运行 GPU）：

```bash
rg -n 'FragmentIterator|WarpIterator|epilogue|sgemmK30' src/kernels/30_kernel_mini_simt.cuh
nl -ba scripts/k32_audit.cu | sed -n '89,139p'
```

源码摘录会说明省略的部分；标为示意的片段仅解释流程，不可直接当完整实现运行。
特别留意：K1/K2/K3 与 K4 之后的 grid 轴约定不同，K30 不在教学编号 dispatcher
里，第七章的 audit 才是相应研究入口。

## 配套练习怎样验证

章节间有前后导航。做题时可运行：

```bash
# 仓库根目录，Python 3 标准库即可；不需要 CUDA 或 GPU。
python3 scripts/check_microbook_examples.py
```

[练习脚本](../../../scripts/check_microbook_examples.py)验证小矩阵、地址、分块等价性、
K10/K30 输出覆盖、标量 bank 模型以及性能单位换算。扩写增加了 K3/K5 的具体
线程追踪、K30 float4 搬运到 fragment 的坐标覆盖与双槽头尾的顺序模型检查。
它不会运行 benchmark，
也不会修改历史实验。源码中的断言适合边读边对照。

## 始终分开的三类内容

- **手算模型**：例如 2×3 乘 3×2，或 32 个 lane 的标量 bank 计算，用于建立直觉。
- **源码事实**：例如 K30 的 lane 坐标公式、两槽 shared 的字节数，可直接检查实现。
- **测量结论**：例如 20.801 TFLOPS，必须回到对应硬件、输入、工具与 CSV。

不能由第一类直接推断第三类。本书既有明确标注的示意片段，也有链接到真实文件
的源码摘录；二者都不是新的完整通用 SGEMM 实现。真正的 GPU 验证使用已有
audit harness。第七章专门区分小规模学习子集与完整报告发布要求。

## 与已有文档的关系

本书是连续讲授；[基础复习](../22-CUDA与SGEMM基础复习.md)适合查词与快速回顾；
[优化故事](../06-RTX3080-K10优化故事.md)按实验发生顺序记录证据；
[复现手册](../20-项目复现与简历讲述.md)提供完整命令；
[模拟面试 QA](../21-SGEMM模拟面试QA.md)用于读完后的表达训练。

项目代码来自开源 SGEMM 教学框架及本分支后续研究；本书是 AI 辅助编写的学习材料，
不是 Simon Boehm 原文翻译，也不是 NVIDIA 官方教材。CUDA API 语义以各章链接的
官方文档为准。更新于 2026-10-14；研究保持暂停，K30 V2 仍为最佳研究基线。

开始阅读：[第 1 章](01-从点积到内存地址.md)。
