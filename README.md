# 从零开始优化 CUDA SGEMM：中文学习版

用 CUDA 逐步优化矩阵乘法。各个 kernel 的原作者讲解见 [Simon Boehm 的 CUDA-MMM 文章](https://siboehm.com/articles/22/CUDA-MMM)。

本分支 `zh-cn-learning` 基于 `5a7dcc513d951ba764d51bc9d587b3163f3a894d`，翻译仓库文档和注释，并加入面向初学者的解释。**不改变 kernel 算法、线程配置、矩阵尺寸、数值校验门槛和运行输出格式。** 原作者英文 README 完整保留在 [README.en.md](README.en.md)，原版继续保留在 `master`。

原项目：[siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)。原作者：Simon Boehm。中文内容属于 AI 辅助翻译与学习补充，并非原作者审校版，也不是原博客的完整译文。`【学习补充】` 表示新增解释，`【注释辨析】` 表示原注释与实际表达式之间需要注意的区别；没有把这些问题偷偷改成另一套实现。

## 概览

以下是**原作者在 NVIDIA A6000（Ampere）上的历史结果**，不是本分支在你的 RTX 3080 上重新测得的数据：

![原作者 A6000 性能曲线；原图及英文图例保留](benchmark_results.png)

方阵边长为 4096 时的 GFLOP/s，保留上游数值和排序：
<!-- benchmark_results -->
| Kernel | GFLOP/s | 相对 cuBLAS 的性能 |
|:--|--:|--:|
| 1：朴素实现（Naive） | `309.0` | 1.3% |
| 2：全局内存合并访问（GMEM Coalescing） | `1986.5` | 8.5% |
| 3：共享内存缓存（SMEM Caching） | `2980.3` | 12.8% |
| 4：一维线程分块（1D Blocktiling） | `8474.7` | 36.5% |
| 5：二维线程分块（2D Blocktiling） | `15971.7` | 68.7% |
| 7：通过重排减少 bank 冲突（Linearize） | `16213.4` | 69.7% |
| 8：通过填充减少 bank 冲突（Offset） | `16459.2` | 70.8% |
| 11：双缓冲（Double Buffering） | `17278.3` | 74.3% |
| 6：向量化内存访问（Vectorized Mem Access） | `18237.3` | 78.4% |
| 9：自动调优（Autotuning） | `19721.0` | 84.8% |
| 10：线程束分块（Warptiling） | `21779.3` | 93.7% |
| 0：cuBLAS | `23249.6` | 100.0% |
<!-- benchmark_results -->

当前源码还包含 12 号异步复制双缓冲版本，但上游这张结果表没有它的数据。编号表示实现版本，不表示性能排名；不要据此要求每个后续版本都更快。

## 环境与构建

上游列出的依赖为 CUDA Toolkit 12、Python（以及 Seaborn）、CMake 和 Ninja，辅助 Python 环境见 [environment.yml](environment.yml)。

**学习补充：** 编译并运行 `sgemm` 主要需要 CUDA Toolkit / cuBLAS、C++ 编译器和 CMake。Python 用于画图等辅助工具，Ninja 或 Make 则是构建后端。`environment.yml` 不是完整且锁定的 CUDA 工具链；`nvidia-smi` 中的 CUDA 版本也不能替代 `nvcc --version` 检查。

检查 GPU 的 [compute capability](https://developer.nvidia.com/cuda-gpus)，再设置 `CMakeLists.txt`。原英文 README 的 `80` 是示例；当前源码实际设置为：

```cmake
set(CUDA_COMPUTE_CAPABILITY 86)
```

这与 RTX 3080 的目标架构匹配。这里填的是计算能力，不是 CUDA 的 `12.4` 版本号。

上游构建方式为 `mkdir build && cd build && cmake .. && cmake --build .`。为了区分调试与性能测试，中文学习示例明确使用 Release，并使用与仓库 Makefile / autotuner 一致的 Makefiles 生成器：

```bash
# 在仓库根目录执行；不要覆盖其他项目或修改已经验证的 liteq3080 环境。
nvcc --version
cmake --version
c++ --version

cmake -S . -B build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2

# 索引入门演示。
./build/simplest_kernel

# 先比较基线、朴素实现、合并访问；每次命令会测试源码中的多个矩阵尺寸。
DEVICE=0 ./build/sgemm 0
DEVICE=0 ./build/sgemm 1
DEVICE=0 ./build/sgemm 2
```

如果已有 `build/` 是 Ninja 生成的，不要在同一目录混用生成器。可以另选构建目录；使用原 `make profile` 或 autotuner 前，则需检查它们固定依赖的 `build/` 和 `make`。Debug 会使用 `-G`，不适合拿来比较吞吐率。当前 CMake 仍使用上游 `find_package(CUDA)`，这里没有顺带升级构建系统；新版本 CMake 的兼容问题应独立处理。

在构建目录内，运行格式仍是 `DEVICE=<设备编号> ./sgemm <kernel编号>`。`DEVICE` 是这个程序读取的变量。通过 [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) 分析性能的上游入口为：

```bash
# 从仓库根目录运行；需事先安装 ncu 并具备性能计数器权限。
mkdir -p benchmark_results
make profile KERNEL=2 PREFIX=learning_
```

性能分析器可能重放 kernel，分析耗时不要当作正常运行延迟。中文分支没有执行 GPU 编译、benchmark 或性能分析，**不包含新的 RTX 3080 性能保证**。

## 中文学习入口

建议按下面顺序读，每次只解释清楚一个变化：

1. [00：阅读路线](docs/zh/00-阅读路线.md)：项目调用链、文件职责、每个阶段要回答的问题。
2. [01：矩阵索引与线程映射](docs/zh/01-矩阵索引与线程映射.md)：`[M,K] × [K,N]`、行主序、thread / block / warp、Kernel 1 与 2。
3. [02：分块与流水线](docs/zh/02-分块与流水线.md)：shared memory、寄存器复用、float4、bank、warp tiling、双缓冲。
4. [03：原版限制与验证](docs/zh/03-原版限制与验证.md)：哪些前提不能忽略、哪些工具会修改源码、如何检查注释变更没有改逻辑。

源码中的 1–12 号 kernel、`sgemm.cu`、`src/runner.cu`、构建与调优脚本均有中文解释。函数名、变量名、CUDA/PTX API、日志字符串和图表程序的英文标签刻意保留，便于对照原文，也避免破坏自动解析。原图和 MIT 许可证不作替换。

## 运行前先注意

这是一组教学/优化实验，不是支持所有尺寸的通用 GEMM 库。多数优化 kernel 缺少通用边界屏蔽，必须同时读 runner 的分块参数。特别是 11 号当前 `BN=256`，主程序却包含 `N=128`；本分支标注这个尺寸问题，但没有修算法或声称 11/12 全部验证通过。

`gen_benchmark_results.sh` 实际只循环 0–10；`plot_benchmark_results.py` 会覆盖 README 结果表和结果图片；autotuner 会用 `sed -i` 改写源文件。9 号 autotuner 还固定 `DEVICE=2`。这些不是适合初学者直接无人值守启动的脚本。详见原版限制文档。

## 致谢与许可证

基准测试框架参考 [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)。感谢 Simon Boehm 的原始实现和逐步优化讲解。

原 [MIT LICENSE](LICENSE) 与 `Copyright (c) 2023 Simon Boehm` 完整保留。中文分支不改变许可条件，也不将原作者的实现或历史性能归为本分支原创。
