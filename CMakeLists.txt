# 【文件导读】全工程目标与编译期实验参数
# 给初学者：构建就是把源代码翻译成可执行文件；编译通过只是第一步，还没有证明计算正确或速度快。
# 使用：cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86，然后 cmake --build build -j2。
# 主目标 sgemm 连接 sgemm.cu 与 src/runner.cu；cuBLAS_sgemm/simplest_kernel 为两个入门示例。
# SGEMM_K10_* 宏决定 K10 的模板参数，SGEMM_K18_VARIANT 等选择实验变体；这些不是可执行程序的运行时参数。
# SGEMM_BUILD_K30_AUDIT/K31_AUDIT/K32_AUDIT 默认关闭；打开后构建独立 audit，不替换 sgemm 的编号分发。
# Release 用于性能对比；Debug 的 -G 用于调试，不能与 Release 的 GFLOPS 混比。
# CMake 的缓存保存在 -B 指定目录，切换变体建议使用新目录；命令执行会写构建产物，不会生成实验报告。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
# 独立实验入口，不进入 sgemm dispatcher，也不替换默认 K10。
option(SGEMM_BUILD_K30_AUDIT "Build isolated K30 mini-SIMT experiment" OFF)
option(SGEMM_BUILD_K31_AUDIT "Build isolated K31 resource/lifetime experiments" OFF)
option(SGEMM_BUILD_K32_AUDIT "Build isolated K32 producer/consumer experiment" OFF)
if(SGEMM_BUILD_K32_AUDIT)
  find_package(CUDAToolkit REQUIRED)
  add_executable(k32_audit scripts/k32_audit.cu)
  set_target_properties(k32_audit PROPERTIES CUDA_ARCHITECTURES 86 CUDA_STANDARD 17)
  target_compile_definitions(k32_audit PRIVATE SGEMM_K10_MIN_BLOCKS=2)
  target_compile_options(k32_audit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xptxas=-v>)
  target_link_libraries(k32_audit PRIVATE CUDA::cublas CUDA::nvToolsExt)
endif()
if(SGEMM_BUILD_K31_AUDIT)
  find_package(CUDAToolkit 12.4 REQUIRED)
  add_executable(k31_audit scripts/k31_audit.cu)
  set_target_properties(k31_audit PROPERTIES CUDA_ARCHITECTURES 86 CUDA_STANDARD 17)
  target_compile_definitions(k31_audit PRIVATE SGEMM_K10_MIN_BLOCKS=2)
  target_compile_options(k31_audit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xptxas=-v>)
  target_link_libraries(k31_audit PRIVATE CUDA::cublas CUDA::nvToolsExt)
endif()
if(SGEMM_BUILD_K30_AUDIT)
  find_package(CUDAToolkit REQUIRED)
  add_executable(k30_audit scripts/k30_audit.cu)
  set_target_properties(k30_audit PROPERTIES CUDA_ARCHITECTURES 86 CUDA_STANDARD 17)
  target_compile_definitions(k30_audit PRIVATE SGEMM_K10_MIN_BLOCKS=2)
  target_compile_options(k30_audit PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xptxas=-v>)
  target_link_libraries(k30_audit PRIVATE CUDA::cublas CUDA::nvToolsExt)
endif()
# 为编辑器/clangd 导出实际编译命令，不影响矩阵乘算法。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# 确认 CUDA 编译语言可用。
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
# 当前上游值是 86，对应本次学习用 RTX 3080 的目标架构。
# 换 GPU 时先查 compute capability，不要把驱动显示的 CUDA 版本填在这里。
set(CUDA_COMPUTE_CAPABILITY 86)

# Debug 模式为设备代码添加调试信息。
# -G 会关闭大部分设备优化，严重影响性能；测吞吐率应使用 Release。
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# 下行为原作者保留的寄存器/共享内存资源报告开关，当前未启用。
# add_compile_options("--ptxas-options=-v")

# 配置头文件搜索路径。
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# 收集 src 中待编译的源文件。
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# 主 benchmark 可执行文件，链接 CUDA runtime 与 cuBLAS。
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Kernel 10 调优参数。仅把显式传给 CMake 的值转成交给 nvcc 的编译期定义；
# 默认构建不设置这些 cache 变量，因此使用 runner/header 中记录的 RTX 3080 默认值。
set(SGEMM_K10_TUNABLES
    NUM_THREADS BN BM BK WN WM WNITER TN TM AS_PAD MIN_BLOCKS)
foreach(K10_TUNABLE IN LISTS SGEMM_K10_TUNABLES)
  if(DEFINED SGEMM_K10_${K10_TUNABLE})
    target_compile_definitions(
      sgemm PRIVATE
      SGEMM_K10_${K10_TUNABLE}=${SGEMM_K10_${K10_TUNABLE}})
  endif()
endforeach()

# K15 复用 K10 的全部计算参数，只允许独立改变 grouped-grid 的 M 方向组宽。
if(DEFINED SGEMM_K15_GROUP_M)
  target_compile_definitions(
    sgemm PRIVATE SGEMM_K15_GROUP_M=${SGEMM_K15_GROUP_M})
endif()

# K16 只搜索限定的 A shared-memory XOR 布局参数。
foreach(K16_TUNABLE IN ITEMS XOR_MASK XOR_SHIFT)
  if(DEFINED SGEMM_K16_${K16_TUNABLE})
    target_compile_definitions(
      sgemm PRIVATE
      SGEMM_K16_${K16_TUNABLE}=${SGEMM_K16_${K16_TUNABLE}})
  endif()
endforeach()

# K18 每次只编译一个手工设计的 warp fragment layout（0..5）。其余 K10
# 参数在 runner/header 中被 static_assert 固定，避免实验越界到 tile/load/pipeline。
if(DEFINED SGEMM_K18_VARIANT)
  target_compile_definitions(
    sgemm PRIVATE SGEMM_K18_VARIANT=${SGEMM_K18_VARIANT})
endif()

# K19 在同一个冻结的 K10 shape 上选择 B-fragment register prefetch 版本。
if(DEFINED SGEMM_K19_VARIANT)
  target_compile_definitions(
    sgemm PRIVATE SGEMM_K19_VARIANT=${SGEMM_K19_VARIANT})
endif()

# K20 在同一 256x128 CTA 上比较两种 one-pass warp 分区和一个分阶段版本。
if(DEFINED SGEMM_K20_VARIANT)
  target_compile_definitions(
    sgemm PRIVATE SGEMM_K20_VARIANT=${SGEMM_K20_VARIANT})
endif()

# K21 V1 隔离 iterator 寻址；V2 才启用 256-thread 双阶段异步流水。
if(DEFINED SGEMM_K21_VARIANT)
  target_compile_definitions(
    sgemm PRIVATE SGEMM_K21_VARIANT=${SGEMM_K21_VARIANT})
endif()

# K22 冻结 K10 的 128-thread ownership，只比较 BK=8 同步、2-stage 与 4-stage。
if(DEFINED SGEMM_K22_VARIANT)
  target_compile_definitions(
    sgemm PRIVATE SGEMM_K22_VARIANT=${SGEMM_K22_VARIANT})
endif()

# K23 V1 reuses K21 V1; V2/V3 change one shared layout, while V4 combines both
# as a separately recorded addendum without rewriting the V1-V3 history.
if(DEFINED SGEMM_K23_VARIANT)
  target_compile_definitions(
    sgemm PRIVATE SGEMM_K23_VARIANT=${SGEMM_K23_VARIANT})
endif()

# 独立 cuBLAS 演示。
# 上游这里误把架构属性再次设置给 sgemm；实验分支改为正确的目标。
add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu )
set_target_properties(cuBLAS_sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# 最简单的线程索引演示；架构属性应设置给它自身。
add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(simplest_kernel PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})
