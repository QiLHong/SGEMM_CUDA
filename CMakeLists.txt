cmake_minimum_required(VERSION 3.19)
project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
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
