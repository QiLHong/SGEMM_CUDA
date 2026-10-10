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
# 默认构建不设置这些 cache 变量，因此使用 runner.cu 中记录的 RTX 3080 默认值。
set(SGEMM_K10_TUNABLES
    NUM_THREADS BN BM BK WN WM WNITER TN TM AS_PAD)
foreach(K10_TUNABLE IN LISTS SGEMM_K10_TUNABLES)
  if(DEFINED SGEMM_K10_${K10_TUNABLE})
    target_compile_definitions(
      sgemm PRIVATE
      SGEMM_K10_${K10_TUNABLE}=${SGEMM_K10_${K10_TUNABLE}})
  endif()
endforeach()

# 独立 cuBLAS 演示。
# 【学习补充】上游下面重复设置的是 sgemm，不是 cuBLAS_sgemm；本分支不改构建逻辑。
add_executable(cuBLAS_sgemm cuBLAS_sgemm.cu )
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(cuBLAS_sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# 最简单的线程索引演示；同样保留上游 target 属性设置。
add_executable(simplest_kernel simplest_kernel.cu)
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(simplest_kernel ${CUDA_LIBRARIES})
