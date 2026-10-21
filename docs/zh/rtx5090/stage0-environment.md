# 阶段0：环境与构建子阶段

原生Release sm_120 GEMM16×16构建运行通过；cuobjdump确认sm_120 cubin；256输出全量CPUdouble最大误差0。memcheck/racecheck/synccheck全部退出0且零错误。本子阶段不提供性能结论。

Ubuntu22.04.5，RTX5090（SM120、170SM），驱动595.71.05，nvcc12.8.93，g++11.4，系统cuBLAS12.8.4.1，torch2.8.0+cu128。完整GPU UUID/资源/CPU/内存/磁盘见environment.json及原始文本。服务器到期时间无法查询，记unknown。

只补充Ninja1.13.2到任务目录rtx5090-tools。作者SM120扩展第一次失败因为稀疏检出漏common/pack.cuh；补齐未修改上游源码后第二次构建成功，五导出函数全部存在。保留两次构建日志及真实-gencode=compute_120/sm_120命令。stub仅供链接，实际libcuda来自系统驱动，未加stub运行路径。

NCU报告ERR_NVGPUCTRPERM（查询命令即使exit0也不能记PASS）。用户已授权修改计数器权限，但宿主机RmProfilingAdminOnly=1，容器缺少SYS_ADMIN/PERFMON bounding capabilities，capsh两次均拒绝，sysfs只读。需AutoDL平台开启权限。当前无实际profile。
以root及绝对路径实跑nvprof明确拒绝CC8.0及以上；sudo未安装但当前已是root，不是PATH或密码问题。

参考（访问2026-10-22）：https://developer.nvidia.com/ERR_NVGPUCTRPERM/ 和 https://docs.nvidia.com/cuda/profiler-users-guide/ 。

历史清单仍在审计，阶段0整体未结束。下一步完成清单和作者绑定的小形状验证，再运行BF16/4096原入口。
