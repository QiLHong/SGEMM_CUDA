#pragma once
// CUDA 12.4 Windows 提供 header-only NVTX3，无需安装旧 nvToolsExt DLL。
// 旧 Linux Toolkit 仍可回退到原有 NVTX 头/链接库。
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
