# 已核对事实

SGEMM HEAD a54435d；vitamin 520f970 工作区干净。现有 BN64 harness 固定 seed=20261020；confirm 顺序不覆盖六种排列。本轮需要独立 host 测试入口。旧成绩只用于历史背景。

工具链实际路径/version已写 preflight/toolchain.txt，CUDA12.4.131、MSVC19.37.32826.1、SDK26100、CMake3.31.6、Ninja1.12.1、driver610.60；NCU/Sanitizer2024.1.1已安装。最初猜测NCU根目录路径不存在，已从旧脚本定位target/windows-desktop-win7-x64/ncu.exe并运行version。旧benchmark每案例进程、固定seed；新入口正在独立实现。

新NCU：2048 SM active/elapsed75.23→89.78，3072 84.46→91.90，4096 95.41→99.29。4096单次profileelapsed提高与fair不稳定领先不一致，必须披露不能因果分解。L2请求~1.5x，4096DRAM读+写406.336→596.207MiB。全部资源/局部spill和SASS保持原样。原FP32864条绝对/SIMT/native标准仍失败，不能称strict FP32通过。
