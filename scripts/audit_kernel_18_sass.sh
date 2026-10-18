#!/usr/bin/env bash
# 【文件导读】离线统计 K18 的 SASS 指令
# 给初学者：这类脚本像读图纸，不负责测 GPU 跑多快；先准备它要求的二进制或报告，再读输出表。
# 用途：对已构建的 K10 与 K18 变体提取目标函数，比较 FFMA/LDS/地址指令及代码哈希。
# 运行：bash scripts/audit_kernel_18_sass.sh；需 cuobjdump，不需要启动 GPU kernel。
# 输入：build-layout-v<variant>/sgemm；先运行对应 build 脚本。
# 输出：CSV 写到标准输出；需要保存时选择新的文件，shell 的 > 会覆盖同名文件。
# marker 是 C++ 模板的修饰符号前缀；awk 在 Function 边界切出函数，count 按 opcode 汇总。
# 哈希清除地址/空白等差异后辅助比较代码；计数是静态数量，不是动态执行量或延迟。
# 函数名或编译器格式改变时须先确认提取非空；本脚本不会执行 correctness，也不会验证全部控制流。
# 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。


# Bash 的安全开关：-e 遇失败尽早停，-u 防止误用未定义变量，pipefail 让管道中间失败也能被发现。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 根据脚本自身位置找仓库，不是把当前 shell 所在目录随意当根目录；后续路径都由这里展开。
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

printf '%s\n' 'mode,static_instructions,FFMA,IMAD,IADD3,LDS,MOV,SHFL,LDG,STG,sass_sha256'
# 二进制里有很多 kernel；先用函数标记切出目标，再统计 opcode，不能把整个二进制的指令数算给一个核。
for mode in k10 v0 v1 v2 v3 v4 v5; do
  variant="${mode#v}"
  [[ "${mode}" != k10 ]] || variant=0
  binary="${ROOT_DIR}/build-layout-v${variant}/sgemm"
  if [[ "${mode}" == k10 ]]; then
    marker='_Z15sgemmWarptilingI'
  else
    marker='_Z21sgemmWarptilingLayoutI'
  fi
  sass="$(cuobjdump --dump-sass "${binary}" | awk -v marker="${marker}" '
    /Function : / {on=index($0, marker)>0}
    on
  ')"
  count() {
    awk -v op="$1" '$0 ~ ("[[:space:]]" op "(\\.|[[:space:]])") {n++} END {print n+0}' <<<"${sass}"
  }
  instructions="$(awk '/\/\*[0-9a-f]+\*\// {n++} END {print n+0}' <<<"${sass}")"
  hash="$(sed -E '/Function :/d; s@/\*[0-9a-f]+\*/@@; s@[[:space:]]+@@g' <<<"${sass}" | sha256sum | cut -d' ' -f1)"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${mode}" "${instructions}" \
    "$(count FFMA)" "$(count IMAD)" "$(count IADD3)" "$(count LDS)" \
    "$(count MOV)" "$(count SHFL)" "$(count LDG)" "$(count STG)" "${hash}"
done
