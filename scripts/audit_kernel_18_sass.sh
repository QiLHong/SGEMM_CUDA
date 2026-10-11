#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

printf '%s\n' 'mode,static_instructions,FFMA,IMAD,IADD3,LDS,MOV,SHFL,LDG,STG,sass_sha256'
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
