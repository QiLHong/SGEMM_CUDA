#!/usr/bin/env bash
set -euo pipefail
csv="$1"
repo="/root/autodl-tmp/SGEMM_CUDA-rtx5090-20261022"
case_dir="$repo/results/rtx5090/stage45-case-snapshots/$(basename "$(dirname "$csv")")-$(basename "$csv" .csv)"
mkdir -p "$case_dir"
cp "$csv" "$case_dir/timings.csv"
cp "$csv.closed.txt" "$case_dir/timings.csv.closed.txt"
printf '{"closed":true,"status":"CASE_COMPLETE","source_csv":"%s"}\n' "$csv" > "$case_dir/CASE_CLOSED.json"
(cd "$repo" && sha256sum "${case_dir#$repo/}/CASE_CLOSED.json" "${case_dir#$repo/}/timings.csv" "${case_dir#$repo/}/timings.csv.closed.txt) > "$case_dir/sha256-manifest.txt"
cd "$repo"
git add "$case_dir"
GIT_AUTHOR_NAME='Hong QiLiang' GIT_AUTHOR_EMAIL='sa24006223@mail.ustc.edu.cn' GIT_COMMITTER_NAME='Hong QiLiang' GIT_COMMITTER_EMAIL='sa24006223@mail.ustc.edu.cn' git commit -m "results(rtx5090): archive closed robustness case $(basename "$csv")"
commit=$(git rev-parse HEAD)
gate=/root/autodl-tmp/stage45-case-gate
mkdir -p "$gate"
printf '%s\n' "$commit" > "$gate/pending"
while [ ! -f "$gate/$commit.published" ]; do sleep 1; done
rm -f "$gate/pending" "$gate/$commit.published"
