#!/usr/bin/env python3
import csv, json, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
docs = root / 'docs/zh/rtx5090'
spec = []
for n, num in [(1024,17),(2048,19),(3072,21),(4096,23)]:
    spec.append(('historical_alpha0.5_beta3', n, num, 'historical'))
for n, num in [(1024,18),(2048,20),(3072,22),(4096,24)]:
    spec.append(('C_equals_AB_alpha1_beta0', n, num, 'ab'))
rows, commits = [], []
for protocol, n, num, tag in spec:
    d = root / f'results/rtx5090/20261022-{num}-strict-{tag}-{n}'
    audit = json.loads((d/'analysis/independent-audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['formal_events'] == 1050
    sha = subprocess.check_output(['git','log','-1','--format=%H','--',str(d.relative_to(root))], cwd=root, text=True).strip()
    commits.append((protocol,n,d,sha))
    with (d/'analysis/summary.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            r['protocol'], r['commit'] = protocol, sha
            rows.append(r)
fields = ['protocol'] + [x for x in rows[0] if x not in ('protocol','commit')] + ['commit']
with (docs/'stage2-summary.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
lines = [
    '# 阶段 2：严格 FP32 核心对照','',
    'RTX 5090，五实现同一进程、同一默认 stream、相同实际输入。每个尺寸和数学问题独立测量；每实现 7 轮，每轮 10 次预热和 30 次正式 CUDA event。八批合计 8400 个正式 event。','',
    '实现为 K10 PAD4、K10 PAD8、K30 V1、K30 V2、PEDANTIC FP32 cuBLAS。历史表使用 alpha=0.5、beta=3；跨库表使用 alpha=1、beta=0（C=AB），两表不混排。输入使用 seed=20261020 的 xorshift32，实际 A/B/C0 字节指纹随批次保存。','',
    '| 协议 | N | 实现 | TFLOPS | 相对 strict（中位延迟比） | GPU参考最大绝对误差 | CPU double最大绝对误差 |',
    '|---|---:|---|---:|---:|---:|---:|'
]
for r in rows:
    lines.append(f"| {r['protocol']} | {r['n']} | {r['implementation']} | {float(r['median_tflops']):.2f} | {float(r['relative_to_strict_medians']):.3f} | {float(r['max_abs']):.6g} | {float(r['direct_cpu_max_abs']):.6g} |")
lines += ['', '统计为每轮 event 均值的跨轮中位数；逐轮配对提升另存于每批 analysis/paired-summary.csv。C0 的 D2D 恢复在 start event 前。每轮末输出做全量 PEDANTIC GPU 参考检查，大尺寸另用固定 256 点 CPU double 直接检查候选输出。所有八批数值验收通过。', '', '## 已核验提交', '']
for protocol,n,d,sha in commits:
    lines.append(f'- {protocol} / N={n}: `{d.relative_to(root).as_posix()}`，`{sha}`。')
(docs/'stage2-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=docs/'progress.json'; x=json.loads(p.read_text()); x['stages'][2]={'stage':2,'status':'passed','notes':'8批、8400正式event；两数学协议分表；全部独立数值验收通过'}; p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
p=docs/'README.md'; s=p.read_text(); link='[阶段2：严格 FP32 核心对照](stage2-results.md)'
if link not in s: p.write_text(s+'\n- '+link+'：K10 PAD4/PAD8、K30 V1/V2、PEDANTIC cuBLAS，两协议八批。\n')
out=root/'results/rtx5090/20261022-25-stage2-report'; out.mkdir(exist_ok=True)
(out/'BATCH_COMPLETE.json').write_text(json.dumps({'closed':True,'status':'COMPLETE','independent_summary_verified':True,'formal_events_summarized':8400},indent=2)+'\n')
print(json.dumps({'rows':len(rows),'events':8400}))
