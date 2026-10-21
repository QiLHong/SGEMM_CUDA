#!/usr/bin/env python3
"""Independent stdlib-only recovery of closed RTX5090 CSV results."""
import argparse, collections, csv, hashlib, json, statistics
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('csv',type=Path)
p.add_argument('--expected-events',type=int)
p.add_argument('--output',type=Path)
a=p.parse_args()
rows=list(csv.DictReader(a.csv.open(encoding='utf-8-sig',newline='')))
if a.expected_events is not None and len(rows)!=a.expected_events:
    raise SystemExit(f'count mismatch {len(rows)} != {a.expected_events}')
result={'source_csv':a.csv.name,'sha256':hashlib.sha256(a.csv.read_bytes()).hexdigest(),'rows':len(rows)}
if not rows: raise SystemExit('empty CSV')
if 'printed_mean_ms' in rows[0]:
    result['protocol']='original wallclock printed values; not CUDA single-call events'
    result['summary']=[{'implementation':r['implementation'],'printed_mean_ms':float(r['printed_mean_ms']),
                        'recomputed_tflops_from_rounded_ms':2*int(r['M'])*int(r['N'])*int(r['K'])/(float(r['printed_mean_ms'])*1e9)} for r in rows]
else:
    group=collections.defaultdict(list)
    for r in rows:
        key=(r['implementation'],int(r['round']))
        value=float(r['milliseconds'])
        if not (0<value<float('inf')): raise SystemExit('invalid event duration')
        group[key].append(value)
    counts={f'{k[0]}/round{k[1]}':len(v) for k,v in group.items()}
    result['group_counts']=counts
    means={k:statistics.mean(v) for k,v in group.items()}
    names=sorted({k[0] for k in group})
    nr=sorted({k[1] for k in group})
    shape=tuple(int(rows[0][x]) for x in ['M','N','K'])
    result['shape']=shape
    result['summary']=[]
    for name in names:
        m=statistics.median(means[name,r] for r in nr)
        s={'implementation':name,'median_round_mean_ms':m,'tflops':2*shape[0]*shape[1]*shape[2]/(m*1e9)}
        if 'hgemm_cublas' in names:
            s['median_paired_speedup_vs_cublas']=statistics.median(means['hgemm_cublas',r]/means[name,r]-1 for r in nr)
        result['summary'].append(s)
text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if a.output: a.output.write_text(text,encoding='utf-8')
print(text)
