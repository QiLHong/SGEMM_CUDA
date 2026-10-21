#!/usr/bin/env python3
"""Execute exactly one historical comparison batch; close/verify before publish."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build',required=True);p.add_argument('--output',required=True)
    p.add_argument('--family',choices=['tensor','grid','ktile','bn64','simt_original','simt_strict','robustness_case','dbf_pair'],required=True)
    p.add_argument('--size',type=int,default=4096);p.add_argument('--m',type=int);p.add_argument('--n',type=int);p.add_argument('--k',type=int)
    p.add_argument('--seed',type=int,default=20261020);p.add_argument('--selection',default='all');p.add_argument('--smoke',action='store_true')
    p.add_argument('--timeout',type=int,default=1800);p.add_argument('--publish-hook',help='absolute executable accepting closed output directory; nonzero stops')
    a=p.parse_args()
    if a.family=='dbf_pair':
        if a.seed!=20261020:p.error('dbf_pair fixes seed20261020 in the frozen binary; other seeds are unsupported')
        if len({a.m or a.size,a.n or a.size,a.k or a.size})!=1:p.error('dbf_pair supports square M=N=K only')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);build=Path(a.build).resolve()
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    M,N,K=a.m or a.size,a.n or a.size,a.k or a.size
    rounds,repeats,warmup=(1,1,1) if a.smoke else (7,30,10)
    selections={'tensor':{'all':8,'tensor_pairs':4},'grid':{'all':10,'grid_w1':6,'grid_w2':6,'grid_w4':6},'ktile':{'all':4,'confirm':3},'bn64':{'all':4,'confirm':3},'simt_original':{'all':8},'simt_strict':{'all':8}}
    records=[];csvs=[];env=os.environ.copy()
    # Process-scoped TF32 policy matches the historical wrapper's distinction.
    inherited_override=env.pop('NVIDIA_TF32_OVERRIDE',None)
    if a.family in ('simt_strict','dbf_pair'):env['NVIDIA_TF32_OVERRIDE']='0'
    else:env.pop('NVIDIA_TF32_OVERRIDE',None)
    meta={'status':'PARTIAL','family':a.family,'shape':[M,N,K],'seed':a.seed,'rounds':rounds,'repeats':repeats,'warmup':warmup,
          'selection':a.selection,'input_contract':'original harness, C=AB alpha1 beta0, xorshift bounded input; original FNV fingerprints retained',
          'inherited_NVIDIA_TF32_OVERRIDE':inherited_override,'effective_NVIDIA_TF32_OVERRIDE':env.get('NVIDIA_TF32_OVERRIDE'),
          'runner_sha256':sha(Path(__file__))}
    save('PARTIAL.json',meta)
    def run(target,args,csvpath,index):
        exe=build/target;command=[str(exe),*map(str,args)];begin=time.time()
        with (out/f'run-{index}.log').open('w',encoding='utf-8') as log:
            try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:code=124
        records.append({'command':command,'binary_sha256':sha(exe),'begin_unix':begin,'end_unix':time.time(),'exit_code':code})
        save('execution.json',records)
        if code:raise RuntimeError(f'{target} exit {code}; partial raw evidence retained')
        csvs.append(csvpath)
    try:
        if a.family=='dbf_pair':
            for r in range(rounds):
                variants=['dbf_default','dbf_maxr128']
                if r%2:variants.reverse()
                for pos,target in enumerate(variants):
                    path=out/f'events-pair{r+1}-{pos+1}-{target}.csv'
                    run(target,[N,1,repeats,warmup,path,'all'],path,f'{r+1}-{pos+1}')
            count=rounds*repeats*2
            meta['process_protocol']='14 fresh processes for7 pairs, variants alternate precedence; each process1 round, as historical DBF cap experiment'
        elif a.family=='robustness_case':
            path=out/'events.csv'
            run('robustness',['--case',M,N,K,a.seed,rounds,repeats,warmup,path],path,0)
            count=rounds*repeats*3
            meta['process_protocol']='one independent case; NOT counted as chapter33 three-process full matrix'
        else:
            if a.seed!=20261020:raise ValueError('historical seed is compile-time20261020 in this harness')
            implcount=selections[a.family].get(a.selection,1);count=rounds*repeats*implcount
            path=out/'events.csv'
            if a.family.startswith('simt'):
                if not(M==N==K):raise ValueError('SIMT harness square-only')
                args=[N,rounds,repeats,warmup,path,a.selection]
            else:args=[M,N,K,rounds,repeats,warmup,path,a.selection]
            run(a.family,args,path,0)
        rows=[]
        for path in csvs:
            with path.open(newline='',encoding='utf-8') as f:
                for x in csv.DictReader(f):
                    ms=float(x.get('milliseconds',x.get('event_ms','nan')))
                    if not math.isfinite(ms) or ms<=0:raise RuntimeError('invalid event retained')
                    x['_ms']=ms;x['_file']=path.name;rows.append(x)
        if len(rows)!=count:raise RuntimeError(f'expected {count} events, got {len(rows)}')
        groups={}
        for x in rows:
            # DBF one-round processes need filename in grouping to retain pair units.
            key=(x.get('configuration',''),x['implementation'],x['round'],x['_file'] if a.family=='dbf_pair' else '')
            groups.setdefault(key,[]).append(x['_ms'])
        summaries=[]
        for config,name in sorted({key[:2] for key in groups}):
            means=[statistics.mean(v) for k,v in groups.items() if k[:2]==(config,name)]
            median=statistics.median(means)
            summaries.append({'configuration':config,'implementation':name,'round_mean_ms':means,'median_round_mean_ms':median,'tflops':2*M*N*K/(median*1e9)})
        save('recomputed-summary.json',summaries)
        meta.update(status='COMPLETE',expected_events=count,actual_events=len(rows),exit_codes=[r['exit_code'] for r in records])
        save('COMPLETED.json',meta)
        # A closed batch must have exactly one terminal marker.
        (out/'PARTIAL.json').unlink(missing_ok=True)
        manifest={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='SHA256.json'}
        save('SHA256.json',manifest)
        if a.publish_hook:
            hook=Path(a.publish_hook)
            if not hook.is_absolute():raise ValueError('publish hook must be absolute')
            code=subprocess.run([str(hook),str(out)],env=env).returncode
            if code:raise RuntimeError(f'publish hook exit {code}; core evidence remains closed/pending')
        print(json.dumps(meta,indent=2));return 0
    except Exception as e:
        meta['error']=str(e);save('FAILURE.json',meta);raise

if __name__=='__main__':sys.exit(main())
