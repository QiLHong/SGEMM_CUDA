"""Bounded Linux stage3 build/run; source kernels unchanged; no Git/publishing.

prepare is CPU-only and portable. build/run execute only when explicitly selected.
Each run closes ONE series/size/beta batch. No background GPU jobs or searching.
"""
import argparse, contextlib, csv, hashlib, json, math, os, re, shutil, signal
import statistics, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
CUTLASS_SHA='f7b19de32c5d1f3cedfc735c2849f12b537522ee'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def csvsave(p,rows,fields=None):
    with Path(p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def run_process(command,log,timeout,cwd,env=None):
    start=time.time();timed_out=False
    with Path(log).open('x',encoding='utf-8') as f:
        proc=subprocess.Popen([str(x) for x in command],stdout=f,stderr=subprocess.STDOUT,
                              cwd=cwd,env=env,start_new_session=os.name=='posix')
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            if os.name=='posix':os.killpg(proc.pid,signal.SIGKILL)
            else:proc.kill()
            rc=proc.wait()
    return dict(command=[str(x) for x in command],exit_code=rc,timed_out=timed_out,elapsed_seconds=time.time()-start)
def verify_sources(repo,spec):
    mismatches=[rel for rel,expected in spec['source_hashes'].items() if not (repo/rel).is_file() or hashlib.sha256((repo/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected]
    if mismatches:raise RuntimeError('Frozen source mismatch: '+', '.join(mismatches))
def source_key(spec):return hashlib.sha256(json.dumps(spec['source_hashes'],sort_keys=True).encode()).hexdigest()

def scrub_cpp(code):
    pattern=r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern,lambda m:' '*(m.end()-m.start()),code,flags=re.S)
def isolate_runner(code,kernel):
    """Disable only unused HOST launch functions, retaining selected body verbatim.

    Necessary for historical K10 tile macros: unrelated K18/K19 fixed-shape
    static_asserts otherwise make the now-expanded master dispatcher unbuildable.
    All kernel headers remain source-identical; no executable algorithm is edited.
    """
    if kernel=='all':return code,[]
    dispatch=dict((int(k),name) for k,name in re.findall(r'case\s+(\d+):\s*(\w+)\(',code))
    selected={dispatch[int(kernel)],dispatch[10]};clean=scrub_cpp(code);changes=[]
    for match in re.finditer(r'^void (run_sgemm_\w+|runSgemm\w+)\s*\(',code,re.M):
        name=match[1]
        if name in selected:continue
        begin=clean.index('{',match.end());depth=1;end=begin+1
        while depth:
            if clean[end]=='{':depth+=1
            elif clean[end]=='}':depth-=1
            end+=1
        changes.append((begin,end,name))
    out=code
    for begin,end,name in reversed(changes):
        body='{ std::fprintf(stderr,"Disabled unrelated host launcher: '+name+'\\n"); std::exit(93); }'
        out=out[:begin]+body+out[end:]
    # Independently verify kept launch function bodies were not touched.
    for name in selected:
        orig=re.search(r'^void '+re.escape(name)+r'\s*\(',code,re.M)
        new=re.search(r'^void '+re.escape(name)+r'\s*\(',out,re.M)
        def body(text,m):
            c=scrub_cpp(text);start=c.index('{',m.end());depth=1;end=start+1
            while depth:
                depth+=(c[end]=='{')-(c[end]=='}');end+=1
            return text[start:end]
        assert body(code,orig)==body(out,new),name
    return out,[name for _,_,name in changes]

def wrapper_text(source,header):
    def quoted(p):return json.dumps(str(p).replace('\\','/'))
    return '\n'.join([
      '#include '+quoted(header),
      '#define cudaEventElapsedTime(...) stage3_capture::elapsed(__VA_ARGS__)',
      '#define cudaMemcpy(...) stage3_capture::copy(__VA_ARGS__)',
      '#define main stage3_legacy_main',
      '#include '+quoted(source),
      '#undef main','#undef cudaMemcpy','#undef cudaEventElapsedTime',
      'int main(int argc,char** argv){stage3_capture::begin();',
      'int rc=stage3_legacy_main(argc,argv);stage3_capture::state().original_exit=rc;',
      'stage3_capture::finish();return rc;}',''])

def adapt_legacy_host(source):
    """Relocate quoted local includes and make renamed main's success return explicit.

    C++ permits falling off main, but not an int-returning renamed function. The
    original independent audits rely on that special rule. Device code is intact.
    """
    code=source.read_text(encoding='utf-8');clean=scrub_cpp(code)
    match=re.search(r'\bint\s+main\s*\(',clean)
    if not match:raise RuntimeError('Cannot locate original main: '+str(source))
    begin=clean.index('{',match.end());depth=1;end=begin+1
    while depth:
        depth+=(clean[end]=='{')-(clean[end]=='}');end+=1
    code=code[:end-1]+'\n  return 0; // Host adapter: renamed main needs an explicit success return.\n'+code[end-1:]
    def relocate(m):
        target=(source.parent/m[1]).resolve()
        return '#include '+json.dumps(target.as_posix()) if target.is_file() else m[0]
    return re.sub(r'^\s*#include\s+"([^"]+)"',relocate,code,flags=re.M)

def prepare_build(args,spec,id):
    conf=spec['builds'][id];root=args.build_root/id
    kind=conf['kind'];repo=args.repo
    if kind=='cutlass' and args.cutlass is None:raise RuntimeError('--cutlass exact v3.5.1 checkout is required')
    if root.exists():
        receipt=root/'prepare.json'
        if not receipt.exists():raise RuntimeError('Existing build directory without our receipt: '+str(root))
        old=json.loads(receipt.read_text())
        if old['source_key']!=source_key(spec) or old['config']!=conf or old['capture_header_sha256']!=sha(HERE/'event_capture.hpp') or old.get('generator_sha256')!=sha(__file__):
            raise RuntimeError('Existing build fingerprint mismatch; choose fresh --build-root')
        for rel,digest in old['generated_files'].items():
            if sha(root/rel)!=digest:raise RuntimeError('Prepared source changed: '+str(root/rel))
        return old
    root.mkdir(parents=True)
    source=repo/('sgemm.cu' if kind=='stock' else f'scripts/{kind}_audit.cu' if kind in ['k30','k31','k32'] else 'scripts/cutlass_simt_reference.cu')
    original_source=source
    if kind!='stock':
        source=root/'legacy_host.cu';source.write_text(adapt_legacy_host(original_source),encoding='utf-8')
    wrapper=root/'captured_main.cu';wrapper.write_text(wrapper_text(source,HERE/'event_capture.hpp'),encoding='utf-8')
    defs={'SGEMM_K10_MIN_BLOCKS':2,**conf['defines']}
    command=[args.nvcc,'-std=c++17','-O3','-DNDEBUG','-arch=sm_120','-lineinfo','-Xptxas=-v']
    command += [f'-D{k}={v}' for k,v in sorted(defs.items())]
    command += ['-I'+str(root),'-I'+str(repo/'src'),str(wrapper)]
    changed=[]
    if kind=='stock':
        derived,changed=isolate_runner((repo/'src/runner.cu').read_text(encoding='utf-8'),conf['kernel'])
        (root/'runner-selected.cu').write_text(derived,encoding='utf-8')
        command.append(str(root/'runner-selected.cu'))
    if kind=='cutlass':
        command += ['-I'+str(args.cutlass/'include')]
        # Compatibility host header only; upstream device source is unchanged.
        (root/'nvToolsExt.h').write_text('#pragma once\n#include <nvtx3/nvToolsExt.h>\n',encoding='utf-8')
    command += ['-lcublas','-o',str(root/'benchmark')]
    if args.nvtx_library and kind in ['k30','k31','k32']:command.append('-lnvToolsExt')
    receipt=dict(id=id,source_key=source_key(spec),source_commit=spec['source_commit'],config=conf,
       actual_source_raw_sha256={rel:sha(repo/rel) for rel in spec['source_hashes']},
       capture_header_sha256=sha(HERE/'event_capture.hpp'),generator_sha256=sha(__file__),wrapper_sha256=sha(wrapper),command=command,
       original_host_source=original_source.relative_to(repo).as_posix(),
       host_adaptation='none' if kind=='stock' else 'relocate local includes; append success return for renamed main',
       generated_files={f.name:sha(f) for f in root.iterdir() if f.is_file()},
       disabled_unrelated_host_launchers=changed,device_headers_unchanged=True,
       host_protocol='original independent-process protocol; host-only elapsed/copy capture',binary=str(root/'benchmark'))
    save(root/'prepare.json',receipt);return receipt

def parse_resources(text):
    records=[];name=None;stack=stores=loads=0
    for line in text.splitlines():
        m=re.search(r"Compiling entry function '([^']+)'",line)
        if m:name=m[1];stack=stores=loads=0
        m=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',line)
        if m:stack,stores,loads=map(int,m.groups())
        m=re.search(r'Used (\d+) registers',line)
        if m:
            shared=re.search(r'(\d+) bytes smem',line)
            records.append(dict(symbol=name,registers=int(m[1]),stack_bytes=stack,spill_store_bytes=stores,spill_load_bytes=loads,static_shared_bytes=int(shared[1]) if shared else None,historical_gate_le240_no_spill=int(m[1])<=240 and stores==0 and loads==0))
    return records
def do_build(args,spec,id):
    rec=prepare_build(args,spec,id);root=args.build_root/id
    old=root/'build.json'
    if old.exists():
        result=json.loads(old.read_text())
        if result['status']=='ready' and sha(root/'benchmark')==result['binary_sha256']:return result
        raise RuntimeError('Prior failed/changed build retained; choose fresh --build-root')
    if spec['builds'][id]['kind']=='cutlass':
        head=subprocess.check_output(['git','-C',str(args.cutlass),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(args.cutlass),'status','--porcelain'],text=True).strip()
        if head!=CUTLASS_SHA or dirty:raise RuntimeError('CUTLASS must be clean at '+CUTLASS_SHA)
    toolchain=run_process([args.nvcc,'--version'],root/'nvcc-version.txt',15,root)
    (root/'cuda-architecture.txt').write_text('sm_120\n',encoding='utf-8')
    result=run_process(rec['command'],root/'build.log',args.build_timeout,root)
    result['toolchain_version_query']=toolchain
    result['status']='build_timeout' if result['timed_out'] else 'build_failed' if result['exit_code'] else 'ready'
    result['resources']=parse_resources((root/'build.log').read_text(errors='replace'))
    if result['status']=='ready':
        result['binary_sha256']=sha(root/'benchmark')
        dump=shutil.which('cuobjdump')
        if dump:result['resource_dump']=run_process([dump,'--dump-resource-usage',root/'benchmark'],root/'resource-usage.txt',60,root)
    save(old,result);return result

@contextlib.contextmanager
def gpu_lock(path):
    import fcntl
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('GPU lock is held; do not run parallel batches')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def telemetry(path):
    exe=shutil.which('nvidia-smi')
    if not exe:return {'status':'missing_tool'}
    return run_process([exe,'--query-gpu=timestamp,name,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv'],path,15,path.parent)
def invoke(args,spec,candidate,n,beta,repeat,warm,folder,timeout=None):
    folder.mkdir();b=spec['builds'][candidate['build']];root=args.build_root/candidate['build']
    rec=json.loads((root/'build.json').read_text())
    if rec['status']!='ready':return {'status':rec['status'],'events':0}
    if sha(root/'benchmark')!=rec['binary_sha256']:raise RuntimeError('Binary changed: '+str(root))
    env=dict(os.environ);env.pop('NVIDIA_TF32_OVERRIDE',None)
    env.update(STAGE3_CAPTURE_PREFIX=str(folder/'capture'),STAGE3_N=str(n),STAGE3_BETA=str(beta),
       STAGE3_OUTPUT_ORDER='actual_first' if b['kind']=='stock' else 'reference_first',
       SGEMM_BENCH_SIZE=str(n),SGEMM_REPEAT=str(repeat),SGEMM_WARMUP=str(warm),SGEMM_ALPHA='0.5',SGEMM_BETA=str(beta),
       SGEMM_RESET_C_EACH_REPEAT='1',SGEMM_SEED=str(spec['series'][args.series]['seed']),DEVICE='0')
    command=[str(root/'benchmark'),candidate['cli']]
    if b['kind']!='stock':command += [str(n),str(beta),str(repeat),str(warm)]
    save(folder/'command.json',dict(command=command,nonsecret_environment={k:env[k] for k in env if k.startswith(('STAGE3_','SGEMM_'))},binary_sha256=rec['binary_sha256']))
    result=run_process(command,folder/'stdout.log',timeout or args.timeout,folder,env)
    text=(folder/'stdout.log').read_text(errors='replace')
    original=re.search(r'^RESULT,([^\n]+)',text,re.M)
    result['original_result']=original[1] if original else None
    eventfile=folder/'capture.events.partial.csv';events=[]
    if eventfile.exists():events=list(csv.DictReader(eventfile.open()))
    valid_events=all(math.isfinite(float(r['ms'])) and float(r['ms'])>0 for r in events)
    checks=[];checkfile=folder/'capture.checks.csv'
    if checkfile.exists():checks=list(csv.DictReader(checkfile.open()))
    numeric_fail=any(r['status']=='FAIL' for r in checks) or 'Divergence!' in text or bool(re.search(r'^FAIL ',text,re.M))
    result['status']='timeout' if result['timed_out'] else 'numerical_failure' if numeric_fail else 'runtime_failure' if result['exit_code'] else 'event_incomplete' if len(events)!=repeat or not valid_events else 'passed'
    result['events']=len(events);result['expected_events']=repeat;result['checks']=checks
    if result['status']=='passed':
        eventfile.rename(folder/'capture.events.csv');result['round_mean_ms']=statistics.mean(float(x['ms']) for x in events)
    result['input_fingerprints']=list(csv.DictReader((folder/'capture.inputs.csv').open())) if (folder/'capture.inputs.csv').exists() else []
    result['resource_spill_present']=any(r['spill_store_bytes'] or r['spill_load_bytes'] for r in rec['resources'])
    save(folder/'result.json',result)
    return result

def do_run(args,spec):
    series=spec['series'][args.series];out=args.output
    if out.exists():raise RuntimeError('Refusing existing output; preserve earlier batch')
    if args.size not in series['sizes']:raise RuntimeError('Shape absent from verified historical menu')
    out.mkdir(parents=True)
    rounds=series['original_rounds'] if args.protocol=='historical-counts' else 7
    repeat=series['original_repeats'] if args.protocol=='historical-counts' else 30
    warm=series['original_warmup'] if args.protocol=='historical-counts' else 10
    cs=series['candidates'];planned=[]
    for r in range(rounds):
        for pos in range(len(cs)):
            c=cs[(r+pos)%len(cs)]
            planned.append(dict(round=r,position=pos,implementation=c['id'],build=c['build'],cli=c['cli'],n=args.size,beta=args.beta,repeats=repeat,warmup=warm))
    csvsave(out/'order-plan.csv',planned)
    save(out/'protocol.json',dict(series=series,protocol=args.protocol,source_key=source_key(spec),source_commit=spec['source_commit'],timing=spec['protocol_note'],timeout_seconds=args.timeout,batch_budget_seconds=args.batch_budget,expected_groups=len(planned),expected_events=len(planned)*repeat,scope='one series/one shape/one beta; smoke separate; all files closed before BATCH_CLOSED'))
    statuses=[];blocked={};start=time.time()
    with gpu_lock(args.gpu_lock):
        save(out/'telemetry-before.json',telemetry(out/'telemetry-before.csv'))
        for c in cs:
            remaining=args.batch_budget-(time.time()-start)
            if remaining<=0:blocked[c['id']]='budget_exhausted';continue
            build_rec=args.build_root/c['build']/'build.json'
            if not build_rec.exists() or json.loads(build_rec.read_text())['status']!='ready':
                blocked[c['id']]='not_built_or_build_failed';continue
            if args.size<c['min_n']:blocked[c['id']]='illegal_shape';continue
            n=c['min_n'];folder=out/('smoke-'+c['id'])
            res=invoke(args,spec,c,n,args.beta,1,0,folder,min(args.timeout,remaining))
            if res['status']!='passed':blocked[c['id']]='smoke_'+res['status']
        for row in planned:
            c=next(x for x in cs if x['id']==row['implementation'])
            if c['id'] in blocked:statuses.append({**row,'status':blocked[c['id']],'events':0});continue
            remaining=args.batch_budget-(time.time()-start)
            if remaining<=0:
                statuses.append({**row,'status':'budget_exhausted','events':0});continue
            folder=out/f"r{row['round']:02d}-p{row['position']:02d}-{c['id']}"
            res=invoke(args,spec,c,args.size,args.beta,repeat,warm,folder,min(args.timeout,remaining))
            statuses.append({**row,'status':res['status'],'events':res['events'],'path':folder.name,'round_mean_ms':res.get('round_mean_ms'),'input_fingerprints':res['input_fingerprints'],'resource_spill_present':res['resource_spill_present']})
            if res['status']!='passed':blocked[c['id']]=res['status']
            save(out/'progress.partial.json',statuses)
        save(out/'telemetry-after.json',telemetry(out/'telemetry-after.csv'))
    # Recompute from closed raw CSVs, independently of the invocation's means.
    for row in statuses:
        if row['status']!='passed':continue
        rawpath=out/row['path']/'capture.events.csv'
        with rawpath.open() as f:raw=list(csv.DictReader(f))
        values=[float(x['ms']) for x in raw]
        if len(raw)!=repeat or [int(x['event_index']) for x in raw]!=list(range(repeat)) or not all(math.isfinite(x) and x>0 for x in values):
            row['status']='closed_raw_event_integrity_failure';continue
        row['round_mean_ms']=statistics.mean(values)
        row['round_stddev_ms']=statistics.stdev(values) if len(values)>1 else 0
        row['raw_sha256']=sha(rawpath)
    # Refuse paired statistics if actual A/B/C0 bytes disagree, even matching seed.
    input_sets={tuple((i['matrix'],i['bytes'],i['fnv1a64']) for i in r['input_fingerprints']) for r in statuses if r['status']=='passed'}
    input_equal=len(input_sets)==1 and all(len(x)==3 for x in input_sets)
    summary=[]
    for c in cs:
        rs=[r for r in statuses if r['implementation']==c['id'] and r['status']=='passed']
        if not rs:continue
        median=statistics.median(r['round_mean_ms'] for r in rs)
        pairs=[]
        if input_equal:
            for r in rs:
                base=next((x for x in statuses if x['round']==r['round'] and x['implementation']=='strict' and x['status']=='passed'),None)
                if base:pairs.append(100*(base['round_mean_ms']/r['round_mean_ms']-1))
        summary.append(dict(implementation=c['id'],completed_rounds=len(rs),expected_rounds=rounds,formal_events=sum(r['events'] for r in rs),median_round_mean_ms=median,mean_round_mean_ms=statistics.mean(r['round_mean_ms'] for r in rs),round_mean_stddev_ms=statistics.stdev(r['round_mean_ms'] for r in rs) if len(rs)>1 else 0,tflops=2*args.size**3/(median*1e9),median_paired_gain_vs_strict_pct=statistics.median(pairs) if pairs else None,inputs_equal_across_group=input_equal))
    save(out/'group-results.json',statuses);save(out/'summary.json',summary)
    if summary:csvsave(out/'summary.csv',summary)
    complete=all(r['status']=='passed' for r in statuses) and input_equal
    save(out/'BATCH_CLOSED.json',dict(complete=complete,series=args.series,n=args.size,beta=args.beta,groups=len(statuses),formal_events=sum(r['events'] for r in statuses),expected_events=len(planned)*repeat,inputs_equal=input_equal,elapsed_seconds=time.time()-start,missing_or_failed=[{'implementation':r['implementation'],'round':r['round'],'status':r['status']} for r in statuses if r['status']!='passed']))
    (out/('BATCH_COMPLETE.txt' if complete else 'BATCH_PARTIAL.txt')).write_text('Closed; see BATCH_CLOSED.json. No publishing performed.\n')
    manifest=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    csvsave(out/'sha256-manifest.csv',manifest)
    print(json.dumps({'output':str(out),'complete':complete,'formal_events':sum(r['events'] for r in statuses)}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['list','prepare','build','run']);p.add_argument('--spec',type=Path,default=HERE/'execution-specs.json')
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--build-root',type=Path,required=True)
    p.add_argument('--series');p.add_argument('--build-id');p.add_argument('--nvcc',default='nvcc');p.add_argument('--cutlass',type=Path)
    p.add_argument('--nvtx-library',action='store_true');p.add_argument('--build-timeout',type=int,default=600)
    p.add_argument('--output',type=Path);p.add_argument('--size',type=int,default=4096);p.add_argument('--beta',type=int,choices=[0,1,3],default=3)
    p.add_argument('--protocol',choices=['events7','historical-counts'],default='events7');p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--batch-budget',type=int,default=1800);p.add_argument('--gpu-lock',type=Path)
    a=p.parse_args();a.repo=a.repo.resolve();a.build_root=a.build_root.resolve()
    if a.output:a.output=a.output.resolve()
    if a.cutlass:a.cutlass=a.cutlass.resolve()
    spec=json.loads(a.spec.read_text(encoding='utf-8'));verify_sources(a.repo,spec)
    if a.action=='list':print(json.dumps({'series':list(spec['series']),'builds':list(spec['builds'])},indent=2));return
    if a.action in ['build','run'] and os.name!='posix':raise RuntimeError('Execution commands are Linux-only; prepare remains portable')
    if a.action=='run':
        if not a.series or not a.output or not a.gpu_lock:p.error('run requires --series --output --gpu-lock (coordinate this same lock with other GPU work)')
        do_run(a,spec);return
    ids=[a.build_id] if a.build_id else sorted({c['build'] for c in spec['series'][a.series]['candidates']}) if a.series else []
    if not ids:p.error('prepare/build requires --series or --build-id')
    for id in ids:
        rec=prepare_build(a,spec,id) if a.action=='prepare' else do_build(a,spec,id)
        print(json.dumps({'id':id,'status':rec.get('status','prepared'),'path':str(a.build_root/id)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
