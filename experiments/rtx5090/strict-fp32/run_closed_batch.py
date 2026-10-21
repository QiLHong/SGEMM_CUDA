#!/usr/bin/env python3
"""Run exactly one isolated strict-FP32 batch; close evidence, never publish it."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BRANCH = 'results/rtx5090-gemm-reproduction-20261022'
TIMEOUT_SECONDS = 180


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['smoke','events'],default='events')
    parser.add_argument('--n',type=int)
    parser.add_argument('--alpha',choices=['0.5','1'],required=True)
    parser.add_argument('--beta',type=int,choices=[0,1,3],required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = args.repo.resolve()
    binary = args.binary.resolve()
    out = args.output.resolve() if args.output.is_absolute() else (repo/args.output).resolve()
    result_root = (repo/'results/rtx5090').resolve()
    if not out.is_relative_to(result_root) or out==result_root:
        raise ValueError('A fresh batch directory under this repository results/rtx5090 is required')
    if not binary.is_file() or not os.access(binary,os.X_OK):
        raise ValueError('Existing executable binary required; this wrapper never builds')
    n = args.n if args.n is not None else (128 if args.mode=='smoke' else 1024)
    if args.mode=='smoke' and n!=128:
        raise ValueError('Smoke mode requires n=128 for full CPU-double output checks')
    if args.mode=='events' and n not in {1024,2048,3072,4096}:
        raise ValueError('Formal stage2 dimensions are 1024, 2048, 3072 or 4096')
    rounds,repeats,warmup = (1,1,1) if args.mode=='smoke' else (7,30,10)
    expected_events = 5*rounds*repeats
    def git(*command):
        return subprocess.check_output(['git','-C',str(repo),*command],text=True).strip()
    if git('branch','--show-current') != BRANCH:
        raise RuntimeError('Wrong execution branch; no GPU work started')
    gitdir = Path(git('rev-parse','--absolute-git-dir'))
    lock_path = gitdir/'rtx5090-gpu.lock'
    with lock_path.open('a+') as gpu_lock:
        fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(parents=True,exist_ok=False)
        def save(name,value):
            path=out/name
            tmp=path.with_name(path.name+'.tmp')
            tmp.write_bytes((json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
            os.replace(tmp,path)
        def fingerprint(path):
            return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
        command=[str(binary),str(n),str(args.beta),str(rounds),str(repeats),str(warmup),
                 str(out/'events.csv'),'paired','--alpha',args.alpha]
        meta={'status':'RUNNING','source_commit':git('rev-parse','HEAD'),'branch':BRANCH,
              'utc_start':dt.datetime.now(dt.timezone.utc).isoformat(),
              'mode':args.mode,'n':n,'alpha':float(args.alpha),'beta':args.beta,'seed':20261020,
              'rounds':rounds,'repeats':repeats,'warmup':warmup,'expected_events':expected_events,
              'selection':'paired','command':command,'timeout_seconds':TIMEOUT_SECONDS,
              'gpu_lock':str(lock_path),'child_environment_override':{'NVIDIA_TF32_OVERRIDE':'0'},
              'timing':'single-call CUDA event, same default stream; C0 D2D reset before start event',
              'correctness':'original abs<=0.01; full PEDANTIC GPU reference and direct CPU-double candidate checks',
              'scope':'one batch only; no commit/push/build/system changes',
              'python':sys.version,'python_executable':sys.executable}
        telemetry = None
        telemetry_stdout = telemetry_stderr = None
        result={'exit_code':None,'timeout':False,'numeric_acceptance_pass':False,
                'independent_summary_verified':False,'formal_events':None}
        wrapper_exit=1
        try:
            source_files=[package/name for name in ['benchmark.cu','derive_benchmark.py',
                          'summarize_batch.py','provenance.json','CMakeLists.txt','run_closed_batch.py']]
            source_files += [repo/'src/kernels/10_kernel_warptiling.cuh',repo/'src/kernels/30_kernel_mini_simt.cuh']
            if not all(p.is_file() for p in source_files):
                raise FileNotFoundError('Missing host package or device headers')
            sources={str(p.relative_to(repo)) if p.is_relative_to(repo) else str(p):fingerprint(p) for p in source_files}
            build={'binary':fingerprint(binary),'build_metadata':{},'build_metadata_missing':[]}
            for name in ['compile_commands.json','CMakeCache.txt','build.ninja']:
                p=binary.parent/name
                if p.is_file(): build['build_metadata'][name]=fingerprint(p)
                else: build['build_metadata_missing'].append(name)
            save('source-fingerprints.json',sources)
            save('build-fingerprint.json',build)
            save('run-command.json',meta)
            save('PARTIAL.json',meta)
            telemetry_command=['nvidia-smi','--query-gpu=timestamp,index,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu,memory.used',
                               '--format=csv','--loop-ms=250']
            save('telemetry-command.json',{'command':telemetry_command,
                 'scope':'GPU-level NVML snapshots only; no process names/PIDs or other tenant details',
                 'limitation':'250 ms snapshots do not prove constant clock during individual CUDA events'})
            try:
                telemetry_stdout=(out/'telemetry.csv').open('w',encoding='utf-8')
                telemetry_stderr=(out/'telemetry.stderr.log').open('w',encoding='utf-8')
                telemetry=subprocess.Popen(telemetry_command,stdout=telemetry_stdout,stderr=telemetry_stderr)
            except Exception as exc:
                result['telemetry_start_error']=str(exc)
            child_env=os.environ.copy()
            child_env['NVIDIA_TF32_OVERRIDE']='0'
            start=time.monotonic()
            with (out/'run.log').open('w',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=repo,env=child_env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    result['exit_code']=child.wait(timeout=TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    result['timeout']=True
                    child.kill()  # Only this wrapper's own benchmark process.
                    result['exit_code']=child.wait(timeout=15)
            result['wall_seconds']=time.monotonic()-start
            (out/'run.exit').write_text(str(result['exit_code'])+'\n',encoding='utf-8')
            result['numeric_acceptance_pass']=result['exit_code']==0 and not result['timeout']
            # All benchmark outputs are closed before any read, hash, or summary.
            if (out/'events.csv').is_file():
                with (out/'events.csv').open(encoding='utf-8-sig',newline='') as f:
                    result['formal_events']=sum(1 for _ in csv.DictReader(f))
                result['events_sha256']=sha(out/'events.csv')
                analysis_command=[sys.executable,str(package/'summarize_batch.py'),
                    '--csv',str(out/'events.csv'),'--out',str(out/'analysis'),
                    '--selection','paired','--rounds',str(rounds),'--repeats',str(repeats)]
                save('analysis-command.json',{'command':analysis_command})
                with (out/'analysis.log').open('w',encoding='utf-8') as log:
                    try:
                        process=subprocess.run(analysis_command,cwd=repo,stdout=log,stderr=subprocess.STDOUT,timeout=40)
                        result['analysis_exit_code']=process.returncode
                    except subprocess.TimeoutExpired:
                        result['analysis_exit_code']=124
                        result['analysis_timeout']=True
                (out/'analysis.exit').write_text(str(result['analysis_exit_code'])+'\n',encoding='utf-8')
                audit_path=out/'analysis/independent-audit.json'
                if result['analysis_exit_code']==0 and audit_path.is_file():
                    audit=json.loads(audit_path.read_text(encoding='utf-8'))
                    result['independent_summary_verified']=(audit['status']=='PASS' and
                        audit['formal_events']==expected_events and audit['raw_sha256']==result['events_sha256'])
            else:
                result['analysis_status']='NOT_RUN_NO_EVENT_CSV'
            changed_sources=[p for p,record in sources.items() if sha(record['path'])!=record['sha256']]
            result['source_files_changed_during_run']=changed_sources
            result['binary_changed_during_run']=sha(binary)!=build['binary']['sha256']
            complete=(result['independent_summary_verified'] and not result['timeout'] and
                      not changed_sources and not result['binary_changed_during_run'])
            result['status']='COMPLETE' if complete else 'PARTIAL'
            wrapper_exit=0 if complete and result['numeric_acceptance_pass'] else (124 if result['timeout'] else 1)
        except Exception:
            result['status']='PARTIAL'
            result['exception']=traceback.format_exc()
            wrapper_exit=1
        finally:
            if telemetry is not None:
                result['telemetry_exited_before_stop']=telemetry.poll() is not None
                if telemetry.poll() is None:
                    telemetry.terminate()
                    try: telemetry.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        telemetry.kill()
                        telemetry.wait(timeout=5)
                result['telemetry_exit_code']=telemetry.returncode
            if telemetry_stdout is not None: telemetry_stdout.close()
            if telemetry_stderr is not None: telemetry_stderr.close()
            if (out/'telemetry.csv').exists():
                lines=(out/'telemetry.csv').read_text(encoding='utf-8').splitlines()
                result['telemetry_rows']=len([line for line in lines if line.strip() and not line.startswith('timestamp,')])
                result['telemetry_status']='SAMPLED' if result['telemetry_rows'] else 'MISSING'
            else: result['telemetry_status']='MISSING'
            result['utc_end']=dt.datetime.now(dt.timezone.utc).isoformat()
            result['wrapper_exit_code']=wrapper_exit
            result['closed']=True
            save('run-result.json',result)
            save('BATCH_COMPLETE.json',{'closed':True,'status':result.get('status','PARTIAL'),
                'independent_summary_verified':result['independent_summary_verified'],
                'numeric_acceptance_pass':result['numeric_acceptance_pass'],
                'exit_code':result['exit_code'],'timeout':result['timeout'],
                'expected_events':expected_events,'formal_events':result['formal_events']})
            # PARTIAL.json remains as the original startup record; this closed
            # marker and run-result.json are the authoritative completion state.
        print(json.dumps({'output':str(out),**result},ensure_ascii=False),flush=True)
        return wrapper_exit


if __name__=='__main__':
    sys.exit(main())
