#!/usr/bin/env python3
"""Run exactly one serial GPU batch with timeout, telemetry and closed evidence."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess, sys, fcntl
p=argparse.ArgumentParser()
p.add_argument('--batch',required=True)
p.add_argument('--timeout',type=int,default=180)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
if pathlib.Path(a.batch).name!=a.batch: raise SystemExit('batch must be one directory name')
gpu_lock=(root/'.git/rtx5090-gpu.lock').open('a+')
fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=root/'results/rtx5090'/a.batch
out.mkdir(parents=True,exist_ok=False)
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd: raise SystemExit('command missing')
def save(name,value): (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
meta={'command':cmd,'cwd':str(root),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
      'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':a.timeout,
      'telemetry_note':'1-second snapshots; does not establish per-event clocks or constancy'}
save('run-command.json',meta)
telemetry=None
with (out/'telemetry.csv').open('w') as t,(out/'run.log').open('w') as log:
    t.write('timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used\n');t.flush()
    try:
        telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used','--format=csv,noheader,nounits','--loop-ms=1000'],stdout=t,stderr=t)
    except OSError as e: meta['telemetry_error']=str(e)
    try:
        process=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
        code=process.returncode
    except subprocess.TimeoutExpired: code=124
    except OSError as e: code=127;log.write(str(e)+'\n')
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try: telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired: telemetry.kill();telemetry.wait()
meta.update(exit_code=code,end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save('run-result.json',meta)
(out/'run.exit').write_text(str(code)+'\n')
data=out/'data'
complete=False
if (data/'COMPLETED.json').exists() and (data/'events.csv').exists():
    info=json.loads((data/'COMPLETED.json').read_text())
    audit=subprocess.run([sys.executable,str(root/'scripts/rtx5090/recompute_results.py'),str(data/'events.csv'),'--expected-events',str(info['expected_events']),'--output',str(out/'independent-summary.json')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    (out/'independent-analysis.log').write_text(audit.stdout+audit.stderr)
    if audit.returncode==0:
        recomputed=json.loads((out/'independent-summary.json').read_text())
        reported={r['implementation']:r for r in json.loads((data/'summary.json').read_text())}
        assert all(abs(r['median_round_mean_ms']-reported[r['implementation']]['median_round_mean_ms'])<1e-10 for r in recomputed['summary'])
        complete=info.get('complete') is True and code in (0,2)
    meta['numeric_acceptance_pass']=info.get('acceptance_pass')
save('BATCH_COMPLETE.json',{'closed':True,'status':'COMPLETE' if complete else 'PARTIAL','independent_summary_verified':complete,
                          'exit_code':code,'numeric_acceptance_pass':meta.get('numeric_acceptance_pass'),'timeout':code==124})
print(json.dumps({'batch':a.batch,'closed':True,'complete':complete,'exit_code':code},ensure_ascii=False),flush=True)
sys.exit(code)
