"""Collect non-secret task environment. Does not change system configuration."""
import datetime, json, os, pathlib, platform, shutil, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[2]
out=root/'results/rtx5090/20261022-01-environment'
out.mkdir(parents=True,exist_ok=True)
commands={
 'os':['cat','/etc/os-release'], 'cpu':['lscpu'], 'memory':['free','-b'],
 'disk':['df','-B1','/root/autodl-tmp'],
 'gpu':['nvidia-smi','--query-gpu=name,uuid,compute_cap,memory.total,memory.used,driver_version,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu','--format=csv'],
 'nvcc':['/usr/local/cuda/bin/nvcc','--version'], 'nvcc_targets':['/usr/local/cuda/bin/nvcc','--list-gpu-code'],
 'compiler':['/usr/bin/g++','--version'],'cmake':['/usr/bin/cmake','--version'],
 'ninja':['/root/autodl-tmp/rtx5090-tools/bin/ninja','--version'],
 'ncu':['/usr/local/cuda/bin/ncu','--version'], 'ncu_permission':['/usr/local/cuda/bin/ncu','--query-metrics'],
 'cuobjdump':['/usr/local/cuda/bin/cuobjdump','--version'],'nvdisasm':['/usr/local/cuda/bin/nvdisasm','--version'],
 'sanitizer':['/usr/local/cuda/bin/compute-sanitizer','--version'],
 'driver_symbols':['nm','-D','/usr/lib/x86_64-linux-gnu/libcuda.so.1'],
}
records={}
for name,cmd in commands.items():
 try:
  p=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=25)
  text=p.stdout
  if name=='driver_symbols': text='\n'.join(x for x in text.splitlines() if 'cuTensorMap' in x)
  (out/(name+'.txt')).write_text(text,encoding='utf-8')
  records[name]={'command':cmd,'exit_code':p.returncode}
 except Exception as e: records[name]={'command':cmd,'error':str(e)}
p=subprocess.run(['nvidia-smi','--query-compute-apps=used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
records['other_gpu_processes']={'count':len(p.stdout.strip().splitlines()) if p.stdout.strip() else 0,'used_memory_MiB':p.stdout.strip().splitlines(),'names_and_pids':'omitted'}
import torch
torch.cuda.init()
records.update({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'torch_build_cuda':torch.version.cuda,'device':str(torch.cuda.get_device_properties(0)),'arch_list':torch.cuda.get_arch_list(),'server_expiry':'not available; unknown','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'loaded_torch_cuda_libraries':sorted(set(l[l.index('/'):].strip() for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if ('libcuda' in l or 'libcublas' in l) and '/' in l))})
(out/'environment.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:records[k] for k in ['torch','device','other_gpu_processes','ncu_permission']},indent=2))

