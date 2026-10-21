#!/usr/bin/env python3
"""Generate isolated Linux/SM120 host adapters; never edit frozen experiments."""
import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

TENSOR={'tensor':'tensorcore','grid':'tensorcore-grid','ktile':'tensorcore-ktile','bn64':'tensorcore-bn64','robustness':'tensorcore-bn64-robustness'}
CHECKPOINT=r'''
// Host-only checkpoint hook: runCase's streams are closed on return. Do not
// publish process.csv here: that batch-wide stream remains open until batch end.
#include <unistd.h>
#include <sys/wait.h>
static void checkpointClosedCase(const std::string& csv) {
    const char* hook=std::getenv("RTX5090_CASE_CHECKPOINT");
    if(!hook || hook[0]!='/') throw std::runtime_error("Set absolute RTX5090_CASE_CHECKPOINT executable before --batch");
    std::ofstream marker(csv+".closed.txt"); marker<<"630 formal events; runCase files closed; callback pending\n"; marker.close();
    pid_t child=fork();
    if(child<0) throw std::runtime_error("checkpoint fork failed");
    if(child==0) { execl(hook,hook,csv.c_str(),static_cast<char*>(nullptr)); _exit(127); }
    int status=0;
    if(waitpid(child,&status,0)!=child || !WIFEXITED(status) || WEXITSTATUS(status)!=0)
        throw std::runtime_error("Checkpoint failed; stopping batch with closed evidence preserved");
}
'''

def h(b):return hashlib.sha256(b).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();repo=Path(a.repo).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest={'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
              'sources':{},'device_headers':{},'adaptation':'host-only SM120 acceptance, POSIX PID, include relocation and closed-case publish callback; no algorithm changes'}
    cm=['cmake_minimum_required(VERSION 3.19)','set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architecture")',
        'project(RTX5090HistoricalGEMM LANGUAGES CXX CUDA)','find_package(CUDAToolkit 12.8 REQUIRED)',
        'set(CMAKE_EXPORT_COMPILE_COMMANDS ON)','set(CMAKE_CUDA_STANDARD 17)','set(CMAKE_CUDA_STANDARD_REQUIRED ON)']
    def dependencies(path):
        for inc in re.findall(r'^#include "([^"]+)"',path.read_text(encoding='utf-8'),re.M):
            dep=(path.parent/inc).resolve()
            if dep.is_file():
                key=dep.relative_to(repo).as_posix()
                if key not in manifest['device_headers']:
                    manifest['device_headers'][key]=h(dep.read_bytes());dependencies(dep)
    def add(name,path,defs='',extra=''):
        raw=path.read_bytes();s=raw.decode('utf-8');old=s
        dependencies(path)
        s=re.sub(r'^#include "([^"]+)"',lambda m:'#include "'+(path.parent/m[1]).resolve().as_posix()+'"',s,flags=re.M)
        if name in TENSOR:
            if name=='robustness':
                assert s.count('#include <process.h>')==1
                s=s.replace('#include <process.h>','#include <unistd.h>\n#define _getpid getpid')
                old_guard='prop.major!=8||prop.minor!=9'
                new_guard='!((prop.major==8 && prop.minor==9) || (prop.major==12 && prop.minor==0))'
                s=s.replace('This archive requires sm_89 hardware','This host adapter accepts sm_89 or sm_120; actual target is recorded')
                marker='int main(int argc,char** argv)try{'
                assert s.count(marker)==1;s=s.replace(marker,CHECKPOINT+'\n'+marker)
                needle='runCase(c,30,10,(out/("timings-"+std::to_string(c.n)+"-"+std::to_string(c.seed)+".csv")).string());'
                assert s.count(needle)==1
                s=s.replace(needle,needle+'checkpointClosedCase((out/("timings-"+std::to_string(c.n)+"-"+std::to_string(c.seed)+".csv")).string());')
                # Fail before opening a GPU context if publication cannot be called.
                check='int batch=std::stoi(argv[2]);if(batch<1||batch>3)'
                assert s.count(check)==1
                s=s.replace(check,'if(!std::getenv("RTX5090_CASE_CHECKPOINT"))throw std::runtime_error("RTX5090_CASE_CHECKPOINT is required for incremental publication");\n    '+check)
            else:
                old_guard='prop.major!=8||prop.minor<0'
                new_guard='!((prop.major==8 && prop.minor>=0) || (prop.major==12 && prop.minor==0))'
                s=s.replace('This experiment requires tested Ampere/Ada sm80+ hardware','This host adapter accepts Ampere/Ada or sm_120; actual target is recorded')
            assert s.count(old_guard)==1;s=s.replace(old_guard,new_guard)
        target=out/(name+'.cu');target.write_text(s,encoding='utf-8',newline='')
        (out/(name+'.host.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=path.relative_to(repo).as_posix(),tofile=target.name)),encoding='utf-8')
        manifest['sources'][name]={'original_path':path.relative_to(repo).as_posix(),'original_sha256':h(raw),'adapted_sha256':h(target.read_bytes())}
        cm.extend([f'add_executable({name} {name}.cu)',f'target_link_libraries({name} PRIVATE CUDA::cublas)',
                   f'target_compile_options({name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo;-Xptxas=-v;--expt-relaxed-constexpr;--expt-extended-lambda{extra}>)'])
        if defs:cm.append(f'target_compile_definitions({name} PRIVATE {defs})')
    for name,d in TENSOR.items():add(name,repo/'experiments'/d/'benchmark.cu')
    add('simt_strict',repo/'scripts/windows_vitamin_benchmark.cu','SGEMM_K10_MIN_BLOCKS=2 VITAMIN_ORIGINAL_CONFIG=0')
    add('simt_original',repo/'scripts/windows_vitamin_benchmark.cu','SGEMM_K10_MIN_BLOCKS=2 VITAMIN_ORIGINAL_CONFIG=1',
        ';--use_fast_math;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__')
    add('dbf_default',repo/'scripts/windows_dbf_register_benchmark.cu','DBF_MAX_REGISTERS=0')
    add('dbf_maxr128',repo/'scripts/windows_dbf_register_benchmark.cu','DBF_MAX_REGISTERS=128',';--maxrregcount=128')
    (out/'CMakeLists.txt').write_text('\n'.join(cm)+'\n',encoding='utf-8')
    (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    names=['vitamin_tf32_dbf','tf32_bn64_bk32','cublas_tf32'];perms=[(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    sizes=[1024,2048,3072,4096];seeds=[20261020,42,12345]
    with (out/'robustness-plan.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['batch','case_index','m','n','k','seed','round','order','implementation'])
        for batch in range(1,4):
            case=0
            for si in range(4):
                for vi in range(3):
                    n=sizes[(si+batch-1)%4];seed=seeds[(vi+batch-1)%3]
                    for r in range(7):
                        pi=batch-1 if r==6 else (r+case+batch-1)%6
                        for order,impl in enumerate(perms[pi]):w.writerow([batch,case,n,n,n,seed,r,order,names[impl]])
                    case+=1
    print(json.dumps({'generated':str(out),'targets':list(manifest['sources']),'plan_groups':756,'formal_events':22680},indent=2))

if __name__=='__main__':main()
