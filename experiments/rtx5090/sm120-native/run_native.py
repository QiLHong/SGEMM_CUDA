#!/usr/bin/env python3
"""One closed native event batch; preserves upstream device code, no autotuning."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from build_native import build

NAMES=['hgemm_bcf_dbf_rw','hgemm_k_stages','hgemm_tma_r_k_stages_64','hgemm_tma_r_k_stages_32','hgemm_cublas']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',required=True)
    p.add_argument('--build-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['smoke','events'],default='smoke')
    p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--size',type=int,default=1024)
    p.add_argument('--k',type=int,default=128,help='smoke K128 or K384, common multiple64 >=128')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--distribution',choices=['normal','uniform'],default='normal')
    p.add_argument('--cpu-threads',type=int,default=4)
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    def save(name,data): (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    meta={'complete':False,'mode':args.mode,'dtype':args.dtype,'seed':args.seed,'distribution':args.distribution,
          'alpha':1,'beta':0,'layout':'row-major contiguous','timing':'native per-call CUDA events, cached descriptor; C0 copy, allocation, input conversion, descriptor encoding, resource setup and validation outside event interval',
          'generator':'torch CPU float32 RNG; cast CPU to input dtype then upload; actual input SHA saved',
          'author_criterion':'abs(reference-candidate)<=.016+.016*abs(candidate); torch.allclose(reference,candidate,atol=.016,rtol=.016)',
          'strict_diagnostic':'abs(reference-candidate)<=.01, independently reported',
          'driver_sha256':sha(__file__),'builder_sha256':sha(Path(__file__).with_name('build_native.py')),
          'native_host_sha256':sha(Path(__file__).with_name('native_host.inc'))}
    checks=[]; resources=[]; bitwise=[]
    save('PARTIAL.json',meta)
    try:
        import torch
        torch.set_grad_enabled(False); torch.set_num_threads(args.cpu_threads)
        if args.mode=='smoke':
            M,N,K=128,1024,args.k
            rounds,warmup,repeat=1,1,1
        else:
            M=N=K=args.size
            rounds,warmup,repeat=7,10,30
        if M<=0 or M%128 or N<=0 or N%1024 or K<128 or K%64: raise ValueError('illegal common shape')
        meta.update(shape=[M,N,K],rounds=rounds,warmup=warmup,repeat=repeat,expected_events=rounds*repeat*5,
                    torch=torch.__version__,cuda=torch.version.cuda)
        lib,source=build(args.upstream,args.build_root)
        save('source-manifest.json',source)
        dtype=torch.bfloat16 if args.dtype=='bf16' else torch.float16
        torch.manual_seed(args.seed)
        def generate(shape):
            return torch.randn(*shape) if args.distribution=='normal' else torch.rand(*shape)*2-1
        raw_a,raw_b=generate((M,K)),generate((K,N))
        t0=time.perf_counter(); a_cpu,b_cpu=raw_a.to(dtype),raw_b.to(dtype)
        meta['input_CPU_conversion_ms']=(time.perf_counter()-t0)*1e3
        c0_cpu=torch.zeros(M,N,dtype=dtype)
        t0=time.perf_counter(); a,b,c0=a_cpu.cuda(),b_cpu.cuda(),c0_cpu.cuda()
        c=torch.empty_like(c0); same=torch.empty_like(c0)
        represented=torch.empty((M,N),device='cuda',dtype=torch.float32)
        original=torch.empty_like(represented)
        torch.cuda.synchronize(); meta['allocation_and_H2D_wall_ms']=(time.perf_counter()-t0)*1e3
        def fp(t):
            x=t.detach().cpu().contiguous()
            return {'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),
                    'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
        save('inputs.json',{'A':fp(a),'B':fp(b),'C0':fp(c0),'raw_FP32_A':fp(raw_a),'raw_FP32_B':fp(raw_b)})
        lib.audit_pedantic(a.float(),b.float(),represented)
        lib.audit_pedantic(raw_a.cuda(),raw_b.cuda(),original)
        lib.audit_measure(a,b,same,c0,4,1,1)
        torch.cuda.synchronize()
        # Freeze references outside all measured intervals.
        same_cpu=same.cpu()
        represented_cpu=represented.cpu().to(dtype)
        original_cpu=original.cpu()
        if args.mode=='smoke':
            cpu_reference=(a_cpu.double()@b_cpu.double()).to(dtype)
            sample_rows=sample_cols=None
            cpu_reference_name='full_CPU_double_actual_inputs_then_output_rounding'
        else:
            flat=random.Random(20261022).sample(range(M*N),min(256,M*N))
            sample_rows=torch.tensor([i//N for i in flat],dtype=torch.int64)
            sample_cols=torch.tensor([i%N for i in flat],dtype=torch.int64)
            cpu_reference=(a_cpu[sample_rows,:].double()*b_cpu[:,sample_cols].T.double()).sum(1).to(dtype)
            save('cpu_sample_coordinates.json',list(zip(sample_rows.tolist(),sample_cols.tolist())))
            cpu_reference_name='fixed256_CPU_double_actual_inputs_then_output_rounding'
        def metrics(ref,candidate,name,reference,round_index):
            x,y=ref.double(),candidate.double()
            diff=(x-y).abs(); finite=torch.isfinite(x)&torch.isfinite(y); finite_diff=diff[finite]
            author_bad=(~finite)|(diff>(.016+.016*y.abs()))
            strict_bad=(~finite)|(diff>.01)
            result={'round':round_index,'implementation':name,'reference':reference,'elements':x.numel(),
                    'max_abs':float(finite_diff.max()) if finite_diff.numel() else None,
                    'max_rel_reference_denominator_floor_1e_30':float((finite_diff/x[finite].abs().clamp_min(1e-30)).max()) if finite_diff.numel() else None,
                    'relative_l2':float(torch.linalg.vector_norm(finite_diff)/torch.linalg.vector_norm(x[finite]).clamp_min(1e-30)),
                    'nan':int(torch.isnan(y).sum()),'inf':int(torch.isinf(y).sum()),'nonfinite':int((~finite).sum()),
                    'author_inequality_FP64_failed_elements':int(author_bad.sum()),'strict_abs_001_failed_elements':int(strict_bad.sum()),
                    'author_allclose':bool(torch.allclose(x,y,atol=.016,rtol=.016)),
                    'strict_abs_001_pass':bool(not strict_bad.any()),'exact_equal_elements':int((x==y).sum())}
            # Original torch allclose is evaluated with original output dtype too.
            if ref.dtype==candidate.dtype:
                result['author_allclose_FP64_diagnostic']=result['author_allclose']
                result['author_allclose']=bool(torch.allclose(ref,candidate,atol=.016,rtol=.016))
                result['author_failed_elements']=int((~torch.isclose(ref,candidate,atol=.016,rtol=.016)).sum())
            checks.append(result); save('correctness.json',checks)
            return result
        orders=[]
        for r in range(rounds):
            order=list(range(5))[r%5:]+list(range(5))[:r%5]
            if r%2: order.reverse()
            orders.append(order)
        save('order.json',[[NAMES[i] for i in row] for row in orders])
        save('PARTIAL.json',meta)
        all_pass=True
        with (out/'events.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['round','order_position','implementation','sample','milliseconds','M','N','K','seed','dtype'])
            writer.writeheader()
            for r,order in enumerate(orders):
                for pos,variant in enumerate(order):
                    measurement=lib.audit_measure(a,b,c,c0,variant,warmup,repeat)
                    values=measurement.pop('milliseconds')
                    if len(values)!=repeat: raise RuntimeError('native event count mismatch')
                    if not all(math.isfinite(ms) and ms>0 for ms in values): raise RuntimeError('nonpositive/nonfinite event')
                    for sample,ms in enumerate(values):
                        writer.writerow(dict(round=r,order_position=pos,implementation=NAMES[variant],sample=sample,milliseconds=ms,M=M,N=N,K=K,seed=args.seed,dtype=args.dtype))
                    f.flush()
                    resources.append(dict(round=r,implementation=NAMES[variant],**measurement))
                    save('resources_and_setup.json',resources)
                    candidate=c.cpu()
                    # Untimed original exported host path, same bytes and same device function.
                    upstream_output=torch.empty_like(c)
                    getattr(lib,NAMES[variant])(a,b,upstream_output)
                    torch.cuda.synchronize()
                    upstream_cpu=upstream_output.cpu()
                    differing=int((candidate.view(torch.int16)!=upstream_cpu.view(torch.int16)).sum())
                    bitwise.append({'round':r,'implementation':NAMES[variant],'different_output_elements':differing,'exact_bitwise_equal':differing==0})
                    save('upstream_bitwise.json',bitwise)
                    all_pass &= differing==0
                    check_same=metrics(same_cpu,candidate,NAMES[variant],'full_same_dtype_cublas_COMPUTE_32F',r)
                    check_represented=metrics(represented_cpu,candidate,NAMES[variant],'full_PEDANTIC_FP32_actual_inputs_then_output_rounding',r)
                    metrics(original_cpu,candidate,NAMES[variant],'full_PEDANTIC_FP32_original_FP32_inputs_no_output_rounding',r)
                    sampled=candidate if sample_rows is None else candidate[sample_rows,sample_cols]
                    check_cpu=metrics(cpu_reference,sampled,NAMES[variant],cpu_reference_name,r)
                    all_pass &= check_same['author_allclose'] and check_represented['author_allclose'] and check_cpu['author_allclose']
                    print(json.dumps({'round':r,'implementation':NAMES[variant],'mean_ms':statistics.mean(values),'author_same_dtype_pass':check_same['author_allclose'],'cpu_double_pass':check_cpu['author_allclose']}),flush=True)
        with (out/'events.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
        if len(rows)!=meta['expected_events']: raise RuntimeError('closed CSV event count mismatch')
        means={name:[statistics.mean(float(x['milliseconds']) for x in rows if x['implementation']==name and int(x['round'])==r) for r in range(rounds)] for name in NAMES}
        summary=[]
        for name in NAMES:
            median=statistics.median(means[name]);paired=[means[NAMES[4]][r]/means[name][r]-1 for r in range(rounds)]
            summary.append({'implementation':name,'round_mean_ms':means[name],'median_round_mean_ms':median,'tflops':2*M*N*K/(median*1e9),'paired_speedup_vs_cublas':paired,'median_paired_speedup_vs_cublas':statistics.median(paired)})
        save('summary.json',summary)
        meta.update(complete=True,acceptance_pass=bool(all_pass),actual_events=len(rows),events_sha256=sha(out/'events.csv'))
        save('COMPLETED.json',meta)
        return 0 if all_pass else 2
    except Exception:
        meta['exception']=traceback.format_exc(); save('PARTIAL.json',meta); raise

if __name__=='__main__': sys.exit(main())
