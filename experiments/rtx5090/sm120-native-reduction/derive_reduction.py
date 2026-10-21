#!/usr/bin/env python3
"""Create a sibling controlled-cuBLAS reduction audit. Source directory stays frozen."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

GUARD=r'''
// Scoped per-handle math mode. Normal return checks restoration; exception
// unwinding also attempts restoration without throwing from a destructor.
struct AuditMathModeGuard {
    cublasHandle_t handle;
    cublasMath_t original;
    bool changed=false;
    explicit AuditMathModeGuard(cublasHandle_t h):handle(h) { AUDIT_BLAS(cublasGetMathMode(handle,&original)); }
    void disallow_reduced_precision_reduction() {
        cublasMath_t requested=static_cast<cublasMath_t>(int(original)|int(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        AUDIT_BLAS(cublasSetMathMode(handle,requested)); changed=true;
    }
    void restore_checked() { if(changed) { AUDIT_BLAS(cublasSetMathMode(handle,original)); changed=false; } }
    ~AuditMathModeGuard() noexcept { if(changed) cublasSetMathMode(handle,original); }
};
'''

def digest(raw):return hashlib.sha256(raw).hexdigest()
def once(s,old,new):
    if s.count(old)!=1:raise RuntimeError(f'Expected exactly one match: {old[:120]!r}, got {s.count(old)}')
    return s.replace(old,new,1)

def derive(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if source==output:raise ValueError('Sibling output required')
    output.mkdir(parents=True,exist_ok=False)
    files={name:(source/name).read_text(encoding='utf-8') for name in ['native_host.inc','build_native.py','run_native.py']}
    original=dict(files)
    h=files['native_host.inc']
    h=once(h,'template<class T>\nstatic py::dict audit_measure_t',GUARD+'\ntemplate<class T>\nstatic py::dict audit_measure_t')
    h=once(h,'    cublasMath_t math_mode; AUDIT_BLAS(cublasGetMathMode(handle,&math_mode));',
'''    AuditMathModeGuard math_guard(handle);
    if(variant==5) math_guard.disallow_reduced_precision_reduction();
    cublasMath_t math_mode; AUDIT_BLAS(cublasGetMathMode(handle,&math_mode));
    if(variant==5) TORCH_CHECK((int(math_mode)&int(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))!=0,"cuBLAS disallow-reduction flag was not applied");''')
    h=once(h,'        case 4: AUDIT_BLAS(cublasGemmEx','        case 4:\n        case 5: AUDIT_BLAS(cublasGemmEx')
    h=once(h,'variant>=0 && variant<=4','variant>=0 && variant<=5')
    h=once(h,'    result["cublas_math_mode"]=int(math_mode);',
'''    result["cublas_math_mode"]=int(math_mode);
    result["cublas_math_mode_before"]=int(math_guard.original);
    result["cublas_reduction_policy"]=(variant==5)?"DISALLOW_REDUCED_PRECISION_REDUCTION":"original handle mode unchanged";
    math_guard.restore_checked();
    cublasMath_t restored_mode; AUDIT_BLAS(cublasGetMathMode(handle,&restored_mode));
    TORCH_CHECK(restored_mode==math_guard.original,"cuBLAS math mode restoration failed");
    result["cublas_math_mode_after_restoration"]=int(restored_mode);''')
    files['native_host.inc']=h
    b=files['build_native.py']
    b=once(b,'name="sm120_native_audit_lib"','name="sm120_native_reduction_audit_lib"')
    b=once(b,'"hgemm_sm120_native.cu"','"hgemm_sm120_native_reduction.cu"')
    # A fresh build directory is mandatory on first invocation; reuse only the same sibling module.
    b=once(b,'    build_root.mkdir(parents=True, exist_ok=True)',
'''    if (build_root / "hgemm_sm120_native.cu").exists():
        raise RuntimeError("Refusing original native build directory; use a new reduction build root")
    build_root.mkdir(parents=True, exist_ok=True)''')
    files['build_native.py']=b
    r=files['run_native.py']
    r=once(r,"'hgemm_cublas']","'hgemm_cublas','hgemm_cublas_no_reduced_reduction']")
    r=once(r,'expected_events=rounds*repeat*5','expected_events=rounds*repeat*len(NAMES)')
    r=once(r,'        lib.audit_measure(a,b,same,c0,4,1,1)',
'''        default_ref=torch.empty_like(c0)
        lib.audit_measure(a,b,default_ref,c0,4,1,1)
        lib.audit_measure(a,b,same,c0,5,1,1)''')
    r=once(r,'        same_cpu=same.cpu()',
'''        same_cpu=same.cpu()
        default_cpu=default_ref.cpu()
        meta['same_precision_reference']='CUBLAS_COMPUTE_32F + CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION, identical input/output dtype'
        meta['performance_baseline']='hgemm_cublas_no_reduced_reduction'
        meta['original_default_cublas_preserved']=True
        meta['controlled_change']='sixth comparator adds only per-handle disallow-reduced-reduction flag; restored after each call; original default row unchanged' ''')
    r=once(r,'            order=list(range(5))[r%5:]+list(range(5))[:r%5]',
             '            order=list(range(len(NAMES)))[r%len(NAMES):]+list(range(len(NAMES)))[:r%len(NAMES)]')
    r=once(r,'        all_pass=True','        all_pass=True\n        per_impl_valid={name:True for name in NAMES}')
    r=once(r,'                    getattr(lib,NAMES[variant])(a,b,upstream_output)',
'''                    getattr(lib,NAMES[variant] if variant<5 else NAMES[4])(a,b,upstream_output)''')
    r=once(r,"bitwise.append({'round':r,'implementation':NAMES[variant],'different_output_elements':differing,'exact_bitwise_equal':differing==0})",
"bitwise.append({'round':r,'implementation':NAMES[variant],'different_output_elements':differing,'exact_bitwise_equal':differing==0,'required_equal_to_original':variant<5,'comparison':'same original variant' if variant<5 else 'new reduction-controlled comparator vs original default cuBLAS; diagnostic only'})")
    r=once(r,'                    all_pass &= differing==0',
'''                    if variant<5:
                        all_pass &= differing==0
                        per_impl_valid[NAMES[variant]] &= differing==0''')
    r=once(r,"                    check_same=metrics(same_cpu,candidate,NAMES[variant],'full_same_dtype_cublas_COMPUTE_32F',r)",
'''                    check_same=metrics(same_cpu,candidate,NAMES[variant],'full_same_dtype_cublas_COMPUTE_32F_DISALLOW_REDUCED_PRECISION_REDUCTION',r)
                    metrics(default_cpu,candidate,NAMES[variant],'diagnostic_original_default_cublas_COMPUTE_32F_unchanged',r)''')
    r=once(r,"                    all_pass &= check_same['author_allclose'] and check_represented['author_allclose'] and check_cpu['author_allclose']",
'''                    valid=bool(check_same['author_allclose'] and check_represented['author_allclose'] and check_cpu['author_allclose'])
                    per_impl_valid[NAMES[variant]] &= valid
                    all_pass &= valid
                    save('per_implementation_validity.json',[{'implementation':name,'valid_fp32_accum_comparison':bool(per_impl_valid[name]),'status_scope':'all completed rounds so far','acceptance_threshold':'unchanged author .016/.016; represented PEDANTIC and CPU double; same dtype disallow-reduction cuBLAS; original bitwise required for variants0..4'} for name in NAMES])''')
    r=once(r,'paired=[means[NAMES[4]][r]/means[name][r]-1 for r in range(rounds)]',
             'paired=[means[NAMES[5]][r]/means[name][r]-1 for r in range(rounds)]')
    r=once(r,"'paired_speedup_vs_cublas':paired,'median_paired_speedup_vs_cublas':statistics.median(paired)",
             "'baseline':NAMES[5],'valid_fp32_accum_comparison':bool(per_impl_valid[name]),'paired_speedup_vs_cublas_no_reduced_reduction':paired,'median_paired_speedup_vs_cublas_no_reduced_reduction':statistics.median(paired)")
    r=once(r,'meta.update(complete=True,acceptance_pass=bool(all_pass),actual_events=len(rows),events_sha256=sha(out/\'events.csv\'))',
'''meta.update(complete=True,acceptance_pass=bool(all_pass),actual_events=len(rows),events_sha256=sha(out/'events.csv'),
                    valid_fp32_accum_comparison_by_implementation={name:bool(per_impl_valid[name]) for name in NAMES},
                    primary_fp32_accum_comparison_pass=all(per_impl_valid[name] for i,name in enumerate(NAMES) if i!=4),
                    default_cublas_numerical_failures_retained=not per_impl_valid[NAMES[4]])''')
    files['run_native.py']=r
    for name,text in files.items():
        if name.endswith('.py'):ast.parse(text)
        (output/name).write_text(text,encoding='utf-8',newline='')
        (output/(name+'.diff')).write_text(''.join(difflib.unified_diff(original[name].splitlines(True),text.splitlines(True),fromfile='original/'+name,tofile='reduction/'+name)),encoding='utf-8')
    manifest={'source_directory':str(source),'created_sibling':str(output),'source_sha256':{n:digest((source/n).read_bytes()) for n in files},
              'derived_sha256':{n:digest((output/n).read_bytes()) for n in files},
              'original_unchanged':all((source/n).read_text(encoding='utf-8')==original[n] for n in files),
              'formal_events':6*7*30,'gpu_run_by_derivation':False,
              'change':'variant5 adds DISALLOW_REDUCED_PRECISION_REDUCTION only; defaultvariant4 unchanged; thresholds unchanged; old baseline frozen; independent refs determine validity'}
    (output/'derivation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();derive(a.source,a.output)
