#!/usr/bin/env python3
"""Independent closed-CSV reconstruction; does not import benchmark driver."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

NAMES=['hgemm_bcf_dbf_rw','hgemm_k_stages','hgemm_tma_r_k_stages_64','hgemm_tma_r_k_stages_32','hgemm_cublas','hgemm_cublas_no_reduced_reduction']
STRICT=NAMES[5]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
    meta=read('COMPLETED.json');checks=read('correctness.json');bits=read('upstream_bitwise.json');resources=read('resources_and_setup.json')
    with (run/'events.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    rounds,repeats=int(meta['rounds']),int(meta['repeat']);expect=6*rounds*repeats
    assert meta['complete'] and len(rows)==expect==meta['actual_events']==meta['expected_events']
    assert hashlib.sha256((run/'events.csv').read_bytes()).hexdigest()==meta['events_sha256']
    assert set(x['implementation'] for x in rows)==set(NAMES)
    groups={}
    for x in rows:
        key=(x['implementation'],int(x['round']))
        assert 0<=key[1]<rounds
        ms=float(x['milliseconds']);assert math.isfinite(ms) and ms>0
        groups.setdefault(key,[]).append((int(x['sample']),ms))
    assert len(groups)==6*rounds
    for v in groups.values():assert len(v)==repeats and sorted(i for i,_ in v)==list(range(repeats))
    # Every call reports before/applied/after math modes. Restoration is explicit evidence.
    assert len(resources)==6*rounds
    for r in resources:
        assert int(r['cublas_math_mode_after_restoration'])==int(r['cublas_math_mode_before'])
        if r['implementation']!=STRICT:assert int(r['cublas_math_mode'])==int(r['cublas_math_mode_before'])
        else:assert r['cublas_reduction_policy']=='DISALLOW_REDUCED_PRECISION_REDUCTION'
    means={name:[statistics.mean(ms for _,ms in groups[name,r]) for r in range(rounds)] for name in NAMES}
    valid={};summaries=[]
    for i,name in enumerate(NAMES):
        independent=[x for x in checks if x['implementation']==name and
                     (x['reference']=='full_same_dtype_cublas_COMPUTE_32F_DISALLOW_REDUCED_PRECISION_REDUCTION' or
                      x['reference']=='full_PEDANTIC_FP32_actual_inputs_then_output_rounding' or
                      'CPU_double_actual_inputs_then_output_rounding' in x['reference'])]
        assert len(independent)==3*rounds
        for r in range(rounds):assert len([x for x in independent if int(x['round'])==r])==3
        bitrows=[x for x in bits if x['implementation']==name];assert len(bitrows)==rounds
        assert all(x['required_equal_to_original']==(i<5) for x in bitrows)
        valid[name]=all(x['author_allclose'] and x['nonfinite']==0 for x in independent) and (i==5 or all(x['exact_bitwise_equal'] for x in bitrows))
        median=statistics.median(means[name]);m,n,k=meta['shape'];pair=[means[STRICT][r]/means[name][r]-1 for r in range(rounds)]
        summaries.append({'implementation':name,'round_mean_ms':means[name],'median_round_mean_ms':median,'tflops':2*m*n*k/(median*1e9),
                          'baseline':STRICT,'paired_speedup_vs_strict_reduction_control':pair,'median_paired_speedup_vs_strict_reduction_control':statistics.median(pair),
                          'valid_fp32_accum_comparison':bool(valid[name]),'independent_correctness_records':len(independent),
                          'same_original_variant_bitwise_required':i<5})
    assert valid==meta['valid_fp32_accum_comparison_by_implementation']
    audit={'formal_event_count':len(rows),'expected_count':expect,'round_groups':len(groups),
           'closed_csv_sha256':meta['events_sha256'],'valid_fp32_accum_comparison_by_implementation':valid,
           'primary_fp32_accum_comparison_pass':all(valid[n] for i,n in enumerate(NAMES) if i!=4),
           'aggregate_all_six_pass':all(valid.values()),'original_default_cublas_fails_independent_checks':not valid[NAMES[4]],
           'math_mode_restoration_all_calls_verified':True,
           'interpretation':'Default comparator remains a measured row; if invalid it is excluded from valid FP32-accumulation performance conclusions. Exact cause beyond the controlled reduction flag requires measured followup; no threshold was changed.'}
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf-8')
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
