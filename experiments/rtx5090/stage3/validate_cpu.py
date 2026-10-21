"""CPU-only source generation/parser/closure checks. Never invokes CUDA tools."""
import argparse, contextlib, copy, csv, json, types
from pathlib import Path
import stage3_runner as r

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=a.repo.resolve();out=a.output.resolve()
    if out.exists():raise RuntimeError('Use a fresh validation output')
    out.mkdir(parents=True)
    spec=json.loads((r.HERE/'execution-specs.json').read_text())
    r.verify_sources(repo,spec)
    args=types.SimpleNamespace(repo=repo,build_root=out/'generated',cutlass=out/'FAKE_CUTLASS_PATH_PREPARE_ONLY',nvcc='nvcc',nvtx_library=False)
    prepared=[]
    for id in spec['builds']:
        rec=r.prepare_build(args,spec,id)
        assert '-arch=sm_120' in rec['command']
        assert rec['device_headers_unchanged']
        if spec['builds'][id]['kind']!='stock':
            code=(args.build_root/id/'legacy_host.cu').read_text(encoding='utf-8')
            assert 'Host adapter: renamed main needs an explicit success return.' in code
        prepared.append(dict(id=id,disabled_host_launchers=len(rec['disabled_unrelated_host_launchers'])))
    # Same prepare is idempotent, while changed generated source is rejected.
    r.prepare_build(args,spec,'k31')
    target=args.build_root/'k31'/'captured_main.cu'
    target.write_text(target.read_text()+'\n// integrity test\n')
    try:r.prepare_build(args,spec,'k31')
    except RuntimeError as e:assert 'Prepared source changed' in str(e)
    else:raise AssertionError('Changed prepared source was accepted')
    resources=r.parse_resources((repo/'docs/results/k31-20261013/ptxas.txt').read_text())
    assert resources and any(x['spill_store_bytes'] or x['spill_load_bytes'] for x in resources)
    assert any(not x['historical_gate_le240_no_spill'] for x in resources)
    # Close synthetic batches through the real statistics and manifest code.
    # The invocation and telemetry functions are replaced; no subprocess runs.
    synthetic=copy.deepcopy(spec)
    synthetic['series']['cpu-fixture']=dict(spec['series']['k31'])
    synthetic['series']['cpu-fixture']['candidates']=spec['series']['k31']['candidates'][:2]
    for candidate in synthetic['series']['cpu-fixture']['candidates']:
        r.save(args.build_root/candidate['build']/'build.json',{'status':'ready'})
    args.series='cpu-fixture';args.size=128;args.beta=3;args.protocol='events7'
    args.timeout=1;args.batch_budget=60;args.gpu_lock=out/'NOT_A_REAL_GPU_LOCK'
    r.gpu_lock=lambda path:contextlib.nullcontext()
    r.telemetry=lambda path:{'status':'synthetic_cpu_fixture_no_gpu'}
    cases=[]
    for case in ['complete','timeout','numerical_failure','input_mismatch']:
        args.output=out/('SYNTHETIC-'+case)
        def fake_invoke(args,spec,c,n,beta,repeat,warm,folder,timeout=None):
            folder.mkdir()
            failed=case in ['timeout','numerical_failure'] and c['id']=='k10' and repeat>1
            status=case if failed else 'passed'
            fingerprints=[dict(matrix=x,bytes='65536',fnv1a64=('different' if case=='input_mismatch' and c['id']=='k10' else x)) for x in ['A','B','C0']]
            if not failed:
                values=[{'event_index':i,'ms':2 if c['id']=='strict' else 1} for i in range(repeat)]
                r.csvsave(folder/'capture.events.csv',values)
            return dict(status=status,events=0 if failed else repeat,round_mean_ms=999,
                input_fingerprints=fingerprints,resource_spill_present=False)
        r.invoke=fake_invoke
        r.do_run(args,synthetic)
        closed=json.loads((args.output/'BATCH_CLOSED.json').read_text())
        summary=json.loads((args.output/'summary.json').read_text())
        assert closed['complete']==(case=='complete')
        if case=='complete':
            assert closed['formal_events']==420
            assert next(x for x in summary if x['implementation']=='k10')['median_paired_gain_vs_strict_pct']==100
            assert next(x for x in summary if x['implementation']=='strict')['median_round_mean_ms']==2
        if case=='input_mismatch':assert all(x['median_paired_gain_vs_strict_pct'] is None for x in summary)
        with (args.output/'sha256-manifest.csv').open() as f:
            assert all(r.sha(args.output/x['path'])==x['sha256'] for x in csv.DictReader(f))
        cases.append(dict(case=case,complete=closed['complete'],formal_events=closed['formal_events']))
    report=dict(status='CPU_VALIDATION_PASS',cuda_compilation='NOT_RUN',gpu_execution='NOT_RUN',
        source_commit=spec['source_commit'],source_hashes_verified=len(spec['source_hashes']),
        prepared_builds=len(prepared),series=len(spec['series']),prepared=prepared,
        archived_ptxas_records_parsed=len(resources),archived_spill_records=sum(bool(x['spill_store_bytes'] or x['spill_load_bytes']) for x in resources),
        synthetic_closure_cases=cases,warning='Synthetic validation directories are not benchmark results; do not publish them as GPU measurements.')
    r.save(out/'validation-report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='prepared'},indent=2))

if __name__=='__main__':main()
