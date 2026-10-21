"""Extract targeted NCU metrics; profiler durations never enter the event table."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

METRICS = {
    'registers': 'launch__registers_per_thread',
    'shared_bytes': 'launch__shared_mem_per_block',
    'occupancy_pct': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'eligible_warps': 'smsp__warps_eligible.avg.per_cycle_active',
    'issue_pct': 'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'tensor_elapsed_pct': 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'tensor_active_pct': 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
    'fma_active_pct': 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
    'barrier_stall': 'smsp__warps_issue_stalled_barrier.avg.per_cycle_active',
    'long_scoreboard_stall': 'smsp__warps_issue_stalled_long_scoreboard.avg.per_cycle_active',
    'short_scoreboard_stall': 'smsp__warps_issue_stalled_short_scoreboard.avg.per_cycle_active',
    'shared_load_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'shared_store_wavefronts': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'shared_load_conflicts': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'shared_store_conflicts': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'local_load_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'local_store_sectors': 'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'l2_read_sectors': 'lts__t_sectors_op_read.sum',
    'dram_read_bytes': 'dram__bytes_read.sum',
    'dram_write_bytes': 'dram__bytes_write.sum',
    'profile_duration': 'gpu__time_duration.sum',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    assert not args.output.exists(), 'Refusing overwrite'
    assert (args.profile/'PROFILE_COMPLETE.txt').exists(), 'Profile incomplete'
    commands=json.loads((args.profile/'commands.json').read_text(encoding='utf-8-sig'))
    assert len(commands)==10 and all(x['exit_code']==0 for x in commands)
    rows=[];manifest=[]
    for command in commands:
        n,impl=command['size'],command['implementation']
        path=args.profile/f'{n}-{impl}.csv'
        text=path.read_text(encoding='utf-8-sig')
        start=text.index('"ID","Process ID"')
        data=list(csv.DictReader(io.StringIO(text[start:])))
        values=[r for r in data if r['ID'].isdigit()]
        assert len(values)==1, path
        units=data[0]
        row={'n':n,'implementation':impl,'kernel':values[0]['Kernel Name']}
        for label,metric in METRICS.items():
            row[label]=float(values[0][metric].replace(',',''))
            row[label+'_unit']=units[metric]
        rows.append(row)
        manifest.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.mkdir(parents=True)
    with (args.output/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'status':'PASS','profiles':len(rows),'raw_sha256':manifest,
            'limits':['One profiled launch per implementation/size, seed20261020; replay, uncontrolled caches/clocks.',
                      'NCU profile duration is separate from uninstrumented CUDA event comparison.',
                      'NCU shared memory may include driver allocation beyond user dynamic shared storage.',
                      'Counter correlations do not establish a single-factor cause.']}
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','profiles':len(rows)}))

if __name__=='__main__':main()
