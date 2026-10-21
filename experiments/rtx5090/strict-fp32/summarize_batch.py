#!/usr/bin/env python3
"""Independently verify one closed strict-FP32 event CSV and recompute statistics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

GROUPS = {
    'paired': {'k10', 'k10_pad8', 'k30_v1', 'k30_v2', 'strict'},
    'all': {'k10', 'k10_pad8', 'k30_v2', 'strict'},
    'sync': {'k10', 'k30_v1', 'k30_v2', 'strict'},
}


def summarize(path, rounds, repeats, selection):
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
    order_path = Path(str(path) + '.order.csv')
    orders = list(csv.DictReader(order_path.open(encoding='utf-8-sig', newline='')))
    names = GROUPS[selection]
    if len(rows) != len(names)*rounds*repeats:
        raise ValueError(f'Expected {len(names)*rounds*repeats} formal events, found {len(rows)}')
    if len(orders) != len(names)*rounds:
        raise ValueError('Incomplete order table')
    conditions = {(r['n'], r['alpha'], r['beta'], r['seed'], r['a_hash'], r['b_hash'], r['c0_hash']) for r in rows}
    if len(conditions) != 1:
        raise ValueError('Mixed mathematical problems or actual input fingerprints')
    n, alpha, beta, seed, ah, bh, ch = next(iter(conditions))
    n = int(n)
    alpha, beta = float(alpha), float(beta)
    if n < 128 or n > 4096 or n % 128 or alpha not in (.5, 1) or beta not in (0, 1, 3) or int(seed) != 20261020:
        raise ValueError('Unexpected input contract')
    expected_cpu_count = n*n if n == 128 else 256
    order_map = {}
    for r in orders:
        key = (int(r['round']), r['implementation'])
        if key in order_map or r['implementation'] not in names:
            raise ValueError('Duplicate/unknown planned implementation')
        if (int(r['n']), float(r['alpha']), float(r['beta']), int(r['seed'])) != (n, alpha, beta, int(seed)):
            raise ValueError('Order metadata differs from event metadata')
        order_map[key] = int(r['order'])
    for round_number in range(1, rounds+1):
        if {order_map.get((round_number, name)) for name in names} != set(range(1, len(names)+1)):
            raise ValueError('Invalid planned order permutation')
        if selection == 'paired':
            for first, second in [('k10','k10_pad8'),('k30_v1','k30_v2')]:
                if abs(order_map[round_number,first]-order_map[round_number,second]) != 1:
                    raise ValueError('Paired implementations are not adjacent')
    groups, seen = {}, set()
    failure_fields = ('round_failure_count','round_nonfinite_count','reference_nonfinite_count',
                      'cpu_reference_failure_count','cpu_reference_nonfinite_count',
                      'direct_cpu_failure_count','direct_cpu_nonfinite_count')
    for row in rows:
        name, rd, iteration = row['implementation'], int(row['round']), int(row['iteration'])
        key = (name,rd,iteration)
        if key in seen or name not in names or rd not in range(1,rounds+1) or iteration not in range(1,repeats+1):
            raise ValueError('Duplicate, missing, or out-of-range event key')
        seen.add(key)
        if int(row['order']) != order_map[rd,name]:
            raise ValueError('Actual order differs from pre-generated table')
        if row['status'] != 'PASS' or any(int(row[field]) != 0 for field in failure_fields):
            raise ValueError('Numerical failure retained in raw data; batch cannot be labelled complete PASS')
        if int(row['cpu_reference_count']) != expected_cpu_count or int(row['direct_cpu_count']) != expected_cpu_count:
            raise ValueError('Unexpected CPU-double coverage')
        if int(row['round_end_unix_ms']) < int(row['round_start_unix_ms']):
            raise ValueError('Invalid round window')
        value = float(row['event_ms'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid event duration')
        for field in ('round_max_abs_error','round_max_rel_error','cpu_reference_max_abs_error',
                      'cpu_reference_max_rel_error','direct_cpu_max_abs_error','direct_cpu_max_rel_error'):
            error = float(row[field])
            if not math.isfinite(error) or error < 0:
                raise ValueError('Invalid error value')
        for field in ('round_max_abs_error','cpu_reference_max_abs_error','direct_cpu_max_abs_error'):
            if float(row[field]) > .01:
                raise ValueError('Original absolute error threshold exceeded')
        groups.setdefault((name,rd), []).append(row)
    round_rows = []
    for (name,rd), members in sorted(groups.items()):
        if len(members) != repeats:
            raise ValueError('Incomplete round')
        values = [float(r['event_ms']) for r in members]
        mean = statistics.mean(values)
        if any(not math.isclose(float(r['round_mean_ms']),mean,rel_tol=1e-9,abs_tol=1e-9) for r in members):
            raise ValueError('Recorded round mean differs from independent raw-event recomputation')
        for field in ('order','round_start_unix_ms','round_end_unix_ms','round_max_abs_error',
                      'round_max_rel_error','direct_cpu_max_abs_error','direct_cpu_max_rel_error'):
            if len({r[field] for r in members}) != 1:
                raise ValueError('Inconsistent round validation/window metadata')
        round_rows.append({'n':n,'alpha':alpha,'beta':beta,'seed':int(seed),'implementation':name,
                           'round':rd,'order':order_map[rd,name],'event_count':len(values),'mean_ms':mean,
                           'min_event_ms':min(values),'max_event_ms':max(values),
                           'max_abs':float(members[0]['round_max_abs_error']),
                           'direct_cpu_max_abs':float(members[0]['direct_cpu_max_abs_error'])})
    round_ms = {(r['implementation'],r['round']):r['mean_ms'] for r in round_rows}
    summaries = []
    strict_median = statistics.median([round_ms['strict',rd] for rd in range(1,rounds+1)])
    for name in sorted(names):
        means = [round_ms[name,rd] for rd in range(1,rounds+1)]
        median = statistics.median(means)
        summaries.append({'n':n,'alpha':alpha,'beta':beta,'seed':int(seed),'implementation':name,
                          'rounds':rounds,'events_per_round':repeats,'formal_events':rounds*repeats,
                          'median_round_mean_ms':median,'median_tflops':2*n*n*n/(median*1e9),
                          'relative_to_strict_medians':strict_median/median,
                          'max_abs':max(r['max_abs'] for r in round_rows if r['implementation']==name),
                          'direct_cpu_max_abs':max(r['direct_cpu_max_abs'] for r in round_rows if r['implementation']==name)})
    comparisons = [('strict',name) for name in sorted(names) if name!='strict']
    if 'k10_pad8' in names: comparisons.append(('k10','k10_pad8'))
    if 'k30_v1' in names: comparisons.append(('k30_v1','k30_v2'))
    paired, pair_summary = [], []
    for baseline,candidate in comparisons:
        ratios = []
        for rd in range(1,rounds+1):
            ratio = round_ms[baseline,rd]/round_ms[candidate,rd]
            ratios.append(ratio)
            paired.append({'n':n,'alpha':alpha,'beta':beta,'seed':int(seed),'baseline':baseline,
                           'candidate':candidate,'round':rd,'baseline_order':order_map[rd,baseline],
                           'candidate_order':order_map[rd,candidate],
                           'adjacent':abs(order_map[rd,baseline]-order_map[rd,candidate])==1,
                           'throughput_ratio':ratio,'throughput_gain_pct':100*(ratio-1)})
        pair_summary.append({'n':n,'alpha':alpha,'beta':beta,'seed':int(seed),'baseline':baseline,
                             'candidate':candidate,'paired_rounds':rounds,'median_paired_gain_pct':100*(statistics.median(ratios)-1),
                             'faster_rounds':sum(r>1 for r in ratios),'slower_rounds':sum(r<1 for r in ratios)})
    return {'rounds.csv':round_rows,'summary.csv':summaries,'paired.csv':paired,'paired-summary.csv':pair_summary}, {
        'status':'PASS','formal_events':len(rows),'round_checks':len(groups),'selection':selection,
        'n':n,'alpha':alpha,'beta':beta,'seed':int(seed),'rounds':rounds,'events_per_round':repeats,
        'input_fingerprints':{'algorithm':'FNV1a64_float32_bytes','a':ah,'b':bh,'c0':ch},
        'cpu_double_elements_per_check':expected_cpu_count,
        'reference_policy':'full GPU PEDANTIC FP32; CPU double checks actual candidate and GPU reference',
        'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'order_sha256':hashlib.sha256(order_path.read_bytes()).hexdigest(),
        'statistic':'median of per-round arithmetic mean event_ms; paired ratios calculated separately',
        'sample_scope':'round-end output checked; other per-call events are not independently validated',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=7)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--selection',choices=list(GROUPS),default='paired')
    a = p.parse_args()
    tables, audit = summarize(a.csv,a.rounds,a.repeats,a.selection)
    a.out.mkdir(parents=True,exist_ok=True)
    for name in [*tables,'independent-audit.json']:
        if (a.out/name).exists(): raise FileExistsError(a.out/name)
    for name,rows in tables.items():
        with (a.out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.out/'independent-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    main()
