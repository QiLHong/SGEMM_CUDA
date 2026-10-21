"""Independently recompute BN64 formal statistics and audit raw evidence.

This file intentionally imports no functions from the summarizer. It uses fsum
and an explicit middle-of-sorted-values median, and verifies every case/round.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def insist(ok, detail):
    if not ok:
        raise ValueError(detail)


def median(values):
    ordered = sorted(values)
    count = len(ordered)
    return ordered[count//2] if count % 2 else (ordered[count//2-1]+ordered[count//2])/2


def same(actual, expected, label):
    insist(math.isclose(float(actual), expected, rel_tol=2e-12, abs_tol=2e-12), f"Independent recomputation disagrees: {label}: {actual} vs {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical", type=Path,
                        default=Path(__file__).resolve().parents[1]/"results/windows/20261021-tensorcore-bn64/confirm01")
    args = parser.parse_args()
    insist(not args.output.exists(), "Refusing to overwrite verification output")
    names = {"vitamin_tf32_dbf", "tf32_bn64_bk32", "cublas_tf32"}
    cases = {(b,s,n) for b in (1,2,3) for s in (20261020,42,12345) for n in (1024,2048,3072,4096)}
    def ck(row):
        return int(row["batch"]),int(row["seed"]),int(row["n"])
    def rk(row):
        return ck(row)+(row["implementation"],int(row["round"]))
    plan = records(args.run/"plan.csv")
    plan_map = {rk(r):(int(r["order"]),int(r["case_index"])) for r in plan}
    insist(len(plan)==len(plan_map)==756, "Incorrect or duplicate plan")
    event_lists = defaultdict(list)
    check_rows, bit_rows, input_rows, window_rows = [],[],[],[]
    input_files = []
    processes = []
    for batch in (1,2,3):
        process_path=args.run/f"batch-{batch}"/"process.csv"
        process_data=records(process_path)
        insist(len(process_data)==1,"Wrong process record count")
        process=process_data[0]
        insist(int(process["batch"])==batch and process["status"]=="COMPLETE" and int(process["cases"])==12 and int(process["formal_events"])==7560,"Incomplete formal process")
        processes.append(process)
        input_files.append(process_path)
        for seed in (20261020,42,12345):
            for n in (1024,2048,3072,4096):
                path = args.run/f"batch-{batch}"/f"timings-{n}-{seed}.csv"
                data = records(path)
                insist(len(data)==630, f"Wrong event count {path}")
                previous = (-1,-1,-1)
                for r in data:
                    insist(ck(r)==(batch,seed,n), f"Mislabelled event {path}")
                    key=rk(r)
                    insist(key in plan_map and plan_map[key]==(int(r["order"]),int(r["case_index"])), "Event contradicts plan")
                    position=(int(r["round"]),int(r["order"]),int(r["iteration"]))
                    insist(position>previous, "Event stream is not in saved execution order")
                    previous=position
                    insist(int(r["m"])==int(r["k"])==n, "Nonsquare formal event")
                    insist(r["cuda_launch_status"]==r["cuda_event_sync_status"]=="0", "Recorded CUDA error")
                    value=float(r["milliseconds"])
                    insist(math.isfinite(value) and value>0, "Nonpositive/nonfinite event")
                    event_lists[key].append((int(r["iteration"]),value))
                input_files.append(path)
                for suffix,target in (("checks",check_rows),("bitwise",bit_rows),("inputs",input_rows),("windows",window_rows)):
                    p=Path(str(path)+f".{suffix}.csv")
                    rows=records(p)
                    for row in rows:
                        row.setdefault("n",str(n))
                        insist(ck(row)==(batch,seed,n), "Sidecar identity mismatch")
                    target.extend(rows)
                    input_files.append(p)
                cpu_path=Path(str(path)+".cpu-indices.csv")
                cpu_positions=records(cpu_path)
                insist(len(cpu_positions)==256,"Wrong CPU sample size")
                expected_indices=[0,n-1,(n-1)*n,n*n-1]
                i=0
                while len(expected_indices)<256:
                    index=(20261020+i*104729) % (n*n)
                    if index not in expected_indices:
                        expected_indices.append(index)
                    i+=1
                insist([int(x["sample_index"]) for x in cpu_positions]==list(range(256)),"CPU sample indices missing/duplicated")
                insist([int(x["linear_index"]) for x in cpu_positions]==expected_indices,"CPU positions differ from fixed historical rule")
                insist(all(int(x["row"])*n+int(x["col"])==int(x["linear_index"]) for x in cpu_positions),"CPU sample coordinate mismatch")
                input_files.append(cpu_path)
                meta_path=Path(str(path)+".meta.txt")
                meta=dict(line.split("=",1) for line in meta_path.read_text(encoding="utf-8-sig").splitlines() if "=" in line)
                insist(meta["pid"]==process["pid"] and meta["shared_tf32_input_pointers"]=="true","Process/input-pointer contract changed")
                input_files.append(meta_path)
    insist(len({x["pid"] for x in processes})==3,"Expected distinct process PIDs")
    insist(all(int(x["start_utc_ms"])<=int(x["end_utc_ms"]) for x in processes),"Bad process time")
    insist(all(int(processes[i]["end_utc_ms"])<=int(processes[i+1]["start_utc_ms"]) for i in range(2)),"GPU formal processes overlapped")
    expected_rounds={case+(name,r) for case in cases for name in names for r in range(7)}
    insist(set(event_lists)==expected_rounds, "Missing formal round")
    computed={}
    for key,items in event_lists.items():
        insist([x[0] for x in items]==list(range(30)), "Duplicate/missing iterations")
        computed[key]=math.fsum(x[1] for x in items)/30
    observed={rk(r):r for r in records(args.analysis/"rounds.csv")}
    insist(set(observed)==expected_rounds,"Round summary coverage differs")
    for key,value in computed.items():
        same(observed[key]["mean_ms"],value,f"round {key}")
        same(observed[key]["tflops"],2*key[2]**3/(value*1e9),f"round TFLOPS {key}")
        same(observed[key]["event_min_ms"],min(x[1] for x in event_lists[key]),"event minimum")
        same(observed[key]["event_max_ms"],max(x[1] for x in event_lists[key]),"event maximum")
    summaries=records(args.analysis/"summary.csv")
    summary_map={ck(r)+(r["implementation"],):r for r in summaries}
    insist(len(summary_map)==len(summaries)==108,"Wrong implementation summary count")
    aggregate={}
    for case in cases:
        for name in names:
            key=case+(name,)
            values=[computed[key+(r,)] for r in range(7)]
            med=median(values);mean=math.fsum(values)/7
            sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/6)
            aggregate[key]=med
            row=summary_map[key]
            same(row["median_round_mean_ms"],med,f"median {key}")
            same(row["tflops"],2*case[2]**3/(med*1e9),f"TFLOPS {key}")
            same(row["round_cv_pct"],100*sd/mean,f"CV {key}")
            same(row["round_min_ms"],min(values),"min round mean")
            same(row["round_max_ms"],max(values),"max round mean")
    pair_rows=records(args.analysis/"paired-rounds.csv")
    pair_map={ck(r)+(r["reference"],int(r["round"])):r for r in pair_rows}
    paired_rows=records(args.analysis/"paired.csv")
    paired_map={ck(r)+(r["reference"],):r for r in paired_rows}
    insist(len(pair_map)==len(pair_rows)==504 and len(paired_map)==len(paired_rows)==72,"Incorrect paired summary count")
    independent_case_gains={}
    independent_round_gains={}
    for case in cases:
        for ref in ("vitamin_tf32_dbf","cublas_tf32"):
            gains=[]
            for r in range(7):
                a=computed[case+(ref,r)]
                b=computed[case+("tf32_bn64_bk32",r)]
                g=(a-b)/b*100
                gains.append(g)
                independent_round_gains[case+(ref,r)]=g
                same(pair_map[case+(ref,r)]["gain_pct"],g,"paired round gain")
            row=paired_map[case+(ref,)]
            same(row["median_paired_gain_pct"],median(gains),"median paired gain")
            same(row["min_gain_pct"],min(gains),"min paired gain")
            same(row["max_gain_pct"],max(gains),"max paired gain")
            same(row["ratio_of_median_throughputs_gain_pct"],(aggregate[case+(ref,)]/aggregate[case+("tf32_bn64_bk32",)]-1)*100,"ratio of medians")
            insist(int(row["positive_rounds"])==sum(g>0 for g in gains),"Wrong positive rounds")
            insist(int(row["negative_rounds"])==sum(g<0 for g in gains),"Wrong negative rounds")
            independent_case_gains[case+(ref,)]=median(gains)
    case_table=records(args.analysis/"cases.csv")
    insist(len(case_table)==36 and {ck(r) for r in case_table}==cases,"Incorrect 36-case table")
    for row in case_table:
        case=ck(row)
        for name in names:
            same(row[name+"_tflops"],2*case[2]**3/(aggregate[case+(name,)]*1e9),"case TFLOPS")
        for ref,tag in (("vitamin_tf32_dbf","author"),("cublas_tf32","cublas")):
            same(row[f"vs_{tag}_median_paired_gain_pct"],independent_case_gains[case+(ref,)],"case paired median")
    size_rows=records(args.analysis/"size-summary.csv")
    insist(len(size_rows)==8,"Incorrect size summary count")
    for row in size_rows:
        n,ref=int(row["n"]),row["reference"]
        values=[independent_case_gains[(b,s,n,ref)] for b in (1,2,3) for s in (20261020,42,12345)]
        round_values=[independent_round_gains[(b,s,n,ref,r)] for b in (1,2,3) for s in (20261020,42,12345) for r in range(7)]
        same(row["median_of_case_paired_medians_pct"],median(values),"size median of case medians")
        same(row["min_case_paired_median_pct"],min(values),"size min case median")
        same(row["max_case_paired_median_pct"],max(values),"size max case median")
        insist(int(row["positive_case_medians"])==sum(x>0 for x in values),"size positive cases")
        insist(int(row["negative_case_medians"])==sum(x<0 for x in values),"size negative cases")
        insist(int(row["positive_rounds"])==sum(x>0 for x in round_values),"size positive rounds")
        insist(int(row["negative_rounds"])==sum(x<0 for x in round_values),"size negative rounds")

    refs={"original_fp32","represented_input_strict_reference","same_precision_cublas","represented_input_cpu_double","vitamin_tf32_baseline"}
    checks=Counter(rk(r)+(r["reference"],) for r in check_rows)
    insist(set(checks)=={case+(name,r,ref) for case in cases for name in names for r in range(-1,7) for ref in refs},"Incomplete correctness checks")
    insist(all(v==1 for v in checks.values()),"Duplicate correctness check")
    for row in check_rows:
        expected_count=256 if row["reference"]=="represented_input_cpu_double" else int(row["n"])**2
        insist(int(row["count"])==expected_count,"Incorrect full/sample scope")
        insist(float(row["native_eps"])==.1,"Changed native threshold")
        insist(int(row["nonfinite"])==0,"Nonfinite detected")
        if row["reference"]!="original_fp32":
            insist(int(row["native_allclose_fail"])==0,"TF32 native gate failed")
    error_groups=defaultdict(list)
    for row in check_rows:
        error_groups[ck(row)+(row["implementation"],row["reference"])].append(row)
    error_rows=records(args.analysis/"errors.csv")
    insist(len(error_rows)==len(error_groups)==540,"Incorrect error summary count")
    for row in error_rows:
        group=error_groups[ck(row)+(row["implementation"],row["reference"])]
        for field in ("max_abs","max_rel","relative_l2","nonfinite","abs_001_fail","simt_allclose_fail","native_allclose_fail"):
            same(row["worst_"+field],max(float(x[field]) for x in group),"error maximum")
        for field in ("abs_001_fail","simt_allclose_fail","native_allclose_fail"):
            insist(int(row["failed_checks_"+field])==sum(int(x[field])>0 for x in group),"error failed-check count")
    bits=Counter(rk(r) for r in bit_rows)
    insist(set(bits)=={case+(name,r) for case in cases for name in ("vitamin_tf32_dbf","tf32_bn64_bk32") for r in range(-1,7)},"Incomplete bitwise checks")
    insist(all(v==1 for v in bits.values()) and all(int(r["bitwise_mismatches"])==0 for r in bit_rows),"Duplicate or failed bitwise check")
    windows=Counter(rk(r) for r in window_rows)
    insist(set(windows)==expected_rounds and all(v==1 for v in windows.values()),"Incomplete round windows")
    for batch in (1,2,3):
        ordered=sorted((r for r in window_rows if int(r["batch"])==batch),key=lambda r:(int(r["case_index"]),int(r["round"]),plan_map[rk(r)][0]))
        previous=int(processes[batch-1]["start_utc_ms"])
        for row in ordered:
            insist(previous<=int(row["start_utc_ms"])<=int(row["end_utc_ms"]),"Recorded execution windows contradict order")
            previous=int(row["end_utc_ms"])
        insist(previous<=int(processes[batch-1]["end_utc_ms"]),"Window exceeds process interval")
    inputs={}
    for row in input_rows:
        key=ck(row)+(row["representation"],row["matrix"],row["stage"])
        insist(key not in inputs,"Duplicate input fingerprint")
        inputs[key]=row["fnv1a64"]
    insist(len(inputs)==288,"Wrong input fingerprint count")
    history_comparisons=[]
    for n in (1024,2048,3072,4096):
        historical={}
        if n!=3072:
            hp=args.historical/f"timings-{n}.csv.meta.txt"
            historical=dict(line.split("=",1) for line in hp.read_text(encoding="utf-8-sig").splitlines() if "=" in line)
            input_files.append(hp)
        for seed in (20261020,42,12345):
            for p,index in (("original_fp32",0),("tf32",1)):
                for matrix in ("A","B"):
                    hashes={inputs[(batch,seed,n,p,matrix,stage)] for batch in (1,2,3) for stage in ("before","after")}
                    insist(len(hashes)==1,"Input differs between batches or before/after")
                    if historical and seed==20261020:
                        old=historical[f"input_{index}_{matrix.lower()}_fnv1a64"]
                        new=next(iter(hashes))
                        insist(old==new,"Historical generator/distribution fingerprint differs")
                        history_comparisons.append(dict(n=n,representation=p,matrix=matrix,fnv1a64=new,match=True))
        for p in ("original_fp32","tf32"):
            for matrix in ("A","B"):
                insist(len({inputs[(1,s,n,p,matrix,"before")] for s in (20261020,42,12345)})==3,"Seeds produced duplicate inputs")
    # Six balanced permutations must be visible in the raw event order.
    for case in cases:
        perms=[]
        for r in range(7):
            names_by_order=sorted((plan_map[case+(name,r)][0],name) for name in names)
            insist([x[0] for x in names_by_order]==[0,1,2],"Invalid order positions")
            perms.append(tuple(x[1] for x in names_by_order))
        insist(len(set(perms[:6]))==6,"Six permutations not exhausted")
    for seed in (20261020,42,12345):
        for n in (1024,2048,3072,4096):
            extra=[tuple(name for _,name in sorted((plan_map[(b,seed,n,name,6)][0],name) for name in names)) for b in (1,2,3)]
            insist(len(set(extra))==3,"Extra permutation did not rotate")
    for entry in json.loads((args.analysis/"raw-files-read-sha256.json").read_text(encoding="utf-8")):
        path=args.run/entry["path"]
        insist(hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"],"Raw file changed after summary")
    original_fail=sum(int(x["abs_001_fail"])>0 for x in check_rows if x["reference"]=="original_fp32")
    audit=dict(status="PASS",events=sum(len(x) for x in event_lists.values()),rounds=len(computed),cases=len(cases),
               implementation_summaries=len(summary_map),paired_rounds=len(pair_map),paired_summaries=len(paired_map),
               full_and_sample_check_records=len(check_rows),baseline_bitwise_checks=len(bit_rows),
               input_fingerprint_records=len(inputs),historical_fingerprint_comparisons=history_comparisons,
               process_ids=[int(x["pid"]) for x in processes],cpu_sample_position_files=36,
               original_fp32_abs_001_failed_checks=original_fail,
               method="Independent raw CSV parse, math.fsum round means, explicit sorted median; no summarizer imports",
               uncertainty="descriptive only; no event-level IID assumptions or confidence intervals",
               excluded="smoke, shape rejection, sanitizer and profiler data excluded from 22680 formal events")
    args.output.mkdir(parents=True)
    (args.output/"verification.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    manifest=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(input_files))]
    (args.output/"independently-read-files.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":
    main()
