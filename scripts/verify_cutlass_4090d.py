"""Independently recompute the CUTLASS comparison from raw CSV files.

This script does not import the summarizer. It uses math.fsum and an explicit
sorted median, checks every output row in the performance/error summaries,
checks the formal plan and actual input fingerprints, and retains precision
failures against original FP32 data. No GPU is used here.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime


def insist(ok, message):
    if not ok:
        raise ValueError(message)


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def median(values):
    values = sorted(values)
    n = len(values)
    return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2


def same(a, b, label):
    insist(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10), f"Disagrees: {label}: {a} vs {b}")


def case(row):
    return row["group"],int(row["batch"]),int(row["seed"]),int(row["n"])


def key(row):
    return case(row)+(row["implementation"],int(row["round"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    insist(not args.output.exists(),"Refusing existing verification output")
    insist((args.run/"RUN_COMPLETE.txt").exists(),"Incomplete run")
    env = json.loads((args.run/"environment.json").read_text(encoding="utf-8-sig"))
    groups = env["groups"]
    batches, seeds, sizes = list(range(1,int(env["batches"])+1)),env["seeds"],env["sizes"]
    insist((env["rounds"],env["repeats"],env["warmup"],env["alpha"],env["beta"]) == (7,30,10,1,0),"Measurement contract changed")
    plan_map, cases = {}, set()
    read_files = [args.run/"environment.json"]
    for group,names in groups.items():
        path = args.run/f"plan-{group}.csv"
        insist(hashlib.sha256(path.read_bytes()).hexdigest().lower() == env["plans"][group]["sha256"].lower(),"Plan hash mismatch")
        for row in records(path):
            row["group"] = group
            k = key(row)
            insist(k not in plan_map,"Duplicate plan key")
            insist(row["implementation"] in names and int(row["m"])==int(row["n"])==int(row["k"]),"Invalid planned shape/implementation")
            plan_map[k] = (int(row["case_index"]),int(row["order"]))
            cases.add(case(row))
        read_files.append(path)
    wanted_cases = {(g,b,s,n) for g in groups for b in batches for s in seeds for n in sizes}
    wanted_keys = {c+(name,r) for c in wanted_cases for name in groups[c[0]] for r in range(7)}
    insist(cases == wanted_cases and set(plan_map)==wanted_keys,"Plan does not cover all formal cases/rounds")
    event_map = defaultdict(list)
    all_checks, all_bits, all_inputs, all_windows, processes = [],[],[],[],[]
    for group,names in groups.items():
        for b in batches:
            folder = args.run/f"{group}-batch-{b}"
            insist((folder/"BATCH_COMPLETE.txt").exists(),"Missing process completion marker")
            rows = records(folder/"process.csv")
            insist(len(rows)==1,"Bad process count")
            process = rows[0]
            insist(process["group"]==group and int(process["batch"])==b and process["status"]=="COMPLETE","Bad process record")
            insist(int(process["cases"])==len(seeds)*len(sizes) and int(process["formal_events"])==len(seeds)*len(sizes)*len(names)*210,"Bad process case/event count")
            processes.append(process)
            read_files.append(folder/"process.csv")
            for s in seeds:
                for n in sizes:
                    p = folder/f"timings-{n}-{s}.csv"
                    data = records(p)
                    insist(len(data)==len(names)*210,"Wrong event count per file")
                    last = (-1,-1,-1)
                    for row in data:
                        k = key(row)
                        insist(case(row)==(group,b,s,n) and k in plan_map,"Wrong event identity")
                        insist(plan_map[k]==(int(row["case_index"]),int(row["order"])),"Execution differs from plan")
                        position = tuple(int(row[t]) for t in ("round","order","iteration"))
                        insist(position>last,"Events not recorded in execution order")
                        last = position
                        insist(int(row["m"])==int(row["k"])==n,"Nonsquare event")
                        insist(row["cuda_launch_status"]==row["cuda_event_sync_status"]=="0","CUDA failure recorded")
                        ms = float(row["milliseconds"])
                        insist(math.isfinite(ms) and ms>0,"Invalid timing sample")
                        same(row["tflops"],2*n**3/(ms*1e9),"raw event throughput")
                        event_map[k].append((int(row["iteration"]),ms))
                    read_files.append(p)
                    for suffix,target in (("checks",all_checks),("bitwise",all_bits),("inputs",all_inputs),("windows",all_windows)):
                        side = Path(str(p)+f".{suffix}.csv")
                        for row in records(side):
                            row.setdefault("n",str(n))
                            insist(case(row)==(group,b,s,n),"Wrong sidecar identity")
                            target.append(row)
                        read_files.append(side)
                    cpufile = Path(str(p)+".cpu-indices.csv")
                    positions = records(cpufile)
                    indices = [0,n-1,(n-1)*n,n*n-1]
                    counter = 0
                    while len(indices)<256:
                        i=(20261020+counter*104729)%(n*n)
                        if i not in indices: indices.append(i)
                        counter+=1
                    insist(len(positions)==256 and [int(x["linear_index"]) for x in positions]==indices,"CPU sample rule differs")
                    insist([int(x["sample_index"]) for x in positions]==list(range(256)),"CPU indices not unique/ordered")
                    insist(all(int(x["row"])*n+int(x["col"])==int(x["linear_index"]) for x in positions),"CPU coordinates differ")
                    read_files.append(cpufile)
                    mp = Path(str(p)+".meta.txt")
                    meta = dict(line.split("=",1) for line in mp.read_text(encoding="utf-8-sig").splitlines() if "=" in line)
                    insist(meta["group"]==group and meta["pid"]==process["pid"] and meta["CASE_COMPLETE"]=="true","Incomplete case/process association")
                    insist(meta["shared_input_pointers_within_group"]=="true" and meta["alpha"]=="1" and meta["beta"]=="0","Input/C=AB contract changed")
                    insist(tuple(int(meta[x]) for x in ("rounds","repeats","warmup"))==(7,30,10),"Case timing counts changed")
                    read_files.append(mp)
    insist(set(event_map)==wanted_keys,"Incomplete actual rounds")
    computed = {}
    for k,values in event_map.items():
        insist([i for i,_ in values]==list(range(30)),"Event samples missing/duplicated")
        computed[k]=math.fsum(v for _,v in values)/30
    insist(sum(len(v) for v in event_map.values())==int(env["expected_formal_events"]),"Event total differs from environment")
    process_order=sorted(processes,key=lambda x:int(x["start_utc_ms"]))
    insist(len({x["pid"] for x in processes})==len(processes),"Repeated process PID")
    insist(all(int(x["start_utc_ms"])<=int(x["end_utc_ms"]) for x in processes),"Reversed process interval")
    insist(all(int(a["end_utc_ms"])<=int(b["start_utc_ms"]) for a,b in zip(process_order,process_order[1:])),"Overlapping formal processes")

    # Recovery must preserve complete original processes, not cherry-pick events.
    recovery_audit = None
    recovery_path = args.run/"recovery.json"
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text(encoding="utf-8-sig"))
        source = args.run.parent / recovery["resume_from"].replace("\\", "/").rstrip("/").split("/")[-1]
        insist(source.is_dir(), "Recovery source archive missing")
        source_env = json.loads((source/"environment.json").read_text(encoding="utf-8-sig"))
        insist(recovery["binary_sha256"] == source_env["binary_sha256"] == env["binary_sha256"], "Recovery binary changed")
        for name in ("environment.json", "plan-tf32.csv", "plan-fp32.csv", "verified-build.json", "shape-rejections.csv"):
            insist((source/name).read_bytes() == (args.run/name).read_bytes(), "Recovery protocol changed: " + name)
        reused = records(args.run/"reused-complete-batch-sha256.csv")
        paths = [r["path"].replace("\\", "/") for r in reused]
        insist(len(paths) == len(set(paths)), "Duplicate recovery manifest entry")
        labels = {p.split("/")[0] for p in paths}
        insist(len(labels) == int(recovery["reused_complete_batches"]), "Recovery batch count differs")
        for row, relative in zip(reused, paths):
            for root in (source, args.run):
                p = root/relative
                insist(p.is_file() and p.stat().st_size == int(row["bytes"]), "Recovery file missing/size changed")
                insist(hashlib.sha256(p.read_bytes()).hexdigest().lower() == row["sha256"].lower(), "Recovery bytes changed")
        old_execution = json.loads((source/"execution.json").read_text(encoding="utf-8-sig"))
        execution = json.loads((args.run/"execution.json").read_text(encoding="utf-8-sig"))
        insist(len(execution) == 6 and all(x["exit_code"] == 0 for x in execution), "Final process exit evidence differs")
        cutoff = datetime.fromisoformat(recovery["completed_before_utc"].replace("Z", "+00:00")) if recovery.get("completed_before_utc") else None
        for label in labels:
            expected_paths = {p.relative_to(source).as_posix() for p in (source/label).rglob("*") if p.is_file()}
            insist(expected_paths == {p for p in paths if p.startswith(label + "/")}, "Partial recovered process")
            g, _, b = label.split("-")
            prior = [x for x in old_execution if x["group"] == g and int(x["batch"]) == int(b)]
            current = [x for x in execution if x["group"] == g and int(x["batch"]) == int(b)]
            insist(len(prior) == len(current) == 1 and prior[0]["exit_code"] == 0, "Recovered execution missing")
            # PowerShell JSON parsing converts timestamp strings to local offsets;
            # compare instants, while requiring every other field to remain exact.
            for field in set(prior[0]) | set(current[0]):
                a, b = prior[0][field], current[0][field]
                if field in ("utc_start", "utc_end"):
                    a, b = datetime.fromisoformat(a), datetime.fromisoformat(b)
                insist(a == b, "Recovered execution changed: " + field)
            if cutoff:
                insist(datetime.fromisoformat(prior[0]["utc_end"]) <= cutoff, "Recovered process ended after cutoff")
            insist((source/label/"BATCH_COMPLETE.txt").is_file(), "Recovery source not complete")
            for name in (label+".log", "telemetry-"+label+".csv", "telemetry-"+label+".stderr.txt"):
                insist((source/name).read_bytes() == (args.run/name).read_bytes(), "Recovered telemetry/log changed")
        for p in (source/"smoke").rglob("*"):
            if p.is_file():
                insist(p.read_bytes() == (args.run/p.relative_to(source)).read_bytes(), "Recovered smoke changed")
        recovery_audit = dict(source=source.name, complete_batches=sorted(labels), manifest_files=len(paths),
                              cutoff=recovery.get("completed_before_utc"), exact_bytes=True)
        read_files.extend([recovery_path,args.run/"reused-complete-batch-sha256.csv",source/"execution.json",args.run/"execution.json"])

    observed_rows = records(args.analysis/"rounds.csv")
    observed = {key(r):r for r in observed_rows}
    insist(len(observed_rows)==len(observed) and set(observed)==wanted_keys,"Round summary coverage differs")
    for k,mean in computed.items():
        row=observed[k]
        values=[v for _,v in event_map[k]]
        same(row["mean_ms"],mean,"round mean")
        same(row["tflops"],2*k[3]**3/(mean*1e9),"round TFLOPS")
        same(row["event_min_ms"],min(values),"event min")
        same(row["event_max_ms"],max(values),"event max")
        same(row["event_median_ms"],median(values),"event median")
        insist(int(row["events"])==30,"Wrong event-count summary")
    summary_rows=records(args.analysis/"summary.csv")
    summary={case(r)+(r["implementation"],):r for r in summary_rows}
    expected_summary={c+(name,) for c in cases for name in groups[c[0]]}
    insist(len(summary)==len(summary_rows) and set(summary)==expected_summary,"Implementation summary coverage differs")
    medians={}
    for k in expected_summary:
        values=[computed[k+(r,)] for r in range(7)]
        m=math.fsum(values)/7
        sd=math.sqrt(math.fsum((v-m)**2 for v in values)/6)
        medians[k]=median(values)
        row=summary[k]
        for field,value in (("median_round_mean_ms",median(values)),("tflops",2*k[3]**3/(median(values)*1e9)),
                            ("round_min_ms",min(values)),("round_max_ms",max(values)),("round_mean_ms",m),
                            ("round_sd_ms",sd),("round_cv_pct",100*sd/m)):
            same(row[field],value,"implementation "+field)

    pr=records(args.analysis/"paired-rounds.csv")
    ps=records(args.analysis/"paired.csv")
    pair_map={case(r)+(r["candidate"],r["reference"],int(r["round"])):r for r in pr}
    paired_map={case(r)+(r["candidate"],r["reference"]):r for r in ps}
    expected_pairs={c+(a,b) for c in cases for a in groups[c[0]] for b in groups[c[0]] if a!=b}
    insist(set(paired_map)==expected_pairs and len(paired_map)==len(ps),"Paired summary coverage differs")
    insist(set(pair_map)=={k+(r,) for k in expected_pairs for r in range(7)} and len(pair_map)==len(pr),"Paired round coverage differs")
    gains_by_case={}
    for k in expected_pairs:
        c,a,b=k[:4],k[4],k[5]
        gains=[]
        for r in range(7):
            av,bv=computed[c+(a,r)],computed[c+(b,r)]
            gain=(bv-av)/av*100
            row=pair_map[k+(r,)]
            same(row["gain_pct"],gain,"paired round gain")
            same(row["candidate_ms"],av,"paired candidate")
            same(row["reference_ms"],bv,"paired reference")
            gains.append(gain)
        gains_by_case[k]=gains
        row=paired_map[k]
        same(row["median_paired_gain_pct"],median(gains),"paired median")
        same(row["min_gain_pct"],min(gains),"paired minimum")
        same(row["max_gain_pct"],max(gains),"paired maximum")
        same(row["ratio_of_median_throughputs_gain_pct"],(medians[c+(b,)]-medians[c+(a,)])/medians[c+(a,)]*100,"ratio of medians")
        for field,count in (("positive_rounds",sum(x>0 for x in gains)),("negative_rounds",sum(x<0 for x in gains)),("tied_rounds",sum(x==0 for x in gains)),("rounds",7)):
            insist(int(row[field])==count,"paired sign/count differs")
    size_rows=records(args.analysis/"size-summary.csv")
    expected_sizes={(g,n,a,b) for g in groups for n in sizes for a in groups[g] for b in groups[g] if a!=b}
    insist({(r["group"],int(r["n"]),r["candidate"],r["reference"]) for r in size_rows}==expected_sizes and len(size_rows)==len(expected_sizes),"Size-pair coverage differs")
    for row in size_rows:
        g,n,a,b=row["group"],int(row["n"]),row["candidate"],row["reference"]
        v=[median(gains_by_case[(g,batch,s,n,a,b)]) for batch in batches for s in seeds]
        rv=[x for batch in batches for s in seeds for x in gains_by_case[(g,batch,s,n,a,b)]]
        same(row["median_of_case_paired_medians_pct"],median(v),"size paired median")
        same(row["min_case_paired_median_pct"],min(v),"size paired min")
        same(row["max_case_paired_median_pct"],max(v),"size paired max")
        for field,count in (("positive_case_medians",sum(x>0 for x in v)),("negative_case_medians",sum(x<0 for x in v)),
                            ("positive_rounds",sum(x>0 for x in rv)),("negative_rounds",sum(x<0 for x in rv)),("total_rounds",len(rv))):
            insist(int(row[field])==count,"size sign/count differs")
    ranges=records(args.analysis/"throughput-ranges.csv")
    insist(len(ranges)==len(sizes)*sum(len(x) for x in groups.values()),"Wrong throughput-range count")
    insist({(x["group"],int(x["n"]),x["implementation"]) for x in ranges}=={(g,n,a) for g in groups for n in sizes for a in groups[g]},"Throughput-range coverage differs")
    for row in ranges:
        g,n,a=row["group"],int(row["n"]),row["implementation"]
        values=[2*n**3/(medians[(g,b,s,n,a)]*1e9) for b in batches for s in seeds]
        same(row["median_case_tflops"],median(values),"case throughput median")
        same(row["min_case_tflops"],min(values),"case throughput min")
        same(row["max_case_tflops"],max(values),"case throughput max")
    case_rows=records(args.analysis/"cases.csv")
    insist(len(case_rows)==len(cases) and {case(r) for r in case_rows}==cases,"Case table differs")
    for row in case_rows:
        c=case(row)
        for name in groups[c[0]]:
            same(row[name+"_tflops"],2*c[3]**3/(medians[c+(name,)]*1e9),"case table throughput")

    refs={g:{"original_fp32","represented_input_strict_reference","same_precision_cublas","represented_input_cpu_double",
             "vitamin_tf32_baseline" if g=="tf32" else "k30_fp32_baseline"} for g in groups}
    check_counts=Counter(key(r)+(r["reference"],) for r in all_checks)
    expected_checks={c+(name,r,ref) for c in cases for name in groups[c[0]] for r in range(-1,7) for ref in refs[c[0]]}
    insist(set(check_counts)==expected_checks and all(v==1 for v in check_counts.values()),"Correctness coverage differs")
    error_groups=defaultdict(list)
    for row in all_checks:
        g=row["group"]
        insist(int(row["count"])==(256 if row["reference"]=="represented_input_cpu_double" else int(row["n"])**2),"Incorrect correctness scope")
        insist(float(row["native_eps"])==(.1 if g=="tf32" else .001),"Native threshold changed")
        insist(int(row["nonfinite"])==0,"Nonfinite detected")
        if g=="fp32":
            insist(int(row["abs_001_fail"])==0,"Strict FP32 absolute-error gate failure")
        elif row["reference"]!="original_fp32":
            insist(int(row["native_allclose_fail"])==0,"Native TF32 gate failure")
        error_groups[case(row)+(row["implementation"],row["reference"])].append(row)
    error_rows=records(args.analysis/"errors.csv")
    insist(len(error_rows)==len(error_groups),"Wrong error summary count")
    insist({case(x)+(x["implementation"],x["reference"]) for x in error_rows}==set(error_groups),"Error summary coverage differs")
    for row in error_rows:
        values=error_groups[case(row)+(row["implementation"],row["reference"])]
        insist(int(row["check_records"])==len(values) and int(row["points_per_check"])==int(values[0]["count"]),"Error record scope differs")
        for field in ("max_abs","max_rel","relative_l2","nonfinite","abs_001_fail","simt_allclose_fail","native_allclose_fail"):
            same(row["worst_"+field],max(float(v[field]) for v in values),"error maximum")
        for field in ("abs_001_fail","simt_allclose_fail","native_allclose_fail"):
            insist(int(row["failed_checks_"+field])==sum(int(v[field])>0 for v in values),"Error failure count differs")
    quality_tables = {}
    for filename in ("correctness-overview.csv","correctness-by-reference.csv"):
        data=records(args.analysis/filename)
        quality_tables[filename]=data
        expected={(g,n,name,"applicable full GPU references plus CPU-double samples") for g in groups for n in sizes for name in groups[g]} if filename=="correctness-overview.csv" else {(g,n,name,ref) for g in groups for n in sizes for name in groups[g] for ref in refs[g]}
        insist(len(data)==len(expected) and {(x["group"],int(x["n"]),x["implementation"],x["reference"]) for x in data}==expected,"Correctness table coverage differs")
        for row in data:
            selected=[x for x in all_checks if x["group"]==row["group"] and x["n"]==row["n"] and x["implementation"]==row["implementation"]]
            if filename=="correctness-overview.csv":
                selected=[x for x in selected if x["group"]=="fp32" or x["reference"]!="original_fp32"]
            else:
                selected=[x for x in selected if x["reference"]==row["reference"]]
            insist(int(row["check_records"])==len(selected),"Correctness table check count differs")
            same(row["max_abs"],max(float(x["max_abs"]) for x in selected),"Correctness table max abs")
            same(row["max_rel"],max(float(x["max_rel"]) for x in selected),"Correctness table max rel")
            for source,label in (("abs_001_fail","abs_001"),("simt_allclose_fail","simt_allclose"),("native_allclose_fail","native_allclose"),("nonfinite","nonfinite")):
                failures=sum(int(x[source])>0 for x in selected)
                insist(row[label+"_status"]==("FAIL" if failures else "PASS"),"Correctness PASS/FAIL changed")
                insist(int(row[label+"_failed_checks"])==failures,"Correctness failed-check count differs")
                insist(int(row[label+"_failure_observations"])==sum(int(x[source]) for x in selected),"Correctness failure observations differ")
    bits=Counter(key(r) for r in all_bits)
    expected_bits={c+(name,r) for c in cases for name in groups[c[0]] for r in range(-1,7)}
    insist(set(bits)==expected_bits and all(v==1 for v in bits.values()),"Bitwise coverage differs")
    for row in all_bits:
        required=row["group"]=="tf32" and row["implementation"] in ("vitamin_tf32_dbf","tf32_bn64_bk32")
        insist(int(row["required"])==int(required) and int(row["count"])==int(row["n"])**2,"Bitwise gate/scope changed")
        if required: insist(int(row["bitwise_mismatches"])==0,"Required bitwise equality failed")
    input_map={}
    for row in all_inputs:
        k=case(row)+(row["representation"],row["matrix"],row["stage"])
        insist(k not in input_map,"Duplicate input hash")
        input_map[k]=row["fnv1a64"]
        insist(int(row["element_count"])==int(row["n"])**2 and int(row["bytes"])==int(row["n"])**2*4,"Wrong input size")
    insist(len(input_map)==len(cases)*8,"Incomplete input fingerprints")
    for s in seeds:
        for n in sizes:
            for representation in ("original_fp32","tf32"):
                for matrix in ("A","B"):
                    hashes={input_map[(g,b,s,n,representation,matrix,stage)] for g in groups for b in batches for stage in ("before","after")}
                    insist(len(hashes)==1,"Actual input bytes differ across groups/processes/before-after")
    for n in sizes:
        for representation in ("original_fp32","tf32"):
            for matrix in ("A","B"):
                insist(len({input_map[("tf32",1,s,n,representation,matrix,"before")] for s in seeds})==len(seeds),"Seeds did not change inputs")
    window_counts=Counter(key(r) for r in all_windows)
    insist(set(window_counts)==wanted_keys and all(v==1 for v in window_counts.values()),"Window coverage differs")
    for process in processes:
        rows=sorted((r for r in all_windows if r["group"]==process["group"] and r["batch"]==process["batch"]),key=lambda r:(int(r["case_index"]),int(r["round"]),plan_map[key(r)][1]))
        end=int(process["start_utc_ms"])
        for row in rows:
            insist(end<=int(row["start_utc_ms"])<=int(row["end_utc_ms"]),"Window order differs")
            end=int(row["end_utc_ms"])
        insist(end<=int(process["end_utc_ms"]),"Window exceeds process end")
    for c in cases:
        permutations=[]
        for r in range(7):
            ordered=sorted((plan_map[c+(name,r)][1],name) for name in groups[c[0]])
            insist([i for i,_ in ordered]==list(range(len(groups[c[0]]))),"Bad order positions")
            permutations.append(tuple(name for _,name in ordered))
        insist(len(set(permutations))==7,"Seven distinct permutations required")

    # Small rectangular CPU checks cover every output coordinate. They remain
    # separate from the square performance table and the 256-point samples.
    smoke = args.run/"smoke"
    insist((smoke/"SMOKE_COMPLETE.txt").exists(),"Small-shape validation incomplete")
    smoke_checks, smoke_coordinates = 0, 0
    for group,names in groups.items():
        for k in (32,64,96):
            base=smoke/f"{group}-small-{k}.csv"
            checks=records(Path(str(base)+".checks.csv"))
            keys={(x["implementation"],int(x["round"]),x["reference"]) for x in checks}
            wanted={(name,r,ref) for name in names for r in (-1,0) for ref in refs[group]}
            insist(keys==wanted and len(checks)==len(wanted),"Smoke correctness coverage differs")
            for x in checks:
                insist(x["group"]==group and int(x["m"])==128 and int(x["n"])==1024 and int(x["k"])==k,"Smoke shape differs")
                insist(int(x["count"])==131072 and int(x["nonfinite"])==0,"Smoke does not check full output or has nonfinite")
                insist(float(x["native_eps"])==(.1 if group=="tf32" else .001),"Smoke threshold changed")
                if group=="fp32": insist(int(x["abs_001_fail"])==0,"Smoke FP32 absolute gate failure")
                elif x["reference"]!="original_fp32": insist(int(x["native_allclose_fail"])==0,"Smoke TF32 native gate failure")
            smoke_checks+=len(checks)
            cp=Path(str(base)+".cpu-indices.csv")
            count=0
            with cp.open(encoding="utf-8-sig",newline="") as f:
                for i,row in enumerate(csv.DictReader(f)):
                    insist(int(row["sample_index"])==int(row["linear_index"])==i,"Smoke CPU output missing/duplicated")
                    insist(int(row["row"])==i//1024 and int(row["col"])==i%1024,"Smoke CPU coordinate differs")
                    count+=1
            insist(count==131072,"Smoke CPU check is not full output")
            smoke_coordinates+=count
            read_files.extend([Path(str(base)+".checks.csv"),cp])

    manifest=json.loads((args.analysis/"raw-files-read-sha256.json").read_text(encoding="utf-8"))
    for entry in manifest:
        path=args.run/entry["path"]
        insist(path.stat().st_size==entry["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"],"Raw data changed after summarizing")
    audit=dict(status="DATA_AUDIT_PASS",events=sum(len(v) for v in event_map.values()),rounds=len(computed),cases=len(cases),
               implementation_summaries=len(summary),paired_rounds=len(pair_map),paired_summaries=len(paired_map),
               check_records=len(all_checks),bitwise_records=len(all_bits),required_bitwise_checks=sum(int(x["required"]) for x in all_bits),
               input_fingerprint_records=len(input_map),process_ids=[int(x["pid"]) for x in process_order],
               raw_manifest_files=len(manifest),removed_events=0,recovery=recovery_audit,
               smoke_full_output_coordinate_checks=smoke_coordinates,smoke_error_check_records=smoke_checks,
               fp32_simt_allclose_failed_checks=sum(x["group"]=="fp32" and int(x["simt_allclose_fail"])>0 for x in all_checks),
               numerical_status={g:{criterion:("FAIL" if any(x[criterion+"_status"]=="FAIL" for x in quality_tables["correctness-overview.csv"] if x["group"]==g) else "PASS")
                                    for criterion in ("abs_001","simt_allclose","native_allclose","nonfinite")} for g in groups},
               method="Independent raw parsing; math.fsum means; explicitly sorted median; no summarizer import",
               uncertainty="Descriptive within-session comparison; no IID-event assumptions or confidence interval")
    args.output.mkdir(parents=True)
    (args.output/"verification.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    hashes=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(read_files))]
    (args.output/"independently-read-files.json").write_text(json.dumps(hashes,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":
    main()
