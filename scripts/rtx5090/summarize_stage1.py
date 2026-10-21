#!/usr/bin/env python3
"""Summarize closed SM120 batches; never launches GPU work or publishes."""
import argparse, csv, json, subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--repo',required=True);a=p.parse_args()
root=Path(a.repo).resolve(); docs=root/'docs/zh/rtx5090'
batches=[('bf16',1024,'08'),('bf16',2048,'09'),('bf16',4096,'10'),('fp16',1024,'11'),('fp16',2048,'12'),('fp16',4096,'13')]
table=[];coverage={}; resources=[];errors=[]
for dt,n,num in batches:
    rel=f'results/rtx5090/20261022-{num}-reduction-{dt}-{n}';d=root/rel
    meta=json.loads((d/'data/COMPLETED.json').read_text());audit=json.loads((d/'reduction-analysis/audit.json').read_text())
    assert meta['complete'] and audit['formal_event_count']==1260
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=root,text=True).strip()
    assert commit
    coverage[dt,n]=(rel,commit)
    for row in json.loads((d/'reduction-analysis/summary.json').read_text()):
        table.append(dict(dtype=dt,N=n,implementation=row['implementation'],tflops=row['tflops'],median_round_mean_ms=row['median_round_mean_ms'],median_paired_gain=row['median_paired_speedup_vs_strict_reduction_control'],valid=row['valid_fp32_accum_comparison'],result_path=rel,commit=commit))
    for row in json.loads((d/'data/resources_and_setup.json').read_text()):
        if row['round']==0 and row['resources']:resources.append(dict(dtype=dt,N=n,implementation=row['implementation'],**row['resources']))
    checks=json.loads((d/'data/correctness.json').read_text())
    for name in {x['implementation'] for x in checks}:
        for ref in {x['reference'] for x in checks}:
            group=[x for x in checks if x['implementation']==name and x['reference']==ref]
            errors.append(dict(dtype=dt,N=n,implementation=name,reference=ref,max_abs=max(x['max_abs'] for x in group),max_relative_l2=max(x['relative_l2'] for x in group),author_all_rounds_pass=all(x['author_allclose'] for x in group),strict_abs001_all_rounds_pass=all(x['strict_abs_001_pass'] for x in group),max_strict_failed_elements=max(x['strict_abs_001_failed_elements'] for x in group)))
def csvsave(name,rows):
    with (docs/name).open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
csvsave('stage1-native-summary.csv',table);csvsave('stage1-errors.csv',errors);csvsave('stage1-resources.csv',resources)
lines=['# 阶段 1：SM120 TMA 实测结果','',
'RTX 5090，CUDA 12.8.93，cuBLAS 12.8.4.1。上游源码冻结于 `520f97093a2abc239977be83d244b327d0e2e2ef`，device 代码区域保持原样；派生 host 增加原生单调用 CUDA event 和可复用 descriptor。', '',
'主表为 CPU FP32 uniform(-1,1)、seed=42，再转换为对应输入类型；输出同类型，FP32 累加，alpha=1、beta=0。每尺寸独立进程，六实现共享实际输入，每实现 7 轮、每轮 10 预热和 30 正式 event，轮间改变顺序。总计 7560 正式 event。C0 恢复、转换、分配、descriptor 准备、校验位于 event 外。', '',
'作者原始 BF16 4096 正态输入、10 预热/200 次墙钟结果单独保存在 `20261022-04-author-bf16-4096`，不可与此表视作相同输入和计时协议。', '',
'基线为另增的 cuBLAS `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` 对照；每次调用验证数学模式恢复。原默认 cuBLAS 始终保留：BF16 1024 未通过独立精度检查，原始五路失败批次也保存在 `20261022-06-native-bf16-1024`。标记 FAIL 的吞吐不用于精度合格的性能结论。', '',
'| 类型 | N | 实现 | TFLOPS | 逐轮配对增益中位数 | 独立精度 |', '|---|---:|---|---:|---:|---|']
for r in table:lines.append(f"| {r['dtype']} | {r['N']} | {r['implementation']} | {r['tflops']:.2f} | {100*r['median_paired_gain']:+.2f}% | {'PASS' if r['valid'] else 'FAIL'} |")
lines+=['','延迟统计为各轮均值的中位数，再按 2N³/time 换算吞吐；配对增益为各轮基线时间/实现时间−1 的中位数。两种统计不可混用。', '',
'正确性同时保留同类型 cuBLAS、实际表示输入的全输出 PEDANTIC、原始 FP32 输入的 PEDANTIC、固定 256 点 CPU double 和原作者路径逐位对照。小形状 128×1024×384 为全输出 CPU double。作者 atol=.016、rtol=.016（相对项为 candidate）与 abs≤.01 各自报告，作者判据通过不代表严格 FP32 通过。详见 stage1-errors.csv。', '',
'资源来自本机 occupancy API：实际寄存器、shared 和每 SM 驻留数在 stage1-resources.csv。TMA32/TMA64、DBF、多 stage 在 BK、stage、寄存器和写回上同时存在差异，因此只能称整体实现比较。低频遥测不能证明逐 event 频率恒定。', '',
'原生构建 SASS 实际包含 UTMALDG.2D、HMMA.16816、LDGSTS 和等待指令；没有将 TMA 当作 TMEM 或 tcgen05。NCU 计数器受宿主权限限制，静态证据不能证明流水充分重叠。原BF16 K384 三项 Sanitizer 通过，其他检查状态以各自日志为准。', '',
'8192 方阵为可选扩展，本阶段未运行；先覆盖已要求的三尺寸与 FP16 对照。', '',
'## 已提交的原始数据','']
for (dt,n),(rel,commit) in coverage.items():lines.append(f'- {dt}/{n}: `{rel}`，提交 `{commit}`；events.csv、输入指纹、顺序、正确性、资源、独立分析齐全。')
(docs/'stage1-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
matrix=docs/'reproduction-matrix.csv'
with matrix.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f));fields=list(rows[0])
for r in rows:
    if r['stage']!='1':continue
    dt='bf16' if 'BF16' in r['id'] else 'fp16';name=r['variant'];selected=[x for x in table if x['dtype']==dt and x['implementation']==name]
    r['status']='已运行；数值失败保留' if not all(x['valid'] for x in selected) else '已完成主尺寸统一测量'
    r['result_path']=';'.join(coverage[dt,n][0] for n in [1024,2048,4096])
    r['verified_remote_commit']=';'.join(coverage[dt,n][1] for n in [1024,2048,4096])
    r['correctness_status']='见stage1-errors.csv；作者判据与abs≤.01分开，默认BF16cuBLAS1024失败'
    r['bitwise_status']='原绑定与派生host，全部原5项逐轮相同；新增归约对照另记'
    r['profile_status']='NCU权限阻塞；实际SASS及occupancy资源已保存'
with matrix.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps({'events':7560,'rows':len(table),'matrix_rows':len(rows)}))
