"""Freeze this experiment's actual bytes and verify preservation of pre-existing files."""
import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={"README.md","scripts/README.md","docs/zh/00-阅读路线.md","results/windows/README.md",
         "docs/zh/26-vitamin-SIMT复现与K30公平对照.md",
         "docs/zh/27-vitamin原入口复跑与寄存器上限实验.md"}


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args();out=args.archive.resolve()
    assert not (out/'archive-sha256.csv').exists(), 'Already frozen'
    changed=[];total=0
    for name,repo in [('sgemm',ROOT),('vitamin',ROOT.parent/'vitamin-cuda')]:
        before=json.loads((out/f'preflight/{name}-hashes.json').read_text(encoding='utf-8'))
        for row in before:
            p=repo/row['path'];total+=1
            assert p.is_file(), f'Missing previous file {p}'
            actual=digest(p)
            if actual!=row['sha256']:
                assert name=='sgemm' and row['path'] in ALLOWED, f'Unexpected old-file change: {p}'
                changed.append(dict(path=row['path'],before=row['sha256'],after=actual))
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo)==(out/f'preflight/{name}-head.txt').read_bytes()
        assert subprocess.check_output(['git','diff','--cached','--binary'],cwd=repo)==(out/f'preflight/{name}-cached.patch').read_bytes()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT.parent/'vitamin-cuda')
    verification=json.loads((out/'verification/verification.json').read_text())
    assert verification['status']=='PASS' and verification['events']==22680
    evidence=json.loads((out/'evidence/audit.json').read_text())
    assert evidence['all_selected_normalized_equal'] and evidence['all_selected_encoded_equal']
    assert evidence['ncu_launches']==6
    build=json.loads((out/'build01/verified-build.json').read_text(encoding='utf-8-sig'))
    assert digest(out/'build01/windows_tensor_bn64_robustness_benchmark.exe')==build['binary_sha256'].lower()
    assert digest(ROOT/'experiments/tensorcore-bn64-robustness/benchmark.cu')==build['source_sha256'].lower()
    # Preserve exact scripts and required include files; no line-ending normalization.
    files=list((ROOT/'experiments/tensorcore-bn64-robustness').glob('*'))
    files += [ROOT/'src/kernels/30_kernel_mini_simt.cuh']
    for folder in ['tensorcore','tensorcore-bn64','tensorcore-ktile']:
        files+=list((ROOT/'experiments'/folder).rglob('*'))
    files+=list((ROOT/'scripts/windows').glob('*TensorBN64Robustness*.ps1'))
    files += [ROOT/'scripts/windows/Initialize-Toolchain.ps1']
    files+=list((ROOT/'scripts').glob('*tensor_bn64_robustness*.py'))
    files += [ROOT/'scripts/extract_vitamin_tensor.py',ROOT/'scripts/prepare_tensor_bn64.py',
              ROOT/'scripts/analyze_tensor_bn64_mapping.py',ROOT/'docs/zh/33-TensorCore-BN64稳健性与尺寸适用范围.md']
    files=sorted(set(f for f in files if f.is_file()))
    source_rows=[]
    with zipfile.ZipFile(out/'source-as-delivered.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:
            name=f.relative_to(ROOT).as_posix();data=f.read_bytes();z.writestr(name,data)
            source_rows.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with zipfile.ZipFile(out/'source-as-delivered.zip') as z:
        for row in source_rows: assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    (out/'source-as-delivered-sha256.json').write_text(json.dumps(source_rows,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'git-status-final.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
    result=dict(previous_files_actually_read_and_hashed=total,allowed_navigation_changes=changed,
        all_other_preexisting_files_unchanged=True,git_heads_and_index_unchanged=True,
        source_snapshot_files=len(files),formal_events=22680,ncu_launches=6,
        device_normalized_and_encoded_sass_equal=True,binary_sha256=build['binary_sha256'],
        sanitizer=json.loads((out/'sanitizer01/result.json').read_text(encoding='utf-8-sig'))['status'],
        no_commit_push_merge=True)
    (out/'delivery-check.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    plan=out/'task_plan.md'
    plan.write_text(plan.read_text(encoding='utf-8').replace('[进行中] 中文','[完成] 中文'),encoding='utf-8')
    with (out/'progress.md').open('a',encoding='utf-8') as f:
        f.write(f'\n交付核验完成：实际读取并哈希{total}个起点文件，除{len(changed)}个授权导航文件外全部未变；Git HEAD和index未变。源码zip逐项回读通过，正式event独立验证和6个NCU证据核验通过；Sanitizer保留TIMEOUT_UNVERIFIED。到此停止，不继续调参。\n')
    rows=[]
    for f in sorted(out.rglob('*')):
        if f.is_file(): rows.append(dict(path=f.relative_to(out).as_posix(),bytes=f.stat().st_size,sha256=digest(f)))
    with (out/'archive-sha256.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,['path','bytes','sha256']);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(previous_files=total,archive_files=len(rows),changed_navigation=len(changed),source_files=len(files))))


if __name__=='__main__':main()
