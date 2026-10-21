"""Derive isolated scripts from the completed grid experiment; no old files edited."""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
outputs={}
s=(ROOT/'experiments/tensorcore-grid/CMakeLists.txt').read_text(encoding='utf-8-sig')
outputs[HERE/'CMakeLists.txt']=s.replace('TensorCoreGridLearning','TensorCoreKTileLearning').replace('tensor_grid','tensor_ktile')
for action in ('Build','Run','Profile'):
    p=ROOT/f'scripts/windows/{action}-TensorGrid.ps1'
    s=p.read_text(encoding='utf-8-sig')
    s=s.replace('tensor-grid','tensor-ktile').replace('tensorcore-grid','tensorcore-ktile')
    s=s.replace('tensor_grid','tensor_ktile').replace('TensorGrid','TensorKTile')
    s=s.replace("'all','grid_w1','grid_w2','grid_w4'", "'all','confirm'")
    if action=='Build':
        s=s.replace('$source,"$repo\\experiments\\tensorcore" -File',
                    '$source,"$repo\\experiments\\tensorcore","$repo\\experiments\\tensorcore-grid" -File')
        s=s.replace('$manifest=[ordered]@{',
                    'Copy-Item "$repo\\experiments\\tensorcore-grid" "$EvidenceDir\\grid-harness-source" -Recurse\n$manifest=[ordered]@{')
    if action=='Run':
        s=s.replace("@('TF32 input / FP32 accumulator+output','FP16 input+output / FP32 accumulator')",
                    "@('TF32 input / FP32 accumulator+output')")
        s=s.replace('C0 reset changes caches; FP16 copies half as many bytes as FP32; cross-precision speed is not a single-factor algorithm claim',
                    'C0 reset changes caches; identical 4-byte-per-element copies for every measured implementation')
        s=s.replace('Scan: rotate/reverse five implementations per precision. Confirmation: baseline/candidate adjacent and swap; cuBLAS rotates outside pair.',
                    'all: dynamic BK16/BK32 adjacent and swap; rotate pair/original/cuBLAS. confirm: original/BK32 adjacent and swap; rotate cuBLAS.')
        s=s.replace('K32 and K64','K32, K64 and K96')
        s=s.replace('foreach ($k in 32,64)', 'foreach ($k in 32,64,96)')
        s=s.replace('@(128,1024,31),','@(128,1024,31),@(128,1024,48),')
    if action=='Profile':
        s=s.replace("@('vitamin_tf32_dbf','tf32_w1','vitamin_fp16_dbf_rw','fp16_w1')",
                    "@('vitamin_tf32_dbf','tf32_dynamic_bk16','tf32_dynamic_bk32','cublas_tf32')")
    outputs[ROOT/f'scripts/windows/{action}-TensorKTile.ps1']=s
s=(ROOT/'scripts/summarize_tensor_grid.py').read_text(encoding='utf-8')
s=s.replace('expected = 10 if meta["selection"] == "all" else 6', 'expected = 4 if meta["selection"] == "all" else 3')
a=s.index('        for kind, baseline in')
b=s.index('                    gain =',a)
s=s[:a]+'''        for kind, baseline in (("tf32", "vitamin_tf32_dbf"),):
            for name in sorted(names):
                if name == "cublas_tf32":
                    continue
                for ref in (baseline, "tf32_dynamic_bk16", "cublas_tf32"):
                    if name == ref or ref not in names:
                        continue
'''+s[b:]
s=s.replace('if "_w" in row["candidate"]', 'if "dynamic" in row["candidate"]')
s=s.replace('vs W8:', 'vs original BK16:')
outputs[ROOT/'scripts/summarize_tensor_ktile.py']=s
for p,s in outputs.items():
    if '--verify' in sys.argv:
        assert p.read_text(encoding='utf-8-sig')==s,p
    else:
        if p.exists():raise FileExistsError(p)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(s,encoding='utf-8',newline='\n')
print('Verified support scripts' if '--verify' in sys.argv else 'Derived support scripts')
