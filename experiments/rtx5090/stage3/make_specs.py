"""Generate a finite source-frozen stage3 menu. Never builds or runs a GPU."""
import csv,json,re,hashlib,argparse
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();repo=a.repo.resolve();builds={};series={}
def build(id,kind,defines=None,kernel=None):
    builds[id]={'kind':kind,'defines':defines or {},'kernel':kernel,'arch':'sm_120','optimization':'-O3 -DNDEBUG','status':'not_built'};return id
build('stock','stock',kernel='all');build('k30','k30');build('k31','k31');build('k32','k32');build('cutlass','cutlass')
def mode(id,b,cli,min_n=128,notes=''):
    return dict(id=id,build=b,cli=str(cli),min_n=min_n,notes=notes)
strict=mode('strict','stock',13);k10=mode('k10','stock',10)
def group(id,candidates,seed=20261013,rounds=7,notes=''):
    series[id]=dict(candidates=candidates,seed=seed,alpha=.5,beta=3,original_rounds=rounds,original_repeats=50,original_warmup=10,event_capture_rounds=7,event_capture_repeats=30,event_capture_warmup=10,sizes=[128,256,512,1024,2048,4096],notes=notes)
group('teaching',[strict]+[mode('k'+str(k),'stock',k,256 if k==11 else 128,'Default K11 BN256 requires N>=256' if k==11 else '') for k in range(1,13)],seed=20261010,rounds=3,notes='Budget slow K1-3 explicitly; each size is a separate closed batch. K11 at 128 is an illegal shape, not a numeric failure.')
for k,variants in {15:[1,2,4,8,16],16:None,17:[0],18:list(range(6)),19:[1,2,3],20:[1,2,3],21:[1,2],22:[1,2,3],23:[1,2,3,4]}.items():
    candidates=[strict,k10]
    if k==16:
        variants=[(m,s) for m in [1,3,7] for s in [1,2,3]]
    for v in variants:
        if k==15:defs={'SGEMM_K15_GROUP_M':v};tag=f'g{v}'
        elif k==16:defs={'SGEMM_K16_XOR_MASK':v[0],'SGEMM_K16_XOR_SHIFT':v[1]};tag=f'm{v[0]}s{v[1]}'
        elif k==17:defs={};tag='default'
        else:defs={f'SGEMM_K{k}_VARIANT':v};tag=f'v{v}'
        b=f'k{k}-{tag}';build(b,'stock',defs,kernel=k);candidates.append(mode(b,b,k))
    group(f'k{k}',candidates,seed=20261010 if k==15 else 20261011 if k<=19 else 20261012 if k<=22 else 20261013,rounds=7 if k==21 else 3,notes='K23 V4 historical addendum used 7 rounds; V1-3 original used 3 rounds. This mixed group is not an exact historical replay.' if k==23 else '')
for k,modes in [(30,['v1','v2','v3','v4']),(31,['control','late','vectors','stage3','stage4','cta256','r220','r225','r230','r235','r240']),(32,['serial','overlap','single'])]:
    cs=[mode('strict',f'k{k}','strict'),mode('k10',f'k{k}','k10')]
    if k!=30:cs.append(mode('k30',f'k{k}','k30'))
    cs += [mode(m,f'k{k}',m,notes='Historical r220 spill 8 B; no automatic resource gate' if m=='r220' else '') for m in modes]
    group(f'k{k}',cs,rounds=6 if k==32 else 7,notes='Independent original audit. Resource gates are historical rules, not current-hardware capability.')
group('cutlass',[mode(m,'cutlass',m) for m in ['strict','k10','k21','cutlass']],notes='CUTLASS v3.5.1 f7b19de32c5d1f3cedfc735c2849f12b537522ee; needs exact clean dependency.')
tiles=list(csv.DictReader((repo/'docs/results/rtx3080-k10-tile-search-pad4-20261010.csv').open()))
for t in tiles:
    defs={f'SGEMM_K10_{k.upper()}':int(t[k]) for k in ['bm','bn','bk','wm','wn','wniter','tm','tn','as_pad','min_blocks']};defs['SGEMM_K10_NUM_THREADS']=int(t['threads'])
    b='k10-tile-'+t['id'];build(b,'stock',defs,kernel=10)
    group(b,[strict,k10,mode(b,b,10,max(128,int(t['bm']),int(t['bn'])))],seed=20261010,rounds=1,notes='Historical frozen search used 15 repeats, winner 50; new 7x30 is a separate protocol. No new parameter search.')
    series[b]['original_repeats']=15
for pad in [0,1,2,3,4,8,12]:
    b=f'k10-pad{pad}';build(b,'stock',{'SGEMM_K10_AS_PAD':pad},kernel=10)
    group(b,[strict,k10,mode(b,b,10)],seed=20261010,rounds=3)
for mb in [0,1,2,3,4]:
    b=f'k10-minblocks{mb}';build(b,'stock',{'SGEMM_K10_MIN_BLOCKS':mb},kernel=10)
    group(b,[strict,k10,mode(b,b,10)],seed=20261010,rounds=3,notes='minBlocks4 had historical spills; preserve current resource and numeric status.')
for wni in [2]:
    for mb in [0,1,2]:
        b=f'k10-sub8x4n2-minblocks{mb}';build(b,'stock',{'SGEMM_K10_WNITER':wni,'SGEMM_K10_MIN_BLOCKS':mb},kernel=10)
        group(b,[strict,k10,mode(b,b,10)],seed=20261010,rounds=1)
group('cublas-math-modes',[mode('legacy','stock',0),strict,mode('tf32','stock',14)],seed=20261010,rounds=3,notes='Different precision modes: keep TF32 performance outside the strict FP32 table. NVIDIA_TF32_OVERRIDE is unset.')
sources=[repo/'sgemm.cu',repo/'src/runner.cu',repo/'src/runner.cuh',repo/'src/kernels.cuh']+list((repo/'src/kernels').rglob('*.cuh'))+[repo/f'scripts/{k}_audit.cu' for k in ['k30','k31','k32']]+[repo/'scripts/cutlass_simt_reference.cu',repo/'scripts/nvtx_compat.h']
hashes={f.relative_to(repo).as_posix():hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in sources}
raw_hashes={f.relative_to(repo).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
spec=dict(schema=1,source_commit='a2b23dabadc79440bf9cbf63f0b566f7eec429ba',source_hashes=hashes,source_hash_normalization='CRLF to LF only; actual build worktree raw hashes recorded separately',source_raw_preparation_hashes=raw_hashes,builds=builds,series=series,protocol_note='Original single-call event boundaries and independent-process launches retained; host capture adds raw events. This is not same-context unified benchmarking. Each implementation uses the default stream within its own process. Actual input fingerprints are checked across processes. events7 uses 7 rounds x 30 events and 10 warmups. historical-counts only selects archived repeat counts: changed group composition, one-size input allocation, smoke order, and absence of the archived 4096 thermal prelude prevent an exact historical replay claim. Main unified comparisons need a separate harness.',excluded_unresolved=['six K10 compiler option families without archived exact values','threads256 tile launch-bounds tuple unknown','broad upstream autotuner grids not proven historically measured'])
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'builds':len(builds),'series':len(series),'output':str(a.output)}))
