#!/usr/bin/env python3
"""Restore fixed published BF16 evidence from GitHub and recompute it without GPU."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

COMMIT = 'f14dc4aadb7d5d0d79f07fb2fc58a29a697314a0'
REPO = 'QiLHong/SGEMM_CUDA'
BATCH = 'results/rtx5090/20261022-06-native-bf16-1024'
SCRIPT = 'scripts/rtx5090/recompute_results.py'
EVENTS = BATCH + '/data/events.csv'
SUMMARY = BATCH + '/independent-summary.json'
MARKER = BATCH + '/BATCH_COMPLETE.json'
MANIFESTS = [
    'docs/zh/rtx5090/manifests/20261022-05-sm120-native-build-smoke.sha256',
    'docs/zh/rtx5090/manifests/20261022-06-native-bf16-1024.sha256',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--relay',type=Path,required=True,help='Read-only local Git object IDs; no local evidence payload is used')
    p.add_argument('--proxy',default='http://127.0.0.1:7877')
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http':a.proxy,'https':a.proxy}))
    def fetch(url):
        request=urllib.request.Request(url,headers={'User-Agent':'RTX5090-evidence-recovery','Accept':'application/vnd.github+json'})
        with opener.open(request,timeout=40) as response:
            if response.status != 200: raise RuntimeError(f'HTTP {response.status}')
            return response.read()
    def git(*args):
        return subprocess.check_output(['git','-C',str(a.relay),*args],text=True).strip()
    def write_json(name,value):
        (a.out/name).write_bytes((json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
    commit_metadata=json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/commits/{COMMIT}'))
    assert commit_metadata['sha']==COMMIT
    assert commit_metadata['tree']['sha']==git('rev-parse',COMMIT+'^{tree}')
    write_json('commit-proof.json',{'commit':COMMIT,'tree':commit_metadata['tree']['sha'],
                                   'github_url':commit_metadata['html_url'],
                                   'proof':'GitHub fixed commit tree agrees with local immutable Git object identities; no local payloads restored'})
    files={}
    for rel in [*MANIFESTS,SCRIPT,EVENTS,SUMMARY,MARKER]:
        url=f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{rel}'
        data=fetch(url)
        expected=git('rev-parse',f'{COMMIT}:{rel}')
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert expected==actual, f'Git blob mismatch: {rel}'
        destination=a.out/'restored'/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(data)
        files[rel]={'url':url,'bytes':len(data),'git_blob':actual,'git_blob_verified':True,
                    'sha256':hashlib.sha256(data).hexdigest()}
    manifest_entries={}
    for rel in MANIFESTS:
        manifest_entries[rel]={}
        for line in (a.out/'restored'/rel).read_text(encoding='utf-8').splitlines():
            digest,path=line.split('  ',1)
            manifest_entries[rel][path]=digest
    for rel in [SCRIPT,EVENTS,SUMMARY,MARKER]:
        # Earlier manifests are snapshots: unrelated mutable navigation/log files
        # legitimately change. Each recovered payload uses its owning checkpoint.
        selected_manifest=MANIFESTS[0] if rel==SCRIPT else MANIFESTS[1]
        expected_digest=manifest_entries[selected_manifest][rel]
        assert files[rel]['sha256']==expected_digest, f'Manifest mismatch: {rel}'
        files[rel]['manifest_verified']=selected_manifest
    restored_script=a.out/'restored'/SCRIPT
    restored_events=a.out/'restored'/EVENTS
    recomputed=a.out/'recomputed-summary.json'
    command=[sys.executable,str(restored_script),str(restored_events),
             '--expected-events','1050','--output',str(recomputed)]
    process=subprocess.run(command,cwd=a.out,text=True,capture_output=True,timeout=40)
    (a.out/'recompute.stdout.log').write_text(process.stdout,encoding='utf-8')
    (a.out/'recompute.stderr.log').write_text(process.stderr,encoding='utf-8')
    (a.out/'recompute.exit').write_text(str(process.returncode)+'\n',encoding='utf-8')
    assert process.returncode==0, 'Recovered script returned nonzero'
    expected_summary=json.loads((a.out/'restored'/SUMMARY).read_text(encoding='utf-8'))
    actual_summary=json.loads(recomputed.read_text(encoding='utf-8'))
    assert actual_summary==expected_summary, 'Recomputed statistics differ from uploaded independent summary'
    marker=json.loads((a.out/'restored'/MARKER).read_text(encoding='utf-8'))
    proof={'status':'RECOVERY_VERIFIED','source_commit':COMMIT,'utc':dt.datetime.now(dt.timezone.utc).isoformat(),
           'runtime':{'python_executable':sys.executable,'version':sys.version},
           'download_origin':'GitHub HTTPS fixed commit; fresh directory',
           'gpu_used':False,'remote_mutations':False,'git_mutations':False,
           'files':files,'recovered_script_exit_code':process.returncode,
           'recomputed_json_equals_uploaded_summary':True,'events':actual_summary['rows'],
           'group_counts':actual_summary.get('group_counts'),
           'recomputed_summary_sha256':hashlib.sha256(recomputed.read_bytes()).hexdigest(),
           'numeric_acceptance_pass':marker.get('numeric_acceptance_pass'),
           'original_run_exit_code':marker.get('exit_code'),
           'interpretation':'Recovery success verifies uploaded bytes and reproducible statistics, not numerical acceptance. Original numerical failure remains recorded.',
           'command':command}
    write_json('recovery.json',proof)
    print(json.dumps({'status':proof['status'],'source_commit':COMMIT,'events':proof['events'],
                      'files_verified':len(files),'numeric_acceptance_pass':proof['numeric_acceptance_pass'],
                      'original_run_exit_code':proof['original_run_exit_code'],'out':str(a.out)},ensure_ascii=False))


if __name__=='__main__':
    main()
