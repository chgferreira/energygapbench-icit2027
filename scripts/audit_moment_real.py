#!/usr/bin/env python3
"""Audit official-pin MOMENT on exactly the already frozen real P pilot queries.

Worker input is rebuilt ONLY from previously verified observed positions. Truth
is opened exclusively after the isolated worker has finished successfully.
No source results or experiment state are mutated; private artifacts stay in runs/.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import math
from zoneinfo import ZoneInfo
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from energygap.data.provenance import sha256_file,write_immutable_json
from energygap.evaluation.reconstruction import reconstruction_metrics
from energygap.runtime.state import atomic_bytes,atomic_json


def run(root=ROOT,experiment_id='real-models-pilot-v2',audit_id='moment-isolated-official-pins-v1',device='auto'):
    root=Path(root).resolve();source=root/'runs'/experiment_id
    probe=subprocess.run([str(root/'.venv-moment/bin/python'),str(root/'scripts/moment_worker.py'),'--device',str(device),'--device-info'],cwd=root,capture_output=True,text=True,timeout=30)
    if probe.returncode:raise RuntimeError('MOMENT isolated device selection failed: '+probe.stdout[-1000:]+probe.stderr[-1000:])
    torch_runtime=json.loads(probe.stdout);device=torch_runtime['device']
    # Read-only SQLite access; no status transitions or source result mutation.
    with sqlite3.connect(f'file:{source / "state.sqlite"}?mode=ro',uri=True) as connection:
        payloads=[json.loads(row[0]) for row in connection.execute('SELECT payload FROM results')]
    rows=sorted([r for r in payloads if r.get('method')=='MOMENT-1-base' and r.get('phase')=='inference'],key=lambda r:(r['site'],r['partition'],r['condition'],r['query_id']))
    expected={(s,p,c) for s in ('H1','H2','I1') for p in ('validation','test') for c in ('C3','C4','C5')}
    if len(rows)!=18 or {(r['site'],r['partition'],r['condition']) for r in rows}!=expected:
        raise ValueError('Audit requires exactly the 18 frozen three-site pilot cells; do not silently change support')
    if any(r.get('status')!='success' or r.get('task')!='R' or r.get('information_tier')!='P' for r in rows):
        raise ValueError('Source rows are not all successful R/P records')
    audit_dir=source/'audits'/audit_id;audit_dir.mkdir(parents=True,exist_ok=True)
    values=[];times=[];available=[];qids=[];provenance=[];shared=[];targets=[]
    for row in rows:
        artifact=source/row['artifact']
        if sha256_file(artifact)!=row['sha256']:raise ValueError('Committed source prediction checksum mismatch')
        manifest=json.loads((source/'manifests'/f'{row["site"]}-fold1.json').read_text())
        start=datetime.fromisoformat(manifest['split']['start'])
        if start.tzinfo is None:raise ValueError('Source split timezone not established')
        with np.load(artifact,allow_pickle=False) as arrays:
            point=np.asarray(arrays['point'],float);t=np.asarray(arrays['times'],float);mask=np.asarray(arrays['available_mask'],bool);target=np.asarray(arrays['target_gap_mask'],bool)
        if point.shape!=(512,) or any(a.shape!=point.shape for a in (t,mask,target)):
            raise ValueError('Source L512 artifact contract mismatch')
        if t[0]<start.timestamp():raise ValueError('Padding provenance unavailable; cannot assume a nonpadding window')
        if not np.isfinite(point[mask]).all() or np.any(mask&target):raise ValueError('Source observation/target masks invalid')
        safe=np.where(mask,point,np.nan)
        values.append(safe);times.append(t);available.append(mask);qids.append(row['query_id']);shared.append(point);targets.append(target)
        provenance.append({'query_id':row['query_id'],'site':row['site'],'partition':row['partition'],'condition':row['condition'],'source_result_key':row['key'],'source_artifact':row['artifact'],'source_sha256':row['sha256'],'original_observations_verified_preserved_in_source_pilot':True,'safe_input_sha256':hashlib.sha256(safe.tobytes()).hexdigest(),'issuance_time_epoch':float(t[-1]),'padding':False})
    # Read source TIMESTAMP columns only, to recover actual reconnection times.
    # The entire I1 file has a few jittered observations, so do not infer query
    # issuance from its site-wide nominal grid. No source power/counter is used.
    import yaml
    reconnection={}
    for site in ('H1','H2','I1'):
        cfgpath=root/'configs/datasets'/f'{site}.yaml'
        config=yaml.safe_load(cfgpath.read_text())
        original_manifest=json.loads((source/'manifests'/f'{site}-fold1.json').read_text())
        if sha256_file(cfgpath)!=original_manifest['dataset_config_sha256']:
            raise ValueError('Dataset contract changed after original pilot')
        if config['timestamp'].get('availability_column'):
            raise ValueError('Arrival-log mapping must be reproduced explicitly for this audit')
        requested={float(t[-1]) for r,t in zip(rows,times) if r['site']==site}
        zone=ZoneInfo(config['timestamp']['timezone'])
        for entry in config['files']:
            raw=root/entry['path']
            if sha256_file(raw)!=entry['sha256']:raise ValueError('Original raw source checksum changed')
            if raw.suffix.lower() not in ('.csv','.txt'):
                raise ValueError('Timestamp-only audit currently supports delimited source files')
            with raw.open(encoding=entry.get('encoding','utf-8-sig'),newline='') as stream:
                for line,source_row in enumerate(csv.DictReader(stream,delimiter=entry.get('delimiter',';')),2):
                    measurement=datetime.fromisoformat(source_row[config['timestamp']['column']])
                    if measurement.tzinfo is None:measurement=measurement.replace(tzinfo=zone)
                    epoch=measurement.timestamp();slot=math.floor(epoch/300+.5)*300
                    if slot not in requested:continue
                    key=(site,float(slot))
                    if key in reconnection:raise ValueError('Reconnection source collision is ambiguous')
                    reconnection[key]={'issuance_time_epoch':epoch,'reported_timestamp':measurement.isoformat(),'source_record_id':entry['sha256']+':'+str(line),'grid_jitter_seconds':epoch-slot,'availability_assumption':'optimistic availability=reported measurement timestamp; no arrival log'}
    issuance_times=[]
    for row,t,record in zip(rows,times,provenance):
        key=(row['site'],float(t[-1]))
        if key not in reconnection:raise ValueError('Original reconnection timestamp not found')
        resolved=reconnection[key]
        if resolved['grid_jitter_seconds']!=0:
            raise ValueError('Selected query has nonzero right-boundary jitter; worker availability/time-axis extension required, no cutoff shifted')
        record.update(resolved);issuance_times.append(resolved['issuance_time_epoch'])
    manifest={'schema_version':1,'source_experiment_id':experiment_id,'audit_id':audit_id,'script_sha256':sha256_file(__file__),'worker_sha256':sha256_file(root/'scripts/moment_worker.py'),'dependency_lock_sha256':sha256_file(root/'environments/moment.lock.txt'),'source_protocol_sha256':sha256_file(source/'protocol.lock.json'),'method':'MOMENT-1-base','variant':'frozen_native_reconstruction_isolated_official_pins','task':'R','information_tier':'P','queries':provenance,'input_policy':'point values at available_mask only, every unavailable position NaN; no old model imputation or evaluator data passed','issuance_policy':'original reconnection timestamps recovered by timestamp-only source scan; selected18 boundaries have zero jitter; availability=reported timestamp is optimistic','physical_semantics':'unknown instantaneous/interval-mean and updated/measured/interval edge; no energy claims'}
    manifest.update(device=device,torch_runtime=torch_runtime,device_helper_sha256=sha256_file(root/'src/energygap/runtime/device.py'))
    write_immutable_json(audit_dir/'manifest.json',manifest)
    buffer=io.BytesIO()
    np.savez_compressed(buffer,values=np.asarray(values),times=np.asarray(times),observed_mask=np.asarray(available),padding_mask=np.zeros((18,512),bool),query_ids=np.asarray(qids))
    atomic_bytes(audit_dir/'input.npz',buffer.getvalue())
    write_immutable_json(audit_dir/'input.json',{'schema_version':1,'task':'R','information_tier':'P','issuance_times':issuance_times})
    command=[str(root/'.venv-moment/bin/python'),str(root/'scripts/moment_worker.py'),'--input',str(audit_dir/'input.npz'),'--metadata',str(audit_dir/'input.json'),'--output',str(audit_dir/'predictions.npz'),'--result-metadata',str(audit_dir/'worker-result.json'),'--checkpoint-cache',str(root/'artifacts/checkpoints/hf'),'--audit-native','--device',device]
    start=perf_counter();worker=subprocess.run(command,capture_output=True,text=True,timeout=120,cwd=root);wall=perf_counter()-start
    atomic_bytes(audit_dir/'worker.log',(worker.stdout+worker.stderr).encode())
    if worker.returncode:raise RuntimeError('Isolated worker failed: '+worker.stderr[-2000:])
    worker_result=json.loads((audit_dir/'worker-result.json').read_text())
    if worker_result['status']!='success' or worker_result['queries']!=18 or worker_result['native_parity_max_abs_error']!=0 or worker_result['environment']['device']!=device:
        raise RuntimeError('Native isolated completion/parity audit failed')
    with np.load(audit_dir/'predictions.npz',allow_pickle=False) as arrays:
        predictions=np.asarray(arrays['point_estimate'],float);returned=arrays['query_ids'].tolist()
    if returned!=qids:raise ValueError('Worker query identity/order mismatch')
    # Only NOW open evaluator truth. It is never read by or serialized to worker.
    scales=defaultdict(list)
    for row in rows:
        if row.get('nmae') and row.get('mae_kw') is not None:scales[row['site']].append(row['mae_kw']/row['nmae'])
    for site,choices in scales.items():
        if not np.allclose(choices,choices[0],rtol=1e-10,atol=1e-12):raise ValueError('Original recorded train normalization ratios inconsistent')
    details=[];groups=defaultdict(list);point_errors=defaultdict(list)
    for index,row in enumerate(rows):
        truth_path=root/'data/evaluation_truth'/experiment_id/(hashlib.sha256(row['query_id'].encode()).hexdigest()+'.json')
        truth=json.loads(truth_path.read_text())
        if truth['query_id']!=row['query_id']:raise ValueError('Evaluator query mismatch')
        y=np.asarray([np.nan if v is None else v for v in truth['values']],float);score=np.asarray(truth['evaluation_mask'],bool)
        if y.shape!=(512,) or score.shape!=y.shape or np.any(score&~targets[index]) or np.any(score&available[index]):raise ValueError('Evaluator support invalid')
        if int(score.sum())!=row['n']:raise ValueError('Source target support changed')
        if not np.array_equal(predictions[index][available[index]],values[index][available[index]]):raise ValueError('Isolated model changed original observations')
        scale=scales[row['site']][0] if scales[row['site']] else None
        metrics=reconstruction_metrics(y,predictions[index],score,scale_train=scale)
        old=reconstruction_metrics(y,shared[index],score,scale_train=scale)
        if not np.isclose(old['mae_kw'],row['mae_kw'],rtol=1e-12,atol=1e-12):raise ValueError('Original evaluator result does not reproduce')
        delta=predictions[index][score]-shared[index][score]
        result={'site':row['site'],'partition':row['partition'],'condition':row['condition'],'duration_minutes':row['duration'],'query_id':row['query_id'],'target_count':int(score.sum()),'isolated':metrics,'shared_override':old,'max_absolute_point_difference_kw':float(np.max(np.abs(delta))),'mean_absolute_point_difference_kw':float(np.mean(np.abs(delta))),'mae_difference_kw':metrics['mae_kw']-old['mae_kw'],'same_scoring_support':True}
        details.append(result);groups[(row['site'],row['partition'])].append(result);point_errors[(row['site'],row['partition'])].extend(np.abs(predictions[index][score]-y[score]).tolist())
    summaries=[]
    for (site,partition),items in sorted(groups.items()):
        summaries.append({'site':site,'partition':partition,'n_episodes':len(items),'n_original_targets':sum(i['target_count'] for i in items),'conditions':sorted(i['condition'] for i in items),'isolated_gap_macro_mae_kw':float(np.mean([i['isolated']['mae_kw'] for i in items])),'isolated_point_micro_mae_kw':float(np.mean(point_errors[(site,partition)])),'shared_override_gap_macro_mae_kw':float(np.mean([i['shared_override']['mae_kw'] for i in items])),'max_absolute_point_difference_kw':max(i['max_absolute_point_difference_kw'] for i in items)})
    report={'schema_version':1,'status':'success','scope':'real-data R/P integrity pilot in official-pin isolated MOMENT; not conference benchmark or inferential result','source_experiment_id':experiment_id,'audit_id':audit_id,'n_sites':3,'n_queries':18,'queries_per_site':dict(Counter(r['site'] for r in rows)),'queries_per_partition':dict(Counter(r['partition'] for r in rows)),'worker':worker_result,'parent_subprocess_wall_seconds':wall,'native_parity_max_abs_error':worker_result['native_parity_max_abs_error'],'observed_preservation':True,'truth_isolation':'evaluator files opened only after successful worker completion; only observed original values serialized','same_support_comparison':True,'cross_environment_max_absolute_target_difference_kw':max(d['max_absolute_point_difference_kw'] for d in details),'equivalence_claim':False,'normalization_source':'train-only normalization denominator recovered from consistent original recorded MAE/nMAE ratios; no refit on audit data','physical_status':'unknown power/timestamp semantics; no energy correction','pretraining_overlap':'unknown','inference_training':'not_applicable frozen checkpoint','conference_complete':False,'private_artifacts':str(audit_dir.relative_to(root)),'source_manifest_sha256':sha256_file(audit_dir/'manifest.json'),'summary':summaries,'episode_results':details}
    atomic_json(audit_dir/'report.json',report)
    atomic_json(root/'artifacts/reports/moment_real_isolated_audit.json',report)
    print(json.dumps({k:report[k] for k in ('status','n_sites','n_queries','native_parity_max_abs_error','cross_environment_max_absolute_target_difference_kw','conference_complete')},indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--experiment-id',default='real-models-pilot-v2');parser.add_argument('--audit-id',default='moment-isolated-official-pins-v1')
    parser.add_argument('--device',default='auto',help='auto, cpu, cuda or cuda:N')
    run(**vars(parser.parse_args()))


if __name__=='__main__':main()
