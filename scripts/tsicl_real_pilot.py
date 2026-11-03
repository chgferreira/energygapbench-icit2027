#!/usr/bin/env python3
"""Supplementary real R/P TS-ICL pilot on the existing immutable 18-query support.

The model executes in isolated subprocesses receiving only sanitized NPZ queries.
Truth is loaded only by a later evaluator after every inference artifact is durable.
No data or outputs are uploaded; model/outputs retain noncommercial restrictions.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from energygap.runtime import StateStore
from energygap.runtime.state import atomic_bytes,atomic_json,fingerprint
from energygap.runtime.scheduler import run_units


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source):
    connection=sqlite3.connect(f'file:{source/"state.sqlite"}?mode=ro',uri=True)
    rows=[json.loads(r[0]) for r in connection.execute('SELECT payload FROM results ORDER BY result_key')]
    connection.close()
    selected=[r for r in rows if r.get('phase')=='inference' and r.get('method')=='MOMENT-1-base']
    if len(selected)!=18 or any(r['status']!='success' for r in selected):raise ValueError('expected 18 successful MOMENT queries as observed-value provenance')
    cases=[]
    for row in selected:
        artifact=source/row['artifact']
        if sha(artifact)!=row['sha256']:raise ValueError('source preserved-observation artifact checksum differs')
        with np.load(artifact,allow_pickle=False) as archive:
            available=archive['available_mask'].astype(bool)
            values=np.where(available,archive['point'],np.nan).astype(np.float64)
            times=archive['times'].astype(np.float64)
        if values.shape!=(512,) or times.shape!=(512,):raise ValueError('source query does not have L512')
        # Source assertion guarantees stored received points equal original observations.
        safe_sha=hashlib.sha256(np.nan_to_num(values,nan=0).tobytes()+available.tobytes()+times.tobytes()).hexdigest()
        metadata={k:row[k] for k in ('site','query_id','gap_id','mask_seed','partition','condition','duration','fold')}
        cases.append({'metadata':metadata,'values':values,'times':times,'safe_input_sha256':safe_sha,
                      'source_artifact':str(artifact.relative_to(ROOT)),'source_artifact_sha256':row['sha256']})
    if {(c['metadata']['site'],c['metadata']['partition'],c['metadata']['condition']) for c in cases}!={(s,p,c) for s in ('H1','H2','I1') for p in ('validation','test') for c in ('C3','C4','C5')}:
        raise ValueError('real pilot support differs from predefined 3sites x2partitions x3durations')
    return cases


def train_scales():
    # Evaluator-only: called after all model subprocesses finish, never delivered to TS-ICL.
    from energygap.data import DatasetConfig,ingest
    from energygap.protocols.splits import rolling_origin_splits
    from energygap.evaluation.reconstruction import training_scale
    result={}
    for site in ('H1','H2','I1'):
        series=ingest(DatasetConfig.load(ROOT/'configs/datasets'/f'{site}.yaml'),root=ROOT,purpose='statistical')
        fold=rolling_origin_splits(series)[0]
        scale,reason=training_scale(np.asarray(series.values)[fold.indices(series,'train')])
        result[site]={'scale_kw':scale,'rule':reason,'fit_partition':'fold1_train_only'}
    return result


def run(source,experiment_id=None,resume='auto',budget_total_hours=.1,device='auto'):
    started=time.monotonic();source=Path(source).resolve();cases=prepare(source)
    worker=ROOT/'scripts/tsicl_worker.py';python=ROOT/'.venv-tsicl/bin/python'
    checkpoint=ROOT/'artifacts/checkpoints/tsicl/tsicl-v1.ckpt'
    probe=subprocess.run([str(python),str(worker),'--device',str(device),'--device-info'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    if probe.returncode:raise RuntimeError('TS-ICL isolated device selection failed: '+probe.stdout[-1000:]+probe.stderr[-1000:])
    torch_runtime=json.loads(probe.stdout);device=torch_runtime['device']
    config={'schema_version':1,'scope':'exploratory_real_native_integration_pilot','source_experiment':source.name,
            'inputs':[{'metadata':c['metadata'],'safe_input_sha256':c['safe_input_sha256']} for c in cases],
            'worker_sha256':sha(worker),'orchestrator_sha256':sha(__file__),'environment_lock_sha256':sha(ROOT/'env/tsicl.lock'),
            'device':device,'torch_runtime':torch_runtime,'device_helper_sha256':sha(ROOT/'src/energygap/runtime/device.py'),
            'checkpoint_sha256':sha(checkpoint),'checkpoint_revision':'19c94031439fb31f36ce395088ee50a6762d3774',
            'local_training':'not_applicable','training_seed':None,'task':'R','information_tier':'P','context':512,
            'conditions':['C3','C4','C5'],'folds':[1],'mask_seeds':[1001,2001],'samples':None,
            'point_estimator':'native_quantile_0.5','quantiles':[.05,.1,.25,.5,.75,.9,.95],
            'forecast':'not_run','physics':'not_run_unconfirmed_semantics',
            'classification':'new model integration on known support; exploratory; no confirmatory ranking',
            'budget_total_hours':budget_total_hours}
    identity=fingerprint(config);run_path=ROOT/'runs'/(experiment_id or 'tsicl-real-pilot-'+identity[:16])
    with StateStore(run_path,config,resume=resume,budget_seconds=budget_total_hours*3600) as store:
        atomic_json(run_path/'protocol.lock.json',config)
        atomic_bytes(run_path/'protocol.lock.sha256',(identity+'\n').encode())
        for site in ('H1','H2','I1'):store.add_job('infer:'+site,6,metadata={'method':'TS-ICL','site':site,'phase':'inference'})
        store.add_job('evaluate',18,dependencies=['infer:H1','infer:H2','infer:I1'],metadata={'phase':'evaluator_only'})
        if store.get_meta('status') not in ('completed','completed_with_failures'):
            store.start_session();store.record_active(time.monotonic()-started)
            for site in ('H1','H2','I1'):
                site_cases=[c for c in cases if c['metadata']['site']==site]
                def infer(index,unused_seed):
                    case=site_cases[index];meta=case['metadata'];key='TS-ICL:'+meta['query_id'];logical=hashlib.sha256(key.encode()).hexdigest()
                    input_path=run_path/'inputs'/(logical+'.npz');output=run_path/'predictions'/(logical+'.npz')
                    buffer=io.BytesIO();np.savez_compressed(buffer,values=case['values'],measurement_times=case['times'],
                                     availability_times=case['times'],issuance_time=case['times'][-1])
                    atomic_bytes(input_path,buffer.getvalue())
                    tic=time.monotonic()
                    process=subprocess.run([str(python),str(worker),'--checkpoint',str(checkpoint),'--input',str(input_path),'--output',str(output),'--device',device],cwd=ROOT,capture_output=True,text=True,timeout=45)
                    elapsed=time.monotonic()-tic
                    if process.returncode!=0:raise RuntimeError('TS-ICL isolated inference failed: '+process.stdout[-1000:]+process.stderr[-1000:])
                    native=json.loads(process.stdout)
                    if native['device']!=device:raise RuntimeError('Isolated inference used another device')
                    with np.load(output,allow_pickle=False) as predictions:
                        point=predictions['point_estimate'];quantiles=predictions['quantiles']
                        received=np.isfinite(case['values'])
                        if not np.array_equal(point[received],case['values'][received]):raise ValueError('received original observations changed')
                        if not np.isfinite(quantiles).all():raise ValueError('nonfinite quantile output')
                    return {'key':key,'phase':'inference','method':'TS-ICL',**meta,'status':'success',
                            'artifact':str(output.relative_to(run_path)),'sha256':sha(output),
                            'safe_input_sha256':case['safe_input_sha256'],'source_preserved_observation_sha256':case['source_artifact_sha256'],
                            'subprocess_wall_seconds':elapsed,'native_inference_seconds':native['inference_seconds'],'device':device,'torch_runtime':native['runtime'],
                            'quantile_crossings':native['quantile_crossings'],'primary_estimator':'native_quantile_0.5',
                            'probabilistic_capability':'marginal_quantiles','joint_samples':False,'local_training':'not_applicable'}
                run_units(store,'infer:'+site,infer,shard_size=1,signature='TS-ICL:L512:subprocess:'+fingerprint(torch_runtime),progress='log')
            # Explicit phase boundary: only now may any evaluation truth be read.
            before_scales=time.monotonic();scales=train_scales();store.record_active(time.monotonic()-before_scales)
            atomic_json(run_path/'train_scales.json',scales)
            predictions={r['query_id']:r for r in store.results() if r.get('phase')=='inference'}
            from energygap.evaluation.reconstruction import reconstruction_metrics
            from energygap.evaluation.probabilistic import pinball,interval_metrics
            def evaluate(index,unused_seed):
                case=cases[index];meta=case['metadata'];prediction=predictions[meta['query_id']]
                output=run_path/prediction['artifact']
                if sha(output)!=prediction['sha256']:raise ValueError('prediction checksum differs before evaluator')
                truth_path=ROOT/'data/evaluation_truth'/source.name/(hashlib.sha256(meta['query_id'].encode()).hexdigest()+'.json')
                truth_sha=sha(truth_path);truth=json.loads(truth_path.read_text())
                if truth['query_id']!=meta['query_id']:raise ValueError('evaluator truth ID mismatch')
                y=np.asarray([np.nan if v is None else v for v in truth['values']],float);mask=np.asarray(truth['evaluation_mask'],bool)
                with np.load(output,allow_pickle=False) as archive:
                    point=archive['point_estimate'];quantiles=archive['quantiles'];levels=archive['quantile_levels']
                scale=scales[meta['site']]['scale_kw'];scores=reconstruction_metrics(y,point,mask,scale_train=scale)
                pins=[pinball(quantiles[:,i],y,float(q),mask) for i,q in enumerate(levels)]
                partial_crps=2*float(np.trapezoid(pins,levels))
                intervals={}
                for coverage,lo,hi in ((.5,.25,.75),(.8,.1,.9),(.9,.05,.95)):
                    lower=quantiles[:,int(np.flatnonzero(np.isclose(levels,lo))[0])]
                    upper=quantiles[:,int(np.flatnonzero(np.isclose(levels,hi))[0])]
                    try:intervals[str(coverage)]=interval_metrics(lower,upper,y,coverage=coverage,scale_train=scale,mask=mask)
                    except ValueError as error:intervals[str(coverage)]={'status':'unsupported','reason':str(error)}
                intervals['0.95']={'coverage':None,'width':None,'reason':'native configured grid lacks 0.025/0.975; no invented quantiles'}
                return {'key':'evaluation:TS-ICL:'+meta['query_id'],'phase':'evaluation','method':'TS-ICL',**meta,
                        'status':'success','task':'R','information_tier':'P','context':512,'training_seed':None,
                        'truth_sha256':truth_sha,'prediction_sha256':prediction['sha256'],'evaluation_count':int(mask.sum()),
                        **scores,'pinball':{str(float(q)):float(p) for q,p in zip(levels,pins)},
                        'approximate_crps_truncated_grid_0.05_0.95_kw':partial_crps,
                        'crps_approximation':'2*trapezoid(pinball) only on quantiles0.05..0.95; tails omitted; not empirical sample CRPS or full CRPS',
                        'intervals_marginal':intervals,'interval_coverage_claim':'descriptive marginal only; no calibration/joint coverage claim'}
            run_units(store,'evaluate',evaluate,shard_size=1,signature='TS-ICL:evaluator:7quantiles',progress='log')
            store.complete()
        inference=[r for r in store.results() if r.get('phase')=='inference'];evaluation=store.results('evaluate')
        summaries=[]
        for site in ('H1','H2','I1'):
            for partition in ('validation','test'):
                rows=[r for r in evaluation if r['site']==site and r['partition']==partition]
                summaries.append({'site':site,'partition':partition,'episodes':len(rows),
                    'duration_weights':'equal_1/3_for_C3_C4_C5',
                    'gap_macro_mae_kw':float(np.mean([r['mae_kw'] for r in rows])) if rows else None,
                    'gap_macro_nmae':float(np.mean([r['nmae'] for r in rows])) if rows else None,
                    'approximate_crps_truncated_grid_0.05_0.95_kw':float(np.mean([r['approximate_crps_truncated_grid_0.05_0.95_kw'] for r in rows])) if rows else None})
        report={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'experiment_id':run_path.name,
            'source_support_experiment':source.name,'status':store.status()['status'],'scientific_fingerprint':identity,
            'scope':'conditional_native_real_P_integration_pilot; exploratory; no converged/scientific superiority claim',
            'n_queries':len(inference),'n_success':sum(r['status']=='success' for r in inference),'n_evaluations':len(evaluation),
            'sites':['H1','H2','I1'],'folds':[1],'task':'R','information_tier':'P','context':512,'conditions':['C3','C4','C5'],
            'mask_seeds':{'validation':1001,'test':2001},'point_estimator':'native_quantile_0.5','local_training':'not_applicable',
            'checkpoint_revision':config['checkpoint_revision'],'checkpoint_sha256':config['checkpoint_sha256'],
            'worker_sha256':config['worker_sha256'],'environment_lock':'env/tsicl.lock','device':device,'torch_runtime':torch_runtime,'probabilistic_capability':'marginal_quantiles',
            'joint_samples':False,'crps':'approximate truncated quantile-grid integral only; no sample/full CRPS claimed',
            'code_and_checkpoint_license':'TS-ICL Non-Commercial License v1.0',
            'outputs_use_restriction':'noncommercial/nonproduction only under upstream terms; no uploads or redistribution performed',
            'stored_power_arrays':'private ignored runs directory only; report contains metrics/hashes',
            'query_construction':'received original values recovered from checksum-verified MOMENT output at available_mask only; all other entries NaN; source model verified observed preservation',
            'arrival_log':'none; availability=measurement optimistic assumption for admitted source support',
            'truth_isolation':'separate evaluator phase after all18 worker outputs committed; no truth tensor in worker NPZ',
            'native_latency_p50_seconds':float(np.median([r['native_inference_seconds'] for r in inference])) if inference else None,
            'subprocess_wall_seconds_total':sum(r['subprocess_wall_seconds'] for r in inference),
            'cumulative_accounted_active_seconds':store.status()['active_seconds'],
            'summaries':summaries,'episode_metrics':evaluation,
            'limits':['18 episodes from known support, one fold/one mask per partition; no confirmatory inference.',
                      'No forecasting or physical correction in this supplementary pilot.',
                      'Configured marginal quantile grid excludes tails and 95% interval endpoints.',
                      'One fresh isolated process per query; timings include repeated model load/import overhead separately.',
                      'Scientific conference matrix remains incomplete.']}
        atomic_json(run_path/'reports/tsicl_real_pilot.json',report)
        atomic_json(ROOT/'artifacts/reports/tsicl_real_pilot.json',report)
        return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'runs/real-models-pilot-recovery')
    parser.add_argument('--experiment-id');parser.add_argument('--resume',default='auto',choices=['auto','never','require'])
    parser.add_argument('--budget-total-hours',type=float,default=.1);parser.add_argument('--device',default='auto',help='auto, cpu, cuda or cuda:N');args=parser.parse_args()
    report=run(**vars(args));print(json.dumps({k:report[k] for k in ('experiment_id','status','n_queries','n_success','n_evaluations','scope')}))
if __name__=='__main__':main()
