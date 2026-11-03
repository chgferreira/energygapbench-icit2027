"""Reproducible small end-to-end runs. Real runs require confirmed metadata."""
from __future__ import annotations
import hashlib
import json
import pickle
from pathlib import Path
from time import perf_counter
import numpy as np
import yaml
from energygap.data import make_fixture, DatasetConfig, ingest
from energygap.protocols.splits import rolling_origin_splits
from energygap.protocols.availability import build_query
from energygap.protocols.windows import match_future_target
from energygap.masking.generators import generate_mask
from energygap.methods.baselines import create_baseline
from energygap.methods.correction import EnergyConstraint, correct_energy
from energygap.methods.calibration import (CORRECTION_GRID, fit_energy_tolerances,
                                          constraint_from_query, select_correction_parameters)
from energygap.evaluation.reconstruction import training_scale, reconstruction_metrics
from energygap.evaluation.energy import energy_episode_metrics
from energygap.evaluation.events import event_metrics
from energygap.forecasting.ridge import DirectRidge, causal_fill, select_alpha
from energygap.runtime.state import StateStore, fingerprint, atomic_json, atomic_bytes, BudgetExhausted
from energygap.runtime.progress import Progress
from energygap.runtime.signals import StopRequested
from energygap.runtime.device import resolve_device, torch_device_metadata, synchronize

class GracefulStop(RuntimeError):
    """Signal observed after all work up to the prior safe boundary is durable."""


BASELINES=('median','locf','linear','seasonal','kalman_filter','kalman_smoother')


def code_hash():
    digest=hashlib.sha256()
    for p in sorted(Path(__file__).parent.rglob('*.py')):
        digest.update(str(p.relative_to(Path(__file__).parent)).encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def prepare_series(profile, root):
    if profile=='smoke':
        return [make_fixture()],[]
    series,blocked=[],[]
    for site in ('H1','H2','I1'):
        path=root/'configs'/'datasets'/f'{site}.yaml'
        try:
            cfg=DatasetConfig.load(path);cfg.validate(purpose='statistical');series.append(ingest(cfg,root=root,purpose='statistical'))
        except (ValueError,FileNotFoundError,TypeError) as exc:
            blocked.append(dict(site=site,status='blocked',reason=str(exc)))
    return series,blocked


def run_pipeline(root=Path('.'),profile='smoke',resume='auto',progress='none',budget_total_hours=.05,max_episodes=2,experiment_id=None,checkpoint_every_steps=50,checkpoint_every_seconds=60,retry_failed=False,device='auto'):
    root=Path(root).resolve()
    device=resolve_device(device)
    execution=dict(device=device,ridge_backend='torch' if device.startswith('cuda:') else 'scipy',
                   ridge_precision='float64',baseline_and_correction_backend='numpy_scipy_cpu')
    if device.startswith('cuda:'):
        execution['torch_runtime']=torch_device_metadata(device)
    series,blocked=prepare_series(profile,root)
    config=dict(schema_version=1,profile=profile,code_hash=code_hash(),context=512,execution=execution,
                max_episodes=max_episodes,conditions=['C3','C4','C5'],methods=list(BASELINES),
                seed=2001,fixture=dict(days=35,seed=7) if profile=='smoke' else None,
                datasets={s.site_id:s.metadata for s in series},blocked=blocked,
                forecast_horizon_seconds=3600,correction=dict(grid=list(CORRECTION_GRID),
                    selection='validation_equal_weight_C3_C4_C5',calibration='train_duration_direction_q95_resolution_floor'),
                scientific_scope='synthetic_validation' if profile=='smoke' else 'restricted_pilot',
                physical_semantics_policy='known_synthetic' if profile=='smoke' else 'statistical_only_unresolved_measurement_semantics',
                budget_total_hours=budget_total_hours,progress=progress,
                checkpoint_every_steps=checkpoint_every_steps,checkpoint_every_seconds=checkpoint_every_seconds)
    # Actual source hashes participate even when the contract blocks canonical ingest.
    if profile!='smoke':
        config['dataset_configs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'configs/datasets').glob('*.yaml'))}
    identity=fingerprint(config)[:16]
    experiment_id=experiment_id or f'{profile}-{identity}'
    path=root/'runs'/experiment_id
    with StateStore(path,config,resume=resume,budget_seconds=budget_total_hours*3600) as store:
        if store.get_meta('status') in ('completed','completed_with_failures'):
            return store.status()
        atomic_bytes(path/'protocol.lock.yaml',yaml.safe_dump(config,sort_keys=True).encode())
        atomic_bytes(path/'protocol.lock.sha256',(fingerprint(config)+'\n').encode())
        for item in blocked:
            job='data:'+item['site'];store.add_job(job,1,metadata=item);store.mark(job,'blocked',item['reason'])
        plan=[]
        masks={}
        for s in series:
            for fold in rolling_origin_splits(s):
                fold_key=f'{s.site_id}:fold{fold.fold_id}'
                prep=fold_key+':fit';store.add_job(prep,1,metadata={'phase':'fit','site':s.site_id,'fold':fold.fold_id})
                atomic_json(path/'manifests'/f'{fold_key}-split.json',fold.manifest())
                for partition in ('validation','test'):
                    for condition in config['conditions']:
                        seed=1001 if partition=='validation' else 2001
                        manifest=generate_mask(s,fold,partition=partition,condition=condition,seed=seed,max_episodes=max_episodes,physical=bool(s.metadata.get('synthetic')))
                        key=(s.site_id,fold.fold_id,partition,condition)
                        masks[key]=manifest
                        atomic_json(path/'manifests'/f'{fold_key}-{partition}-{condition}.json',manifest.to_dict())
                        job=f'{fold_key}:{partition}:{condition}'
                        store.add_job(job,len(manifest.episodes),dependencies=[prep],metadata={'phase':'inference','site':s.site_id,'condition':condition,'partition':partition})
                        plan.append(dict(job_id=job,dependencies=[prep],units=len(manifest.episodes),unit='queries',resource='gpu' if device.startswith('cuda:') else 'cpu',device=device,estimated_seconds=None))
        atomic_json(path/'plan.json',dict(experiment_id=experiment_id,jobs=plan,blocked=blocked,forecast_fit='one_fixed_weight_model_per_fold',complete_scientific_experiment=False))
        if not series:
            store.complete()
            from energygap.reporting.report import generate_report
            generate_report(path)
            return store.status()
        store.start_session()
        stop=StopRequested();stop.__enter__()
        try:
            for s in series:
                selected_alpha=1.
                for fold in rolling_origin_splits(s):
                    if stop.requested: raise GracefulStop('signal requested at fold boundary')
                    fold_key=f'{s.site_id}:fold{fold.fold_id}';prep=fold_key+':fit'
                    cache=path/'models'/f'{fold_key}.pickle'
                    train=fold.indices(s,'train');val=fold.indices(s,'validation')
                    values=np.asarray(s.values);times=np.asarray(s.times)
                    if store.job(prep)['status']=='succeeded':
                        payload=pickle.loads(cache.read_bytes())
                    else:
                        with store.claim(prep) as lease:
                            store.check_budget()
                            reservation=store.begin_work(prep,lease.token)
                            started=perf_counter()
                            models={name:create_baseline(name).fit(values[train],times[train]) for name in BASELINES}
                            scale,scale_reason=training_scale(values[train])
                            if fold.fold_id==1:
                                selected_alpha,scores=select_alpha(values[train],times[train],values[val],times[val],device=device)
                            forecast=DirectRidge(alpha=selected_alpha,device=device).fit(values[train],times[train])
                            physical=bool(s.metadata.get('synthetic'))
                            calibration=fit_energy_tolerances(s,train) if physical else {}
                            validation_cases=[]
                            for validation_condition in config['conditions'] if physical else ():
                                validation_manifest=masks[(s.site_id,fold.fold_id,'validation',validation_condition)]
                                for validation_episode in validation_manifest.episodes:
                                    validation_query,validation_truth=build_query(s,validation_episode,information_tier='E',artificial_hidden=validation_manifest.hidden_indices)
                                    validation_cases.append((validation_query,validation_truth,validation_episode.duration_minutes))
                            correction_selection=select_correction_parameters(models,validation_cases,calibration,scale,metadata=s.metadata) if physical else {}
                            payload=dict(models=models,forecast=forecast,scale=scale,scale_reason=scale_reason,alpha=selected_alpha,
                                         calibration=calibration,correction_selection=correction_selection)
                            atomic_json(path/'models'/f'{fold_key}-calibration-selection.json',dict(calibration=calibration,correction_selection=correction_selection))
                            checksum=atomic_bytes(cache,pickle.dumps(payload,protocol=5))
                            store.publish(prep,lease.token,[dict(key=prep,artifact=str(cache.relative_to(path)),sha256=checksum,alpha=selected_alpha,scale=scale,scale_reason=scale_reason)],1)
                            synchronize(device)
                            store.finish_work(reservation,perf_counter()-started);store.finish(prep,lease.token)
                    selected_alpha=payload['alpha'];models=payload['models'];forecast=payload['forecast'];scale=payload['scale']
                    for partition in ('validation','test'):
                        for condition in config['conditions']:
                            manifest=masks[(s.site_id,fold.fold_id,partition,condition)]
                            job=f'{fold_key}:{partition}:{condition}'
                            job_info=store.job(job)
                            if stop.requested: raise GracefulStop('signal requested at query boundary')
                            if job_info['status']=='succeeded':continue
                            with store.claim(job) as lease, Progress(len(manifest.episodes),job_info['completed_units'],progress,job) as bar:
                                for index in range(job_info['completed_units'],len(manifest.episodes)):
                                    if stop.requested: raise GracefulStop('signal requested at query boundary')
                                    store.check_budget();reservation=store.begin_work(job,lease.token)
                                    started=perf_counter();episode=manifest.episodes[index]
                                    query,truth=build_query(s,episode,information_tier='E' if s.metadata.get('synthetic') else 'P',artificial_hidden=manifest.hidden_indices)
                                    qvalues=np.asarray(query.values);qtimes=np.asarray(query.times)
                                    target=np.asarray(query.target_gap_mask);evaluation=np.asarray(truth.evaluation_mask)
                                    # Separate evaluator storage; no truth is passed to fit/impute/correction.
                                    truthpath=root/'data/evaluation_truth'/experiment_id/f'{query.query_id.replace(":","_")}.json'
                                    atomic_json(truthpath,dict(query_id=query.query_id,values=[float(v) if np.isfinite(v) else None for v in truth.values],evaluation_mask=list(truth.evaluation_mask)))
                                    predictions={};records=[];correction_diagnostics={}
                                    for name,model in models.items():
                                        tic=perf_counter();fallback=None
                                        try:
                                            prediction=model.impute(qvalues,qtimes,task='R')
                                            if not np.isfinite(prediction[evaluation]).all():raise ValueError('nonfinite_target_output')
                                            status='success'
                                        except Exception as exc:
                                            prediction=models['seasonal'].impute(qvalues,qtimes,task='R');status='fallback';fallback=type(exc).__name__+': '+str(exc)
                                        predictions[name]=(prediction,status,fallback,perf_counter()-tic)
                                    for name in BASELINES if s.metadata.get('synthetic') else ():
                                        base=predictions[name][0]
                                        parameters=payload['correction_selection'][name]['selected']
                                        constraint=constraint_from_query(query,payload['calibration'],episode.duration_minutes,
                                            tau_multiplier=parameters['tau_multiplier'],metadata=s.metadata)
                                        result=correct_energy(base,qtimes,target,constraint,scale_train=scale,
                                            lambda_energy=parameters['lambda_energy'],gamma=parameters['gamma'],
                                            seasonal_start=predictions['seasonal'][0])
                                        correction_diagnostics[name+'+energy_consistent']=dict(result.diagnostics,
                                            correction_status=result.status,base_status=predictions[name][1],
                                            base_fallback_reason=predictions[name][2],selected_parameters=parameters)
                                        combined_status='fallback' if predictions[name][1]=='fallback' else result.status
                                        predictions[name+'+energy_consistent']=(result.values,combined_status,result.diagnostics.get('reason'),result.diagnostics.get('elapsed_seconds',0.))
                                    forecast_target,exclusion=match_future_target(s,query.issuance_time,3600,fold=fold) if partition=='test' else (None,'validation_not_downstream_endpoint')
                                    for name,(prediction,status,fallback,latency) in predictions.items():
                                        metrics=reconstruction_metrics(truth.values,prediction,evaluation,scale_train=scale)
                                        metrics.update(event_metrics(truth.values,prediction,qtimes,evaluation))
                                        positions=np.flatnonzero(target)
                                        if s.metadata.get('synthetic') and len(positions) and positions[0]>0 and positions[-1]+1<len(prediction):
                                            bounds=slice(positions[0]-1,positions[-1]+2)
                                            original=qvalues.copy();original[evaluation]=np.asarray(truth.values)[evaluation]
                                            energy_interval=query.energy_intervals[0] if query.energy_intervals else {}
                                            metrics.update(energy_episode_metrics(original[bounds],prediction[bounds],qtimes[bounds],
                                                counter_import=energy_interval.get('import_kwh'),counter_export=energy_interval.get('export_kwh'),
                                                semantics=s.metadata.get('power_semantics','instantaneous')))
                                        forecast_error=None;reference_error=None
                                        if forecast_target is not None:
                                            issuance=query.issuance_time.timestamp()
                                            forecast_error=abs(forecast.predict(prediction[-288:],issuance)-forecast_target['value'])
                                            reference=values[max(0,episode.right_index-287):episode.right_index+1]
                                            reference_error=abs(forecast.predict(reference,issuance)-forecast_target['value'])
                                        row=dict(key=f'{job}:{episode.gap_id}:{name}',run_id=fingerprint(dict(config=config,site=s.site_id,fold=fold.fold_id,method=name))[:20],site=s.site_id,fold=fold.fold_id,partition=partition,task='R',information_tier='E' if '+' in name else 'P',method=name,variant='synthetic_validation' if s.metadata.get('synthetic') else 'pilot',checkpoint=None,context=512,gap_id=episode.gap_id,query_id=query.query_id,issuance_time=query.issuance_time.isoformat(),day=query.issuance_time.date().isoformat(),duration=episode.duration_minutes,condition=condition,geometry='block',missingness_mechanism='MCAR',requested_rate=manifest.requested_rate,realized_rate=manifest.realized_rate,mask_seed=manifest.seed,train_seed=None,sample_seed=None,status=status,fallback_reason=fallback,n_original_targets=int(evaluation.sum()),latency_seconds=latency,forecast_mae_kw=forecast_error,forecast_reference_mae_kw=reference_error,forecast_nmae=forecast_error/scale if forecast_error is not None and scale else None,forecast_exclusion_reason=exclusion,forecast_target_offset_seconds=forecast_target['offset_seconds'] if forecast_target else None,**metrics)
                                        row['prediction']=[float(v) if np.isfinite(v) else None for v in prediction[evaluation]]
                                        if name in correction_diagnostics:
                                            row['correction_diagnostics']=correction_diagnostics[name]
                                        records.append(row)
                                    synchronize(device)
                                    store.publish(job,lease.token,records,index+1);store.finish_work(reservation,perf_counter()-started);bar.update(index+1)
                                store.finish(job,lease.token)
            store.complete();store.end_session()
        except GracefulStop:
            store.end_session('interrupted')
        except BudgetExhausted:
            store.end_session('paused_budget')
        except BaseException:
            store.end_session('interrupted');raise
        finally:
            stop.__exit__(None,None,None)
    from energygap.reporting.report import generate_report
    generate_report(path)
    with StateStore(path,resume='require') as final_store:
        return final_store.status()
