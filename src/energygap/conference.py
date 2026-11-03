"""Executable available-method matrix; hypothetical metrology remains exploratory.

Scientific settings are immutable across resume. The single GPU is used serially;
CPU pools may use the requested thread count. Native fit state is delegated to the
audited epoch backends and queries/selection/bootstrap use transactional shards.
"""
from __future__ import annotations

from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
import yaml

from energygap.pipeline import BASELINES, code_hash, prepare_series, GracefulStop
from energygap.data import SeriesData
from energygap.masking.generators import generate_mask, CONDITIONS
from energygap.protocols.availability import build_query
from energygap.methods.baselines import create_baseline
from energygap.methods.calibration import CORRECTION_GRID, fit_energy_tolerances, constraint_from_query
from energygap.methods.correction import EnergyConstraint
from energygap.methods.cpi import CPIEnergyDonorAdapted
from energygap.methods.validation_panel import select_epoch_validation_panel
from energygap.evaluation.reconstruction import training_scale, reconstruction_metrics
from energygap.evaluation.events import event_metrics
from energygap.evaluation.probabilistic import summarize_samples, pinball, interval_metrics
from energygap.conference_energy import conditional_energy_metrics, cpi_energy_records
from energygap.forecasting.ridge import DirectRidge, select_alpha
from energygap.protocols.splits import rolling_origin_splits
from energygap.runtime.device import resolve_device, torch_device_metadata, synchronize
from energygap.runtime.threads import configure_cpu_threads
from energygap.runtime.state import StateStore, atomic_bytes, atomic_json, fingerprint, stable_seed, BudgetExhausted, TERMINAL
from energygap.runtime.signals import StopRequested
from energygap.runtime.progress import Progress
from energygap.runtime.conference_progress import ConferenceProgress as GlobalProgress
from energygap.runtime.energy_workers import EnergyCorrectionPool, EnergyEpisodeTask, EnergyWorkCancelled

LOG=logging.getLogger(__name__)
DEFAULT_CONFIG='configs/experiments/server_full.yaml'
BATCHED_METHODS=('BRITS','SAITS','ImputeFormer','T1','MOMENT-1-base')


def load_settings(root,config_path=None,settings=None):
    if settings is None:
        path=Path(config_path or DEFAULT_CONFIG)
        if not path.is_absolute():path=Path(root)/path
        settings=yaml.safe_load(path.read_text())
    cfg=dict(settings)
    required=('sites','folds','conditions','methods','train_seeds','validation_mask_seeds','test_mask_seeds')
    if any(not cfg.get(key) for key in required):raise ValueError('Empty or missing conference dimensions')
    for key in ('sites','folds','conditions','methods','train_seeds','validation_mask_seeds','test_mask_seeds'):
        if len(set(cfg[key]))!=len(cfg[key]):raise ValueError('Duplicate conference dimension: '+key)
    if set(cfg['conditions'])-set(CONDITIONS):raise ValueError('Unknown condition')
    if set(cfg['folds'])-{1,2,3}:raise ValueError('Unknown rolling fold')
    if cfg.get('context',512)!=512:raise ValueError('Core context is fixed at 512')
    if cfg.get('physical_confirmatory',False):raise ValueError('Unconfirmed measurement semantics cannot enable confirmatory physics')
    if set(cfg.get('physical_hypotheses',[]))-{'instantaneous','interval_mean_start','interval_mean_end'}:
        raise ValueError('Unknown conditional physical hypothesis')
    for key,default in [('max_epochs',100),('patience',10),('batch_size',32),('inference_batch_size',32),('csdi_sample_batch_size',32),('samples',100),('hpo_trials',6),('training_stride',288),('shard_size',16),('frozen_shard_size',128),('bootstrap_attempts',10000),('block_days',7)]:
        cfg.setdefault(key,default)
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<=0:raise ValueError('Positive integer required: '+key)
    for key,default in [('progress_refresh_seconds',1),('progress_persist_seconds',10),('status_snapshot_seconds',60),('heartbeat_seconds',15)]:
        cfg.setdefault(key,default)
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],(int,float)) or not np.isfinite(cfg[key]) or cfg[key]<=0:
            raise ValueError('Positive finite interval required: '+key)
    cfg.setdefault('energy_cpu_workers',1)
    if isinstance(cfg['energy_cpu_workers'],bool) or not isinstance(cfg['energy_cpu_workers'],int) or cfg['energy_cpu_workers']<=0:
        raise ValueError('Positive integer required: energy_cpu_workers')
    cfg.setdefault('energy_cpu_max_pending',2*cfg['energy_cpu_workers'])
    for key in ('energy_cpu_workers','energy_cpu_max_pending'):
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<=0:
            raise ValueError('Positive integer required: '+key)
    cfg.setdefault('brits_execution','vectorized')
    if cfg['brits_execution'] not in ('vectorized','native','vectorized_cuda_graph'):
        raise ValueError('brits_execution must be vectorized, native or vectorized_cuda_graph')
    if cfg['hpo_trials']>6:raise ValueError('Only six prespecified HPO configurations exist')
    cfg.setdefault('hpo_trial_indices',list(range(cfg['hpo_trials'])))
    indices=cfg['hpo_trial_indices']
    if (not isinstance(indices,(list,tuple)) or len(indices)!=cfg['hpo_trials']
            or any(isinstance(index,bool) or not isinstance(index,int) or not 0<=index<6 for index in indices)
            or len(set(indices))!=len(indices)):
        raise ValueError('hpo_trial_indices must contain hpo_trials distinct integer indices from 0 to 5')
    cfg['hpo_trial_indices']=list(indices)
    for key,default in [('hpo_max_epochs',cfg['max_epochs']),('hpo_patience',cfg['patience'])]:
        cfg.setdefault(key,default)
        if isinstance(cfg[key],bool) or not isinstance(cfg[key],int) or cfg[key]<=0:
            raise ValueError('Positive integer required: '+key)
    if cfg['hpo_patience']>cfg['hpo_max_epochs']:
        raise ValueError('hpo_patience cannot exceed hpo_max_epochs')
    cfg.setdefault('epoch_validation_max_per_stratum',None)
    cap=cfg['epoch_validation_max_per_stratum']
    if cap is not None and (isinstance(cap,bool) or not isinstance(cap,int) or cap<=0):
        raise ValueError('epoch_validation_max_per_stratum must be a positive integer or null')
    cfg.setdefault('epoch_validation_seed',1729)
    if isinstance(cfg['epoch_validation_seed'],bool) or not isinstance(cfg['epoch_validation_seed'],int):
        raise ValueError('epoch_validation_seed must be an integer')
    for key in ('max_episodes','max_training_windows'):
        cfg.setdefault(key,None)
        if cfg[key] is not None and (not isinstance(cfg[key],int) or cfg[key]<=0):raise ValueError('Positive cap or null required: '+key)
    cfg.setdefault('forecast_horizons_seconds',[1800,3600,10800,21600])
    if 3600 not in cfg['forecast_horizons_seconds'] or any(h<=0 or h%300 for h in cfg['forecast_horizons_seconds']):
        raise ValueError('Forecast horizons must include 1h and use positive five-minute multiples')
    cfg.setdefault('physical_conditions',['C3','C4','C5'])
    cfg.setdefault('physical_hypotheses',[])
    cfg.setdefault('require_available',[])
    cfg.setdefault('environment_policy','single_cuda_env')
    if cfg['environment_policy']!='single_cuda_env':raise ValueError('Conference execution requires one active cuda_env')
    if set(cfg['methods']) & {'BRITS','SAITS','CSDI','ImputeFormer','T1'} and not {'C3','C4','C5'}<=set(cfg['conditions']):
        raise ValueError('Trainable conference methods require C3, C4 and C5 for epoch and HPO selection')
    return cfg



def _job(store,jid,n,phase,deps=(),**meta):
    store.add_job(jid,n,dependencies=deps,metadata=dict(phase=phase,signature=meta.pop('signature',phase),**meta))


def _finish_unit(store,jid,lease,index,records,compute,display,signature,stop,*,units=1):
    if stop.requested:raise GracefulStop('Signal requested at durable boundary')
    store.check_budget();reservation=store.begin_work(jid,lease.token);tic=perf_counter();published=False
    try:
        result=compute()
        if result is not None:records=result
        store.publish(jid,lease.token,records,index)
        published=True
        return records
    finally:
        elapsed=perf_counter()-tic
        store.finish_work(reservation,elapsed)
        display.update(signature,units if published else 0,elapsed,job_id=jid)



def _query_outputs(adapter,queries,seeds,settings,stop):
    """Only sanitized queries cross into native inference; truth stays with evaluator.

    A failed deterministic batch is retried item by item to isolate invalid items.
    Stochastic methods are supplied one query at a time by the caller, preserving
    their per-query RNG contract. All retry work remains in ledger wall timings.
    """
    if adapter is None:
        return [dict(error='native_fit_failed') for _ in queries]
    try:
        outputs=adapter.infer_queries(queries,samples=settings['samples'],sample_seeds=seeds,
                                      batch_size=settings['inference_batch_size'])
        if len(outputs)!=len(queries):raise ValueError('Native batch output count mismatch')
        return outputs
    except Exception as exc:
        if stop.requested:raise GracefulStop('Signal during native query batch') from exc
        if len(queries)==1:return [dict(error=type(exc).__name__+': '+str(exc))]
        LOG.exception('Native batch failed; retrying individual queries without changing the matrix')
    outputs=[]
    for query,seed in zip(queries,seeds):
        if stop.requested:raise GracefulStop('Signal between individual batch retries')
        try:
            output=adapter.infer_query(query,samples=settings['samples'],sample_seed=seed)
            output.setdefault('timing',{})['batch_retry_scalar']=True
            outputs.append(output)
        except Exception as exc:
            if stop.requested:raise GracefulStop('Signal during batch retry') from exc
            outputs.append(dict(error=type(exc).__name__+': '+str(exc)))
    return outputs


def _load_selection_scores(run_dir, fit_record, cases):
    """Reuse only evaluator scores tied to the exact saved best checkpoint.

    The cache contains scalar validation losses and episode identities, never
    hidden values or inference inputs. Missing/stale caches use normal scoring.
    """
    relative=fit_record.get('selection_scores_artifact')
    if not relative:return {}
    if not fit_record.get('epoch_validation_covers_all_candidates',True):return {}
    try:
        if Path(relative).is_absolute() or '..' in Path(relative).parts:return {}
        cache=Path(run_dir)/relative
        raw=cache.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=fit_record['selection_scores_sha256']:return {}
        saved=json.loads(raw)
        if saved.get('scope','full_prespecified_validation_episodes')!='full_prespecified_validation_episodes':return {}
        if saved['model_sha256']!=fit_record['sha256'] or saved['best_epoch']!=fit_record['training_metadata']['best_epoch']:
            return {}
        scores={(row['mask_seed'],row['gap_id']):row for row in saved['scores']}
        expected={(manifest.seed,episode.gap_id) for manifest,episode in cases}
        if len(scores)!=len(cases) or set(scores)!=expected:return {}
        if any(not np.isfinite(row['nmae']) for row in scores.values()):return {}
        return scores
    except (OSError,ValueError,KeyError,TypeError):
        return {}

def _conditional_constraint(query,calibration,duration,hypothesis,multiplier=1.):
    meta={'power_semantics':hypothesis}
    con=constraint_from_query(query,calibration,duration,tau_multiplier=multiplier,metadata=meta)
    if hypothesis=='instantaneous':return con
    interval=query.energy_intervals[0] if query.energy_intervals else {}
    imp,exp=interval.get('import_kwh'),interval.get('export_kwh')
    valid=all(v is not None for v in (imp,exp,con.tau_import,con.tau_export))
    return EnergyConstraint(net_kwh=imp-exp if valid else None,
                            tau_net=con.tau_import+con.tau_export if valid else None,
                            semantics=hypothesis,reliable=valid,
                            reason='conditional_net_mean_train_internal_residual_tolerance')


def _restore(query,row):
    x=np.asarray(query.values,float).copy()
    missing=row.get('filled_missing_indices',[])
    if missing:x[np.asarray(missing,int)]=row['filled_missing_values']
    else:x[np.asarray(query.target_gap_mask,bool)]=row['prediction']
    return x


def _forecast_target(series,fold,query,horizon,arrays):
    times,values,observed=arrays;wanted=query.issuance_time.timestamp()+horizon
    position=int(np.searchsorted(times,wanted));candidates=[]
    for j in (position-1,position,position+1):
        if 0<=j<len(times) and observed[j] and abs(times[j]-wanted)<=90 and fold.partition(series.records[j].grid_time)=='test':
            candidates.append(j)
    if not candidates:return None,'no_original_target_within_90s_in_test_fold'
    j=min(candidates,key=lambda k:(abs(times[k]-wanted),times[k],k))
    return (values[j],float(times[j]-wanted)),None


def _record(series,fold,manifest,episode,query,truth,prediction,method,seed,variant,scale,forecasts,arrays,
            latency=0.,status='success',reason=None,output=None,keep_fill=False,diagnostics=None,execution_timing=None,
            forecast_queue=None):
    x=np.asarray(prediction,float).reshape(-1);mask=np.asarray(truth.evaluation_mask,bool)
    metrics=reconstruction_metrics(truth.values,x,mask,scale_train=scale)
    metrics['metric_reason']=metrics.pop('reason',None)
    metrics.update(event_metrics(truth.values,x,query.times,mask))
    forecast_rows={};pending_forecasts=[]
    for horizon,forecaster in forecasts.items():
        target,exclusion=_forecast_target(series,fold,query,int(horizon),arrays) if manifest.partition=='test' else (None,'validation_not_downstream_test')
        if target is None or forecast_queue is not None:
            error=None
            if target is not None:pending_forecasts.append((horizon,target))
        else:error=abs(forecaster.predict(x[-288:],query.issuance_time.timestamp())-target[0])
        forecast_rows[str(horizon)]={'mae_kw':error,'nmae':error/scale if error is not None and scale else None,
            'target_offset_seconds':target[1] if target else None,'exclusion_reason':exclusion}
    row=dict(phase='inference',site=series.site_id,fold=fold.fold_id,partition=manifest.partition,condition=manifest.condition,
             duration=episode.duration_minutes,mask_seed=manifest.seed,train_seed=seed,method=method,variant=variant,
             information_tier='P' if variant=='P' else 'E',task='R',context=512,gap_id=episode.gap_id,
             query_id=query.query_id,issuance_time=query.issuance_time.isoformat(),day=query.issuance_time.date().isoformat(),
             status=status,reason=reason,physical_confirmatory=False,requested_rate=manifest.requested_rate,realized_rate=manifest.realized_rate,
             latency_seconds=latency,prediction=[float(v) for v in x[mask]],n_original_targets=int(mask.sum()),
             forecast_mae_kw=forecast_rows['3600']['mae_kw'],forecast_nmae=forecast_rows['3600']['nmae'],
             forecasts=forecast_rows,**metrics)
    timing=execution_timing if execution_timing is not None else (output or {}).get('timing',{})
    raw_timing_mode=timing.get('timing_mode','single_query')
    timing_mode='single_query' if raw_timing_mode=='query_wall_mc_batched' else raw_timing_mode
    row.update(timing_mode=timing_mode,inference_batch_size=int(timing.get('batch_size',1)),
               inference_amortized_seconds=latency,batch_wall_seconds=timing.get('batch_wall_seconds'))
    for key in ('requested_batch_size','adaptive_batch_size','oom_retries','oom_retry_seconds','oom_retry_timing_policy'):
        if key in timing:row[key]=timing[key]
    if raw_timing_mode=='query_wall_mc_batched':
        row.update(sampling_timing_mode=raw_timing_mode,csdi_sample_batch_size=timing['sample_batch_size'],
                   n_joint_samples=timing['n_joint_samples'])
    if timing_mode!='single_query':
        row['latency_seconds']=None
        row['performance_reason']=timing.get('reason','batched compute cost is amortized throughput; isolated-query latency not measured')
        if timing_mode!='batch_amortized':row['inference_amortized_seconds']=None
    if timing.get('batch_retry_scalar'):
        row['batch_retry_scalar']=True
        row['performance_reason']='individual query retry after failed batch; ledger wall includes failed attempt'
    if keep_fill:
        positions=np.flatnonzero(~np.isfinite(query.values)&np.isfinite(x))
        row.update(filled_missing_indices=positions.tolist(),filled_missing_values=x[positions].tolist())
    if output and output.get('samples') is not None:
        draws=np.asarray(output['samples'],float)
        if draws.ndim==3:draws=draws[:,:,0]
        scores=summarize_samples(draws,truth.values,mask,scale_train=scale)
        scores.pop('point',None);scores.pop('quantiles',None);row['probabilistic']=scores
    if output and output.get('quantiles') is not None:
        row['probabilistic_capability']='marginal_quantiles_not_joint_trajectories'
        levels=np.asarray(output['quantile_levels'],float);quantiles=np.asarray(output['quantiles'],float).squeeze()
        if quantiles.shape==(len(x),len(levels)):quantiles=quantiles.T
        if quantiles.shape!=(len(levels),len(x)):raise ValueError('Unexpected native marginal quantile shape')
        scores=[pinball(q,truth.values,float(level),mask) for level,q in zip(levels,quantiles)]
        intervals={}
        for coverage in (.5,.8,.9,.95):
            lower=np.flatnonzero(np.isclose(levels,(1-coverage)/2));upper=np.flatnonzero(np.isclose(levels,(1+coverage)/2))
            intervals[str(coverage)]=interval_metrics(quantiles[lower[0]],quantiles[upper[0]],truth.values,coverage=coverage,scale_train=scale,mask=mask) if len(lower) and len(upper) else {'coverage':None,'reason':'native_quantile_grid_does_not_include_interval_endpoints'}
        row['probabilistic']={'pinball':{str(float(k)):v for k,v in zip(levels,scores)},'intervals':intervals,
                              'crps_truncated_quantile_approximation':float(2*np.trapezoid(scores,levels)),
                              'crps_kind':'approximate_integrated_pinball_on_native_0.05_to_0.95_grid_tails_not_scored',
                              'joint_samples':False}
    if diagnostics is not None:row['correction_diagnostics']=diagnostics
    if forecast_queue is not None:
        for horizon,target in pending_forecasts:
            forecast_queue.add(horizon,x[-288:],query.issuance_time.timestamp(),target,scale,row)
    return row


def _prepare_fold(series,fold,settings,device,notify=None):
    notify=notify or (lambda message:None)
    train=fold.indices(series,'train');val=fold.indices(series,'validation')
    values=np.asarray(series.values);times=np.asarray(series.times)
    def admissible(indices,cutoff):
        return np.asarray([r.value_canonical if r.observed_original and not r.natural_missing and r.available_to_method
                           and r.measurement_time<cutoff and r.availability_time<cutoff else np.nan
                           for r in (series.records[i] for i in indices)],float)
    train_values=admissible(train,fold.train_end);validation_values=admissible(val,fold.validation_end)
    models={}
    for name in BASELINES:
        notify(f'Preparação {series.site_id}/fold{fold.fold_id}: ajustando baseline {name}.')
        models[name]=create_baseline(name).fit(train_values,times[train])
    scale,reason=training_scale(train_values);forecasts={};selection={}
    for horizon in settings['forecast_horizons_seconds']:
        notify(f'Preparação {series.site_id}/fold{fold.fold_id}: seleção ridge, horizonte {horizon//60} min.')
        alpha,scores=select_alpha(train_values,times[train],validation_values,times[val],device=device,horizon_steps=horizon//300)
        forecasts[horizon]=DirectRidge(alpha=alpha,horizon_steps=horizon//300,device=device).fit(train_values,times[train])
        selection[str(horizon)]={'alpha':alpha,'validation_scores':scores}
    calibrations={}
    for hypothesis in settings['physical_hypotheses']:
        notify(f'Preparação {series.site_id}/fold{fold.fold_id}: calibração exploratória {hypothesis}.')
        conditional=SeriesData(series.records,series.site_id,series.nominal_seconds,{**series.metadata,'power_semantics':hypothesis,'hypothesis_only':True})
        calibrations[hypothesis]=fit_energy_tolerances(conditional,train,cutoff_time=fold.train_end)
    cpi={}
    if settings['physical_hypotheses']:
        for direction in ('import','export'):
            counter=np.asarray([getattr(r,direction+'_counter') if (r.counter_availability_time or r.availability_time)<fold.train_end else np.nan for r in (series.records[i] for i in train)],float)
            cpi[direction]=CPIEnergyDonorAdapted(frequency_seconds=series.nominal_seconds).fit(counter,times[train])
    return dict(models=models,scale=scale,scale_reason=reason,forecasts=forecasts,forecast_selection=selection,
                calibrations=calibrations,cpi=cpi)


def run_conference(root=Path('.'),*,config_path=None,settings=None,device='auto',threads=None,
                   experiment_id=None,resume='auto',progress='tqdm',budget_total_hours=10.,
                   dry_run=False,series_override=None,energy_cpu_workers=None,
                   energy_cpu_max_pending=None,allow_runtime_patch=False,
                   report_workers=None,report_chunk_size=10000,report_bootstrap_chunk_size=1000):
    def startup(message):
        if progress!='none':LOG.info('%s',message)
    startup('Inicialização: carregando contratos dos modelos...')
    from energygap.methods.conference_models import (model_availability,LOCAL_METHODS,FROZEN_METHODS,
        training_windows,hpo_configurations,create_adapter,load_adapter)
    root=Path(root).resolve();cfg=load_settings(root,config_path,settings)
    from energygap.runtime.execution_patch import pool_options, authorize_resume, record_execution
    effective_pool=pool_options(cfg,energy_cpu_workers,energy_cpu_max_pending)
    if allow_runtime_patch and (resume!='require' or not experiment_id or dry_run):
        raise ValueError('An audited runtime patch requires an explicit existing ID and resume=require')
    startup(f'Inicialização: verificando dispositivo {device}...')
    device=resolve_device(device);threads=int(threads or 1);configure_cpu_threads(threads)
    import os
    report_available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    effective_report=dict(workers=min(16,threads,report_available) if report_workers is None else report_workers,
        chunk_size=report_chunk_size,bootstrap_chunk_size=report_bootstrap_chunk_size)
    for name,value in effective_report.items():
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Report execution options must be positive integers: '+name)
    if effective_report['workers']>report_available:
        raise ValueError('Report workers exceed the available CPU affinity')
    if report_chunk_size>100000 or report_bootstrap_chunk_size>10000:
        raise ValueError('Report export batches must be at most 100000 rows; bootstrap publications at most 10000 replicas')
    if budget_total_hours<=0:raise ValueError('Budget must be positive')
    startup('Inicialização: lendo os CSV e conferindo hashes dos dados...')
    series,blocked=prepare_series('conference',root) if series_override is None else (list(series_override),[])
    startup('Inicialização: dados lidos; conferindo fontes, ambientes e hashes dos checkpoints...')
    series=[s for s in series if s.site_id in cfg['sites']]
    missing=set(cfg['sites'])-{s.site_id for s in series}
    if missing:blocked.extend({'site':s,'reason':'dataset_missing_or_invalid'} for s in sorted(missing))
    availability=model_availability(root,device=device,threads=threads)
    for name in BASELINES:availability[name]={'status':'available','available':True,'reason':None}
    unknown=set(cfg['methods'])-set(availability)
    if unknown:raise ValueError('Unknown methods: '+str(sorted(unknown)))
    for name in cfg['require_available']:
        if name not in cfg['methods']:raise ValueError('Required model absent from methods: '+name)
        if availability[name]['status']!='available':blocked.append({'method':name,'reason':availability[name]['reason']})
    config=dict(schema_version=2,profile='conference',scope=cfg.get('scope','explicit_conference_configuration'),
                experiment_settings=cfg,code_hash=code_hash(),execution={'device':device,'threads':threads,'thread_policy':'intraop_requested_interop_1_serial_gpu'},
                sites=cfg['sites'],folds=cfg['folds'],conditions=cfg['conditions'],methods=cfg['methods'],
                train_seeds=cfg['train_seeds'],validation_mask_seeds=cfg['validation_mask_seeds'],test_mask_seeds=cfg['test_mask_seeds'],
                context=512,max_episodes=cfg['max_episodes'],availability=availability,
                physical_hypotheses=cfg['physical_hypotheses'],physical_confirmatory=False,
                physical_semantics_policy='exploratory_conditional_unconfirmed_measurement_semantics',
                datasets={s.site_id:s.metadata for s in series},blocked=blocked,
                budget_total_hours=budget_total_hours,progress=progress,
                bootstrap_attempts=cfg['bootstrap_attempts'],block_days=cfg['block_days'])
    packages={}
    for package in ('numpy','scipy','pandas','scikit-learn','pypots','torch','pyarrow','momentfm','tsicl','transformers','huggingface-hub'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    config['execution'].update(packages=packages,python_version=sys.version.split()[0],
        inference_batch_size=cfg['inference_batch_size'],csdi_sample_batch_size=cfg['csdi_sample_batch_size'],
        inference_policy='deterministic_query_batches_oom_halving_cached_BRITS_CSDI_native_sample_batches_MOMENT_batches_TSICL_serial',
        brits_execution=cfg['brits_execution'],energy_cpu_workers=cfg['energy_cpu_workers'],
        energy_cpu_max_pending=cfg['energy_cpu_max_pending'],energy_worker_policy='spawn_CPU_only_one_BLAS_thread_ordered_episode_tasks',
        validation_policy=('all_prespecified_validation_episodes_every_epoch_best_checkpoint_score_reuse'
                           if cfg['epoch_validation_max_per_stratum'] is None else
                           'fixed_metadata_stratified_panel_every_epoch;full_HPO_final_scoring_and_final_validation_test'),
        epoch_validation_max_per_stratum=cfg['epoch_validation_max_per_stratum'],
        epoch_validation_seed=cfg['epoch_validation_seed'],hpo_trial_indices=cfg['hpo_trial_indices'],
        hpo_max_epochs=cfg['hpo_max_epochs'],hpo_patience=cfg['hpo_patience'],
        batched_timing_policy='amortized_compute_cost_separate_from_isolated_query_latency')
    if device.startswith('cuda:'):config['execution']['torch_runtime']=torch_device_metadata(device)
    if series_override is not None:config['scope']='synthetic_executor_validation_not_real_science'
    identity=fingerprint(config);experiment_id=experiment_id or 'conference-'+identity[:16]
    path=root/'runs'/experiment_id
    if blocked:
        for item in blocked:startup('Pré-checagem bloqueada: '+json.dumps(item,ensure_ascii=False))
        result=dict(status='blocked_preflight',experiment_id=experiment_id,blocked=blocked,availability=availability,
                    reason='Required data/environment/checkpoints are missing; no reduced matrix launched')
        atomic_json(root/'artifacts/reports/server_preflight.json',result);return result
    guard=nullcontext();patch=None
    if allow_runtime_patch:
        config,guard=authorize_resume(path,config)
        patch=guard.audit
        identity=fingerprint(config)
    startup(f'Inicialização: abrindo estado {experiment_id}; conferindo artefatos já existentes, se houver...')
    with guard, StateStore(path,config,resume=resume,budget_seconds=budget_total_hours*3600) as store, EnergyCorrectionPool(
            effective_pool['workers'],effective_pool['max_pending']) as energy_pool:
        if patch is not None:guard.attach(store)
        if store.get_meta('status') in ('completed','completed_with_failures'):
            return store.status()
        atomic_bytes(path/'protocol.lock.yaml',yaml.safe_dump(config,sort_keys=True).encode())
        atomic_json(path/'protocol.lock.json',config);atomic_bytes(path/'protocol.lock.sha256',(identity+'\n').encode())
        store.snapshot_interval_seconds=cfg['status_snapshot_seconds']
        masks={};folds={};fitjobs={};inferjobs={};tunejobs={};hpojobs={};selectjobs={};prep_jobs={};epoch_panels={}
        usable=[n for n in cfg['methods'] if availability[n]['status']=='available']
        local=[n for n in usable if n in LOCAL_METHODS]
        trial_parameters={}
        for name in local:
            catalogue=hpo_configurations(name)
            if any(index>=len(catalogue) for index in cfg['hpo_trial_indices']):
                raise ValueError('Configured HPO trial index absent from the verified catalogue: '+name)
            trial_parameters[name]=[(index,catalogue[index]) for index in cfg['hpo_trial_indices']]
        if progress!='none':LOG.info('Planning: registering method catalogue and jobs in one SQLite transaction')
        # Commit/rollback before exporting progress, so readers never see a half-plan.
        with store.defer_snapshots(),store.transaction():
            for name in cfg['methods']:
                if name not in usable:
                    jid='catalogue:'+name;_job(store,jid,0,'catalogue')
                    store.mark(jid,'unavailable',availability[name]['reason'])
            for series_item in series:
                site=series_item.site_id
                for fold in rolling_origin_splits(series_item):
                    if fold.fold_id not in cfg['folds']:continue
                    f=fold.fold_id;folds[(site,f)]=fold;fk=f'{site}:fold{f}'
                    if progress!='none':
                        LOG.info('Planning masks: %s; %d conditions, %d validation seeds, %d test seeds',
                                 fk,len(cfg['conditions']),len(cfg['validation_mask_seeds']),len(cfg['test_mask_seeds']))
                    atomic_json(path/'manifests'/f'{fk}-split.json',fold.manifest())
                    prep_jobs[(site,f)]=fk+':prepare';_job(store,fk+':prepare',1,'prepare',signature='prepare:ridge_and_baselines')
                    for partition,seeds in [('validation',cfg['validation_mask_seeds']),('test',cfg['test_mask_seeds'])]:
                        for condition in cfg['conditions']:
                            for mask_seed in seeds:
                                manifest=generate_mask(series_item,fold,partition=partition,condition=condition,seed=mask_seed,max_episodes=cfg['max_episodes'],max_gap_positions=510)
                                masks[(site,f,partition,condition,mask_seed)]=manifest
                                atomic_json(path/'manifests'/f'{fk}-{partition}-{condition}-{mask_seed}.json',manifest.to_dict())
                    if progress!='none':LOG.info('Planning masks: %s ready; %d mask manifests generated so far',fk,len(masks))
                    panel=select_epoch_validation_panel(
                        [dict(duration=e.duration_minutes,mask_seed=m.seed,gap_id=e.gap_id)
                         for key,m in masks.items() if key[:3]==(site,f,'validation') and key[3] in ('C3','C4','C5')
                         for e in m.episodes],site=site,fold=f,
                        max_per_stratum=cfg['epoch_validation_max_per_stratum'],seed=cfg['epoch_validation_seed'],
                        mask_seeds=cfg['validation_mask_seeds'])
                    if local and not {15.,60.,180.}<={row['duration'] for row in panel['strata'] if row['candidate_count']}:
                        raise ValueError(f'{fk}: no eligible validation support for every C3/C4/C5 duration; training was not started')
                    panel_relative=Path('manifests')/f'{fk}-epoch-validation-panel.json'
                    panel_digest=atomic_json(path/panel_relative,panel)
                    epoch_panels[(site,f)]=dict(manifest=panel,artifact=str(panel_relative),sha256=panel_digest)
                    if progress!='none':
                        LOG.info('Epoch validation %s: %d/%d episodes; policy=%s; panel=%s; missing_strata=%d',
                            fk,panel['selected_count'],panel['candidate_count'],panel['policy'],panel['panel_sha256'],len(panel['missing_required_strata']))
                if progress!='none':LOG.info('Planning jobs: %s; %d available methods, %d trainable methods',site,len(usable),len(local))
                first=min(cfg['folds'])
                for name in local:
                    candidates=[]
                    hpo_queries=sum(len(m.episodes) for k,m in masks.items() if k[0:3]==(site,first,'validation') and k[3] in ('C3','C4','C5'))
                    for trial in cfg['hpo_trial_indices']:
                        jid=f'hpo:{site}:{name}:{trial}';candidates.append(jid);hpojobs[(site,name,trial)]=jid
                        _job(store,jid,hpo_queries+1,'hpo',[prep_jobs[(site,first)]],signature='hpo:'+name,site=site,method=name,trial=trial,
                             max_epochs=cfg['hpo_max_epochs'],patience=cfg['hpo_patience'],hpo_trial_index=trial)
                    jid=f'select:{site}:{name}';selectjobs[(site,name)]=jid
                    _job(store,jid,1,'selection',candidates,site=site,method=name)
                for f in cfg['folds']:
                    for name in usable:
                        seeds=cfg['train_seeds'] if name in local else [None]
                        validation_jobs=[]
                        tune_id=f'{site}:fold{f}:{name}:correction_selection'
                        for seed in seeds:
                            prefix=f'{site}:fold{f}:{name}:seed{seed}'
                            deps=[prep_jobs[(site,f)]]
                            if name in local:
                                jid=prefix+':fit';fitjobs[(site,f,name,seed)]=jid
                                _job(store,jid,1,'fit',deps+[selectjobs[(site,name)]],signature='fit:'+name,site=site,method=name,fold=f,train_seed=seed,
                                     max_epochs=cfg['max_epochs'],patience=cfg['patience'])
                                deps=[jid]
                            for partition in ('validation','test'):
                                for condition in cfg['conditions']:
                                    for ms in cfg[partition+'_mask_seeds']:
                                        manifest=masks[(site,f,partition,condition,ms)]
                                        jid=prefix+f':{partition}:{condition}:mask{ms}';inferjobs[(site,f,name,seed,partition,condition,ms)]=jid
                                        inference_deps=deps+([tune_id] if partition=='test' and cfg['physical_hypotheses'] else [])
                                        _job(store,jid,len(manifest.episodes),'inference',inference_deps,signature=f'infer:{name}:{partition}:{condition}',site=site,fold=f,method=name,partition=partition,condition=condition,mask_seed=ms,train_seed=seed)
                                        if partition=='validation':validation_jobs.append(jid)
                        if cfg['physical_hypotheses']:
                            tunejobs[(site,f,name)]=tune_id
                            _job(store,tune_id,len(CORRECTION_GRID)*len(cfg['physical_hypotheses']),'correction_selection',validation_jobs,
                                 signature='correction_selection',site=site,fold=f,method=name,seed_loss_aggregation='mean_before_episode_and_duration')
                if progress!='none':LOG.info('Planning jobs: %s ready; %d jobs registered so far',site,store.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0])
            _job(store,'report',1,'report',sorted(inferjobs.values()))
        if progress!='none':LOG.info('Planning committed: %d jobs and %d mask manifests',store.db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],len(masks))
        # Full pre-evaluation immutable plan includes every unavailable catalogue entry.
        planned_jobs=store.status()['jobs']
        plan={'experiment_id':experiment_id,'scope':config['scope'],'execution_supported':True,
              'scientific_complete':False,'physical_confirmatory':False,'availability':availability,
              'jobs':planned_jobs,'mask_manifests':len(masks),
              'epoch_validation_panels':{f'{site}:fold{f}':dict(artifact=p['artifact'],sha256=p['sha256'],
                  panel_sha256=p['manifest']['panel_sha256'],policy=p['manifest']['policy'],
                  candidate_count=p['manifest']['candidate_count'],selected_count=p['manifest']['selected_count'],
                  covers_all_candidates=p['manifest']['covers_all_candidates'],missing_required_strata=p['manifest']['missing_required_strata'])
                  for (site,f),p in epoch_panels.items()},
              'expected_queries':sum(j['total_units'] for j in planned_jobs if j['metadata']['phase']=='inference')}
        import shutil
        core_test=sum(len(m.episodes) for k,m in masks.items() if k[2]=='test' and k[3] in cfg['physical_conditions'])
        variants=sum(len(cfg['train_seeds']) if n in local else 1 for n in usable)
        corrected=core_test*variants*len(cfg['physical_hypotheses'])
        corrected_samples=core_test*len(cfg['train_seeds'])*len(cfg['physical_hypotheses']) if 'CSDI' in local else 0
        plan['resource_summary']={'free_disk_bytes_at_planning':shutil.disk_usage(root).free,
            'p_query_units':plan['expected_queries'],'corrected_point_rows':corrected,
            'corrected_sample_summary_rows':corrected_samples,
            'forecast_long_rows':(plan['expected_queries']+corrected+corrected_samples)*len(cfg['forecast_horizons_seconds']),
            'CPI_track_rows':core_test*2 if 'linear' in usable and cfg['physical_hypotheses'] else 0,
            'duration_estimate_seconds':None,'disk_note':'Uncapped matrix can create many GB; counts are not storage/time estimates'}
        atomic_json(path/'plan.json',plan)
        if dry_run:return {'status':'planned','experiment_id':experiment_id,'total_jobs':len(plan['jobs']),'expected_queries':plan['expected_queries'],'resource_summary':plan['resource_summary'],'availability':availability}
        store.start_session()
        record_execution(store,cfg,effective_pool,code_hash=code_hash(),patch=patch,reporting=effective_report)
        startup(f'Correções CPU: {effective_pool["workers"]} workers; fila {effective_pool["max_pending"]}; lote de inferência {cfg["inference_batch_size"]}.')
        startup(f'Relatórios: {effective_report["workers"]} workers CPU; exportação em lotes de {report_chunk_size}; bootstrap confirmado a cada {report_bootstrap_chunk_size} réplicas.')
        display=GlobalProgress(store,progress,
            refresh_seconds=cfg['progress_refresh_seconds'],persist_seconds=cfg['progress_persist_seconds']);display.update()
        stop=StopRequested();stop.__enter__()
        try:
            for s in series:
                site_jobs=[jid for key,jid in inferjobs.items() if key[0]==s.site_id]
                if site_jobs and all(store.job(jid)['status']=='succeeded' for jid in site_jobs):
                    startup(f'Resultados {s.site_id}: validação e teste já confirmados; seguindo para os trabalhos pendentes.')
                    continue
                site=s.site_id;arrays=(np.asarray(s.times),np.asarray(s.values),np.asarray([r.observed_original and not r.natural_missing for r in s.records]))
                for f in cfg['folds']:
                    fold=folds[(site,f)];prep=prep_jobs[(site,f)];cache=path/'models'/f'{site}-fold{f}-common.pickle'
                    if store.job(prep)['status']!='succeeded':
                        with store.claim(prep) as lease:
                            def prepare():
                                payload=_prepare_fold(s,fold,cfg,device,notify=startup)
                                checksum=atomic_bytes(cache,pickle.dumps(payload,protocol=5))
                                return [dict(key=prep,phase='prepare',artifact=str(cache.relative_to(path)),sha256=checksum)]
                            _finish_unit(store,prep,lease,1,[],prepare,display,'prepare:ridge_and_baselines',stop);store.finish(prep,lease.token)
                    payload=pickle.loads(cache.read_bytes())
                    from energygap.runtime.startup import finish_startup_trace
                    finish_startup_trace()
                    startup(f'Preparação {site}/fold{f} concluída; iniciando os jobs pendentes.')
                    windows=validation_windows=None
                    if local:
                        windows=training_windows(s,fold.indices(s,'train'),context=512,stride=cfg['training_stride'],max_windows=cfg['max_training_windows'],cutoff_time=fold.train_end)
                        validation_windows=training_windows(s,fold.indices(s,'validation'),context=512,stride=cfg['training_stride'],max_windows=cfg['max_training_windows'],cutoff_time=fold.validation_end)
                    panel_record=epoch_panels[(site,f)];panel=panel_record['manifest']
                    selected_epoch_keys=frozenset((row['mask_seed'],row['gap_id']) for row in panel['selected'])
                    def create(name,seed,parameters,*,hpo=False):
                        return create_adapter(name,root=root,device=device,threads=threads,seed=seed,
                                              params=parameters,epochs=cfg['hpo_max_epochs'] if hpo else cfg['max_epochs'],
                                              patience=cfg['hpo_patience'] if hpo else cfg['patience'],batch_size=cfg['batch_size'],
                                              inference_batch_size=cfg['inference_batch_size'],csdi_sample_batch_size=cfg['csdi_sample_batch_size'],
                                              brits_execution=cfg['brits_execution'])
                    def fit_one(jid,name,seed,parameters):
                        artifact=path/'models'/(hashlib.sha256(jid.encode()).hexdigest()+'.pt')
                        is_hpo=jid.startswith('hpo:')
                        fit_epochs=cfg['hpo_max_epochs'] if is_hpo else cfg['max_epochs']
                        fit_patience=cfg['hpo_patience'] if is_hpo else cfg['patience']
                        selection_metadata=dict(training_role='hpo' if is_hpo else 'final',max_epochs=fit_epochs,patience=fit_patience,
                            epoch_validation_policy=panel['policy'],epoch_validation_candidates=panel['candidate_count'],
                            epoch_validation_selected=panel['selected_count'],epoch_validation_panel_sha256=panel['panel_sha256'],
                            epoch_validation_manifest=panel_record['artifact'],epoch_validation_manifest_sha256=panel_record['sha256'],
                            epoch_validation_covers_all_candidates=panel['covers_all_candidates'])
                        if store.job(jid)['completed_units']>=1:
                            result=store.results(jid)[0]
                            return load_adapter(artifact,root=root,device=device,threads=threads) if result.get('status')=='success' else None
                        with store.claim(jid) as lease:
                            fit_active_before=store.get_meta('active_seconds',0.)
                            def compute_fit():
                                tic=perf_counter();active_before=fit_active_before
                                last_heartbeat=tic
                                best_scores={'metric':float('inf'),'epoch':None,'scores':[]}
                                def check_training_boundary():
                                    if stop.requested:raise GracefulStop('Signal during validation selection; unfinished epoch will repeat')
                                    ceiling=store.get_meta('budget_seconds')
                                    if ceiling is not None and active_before+perf_counter()-tic>=ceiling:
                                        raise BudgetExhausted('Budget during validation selection; unfinished epoch will repeat')
                                def heartbeat(force=False):
                                    nonlocal last_heartbeat
                                    now=perf_counter()
                                    if force or now-last_heartbeat>=cfg['heartbeat_seconds']:
                                        store.heartbeat(jid,lease.token);last_heartbeat=now
                                def boundary(info):
                                    display.training({'job_id':jid,**{k:v for k,v in info.items() if k not in ('history','checkpoint_path')}})
                                    heartbeat(force=info.get('stage')=='epoch_complete')
                                    check_training_boundary()
                                def validation_endpoint(model,epoch):
                                    # Evaluator-owned cache: only the sanitized Query crosses
                                    # into inference. Truth stays in this validation callback.
                                    # Reuse across HPO trials/seeds without a test-derived cap.
                                    if '_epoch_selection_cases' not in payload:
                                        prepared=[]
                                        for key,manifest in sorted(masks.items()):
                                            if key[:3]!=(site,f,'validation') or key[3] not in ('C3','C4','C5'):continue
                                            hidden=frozenset(manifest.hidden_indices)
                                            for episode in manifest.episodes:
                                                if (manifest.seed,episode.gap_id) not in selected_epoch_keys:continue
                                                check_training_boundary()
                                                query,truth=build_query(s,episode,artificial_hidden=hidden)
                                                prepared.append((manifest,episode,query,truth))
                                        payload['_epoch_selection_cases']=prepared
                                    cases=payload['_epoch_selection_cases'];losses=defaultdict(list);episode_scores=[]
                                    display.training({'job_id':jid,'stage':'validation','epoch':epoch,
                                        'max_epochs':fit_epochs,'validation_queries_done':0,
                                        'validation_queries_total':len(cases),
                                        'validation_scope':'full' if panel['covers_all_candidates'] else 'fixed_panel',
                                        'validation_full_queries_total':panel['candidate_count']})
                                    batch_size=cfg['inference_batch_size'] if name in BATCHED_METHODS else 1
                                    for first in range(0,len(cases),batch_size):
                                        check_training_boundary()
                                        batch=cases[first:first+batch_size]
                                        display.training({'job_id':jid,'epoch':epoch,'stage':'validation',
                                            'validation_queries_done':first,'validation_queries_total':len(cases),
                                            'inference_batch_index':first//batch_size+1,
                                            'inference_batches_total':(len(cases)+batch_size-1)//batch_size,
                                            'inference_batch_size':len(batch)})
                                        sampling=[stable_seed(identity,'epoch_selection',site,f,name,m.seed,e.gap_id)
                                                  for m,e,q,t in batch]
                                        outputs=_query_outputs(model,[q for m,e,q,t in batch],sampling,cfg,stop)
                                        for (manifest,episode,query,truth),output in zip(batch,outputs):
                                            if 'error' in output:raise RuntimeError(output['error'])
                                            prediction=np.asarray(output['point_estimate'],float).reshape(-1)
                                            score=reconstruction_metrics(truth.values,prediction,truth.evaluation_mask,scale_train=payload['scale'])['nmae']
                                            if score is None or not np.isfinite(score):
                                                raise ValueError('Undefined original-support validation gap nMAE')
                                            losses[float(episode.duration_minutes)].append(float(score))
                                            if is_hpo and panel['covers_all_candidates']:
                                                episode_scores.append(dict(mask_seed=manifest.seed,gap_id=episode.gap_id,nmae=float(score)))
                                        done=first+len(batch)
                                        display.training({'job_id':jid,'epoch':epoch,'max_epochs':fit_epochs,
                                            'stage':'validation','selection':'validation_equal_duration_gap_nmae',
                                            'validation_queries_done':done,'validation_queries_total':len(cases)})
                                        heartbeat();store.snapshot()
                                    if set(losses)!={15.,60.,180.}:
                                        raise ValueError('Validation epoch selection requires every C3/C4/C5 duration stratum')
                                    metric=float(np.mean([np.mean(losses[duration]) for duration in (15.,60.,180.)]))
                                    if episode_scores and metric<best_scores['metric']:
                                        best_scores.update(metric=metric,epoch=epoch,scores=episode_scores)
                                    return metric
                                try:
                                    adapter=create(name,seed,parameters,hpo=is_hpo)
                                    adapter.metadata.update(selection_metadata)
                                    adapter.fit_windows(windows,validation_windows=validation_windows,checkpoint_path=artifact.with_suffix('.epoch.pt'),boundary_callback=boundary,
                                                        selection_callback=validation_endpoint,progress_callback=boundary)
                                    adapter.save_fitted(artifact)
                                    sidecar=Path(str(artifact)+'.sha256')
                                    fitted_digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
                                    record=dict(key=jid,phase='fit',status='success',method=name,site=site,fold=f,train_seed=seed,params=parameters,
                                                 artifact=str(artifact.relative_to(path)),sha256=fitted_digest,
                                                 artifact_sidecar=str(sidecar.relative_to(path)),artifact_sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                                                 training_metadata=adapter.metadata,
                                                 epoch_selection_validation_mask_seeds=cfg['validation_mask_seeds'],**selection_metadata)
                                    # After an interrupted fit the best epoch can precede this
                                    # invocation. In that case score normally rather than using
                                    # scores from a different epoch/model.
                                    if (panel['covers_all_candidates'] and best_scores['epoch']==adapter.best_epoch
                                            and best_scores['metric']==adapter.best_metric):
                                        scores_path=artifact.with_suffix('.selection.json')
                                        try:
                                            digest=atomic_json(scores_path,dict(schema_version=1,model_sha256=fitted_digest,
                                                best_epoch=adapter.best_epoch,scores=best_scores['scores'],
                                                scope='full_prespecified_validation_episodes',
                                                candidate_identity_sha256=panel['candidate_identity_sha256']))
                                        except (OSError,ValueError):
                                            LOG.warning('Optional validation-score cache could not be saved; native HPO scoring will be used: %s',jid,exc_info=True)
                                        else:
                                            record.update(selection_scores_artifact=str(scores_path.relative_to(path)),selection_scores_sha256=digest)
                                    return [record]
                                except (GracefulStop,BudgetExhausted):raise
                                except Exception as exc:
                                    if stop.requested:raise GracefulStop('Signal during native fit; unfinished epoch will repeat') from exc
                                    LOG.exception('Fit failed; operational seasonal fallback retained: %s',jid)
                                    if device.startswith('cuda:'):
                                        import gc,torch
                                        gc.collect();torch.cuda.empty_cache()
                                    return [dict(key=jid,phase='fit',status='fit_failed',method=name,site=site,fold=f,train_seed=seed,params=parameters,
                                                 reason=type(exc).__name__+': '+str(exc),**selection_metadata)]
                            records=_finish_unit(store,jid,lease,1,[],compute_fit,display,'fit:'+name,stop)
                            display.training(None)
                            # HPO jobs continue with validation scoring after the fitted unit.
                            if store.job(jid)['total_units']==1:store.finish(jid,lease.token)
                        return load_adapter(artifact,root=root,device=device,threads=threads) if records[0]['status']=='success' else None

                    if f==min(cfg['folds']):
                        for name in local:
                            for ordinal,(trial,params) in enumerate(trial_parameters[name],1):
                                jid=hpojobs[(site,name,trial)]
                                if store.job(jid)['status']=='succeeded':continue
                                startup(f'HPO {site}/{name}: configuração {ordinal}/{cfg["hpo_trials"]} (índice {trial}); treino e seleção por validação.')
                                adapter=fit_one(jid,name,cfg['train_seeds'][0],params)
                                cases=[(m,e) for k,m in masks.items() if k[:3]==(site,f,'validation') and k[3] in ('C3','C4','C5') for e in m.episodes]
                                if not cases:
                                    del adapter;continue
                                fit_record=next((row for row in store.results(jid) if row.get('phase')=='fit'),{})
                                saved_scores=_load_selection_scores(path,fit_record,cases)
                                with store.claim(jid) as lease:
                                    start=store.job(jid)['completed_units']-1
                                    batch_size=cfg['inference_batch_size'] if name in BATCHED_METHODS else 1
                                    for first in range(start,len(cases),batch_size):
                                        batch=cases[first:first+batch_size]
                                        display.stage('hpo_score:'+name,job_id=jid,
                                            inference_batch_index=first//batch_size+1,
                                            inference_batches_total=(len(cases)+batch_size-1)//batch_size,
                                            inference_batch_size=len(batch))
                                        def score():
                                            if saved_scores:
                                                return [dict(key=f'{jid}:score:{index}',phase='hpo_score',method=name,site=site,fold=f,trial=trial,
                                                    duration=e.duration_minutes,gap_id=e.gap_id,mask_seed=m.seed,
                                                    nmae=saved_scores[(m.seed,e.gap_id)]['nmae'],status='success',reason=None,
                                                    score_source='best_checkpoint_validation_cache',source_model_sha256=fit_record['sha256'])
                                                    for index,(m,e) in enumerate(batch,first)]
                                            prepared=[build_query(s,e,artificial_hidden=frozenset(m.hidden_indices)) for m,e in batch]
                                            sampling=[stable_seed(identity,'epoch_selection',site,f,name,m.seed,e.gap_id) for m,e in batch]
                                            outputs=_query_outputs(adapter,[q for q,t in prepared],sampling,cfg,stop)
                                            records=[]
                                            for index,((m,e),(q,t),out) in enumerate(zip(batch,prepared,outputs),first):
                                                status='success';reason=None;score_value=None
                                                try:
                                                    if 'error' in out:raise RuntimeError(out['error'])
                                                    pred=np.asarray(out['point_estimate']).reshape(-1)
                                                    score_value=reconstruction_metrics(t.values,pred,t.evaluation_mask,scale_train=payload['scale'])['nmae']
                                                except Exception as exc:
                                                    if stop.requested:raise GracefulStop('Signal during HPO scoring') from exc
                                                    status='failed';reason=str(exc)
                                                records.append(dict(key=f'{jid}:score:{index}',phase='hpo_score',method=name,site=site,fold=f,trial=trial,
                                                    duration=e.duration_minutes,gap_id=e.gap_id,mask_seed=m.seed,nmae=score_value,status=status,reason=reason,
                                                    score_source='native_best_checkpoint_inference'))
                                            return records
                                        _finish_unit(store,jid,lease,first+len(batch)+1,[],score,display,'hpo_score:'+name,stop,units=len(batch))
                                    store.finish(jid,lease.token)
                                del adapter
                            selection=selectjobs[(site,name)]
                            if store.job(selection)['status']!='succeeded':
                                def choose():
                                    trials=[]
                                    expected={(m.seed,e.gap_id) for key,m in masks.items()
                                              if key[:3]==(site,f,'validation') and key[3] in ('C3','C4','C5') for e in m.episodes}
                                    for t,params in trial_parameters[name]:
                                        scores=store.results(hpojobs[(site,name,t)]);losses=defaultdict(list);failed=False
                                        scored=[]
                                        for r in scores:
                                            if r.get('phase')=='hpo_score':
                                                scored.append((r['mask_seed'],r['gap_id']))
                                                if r.get('nmae') is None:failed=True
                                                else:losses[r['duration']].append(r['nmae'])
                                        complete=len(scored)==len(expected) and set(scored)==expected
                                        loss=float(np.mean([np.mean(v) for v in losses.values()])) if set(losses)=={15.,60.,180.} and not failed and complete else None
                                        trials.append({'trial':t,'params':params,'validation_equal_duration_nmae':loss,
                                            'selection_scope':'full_prespecified_validation_episodes','expected_queries':len(expected),
                                            'scored_queries':len(scored),'missing_queries':len(expected-set(scored)),
                                            'coverage_complete':complete})
                                    valid=[r for r in trials if r['validation_equal_duration_nmae'] is not None]
                                    best=min(valid,key=lambda r:(r['validation_equal_duration_nmae'],r['trial'])) if valid else trials[0]
                                    return [dict(key=selection,phase='selection',selected_params=best['params'],selected_trial_index=best['trial'],
                                        hpo_trial_indices=cfg['hpo_trial_indices'],trials=trials,
                                        selection_scope='full_prespecified_validation_episodes',
                                        status='selected_validation_only' if valid else 'no_valid_trial_operational_default',selection_fold=f)]
                                with store.claim(selection) as lease:
                                    _finish_unit(store,selection,lease,1,[],choose,display,'selection',stop);store.finish(selection,lease.token)
                    for name in usable:
                        seeds=cfg['train_seeds'] if name in local else [None]
                        for partition in ('validation','test'):
                            if partition=='test' and cfg['physical_hypotheses']:
                                _tune_correction(store,path,s,fold,name,cfg,payload,masks,tunejobs[(site,f,name)],display,stop,energy_pool=energy_pool)
                            selected=store.results(tunejobs[(site,f,name)]) if partition=='test' and cfg['physical_hypotheses'] else []
                            correction={h:min([r for r in selected if r.get('hypothesis')==h],key=lambda r:(r['loss'] is None,r['loss'] or 0,r['trial']))['parameters'] for h in cfg['physical_hypotheses']} if selected else {}
                            for seed in seeds:
                                # A completed validation pass is never inferred again on resume.
                                pending=[inferjobs[(site,f,name,seed,partition,c,ms)] for c in cfg['conditions'] for ms in cfg[partition+'_mask_seeds']]
                                if all(store.job(j)['status']=='succeeded' for j in pending):continue
                                adapter=None
                                if name in local:
                                    params=store.results(selectjobs[(site,name)])[0]['selected_params']
                                    adapter=fit_one(fitjobs[(site,f,name,seed)],name,seed,params)
                                elif name not in BASELINES:adapter=create(name,0,{})
                                for condition in cfg['conditions']:
                                    for ms in cfg[partition+'_mask_seeds']:
                                        jid=inferjobs[(site,f,name,seed,partition,condition,ms)]
                                        manifest=masks[(site,f,partition,condition,ms)]
                                        if store.job(jid)['status']=='succeeded':continue
                                        _infer(store,jid,s,fold,manifest,name,seed,adapter,payload,cfg,identity,arrays,correction,display,stop,energy_pool=energy_pool)
                                del adapter
                                if device.startswith('cuda:'):
                                    import torch
                                    torch.cuda.empty_cache()
                del payload
            from energygap.reporting.conference import generate_conference_report,refresh_conference_status
            from energygap.runtime.startup import finish_startup_trace
            finish_startup_trace()
            if store.job('report')['status']!='succeeded':
                display.training(None);display.stage('report',job_id='report')
                with store.claim('report') as lease:
                    reservation=store.begin_work('report',lease.token);started=perf_counter();published=False
                    active_before=store.get_meta('active_seconds',0.)
                    try:
                        with store.child_work_budget(reservation):
                            generate_conference_report(path,bootstrap_attempts=cfg['bootstrap_attempts'],block_days=cfg['block_days'],store=store,progress=progress,progress_observer=display.update,stop=stop,
                                report_workers=effective_report['workers'],report_chunk_size=report_chunk_size,
                                bootstrap_chunk_size=report_bootstrap_chunk_size)
                        from energygap.reporting.conference_figures import generate_conference_figures
                        display.update('report',job_id='report',kind='report',report_info=dict(
                            stage='figures',label='Gerando figuras',done=0,total=None,unit='figura',status='running'))
                        figures=generate_conference_figures(path)
                        manifest=path/'reports/report_complete.json'
                        checksum=atomic_json(manifest,{'fingerprint':identity,'figures':figures,'tables':'reports/','physical_confirmatory':False})
                        records=[dict(key='report',phase='report',status='success',artifact=str(manifest.relative_to(path)),sha256=checksum)]
                        report_files=[file for file in sorted((path/'reports').rglob('*'))
                            if file.is_file() and file.name not in ('summary.json','report_complete.json')]
                        hash_total=sum(file.stat().st_size for file in report_files);hashed=0;hash_tick=perf_counter()
                        display.update('report',job_id='report',kind='report',report_info=dict(
                            stage='output_checksums',label='Conferindo arquivos finais',done=0,total=hash_total,unit='bytes',status='running'))
                        for file in report_files:
                            digest=hashlib.sha256()
                            with file.open('rb') as stream:
                                for chunk in iter(lambda:stream.read(1024*1024),b''):
                                    digest.update(chunk);hashed+=len(chunk)
                                    if stop.requested:raise GracefulStop('Signal during final report integrity checks; scientific results retained')
                                    if perf_counter()-hash_tick>=cfg['progress_refresh_seconds']:
                                        display.update('report',job_id='report',kind='report',report_info=dict(
                                            stage='output_checksums',label='Conferindo arquivos finais',done=hashed,total=hash_total,unit='bytes',status='running'))
                                        hash_tick=perf_counter()
                            records.append(dict(key='report:artifact:'+str(file.relative_to(path)),phase='report_artifact',artifact=str(file.relative_to(path)),sha256=digest.hexdigest()))
                        display.update('report',job_id='report',kind='report',report_info=dict(
                            stage='output_checksums',label='Conferindo arquivos finais',done=hashed,total=hash_total,unit='bytes',status='complete'))
                        store.publish('report',lease.token,records,1)
                        published=True
                    finally:
                        # Child bootstrap stages account their own active work.
                        # Charge only export/aggregation/plot overhead to this job.
                        child_work=max(0.,store.get_meta('active_seconds',0.)-active_before)
                        overhead=max(0.,perf_counter()-started-child_work)
                        store.finish_work(reservation,overhead)
                        display.update('report',1 if published else 0,overhead,job_id='report')
                    store.finish('report',lease.token)
            store.complete();store.snapshot(force=True);display.update()
            refresh_conference_status(path,store=store)
        except (BudgetExhausted,GracefulStop) as exc:
            store.end_session('paused_budget' if isinstance(exc,BudgetExhausted) else 'interrupted');store.snapshot(force=True)
        except BaseException:
            store.end_session('interrupted');store.snapshot(force=True);raise
        finally:stop.__exit__(None,None,None);display.close()
        return store.status()


def _tune_correction(store,path,series,fold,name,cfg,payload,masks,jid,display,stop,*,energy_pool=None):
    if store.job(jid)['status']=='succeeded':return
    rows=[]
    # Per-job queries avoid scanning every result in the growing experiment.
    prefix=f'{series.site_id}:fold{fold.fold_id}:{name}:seed%:validation:'
    for job in store.db.execute('SELECT job_id FROM jobs WHERE job_id LIKE ?',(prefix+'%',)):
        rows.extend(r for r in store.results(job[0]) if r.get('phase')=='inference' and r['variant']=='P' and r['condition'] in cfg['physical_conditions'])
    lookup={(m.condition,m.seed,e.gap_id):(m,e) for k,m in masks.items() if k[:3]==(series.site_id,fold.fold_id,'validation') for e in m.episodes}
    cases=[]
    for row in rows:
        manifest,episode=lookup[(row['condition'],row['mask_seed'],row['gap_id'])]
        query,truth=build_query(series,episode,information_tier='E',artificial_hidden=frozenset(manifest.hidden_indices))
        base=_restore(query,row);seasonal=payload['models']['seasonal'].impute(query.values,query.times)
        cases.append((query,truth,row['duration'],base,seasonal,row['gap_id'],row['mask_seed']))
    trials=[(h,i,p) for h in cfg['physical_hypotheses'] for i,p in enumerate(CORRECTION_GRID)]
    disabled_scores={}
    with store.claim(jid) as lease:
        for index in range(store.job(jid)['completed_units'],len(trials)):
            hypothesis,trial,parameters=trials[index]
            trial_started=perf_counter();trial_active_before=store.get_meta('active_seconds',0.)
            trial_budget=store.get_meta('budget_seconds');last_heartbeat=last_snapshot=trial_started
            def compute():
                # All disabled candidates are the same base identity. The tau
                # multiplier and gamma cannot affect its values or reliability.
                # Keep every declared trial/result while evaluating it once per
                # hypothesis, including the same failure accounting.
                if parameters['lambda_energy']==0 and hypothesis in disabled_scores:
                    return [dict(key=f'{jid}:{hypothesis}:{trial}',phase='correction_selection',hypothesis=hypothesis,trial=trial,
                        parameters=parameters,**disabled_scores[hypothesis],selection_partition='validation',physical_confirmatory=False)]
                seed_losses=defaultdict(list);failures=0
                def tasks():
                    for query,truth,duration,base,seasonal,gap_id,mask_seed in cases:
                        constraint=_conditional_constraint(query,payload['calibrations'][hypothesis],duration,hypothesis,parameters['tau_multiplier'])
                        yield EnergyEpisodeTask(base,query.times,query.target_gap_mask,seasonal,payload['scale'],
                                                ((hypothesis,constraint,parameters),))
                pool=energy_pool or EnergyCorrectionPool()
                def check_correction():
                    nonlocal last_heartbeat,last_snapshot
                    if stop.requested:raise EnergyWorkCancelled('Signal during validation correction')
                    now=perf_counter()
                    if trial_budget is not None and trial_active_before+now-trial_started>=trial_budget:
                        raise BudgetExhausted('Budget during correction grid; unfinished trial will repeat')
                    if now-last_heartbeat>=cfg.get('heartbeat_seconds',15):
                        store.heartbeat(jid,lease.token);last_heartbeat=now
                    if now-last_snapshot>=cfg.get('status_snapshot_seconds',60):
                        store.snapshot();last_snapshot=now
                try:
                    for case,results in zip(cases,pool.map_ordered(tasks(),stop=stop,check=check_correction)):
                        query,truth,duration,base,seasonal,gap_id,mask_seed=case
                        result=results[0]['point']
                        failures+=result.status in ('failed','unavailable','unsupported')
                        loss=reconstruction_metrics(truth.values,result.values,truth.evaluation_mask,scale_train=payload['scale'])['nmae']
                        seed_losses[(duration,gap_id,mask_seed)].append(loss)
                except EnergyWorkCancelled as exc:
                    raise GracefulStop('Signal during validation correction grid; current trial will repeat') from exc
                losses=defaultdict(list)
                for (duration,gap_id,mask_seed),values in seed_losses.items():
                    if all(v is not None for v in values):losses[duration].append(float(np.mean(values)))
                loss=float(np.mean([np.mean(v) for v in losses.values()])) if set(losses)=={15.,60.,180.} else None
                if parameters['lambda_energy']==0:disabled_scores[hypothesis]=dict(loss=loss,failures=failures)
                return [dict(key=f'{jid}:{hypothesis}:{trial}',phase='correction_selection',hypothesis=hypothesis,trial=trial,
                             parameters=parameters,loss=loss,failures=failures,selection_partition='validation',physical_confirmatory=False)]
            _finish_unit(store,jid,lease,index+1,[],compute,display,'correction_selection',stop)
        store.finish(jid,lease.token)


def _infer(store,jid,series,fold,manifest,name,seed,adapter,payload,cfg,identity,arrays,correction,display,stop,*,energy_pool=None):
    from energygap.forecasting.batched_evaluation import ForecastQueue
    hidden=frozenset(manifest.hidden_indices);initial=store.job(jid)['completed_units']
    frozen=name not in BASELINES and seed is None
    batched=name in BATCHED_METHODS and adapter is not None
    shard=cfg['inference_batch_size'] if batched else cfg['frozen_shard_size'] if frozen else cfg['shard_size']
    with store.claim(jid) as lease,Progress(len(manifest.episodes),initial,'none',jid):
        for first in range(initial,len(manifest.episodes),shard):
            if stop.requested:raise GracefulStop('Signal at committed query shard')
            store.check_budget();active_before=store.get_meta('active_seconds',0.)
            reservation=store.begin_work(jid,lease.token);started=perf_counter();records=[];completed=0;published=False
            forecast_queue=ForecastQueue(payload['forecasts'])
            try:
                episodes=manifest.episodes[first:first+shard]
                cases=[build_query(series,e,artificial_hidden=hidden) for e in episodes]
                display.stage(f'infer:{name}:{manifest.partition}:{manifest.condition}',job_id=jid,
                    shard_queries_done=0,shard_queries_total=len(episodes))
                outputs=None
                if batched:
                    display.training({'job_id':jid,'stage':'inference','shard_queries_done':0,
                        'shard_queries_total':len(episodes),'inference_batch_index':first//shard+1,
                        'inference_batches_total':(len(manifest.episodes)+shard-1)//shard,
                        'inference_batch_size':len(episodes)})
                    outputs=_query_outputs(adapter,[q for q,t in cases],
                        [stable_seed(identity,jid,e.gap_id) for e in episodes],cfg,stop)
                contexts=deque()
                physical=manifest.partition=='test' and manifest.condition in cfg['physical_conditions']
                pool=(energy_pool or EnergyCorrectionPool()) if physical and correction else EnergyCorrectionPool()
                last_heartbeat=last_snapshot=perf_counter()
                def check_energy():
                    nonlocal last_heartbeat,last_snapshot
                    if stop.requested:raise EnergyWorkCancelled('Signal during test correction')
                    now=perf_counter()
                    # The current shard may finish and publish its completed
                    # episodes on budget exhaustion, matching the serial path.
                    if now-last_heartbeat>=cfg.get('heartbeat_seconds',15):
                        store.heartbeat(jid,lease.token);last_heartbeat=now
                    if now-last_snapshot>=cfg.get('status_snapshot_seconds',60):
                        store.snapshot();last_snapshot=now
                def prepare_tasks():
                    # Only this producer calls neural models. Spawned CPU workers
                    # overlap older corrections without touching GPU/RNG/truth.
                    for j,(episode,(query,truth)) in enumerate(zip(episodes,cases)):
                        if stop.requested:raise GracefulStop('Signal before native inference')
                        tic=perf_counter();output=None;reason=None;status='success';execution_timing=None;seasonal=None
                        try:
                            if name in BASELINES:prediction=payload['models'][name].impute(query.values,query.times)
                            else:
                                if adapter is None:raise RuntimeError('native_fit_failed')
                                output=outputs[j] if outputs is not None else adapter.infer_query(query,samples=cfg['samples'],sample_seed=stable_seed(identity,jid,episode.gap_id))
                                if 'error' in output:raise RuntimeError(output['error'])
                                prediction=np.asarray(output['point_estimate']).reshape(-1)
                            mask=np.asarray(query.target_gap_mask,bool)
                            if prediction.shape!=(512,) or not np.isfinite(prediction[mask]).all():raise ValueError('Invalid native target output')
                        except Exception as exc:
                            if stop.requested:raise GracefulStop('Signal during native inference') from exc
                            reason=type(exc).__name__+': '+str(exc);status='fallback';output=None
                            if batched:
                                execution_timing=dict(timing_mode='unavailable_after_batch_failure',batch_size=len(episodes),
                                    reason='native batch/retry failed; fallback-loop time does not measure native query latency')
                            seasonal=payload['models']['seasonal'].impute(query.values,query.times);prediction=seasonal
                        latency=perf_counter()-tic
                        if (frozen or batched) and output:latency=float(output.get('timing',{}).get('inference_seconds',latency))
                        eq=None;hypotheses=[]
                        if physical:
                            eq,_=build_query(series,episode,information_tier='E',artificial_hidden=hidden)
                            if seasonal is None:seasonal=payload['models']['seasonal'].impute(query.values,query.times)
                            for hypothesis,parameters in correction.items():
                                con=_conditional_constraint(eq,payload['calibrations'][hypothesis],episode.duration_minutes,hypothesis,parameters['tau_multiplier'])
                                hypotheses.append((hypothesis,con,parameters))
                        contexts.append((episode,query,truth,prediction,output,status,reason,execution_timing,latency,eq))
                        yield EnergyEpisodeTask(prediction,query.times,query.target_gap_mask,seasonal,payload['scale'],
                            tuple(hypotheses),output.get('samples') if output and physical else None)
                mapped=pool.map_ordered(prepare_tasks(),stop=stop,check=check_energy)
                try:
                    for energy_results in mapped:
                        episode,query,truth,prediction,output,status,reason,execution_timing,latency,eq=contexts.popleft()
                        row=_record(series,fold,manifest,episode,query,truth,prediction,name,seed,'P',payload['scale'],payload['forecasts'],arrays,
                                    latency,status,reason,output,keep_fill=manifest.partition=='validation' and manifest.condition in cfg['physical_conditions'],
                                    execution_timing=execution_timing,forecast_queue=forecast_queue)
                        row['key']=f'{jid}:{episode.gap_id}:P';row['sample_seed']=stable_seed(identity,jid,episode.gap_id);records.append(row)
                        for energy in energy_results:
                            hypothesis=energy['hypothesis'];result=energy['point']
                            cpu_timing=(dict(timing_mode='parallel_cpu_service_cost',batch_size=1,
                                reason='CPU correction service cost excludes queue waiting and is not isolated end-to-end latency')
                                if pool.workers>1 else None)
                            erow=_record(series,fold,manifest,episode,eq,truth,result.values,name+'+energy_consistent',seed,'E:'+hypothesis,payload['scale'],payload['forecasts'],arrays,
                                latency+result.diagnostics.get('elapsed_seconds',0),status if status=='fallback' else result.status,
                                reason or result.diagnostics.get('reason'),diagnostics=result.diagnostics,
                                execution_timing=cpu_timing if cpu_timing is not None else output.get('timing',{}) if output else execution_timing,forecast_queue=forecast_queue)
                            if pool.workers>1:
                                erow.update(correction_service_seconds=result.diagnostics.get('elapsed_seconds'),energy_cpu_workers=pool.workers)
                            erow.update(conditional_energy_metrics(eq,truth,result.values,hypothesis))
                            erow['key']=f'{jid}:{episode.gap_id}:E:{hypothesis}';records.append(erow)
                            corrected=energy['samples']
                            if corrected is not None:
                                point=corrected['summary']['point']
                                drow=_record(series,fold,manifest,episode,eq,truth,point,name+'+energy_samples',seed,'E:'+hypothesis,
                                    payload['scale'],payload['forecasts'],arrays,latency+energy['sample_correction_seconds'],status,reason,output={'samples':corrected['samples']},
                                    diagnostics={'parameter_selection':'validation_point_endpoint_shared_correction_grid',
                                                 'coverage_calibration_guaranteed':False,'sample_statuses':corrected['sample_statuses']},
                                    execution_timing=cpu_timing,forecast_queue=forecast_queue)
                                if pool.workers>1:
                                    drow.update(correction_service_seconds=energy['sample_correction_seconds'],energy_cpu_workers=pool.workers)
                                drow.update(conditional_energy_metrics(eq,truth,point,hypothesis))
                                drow['key']=f'{jid}:{episode.gap_id}:E_samples:{hypothesis}'
                                drow['point_estimator']='median_of_transformed_joint_samples';records.append(drow)
                        # Donor selection/evaluation remains coordinator-owned.
                        if physical and name=='linear' and payload.get('cpi'):
                            for cpi in cpi_energy_records(eq,truth,payload['cpi'],payload['scale']):
                                cpi.update(key=f'{jid}:{episode.gap_id}:CPI:{cpi["physical_hypothesis"]}',site=series.site_id,fold=fold.fold_id,
                                           gap_id=episode.gap_id,mask_seed=manifest.seed,partition=manifest.partition,condition=manifest.condition,
                                           duration=episode.duration_minutes,day=query.issuance_time.date().isoformat())
                                records.append(cpi)
                        completed+=1
                        display.training({'job_id':jid,'stage':'inference',
                            'shard_queries_done':completed,'shard_queries_total':len(episodes)})
                        if stop.requested:break
                        ceiling=store.get_meta('budget_seconds')
                        if ceiling is not None and active_before+perf_counter()-started>=ceiling:break
                except EnergyWorkCancelled as exc:
                    raise GracefulStop('Signal during CPU energy correction; uncommitted shard will repeat') from exc
                finally:
                    mapped.close()
                forecast_queue.flush()
                store.publish(jid,lease.token,records,first+completed)
                published=True
            finally:
                elapsed=perf_counter()-started;store.finish_work(reservation,elapsed)
                display.update(f'infer:{name}:{manifest.partition}:{manifest.condition}',completed if published else 0,elapsed,job_id=jid,kind='inference')
            if stop.requested:raise GracefulStop('Signal after committed query')
            store.check_budget()
        store.finish(jid,lease.token);display.update()
