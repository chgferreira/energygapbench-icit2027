"""CLI commands are usable without importing torch or starting models."""
from __future__ import annotations
import argparse
import importlib.util
import json
import logging
from pathlib import Path
import sys


def output(value):
    print(json.dumps(value,indent=2,default=str,allow_nan=False))


def main(argv=None):
    parser=argparse.ArgumentParser(prog='energygap')
    parser.add_argument('--root',type=Path,default=Path('.'),help='repository root')
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('doctor','gpu-smoke','inspect-data','validate-config','prepare','make-splits','make-masks','plan','run','status','resume','evaluate','forecast-evaluate','statistics','report','reproduce','recovery-demo'):
        p=sub.add_parser(cmd)
        p.add_argument('--profile',choices=['smoke','pilot','conference','extended'],default='smoke' if cmd not in ('plan',) else 'conference')
        p.add_argument('--experiment-id')
        p.add_argument('--resume',choices=['auto','never','require'],default='auto')
        p.add_argument('--progress',choices=['tqdm','log','none'],default='log')
        p.add_argument('--budget-total-hours',type=float,default=None)
        p.add_argument('--max-episodes',type=int,default=2)
        p.add_argument('--checkpoint-every-steps',type=int,default=50)
        p.add_argument('--checkpoint-every-seconds',type=float,default=60)
        p.add_argument('--retry-failed',action='store_true')
        p.add_argument('--dry-run',action='store_true')
        p.add_argument('--json',action='store_true')
        p.add_argument('--watch',action='store_true')
        p.add_argument('--threads',type=int,default=None,help='CPU threads per process (e.g. 32); GPU jobs remain serial')
        p.add_argument('--config',type=Path,default=None,help='Explicit scientific YAML settings for the conference executor')
        if cmd in ('run','resume'):
            p.add_argument('--energy-cpu-workers',type=int,default=None,help='Operational CPU correction workers; model training/inference unchanged')
            p.add_argument('--energy-cpu-max-pending',type=int,default=None,help='Bounded pending CPU correction tasks')
            p.add_argument('--allow-runtime-patch',action='store_true',help='Resume only an exact audited predecessor release, retaining scientific identity')
            p.add_argument('--report-workers',type=int,default=None,help='CPU bootstrap workers (default: at most 16, bounded by requested threads and CPU affinity)')
            p.add_argument('--report-chunk-size',type=int,default=10000,help='Bounded rows per report export batch; no scientific sampling cap')
            p.add_argument('--report-bootstrap-chunk-size',type=int,default=1000,help='Bootstrap replicas per durable ordered publication')
        if cmd in ('doctor','gpu-smoke','run','resume','reproduce','plan'):
            p.add_argument('--device',default=None,help='auto, cpu, cuda, or cuda:N; cuda:0 selects the single server GPU')
        if cmd=='inspect-data':p.add_argument('--root',dest='data_root',type=Path,default=Path('raw_data'))
        if cmd=='recovery-demo':p.add_argument('--output',type=Path,default=None)
    args=parser.parse_args(argv);root=args.root.resolve()
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    try:
        from energygap.runtime.threads import configure_cpu_threads
        thread_configuration=configure_cpu_threads(args.threads)
        if args.command in ('run','resume','reproduce'):
            from energygap.runtime.startup import begin_startup_trace
            begin_startup_trace()
            if args.progress!='none':
                logging.getLogger('energygap.startup').info('EnergyGap: comando=%s Python=%s projeto=%s',args.command,sys.executable,root)
        if args.command=='gpu-smoke':
            from energygap.runtime.gpu_smoke import run_gpu_smoke
            output(run_gpu_smoke(args.device or 'cuda:0'));return 0
        if args.command=='doctor':
            import platform,importlib.metadata
            packages={name:importlib.metadata.version(name) if importlib.util.find_spec(module) else None for name,module in [('numpy','numpy'),('pandas','pandas'),('scipy','scipy'),('PyYAML','yaml'),('pytest','pytest'),('tqdm','tqdm'),('torch','torch'),('pypots','pypots')]}
            result=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=packages,local_filesystem_only=True,threads=thread_configuration)
            if args.device is not None:
                from energygap.runtime.device import resolve_device,torch_device_metadata,cuda_probe
                selected=resolve_device(args.device)
                result['execution']=cuda_probe(selected) if selected.startswith('cuda:') else torch_device_metadata(selected)
            if thread_configuration.get('requested') is not None and 'torch' in sys.modules:
                from energygap.runtime.threads import configure_torch_threads
                result['threads']['torch']=configure_torch_threads(sys.modules['torch'],thread_configuration['requested'])
            output(result);return 0
        if args.command=='inspect-data':
            from energygap.data.audit import inspect_data
            result=inspect_data(args.data_root,root/'artifacts/reports',root/'configs/datasets')
            output(result);return 0
        if args.command=='validate-config':
            from energygap.data import DatasetConfig
            from energygap.methods.registry import load_registry
            results={site:DatasetConfig.load(root/'configs/datasets'/f'{site}.yaml').unresolved() for site in ('H1','H2','I1')}
            load_registry(root)
            output(dict(valid=not any(results.values()),dataset_unresolved=results));return 2 if any(results.values()) else 0
        if (args.command=='plan' or args.dry_run) and args.profile=='conference' and args.command!='resume' and (args.command!='plan' or args.config is not None):
            from energygap.conference import run_conference
            result=run_conference(root,config_path=args.config,device=args.device or 'auto',threads=args.threads or 1,
                experiment_id=args.experiment_id,resume=args.resume,progress='none',budget_total_hours=args.budget_total_hours or 10.,dry_run=True)
            output(result);return 2 if result.get('status')=='blocked_preflight' else 0
        if args.command=='plan' or (args.dry_run and args.command not in ('resume',)):
            from energygap.runtime.planner import plan
            from energygap.runtime.state import atomic_json
            result=plan(root,args.profile,args.budget_total_hours or 1.)
            atomic_json(root/'artifacts/reports'/f'{args.profile}_plan.json',result)
            output({k:v for k,v in result.items() if k!='jobs'});return 0
        if args.command=='status':
            if not args.experiment_id:raise ValueError('--experiment-id required')
            from energygap.runtime.state import StateStore
            import time
            while True:
                with StateStore(root/'runs'/args.experiment_id,resume='require',reconcile_on_open=False) as store:
                    status=store.status()
                    output(status if args.json else {k:v for k,v in status.items() if k not in ('jobs','sessions')})
                if not args.watch:break
                time.sleep(1)
            return 0
        if args.command=='recovery-demo':
            from energygap.runtime.demo import demonstrate
            from datetime import datetime,timezone
            directory=args.output or root/'runs'/('recovery-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
            output(demonstrate(directory));return 0
        if args.command in ('run','reproduce','resume'):
            if args.progress!='none':logging.getLogger('energygap.startup').info('Inicialização: importando executor e bibliotecas científicas...')
            from energygap.pipeline import run_pipeline
            params={};cfg=None;saved_budget=None
            if args.command=='resume':
                if not args.experiment_id:raise ValueError('--experiment-id required')
                from energygap.runtime.state import StateStore
                with StateStore(root/'runs'/args.experiment_id,resume='require',reconcile_on_open=False) as store:
                    cfg=store.get_meta('config');params=dict(profile=cfg['profile'],max_episodes=cfg['max_episodes'],device=args.device or cfg.get('execution',{}).get('device','cpu'))
                    if args.budget_total_hours is not None:store.extend_budget(args.budget_total_hours*3600)
                    saved_budget=store.get_meta('budget_seconds')
                args.resume='require'
            else:params=dict(profile=args.profile,max_episodes=args.max_episodes,device=args.device or 'auto')
            if params['profile']!='conference' and (getattr(args,'energy_cpu_workers',None) is not None
                    or getattr(args,'energy_cpu_max_pending',None) is not None or getattr(args,'allow_runtime_patch',False)
                    or getattr(args,'report_workers',None) is not None
                    or getattr(args,'report_chunk_size',10000)!=10000
                    or getattr(args,'report_bootstrap_chunk_size',1000)!=1000):
                raise ValueError('CPU execution overrides/runtime patches require the conference executor')
            if params['profile']=='conference':
                from energygap.conference import run_conference
                if args.command=='resume' and not cfg.get('experiment_settings'):
                    raise ValueError('This state predates the conference executor; start a new conference experiment')
                result=run_conference(root=root,config_path=args.config,
                    settings=cfg['experiment_settings'] if cfg else None,
                    device=params['device'],threads=args.threads or (cfg.get('execution',{}).get('threads',1) if cfg else 1),
                    experiment_id=args.experiment_id,resume=args.resume,progress=args.progress,
                    budget_total_hours=args.budget_total_hours or (saved_budget/3600 if saved_budget else 10.),dry_run=args.dry_run,
                    energy_cpu_workers=getattr(args,'energy_cpu_workers',None),
                    energy_cpu_max_pending=getattr(args,'energy_cpu_max_pending',None),
                    allow_runtime_patch=getattr(args,'allow_runtime_patch',False),
                    report_workers=getattr(args,'report_workers',None),
                    report_chunk_size=getattr(args,'report_chunk_size',10000),
                    report_bootstrap_chunk_size=getattr(args,'report_bootstrap_chunk_size',1000))
                output({k:v for k,v in result.items() if k not in ('jobs','sessions')});return 2 if result.get('status')=='blocked_preflight' else 0
            if params['profile']=='extended':
                from energygap.runtime.planner import plan
                from energygap.runtime.state import atomic_json
                result=plan(root,params['profile'],args.budget_total_hours or 1.)
                atomic_json(root/'artifacts/reports'/f'{params["profile"]}_plan.json',result)
                output(dict(status='blocked',reason='Conference execution is incomplete: resolve all data/method/recovery blockers in the explicit plan before scientific execution.',plan=str(root/'artifacts/reports'/f'{params["profile"]}_plan.json')));return 2
            result=run_pipeline(root=root,resume=args.resume,progress=args.progress,budget_total_hours=args.budget_total_hours or .05,experiment_id=args.experiment_id,checkpoint_every_steps=args.checkpoint_every_steps,checkpoint_every_seconds=args.checkpoint_every_seconds,retry_failed=args.retry_failed,**params)
            output(result);return 0
        if args.command in ('prepare','make-splits','make-masks'):
            from energygap.pipeline import prepare_series
            from energygap.protocols.splits import rolling_origin_splits
            from energygap.masking.generators import generate_mask
            from energygap.runtime.state import atomic_json
            series,blocked=prepare_series(args.profile,root);counts={}
            for s in series:
                counts[s.site_id]=len(s.records)
                if args.command=='prepare':
                    import pandas as pd
                    from dataclasses import asdict
                    destination=root/'data/processed'/f'{s.site_id}.parquet';destination.parent.mkdir(parents=True,exist_ok=True)
                    pd.DataFrame([asdict(r) for r in s.records]).to_parquet(destination,index=False)
                else:
                    for fold in rolling_origin_splits(s):
                        atomic_json(root/'data/splits'/f'{s.site_id}-fold{fold.fold_id}.json',fold.manifest())
                        if args.command=='make-masks':
                            for condition in ('C1','C2','C3','C4','C5','C6'):
                                m=generate_mask(s,fold,condition=condition,max_episodes=args.max_episodes)
                                m.write(root/'data/masks'/f'{m.mask_id}.json')
            output(dict(prepared=counts,blocked=blocked,synthetic=args.profile=='smoke'));return 2 if blocked and not series else 0
        if args.command in ('report','evaluate','forecast-evaluate','statistics'):
            if not args.experiment_id:raise ValueError('--experiment-id required')
            from energygap.runtime.state import StateStore
            with StateStore(root/'runs'/args.experiment_id,resume='require',reconcile_on_open=False) as store:
                cfg=store.get_meta('config')
            if cfg.get('experiment_settings'):
                from energygap.reporting.conference import generate_conference_report
                output(generate_conference_report(root/'runs'/args.experiment_id,
                    bootstrap_attempts=cfg['experiment_settings']['bootstrap_attempts'],
                    block_days=cfg['experiment_settings']['block_days'],progress=args.progress));return 0
            from energygap.reporting.report import generate_report
            output(generate_report(root/'runs'/args.experiment_id));return 0
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as exc:
        output(dict(status='error',reason=type(exc).__name__+': '+str(exc)));return 2
    finally:
        if args.command in ('run','resume','reproduce'):
            from energygap.runtime.startup import finish_startup_trace
            finish_startup_trace()
    return 0


if __name__=='__main__':raise SystemExit(main())
