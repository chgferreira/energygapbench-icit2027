#!/usr/bin/env python3
"""Historical TS-ICL 0.2.1 worker for the former isolated audit environment.

The supported server matrix uses native TS-ICL 0.2.0 in cuda_env through
frozen_active.py; see docs/single_cuda_env.md. This preserves the earlier audit.

Inputs are sanitized NPZ arrays, never evaluation truth. Outputs are median points
and marginal quantiles, never joint trajectory samples. No implicit downloads.
The TS-ICL Model is licensed by EDF SA under the TS-ICL Non-Commercial License.
This wrapper is an independent research integration and implies no EDF endorsement.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from energygap.runtime.device import resolve_device,torch_device_metadata,synchronize
from energygap.runtime.threads import configure_cpu_threads,configure_torch_threads,requested_threads

CHECKPOINT_SHA256='a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431'
CHECKPOINT_REVISION='19c94031439fb31f36ce395088ee50a6762d3774'
LEVELS=[.05,.1,.25,.5,.75,.9,.95]


class NativeTSICLWorker:
    def __init__(self,checkpoint,device='cpu',threads=None):
        configure_cpu_threads(threads)
        os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
        import numpy as np
        import torch
        self.device=resolve_device(device,allow_missing_torch=False)
        from tsicl import TSICL
        if importlib.metadata.version('tsicl')!='0.2.1':raise ValueError('audited tsicl==0.2.1 required')
        if str(torch.__version__).split('+')[0]!='2.9.1':raise ValueError('isolated audited torch==2.9.1 required')
        self.checkpoint=Path(checkpoint)
        if not self.checkpoint.is_file():raise FileNotFoundError('verified local TS-ICL checkpoint required; auto-download disabled')
        if hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:raise ValueError('TS-ICL checkpoint SHA256 mismatch')
        license_path=self.checkpoint.parent/'LICENSE'
        if not license_path.is_file() or 'TS-ICL NON-COMMERCIAL LICENSE' not in license_path.read_text():raise ValueError('original TS-ICL noncommercial model license must accompany local checkpoint')
        checkpoint_data=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        if 'imputer' not in checkpoint_data or 'config' not in checkpoint_data:raise ValueError('checkpoint lacks pretrained imputation state; random fallback forbidden')
        self.checkpoint_keys=list(checkpoint_data);del checkpoint_data
        self.threads=requested_threads(threads)
        configure_torch_threads(torch,self.threads);torch.manual_seed(20261031);np.random.seed(20261031)
        if self.device.startswith('cuda'):torch.cuda.manual_seed_all(20261031)
        torch.use_deterministic_algorithms(True)
        synchronize(self.device)
        started=time.monotonic();self.model=TSICL(model_path=self.checkpoint,allow_auto_download=False)
        synchronize(self.device)
        self.load_seconds=time.monotonic()-started
        self.runtime=torch_device_metadata(self.device)
        if self.model.max_context_length<512:raise ValueError('native context limit below required L512; no compression allowed')

    def infer(self,values,*,measurement_times=None,availability_times=None,issuance_time=None,task='R'):
        import numpy as np
        import torch
        if task not in ('R','O'):raise ValueError('this bidirectional native imputation adapter is unsupported for C')
        values=np.asarray(values,dtype=np.float64).copy()
        if values.shape!=(512,) or np.isinf(values).any():raise ValueError('univariate context L512 required; no compression/padding-as-observation')
        if task=='R' and measurement_times is not None:
            times=np.asarray(measurement_times,dtype=np.float64)
            if times.shape!=(512,) or not np.all(np.diff(times)>0):raise ValueError('measurement times must be monotone L512')
            if not np.allclose(np.diff(times),300,rtol=0,atol=1e-6):raise ValueError('native regular 5min indexed core required; irregular extension separate')
            if issuance_time is None or times[-1]>float(issuance_time):raise ValueError('R context extends beyond reconnection issuance')
        if availability_times is not None:
            arrival=np.asarray(availability_times,dtype=np.float64)
            if arrival.shape!=(512,) or issuance_time is None:raise ValueError('availability/issuance contract required')
            values[arrival>float(issuance_time)]=np.nan
        if np.isfinite(values).sum()<2:raise ValueError('insufficient admitted observations')
        observed=np.isfinite(values)
        synchronize(self.device);started=time.monotonic()
        with torch.inference_mode():
            point,quantiles=self.model.impute(inputs=values.copy(),covars=None,batch_size=1,
                     quantile_levels=LEVELS,device=torch.device(self.device),denormalize=True,
                     point_estimator='median',allow_auto_complete=False,replace_by_gt=False,squeeze_output=True)
        point=point.detach().cpu().numpy().astype(np.float64)
        quantiles=quantiles.detach().cpu().numpy().astype(np.float64)
        synchronize(self.device)
        inference_seconds=time.monotonic()-started
        if point.shape!=(512,) or quantiles.shape!=(512,len(LEVELS)):raise ValueError('native output shape mismatch')
        if not np.isfinite(point).all() or not np.isfinite(quantiles).all():raise ValueError('nonfinite native imputation output')
        # Preserve only received inputs. Hidden values are NaN and never restored.
        point=np.where(observed,values,point)
        quantiles=np.where(observed[:,None],values[:,None],quantiles)
        return {'point_estimate':point,'quantiles':quantiles,'quantile_levels':np.asarray(LEVELS),
                'samples':None,'capability':'marginal_quantiles','status':'success',
                'inference_seconds':inference_seconds,'device':self.device,'runtime':self.runtime,
                'quantile_crossings':int(np.sum(np.diff(quantiles,axis=-1)<0)),
                'admitted_observations':int(observed.sum()),'imputed_mask':~observed,
                'primary_estimator':'native_quantile_0.5','local_training':'not_applicable'}


def audit(checkpoint,output,device='cpu'):
    import numpy as np
    import torch
    from tsicl import TSICL
    worker=NativeTSICLWorker(checkpoint,device=device)
    original=np.sin(np.arange(1024)/17)+.2*np.cos(np.arange(1024)/41)
    gap=np.zeros(512,bool);gap[499:511]=True
    hidden=gap.copy();hidden[100:102]=True
    values=original[:512].copy();values[hidden]=np.nan
    times=np.arange(512,dtype=float)*300;issue=times[-1]
    wrapped=worker.infer(values,measurement_times=times,availability_times=times,issuance_time=issue)
    with torch.inference_mode():
        raw_point,raw_q=worker.model.impute(inputs=values.copy(),covars=None,batch_size=1,quantile_levels=LEVELS,
            device=torch.device(worker.device),denormalize=True,point_estimator='median',allow_auto_complete=False,replace_by_gt=False,squeeze_output=True)
    native=raw_point.detach().cpu().numpy().astype(float)
    parity=float(np.max(np.abs(wrapped['point_estimate'][hidden]-native[hidden])))
    poisoned=original.copy();poisoned[512:]=1e12;poisoned[:512][hidden]=-1e12
    sanitized=poisoned[:512].copy();sanitized[hidden]=np.nan
    repeated=worker.infer(sanitized,measurement_times=times,availability_times=times,issuance_time=issue)
    poisoning=float(np.max(np.abs(wrapped['point_estimate']-repeated['point_estimate'])))
    late_values=values.copy();late_values[490]=1e10;arrivals=times.copy();arrivals[490]=issue+100
    late=worker.infer(late_values,measurement_times=times,availability_times=arrivals,issuance_time=issue)
    late_values[490]=-1e10
    late_again=worker.infer(late_values,measurement_times=times,availability_times=arrivals,issuance_time=issue)
    delay=float(np.max(np.abs(late['point_estimate']-late_again['point_estimate'])))
    med=float(np.max(np.abs(wrapped['point_estimate'][hidden]-wrapped['quantiles'][hidden,LEVELS.index(.5)])))
    durations=[]
    for _ in range(3):durations.append(worker.infer(values)['inference_seconds'])
    pipeline=Path(inspect.getfile(TSICL));code=pipeline.read_bytes()
    git_blob=hashlib.sha1(b'blob '+str(len(code)).encode()+b'\0'+code).hexdigest()
    report={'schema_version':1,'verified_at':datetime.now(timezone.utc).isoformat(),'method':'TS-ICL',
        'implementation_status':'isolated_native_imputation_worker','validation_status':'synthetic_L512_native_parity_and_poisoning_passed',
        'source':'https://github.com/EDF-Lab/ts-icl','source_commit':'61d1e891a4aeab4b8aa4833c57480025c225b357','source_tag':'v0.2.1',
        'source_commit_additionally_inspected':'349f3eae4f01f78536b16a6ea53c0837760166ec',
        'installed_pipeline_git_blob':git_blob,'release_pipeline_git_blob':'acd8c53f627e765429ba812d4ce14791c407c14d',
        'release_pipeline_exact_match':git_blob=='acd8c53f627e765429ba812d4ce14791c407c14d','installed_pipeline_sha256':hashlib.sha256(code).hexdigest(),
        'package_version':importlib.metadata.version('tsicl'),'checkpoint':'taharnbl/TS-ICL/tsicl-v1.ckpt',
        'checkpoint_revision':CHECKPOINT_REVISION,'checkpoint_sha256':CHECKPOINT_SHA256,'checkpoint_bytes':Path(checkpoint).stat().st_size,
        'checkpoint_keys':worker.checkpoint_keys,'checkpoint_component_used':'imputer (separate forecaster state not used for imputation)',
        'code_license':'TS-ICL NON-COMMERCIAL LICENSE v1.0','checkpoint_license':'TS-ICL NON-COMMERCIAL LICENSE v1.0',
        'research_terms_source':'https://huggingface.co/taharnbl/TS-ICL/blob/'+CHECKPOINT_REVISION+'/LICENSE',
        'distribution_policy':'external optional environment and local ignored checkpoint; not bundled in MIT release',
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                       'device':worker.device,'runtime':worker.runtime,'threads':worker.threads,'precision':'float32_native','isolation':'.venv-tsicl','lockfile':'env/tsicl.lock'},
        'context':512,'native_max_context_length':worker.model.max_context_length,'native_max_target_length':worker.model.max_target_length,
        'query_contract':{'task':'R','n_features':1,'right_edge_index':511,'target_gap_indices':[499,511],
                          'all_input_missing_count':int(hidden.sum()),'evaluation_count':int(gap.sum()),
                          'future_samples_delivered':0,'calendar_or_auxiliary_covariates':None,'irregular_native':'not_audited'},
        'point_estimator':'median_native_quantile_0.5','probabilistic_capability':'marginal_quantiles','quantile_levels':LEVELS,
        'joint_samples':False,'sample_CRPS':'not_applicable_without_native_samples; pinball-grid CRPS integration would be approximate',
        'energy_correction_of_joint_paths':'unsupported_no_joint_paths','normalization':'native CustomStandardScaler on admitted observed context; unavailable values passed as NaN',
        'observed_exactly_preserved':bool(np.array_equal(wrapped['point_estimate'][~hidden],values[~hidden])),
        'native_parity_max_abs_error_hidden':parity,'future_and_hidden_truth_poisoning_max_abs_error':poisoning,
        'availability_delay_poisoning_max_abs_error':delay,'median_quantile_consistency_error':med,
        'quantile_crossings':wrapped['quantile_crossings'],'hidden_mae_synthetic_kw':float(np.mean(np.abs(wrapped['point_estimate'][gap]-original[:512][gap]))),
        'load_seconds':worker.load_seconds,'first_query_seconds':wrapped['inference_seconds'],'warm_query_seconds':durations,
        'query_p50_seconds':float(np.median(durations)),'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'local_training':'not_applicable','training_seeds':None,'supports_resume':{'training':'not_applicable','inference':'external_runner_query_shards'},
        'pretraining_overlap':'unknown','scientific_benchmark_executed':False,
        'limitations':['Synthetic native parity is not a real H1/H2/I1 benchmark.',
                       'Quantiles are marginal and not correlated sample trajectories or evidence of calibrated coverage.',
                       'Isolated environment differs from main environment because upstream requires torch<2.10.',
                       'Subprocess worker integration is separate from the current restricted pilot runner.',
                       'No joint probabilistic energy correction or exact sample CRPS is claimed.']}
    report['worker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['passed']=report['release_pipeline_exact_match'] and parity==0 and poisoning==0 and delay==0 and med==0 and report['observed_exactly_preserved']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,default=Path('artifacts/checkpoints/tsicl/tsicl-v1.ckpt'))
    parser.add_argument('--audit',action='store_true');parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cpu',help='auto, cpu, cuda or cuda:N; explicit CUDA never falls back to CPU')
    parser.add_argument('--threads',type=int,default=None,help='CPU pool budget; defaults to inherited ENERGYGAP_NUM_THREADS or1')
    parser.add_argument('--device-info',action='store_true',help='Print the selected isolated PyTorch runtime without loading weights')
    args=parser.parse_args()
    try:
        configure_cpu_threads(args.threads)
        if args.device_info:
            device=resolve_device(args.device,allow_missing_torch=False)
            import torch
            torch.use_deterministic_algorithms(True)
            print(json.dumps(torch_device_metadata(device)));return 0
        if args.output is None:parser.error('--output is required unless --device-info is used')
        if args.audit:
            report=audit(args.checkpoint,args.output,device=args.device);print(json.dumps({'status':'success' if report['passed'] else 'failed','report':str(args.output),'passed':report['passed']}));return 0 if report['passed'] else 2
        if args.input is None:parser.error('--input NPZ or --audit required')
        import numpy as np
        worker=NativeTSICLWorker(args.checkpoint,device=args.device)
        with np.load(args.input,allow_pickle=False) as data:
            allowed={'values','measurement_times','availability_times','issuance_time'}
            if set(data.files)!=allowed:raise ValueError('worker requires only admissible values, measurement/availability times and issuance; truth/extra tensors refused')
            result=worker.infer(data['values'],measurement_times=data['measurement_times'] if 'measurement_times' in data else None,
                availability_times=data['availability_times'] if 'availability_times' in data else None,
                issuance_time=float(data['issuance_time']) if 'issuance_time' in data else None)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.output.with_name('.'+args.output.name+'.'+str(os.getpid())+'.tmp')
        with temporary.open('xb') as stream:
            np.savez_compressed(stream,point_estimate=result['point_estimate'],quantiles=result['quantiles'],quantile_levels=result['quantile_levels'],imputed_mask=result['imputed_mask'])
            stream.flush();os.fsync(stream.fileno())
        with np.load(temporary,allow_pickle=False) as check:
            if check['point_estimate'].shape!=(512,) or not np.isfinite(check['point_estimate']).all():raise ValueError('NPZ publication validation failed')
        os.replace(temporary,args.output)
        print(json.dumps({k:v for k,v in result.items() if k not in ('point_estimate','quantiles','quantile_levels','imputed_mask')}));return 0
    except (ImportError,FileNotFoundError,ValueError,RuntimeError) as error:
        print(json.dumps({'status':'unavailable' if isinstance(error,(ImportError,FileNotFoundError)) else 'unsupported','reason':str(error)}));return 2
if __name__=='__main__':raise SystemExit(main())
