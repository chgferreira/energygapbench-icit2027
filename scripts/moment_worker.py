#!/usr/bin/env python3
"""Historical version-1 worker for the former isolated MOMENT 0.1.4 audit.

The supported server matrix uses frozen_active.py in cuda_env; see
docs/single_cuda_env.md. This script preserves the earlier audited protocol.

Input contains only admitted/masked observations. Unknown arrays (including truth)
are rejected. Successful metadata is published last as the completion marker.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from energygap.runtime.device import resolve_device,torch_device_metadata,synchronize
from energygap.runtime.threads import configure_cpu_threads,configure_torch_threads,requested_threads


def _sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.part-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as out:
            json.dump(data,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
        os.replace(temporary,path)
    except BaseException:
        if os.path.exists(temporary):os.unlink(temporary)
        raise


def load_input(input_path,metadata_path):
    import numpy as np
    metadata=json.loads(Path(metadata_path).read_text())
    allowed_meta={'schema_version','task','information_tier','issuance_times','lookahead_seconds'}
    if set(metadata)-allowed_meta:raise ValueError('Unknown metadata fields; truth/targets are never accepted')
    if metadata.get('schema_version')!=1:raise ValueError('Unsupported worker contract version')
    task=metadata.get('task','R')
    if task not in ('R','O'):raise ValueError('Native bidirectional MOMENT worker supports audited R/O only')
    with np.load(input_path,allow_pickle=False) as archive:
        allowed={'values','times','observed_mask','padding_mask','query_ids','availability_times'}
        if set(archive.files)-allowed:raise ValueError('Unknown input arrays; no truth or evaluator masks allowed')
        required={'values','times','observed_mask','padding_mask','query_ids'}
        if not required<=set(archive.files):raise ValueError('Missing required worker input arrays')
        arrays={name:archive[name].copy() for name in archive.files}
    values=np.asarray(arrays['values'],float);times=np.asarray(arrays['times'],float)
    observed=np.asarray(arrays['observed_mask'],bool);padding=np.asarray(arrays['padding_mask'],bool)
    if values.ndim==1:values=values[None,:];times=times[None,:];observed=observed[None,:];padding=padding[None,:]
    if values.ndim!=2 or values.shape[1]!=512 or any(a.shape!=values.shape for a in (times,observed,padding)):raise ValueError('Expected matching [queries,512] arrays')
    qids=np.asarray(arrays['query_ids'])
    if qids.dtype.kind not in ('U','S') or qids.shape!=(len(values),) or len(set(map(str,qids)))!=len(qids):raise ValueError('query_ids must be unique nonobject strings')
    if not np.isfinite(times).all() or not np.allclose(np.diff(times,axis=1),300,atol=1e-6,rtol=0):raise ValueError('Core MOMENT contract requires a complete 5-minute L=512 time grid')
    if np.any(observed&padding) or not np.isfinite(values[observed]).all():raise ValueError('Padding/nonfinite entries cannot be marked observed')
    issuance=np.asarray(metadata.get('issuance_times',[]),float)
    if issuance.shape!=(len(values),) or not np.isfinite(issuance).all():raise ValueError('One finite issuance time per query is mandatory')
    lookahead=float(metadata.get('lookahead_seconds',0))
    if task=='R' and lookahead!=0 or task=='O' and lookahead<=0:raise ValueError('Offline look-ahead must be explicit and positive only for O')
    cutoff=issuance+lookahead
    if np.any(observed&(times>cutoff[:,None])):raise ValueError('Observed input extends beyond admissible issuance/look-ahead')
    if 'availability_times' in arrays:
        arrival=np.asarray(arrays['availability_times'],float)
        if arrival.shape!=values.shape or np.any(observed&(~np.isfinite(arrival)|(arrival>cutoff[:,None]))):raise ValueError('An observation has not arrived by its admissible cutoff')
    # All unavailable placeholders are removed before any imported model code.
    safe=np.where(observed&~padding,values,np.nan)
    return metadata,dict(values=safe,times=times,observed_mask=observed,padding_mask=padding,query_ids=qids.astype(str))


def check_environment():
    required={'momentfm':'0.1.4','numpy':'1.25.2','huggingface-hub':'0.24.0','transformers':'4.33.3','torch':'2.5.1'}
    actual={name:importlib.metadata.version(name) for name in required}
    if sys.version_info[:2]!=(3,11) or any(actual[name].split('+')[0]!=version for name,version in required.items()):
        raise RuntimeError('Incompatible MOMENT worker environment; use Python3.11 and environments/moment.lock.txt: '+str(actual))
    return actual


def run(args):
    configure_cpu_threads(args.threads)
    import numpy as np
    metadata,arrays=load_input(args.input,args.metadata)
    input_sha=_sha(args.input);metadata_sha=_sha(args.metadata)
    output=Path(args.output);result_meta=Path(args.result_metadata)
    device=resolve_device(getattr(args,'device','cpu'),allow_missing_torch=False)
    # CPU completion checks stay independent of model/Torch imports. CUDA/auto
    # validate the active isolated runtime before accepting a completed shard.
    if device.startswith('cuda'):
        import torch
        torch.use_deterministic_algorithms(True)
    runtime=({'device':'cpu','backend':'pytorch','torch_version':importlib.metadata.version('torch')}
             if device=='cpu' else torch_device_metadata(device))
    identity={'input_sha256':input_sha,'input_metadata_sha256':metadata_sha,'threads':args.threads,'checkpoint_revision':'5e44b0ea26376a176360f87831124e018f876d96','precision':'float32','worker_sha256':_sha(__file__),'device':device,'runtime':runtime,'device_helper_sha256':_sha(ROOT/'src/energygap/runtime/device.py')}
    if output.exists() or result_meta.exists():
        if output.exists() and result_meta.exists():
            prior=json.loads(result_meta.read_text())
            if prior.get('input_identity')==identity and prior.get('output_sha256')==_sha(output) and prior.get('status')=='success':
                print(json.dumps({'status':'already_complete','queries':len(arrays['values'])}));return 0
        raise FileExistsError('Worker output already exists or is partial/incompatible; choose a new shard path')
    check_environment()
    cache_root=str(Path(args.checkpoint_cache).resolve())
    os.environ['HF_HOME']=str(Path(cache_root).parent);os.environ['HF_HUB_CACHE']=cache_root;os.environ['TRANSFORMERS_CACHE']=cache_root
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from energygap.methods.foundation import MOMENTFrozen,prepare_moment_input,MOMENT_CHECKPOINT,MOMENT_REVISION,MOMENT_SHA256
    configure_torch_threads(torch,args.threads)
    torch.use_deterministic_algorithms(True)
    model=MOMENTFrozen(cache_dir=args.checkpoint_cache,device=device,allow_network=False)
    synchronize(device);started=perf_counter();model._load();synchronize(device);load_seconds=perf_counter()-started
    predictions=[];latencies=[];parity=[]
    for values,times,observed,padding in zip(arrays['values'],arrays['times'],arrays['observed_mask'],arrays['padding_mask']):
        synchronize(device);start=perf_counter();pred=model.impute(values,times,task=metadata.get('task','R'),observed_mask=observed,padding_mask=padding);synchronize(device);latencies.append(perf_counter()-start)
        if not np.array_equal(pred[observed],values[observed]):raise RuntimeError('Observed preservation failed')
        if args.audit_native:
            batch=prepare_moment_input(values,observed,padding)
            with torch.inference_mode():native=model.model.reconstruction(**{k:torch.as_tensor(v,device=device) for k,v in batch.items()}).reconstruction[0,0].cpu().numpy()
            score=~observed&~padding
            parity.append(float(np.max(np.abs(pred[score]-native[score]))) if np.any(score) else 0.)
        predictions.append(pred)
    predictions=np.asarray(predictions)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix=output.name+'.part-',dir=output.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez_compressed(stream,point_estimate=predictions,imputed_mask=~arrays['observed_mask']&~arrays['padding_mask'],query_ids=arrays['query_ids']);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,output)
    except BaseException:
        if os.path.exists(temporary):os.unlink(temporary)
        raise
    versions={name:importlib.metadata.version(name) for name in ('momentfm','numpy','huggingface-hub','transformers','torch')}
    result={'schema_version':1,'status':'success','method':'MOMENT-1-base','variant':'frozen_native_reconstruction_isolated','input_identity':identity,'output_sha256':_sha(output),'queries':len(predictions),'checkpoint':MOMENT_CHECKPOINT,'checkpoint_revision':MOMENT_REVISION,'checkpoint_sha256':MOMENT_SHA256,'environment':{'python':sys.version,'executable':sys.executable,'packages':versions,'device':device,'runtime':torch_device_metadata(device),'precision':'float32','threads':args.threads,'dependency_override':False},'load_seconds':load_seconds,'inference_seconds':latencies,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'platform':platform.platform(),'native_parity_max_abs_error':max(parity) if parity else None,'observed_preserved':True,'normalization':'masked official RevIN; missing placeholders removed before model','resume_capabilities':{'training':'not_applicable','inference':'whole_query_shard_commit; no training state'},'task':metadata.get('task','R'),'information_consumed':'P'}
    _atomic_json(result_meta,result)
    print(json.dumps({'status':'success','queries':len(predictions),'native_parity_max_abs_error':result['native_parity_max_abs_error']}))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','metadata','output','result-metadata','checkpoint-cache'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--device',default='cpu',help='auto, cpu, cuda or cuda:N; explicit CUDA never falls back to CPU')
    parser.add_argument('--device-info',action='store_true',help='Print the selected isolated PyTorch runtime without loading weights')
    parser.add_argument('--threads',type=int,default=requested_threads());parser.add_argument('--audit-native',action='store_true')
    args=parser.parse_args(argv)
    if args.threads<1:parser.error('--threads must be positive')
    try:
        configure_cpu_threads(args.threads)
        if args.device_info:
            device=resolve_device(args.device,allow_missing_torch=False)
            import torch
            torch.use_deterministic_algorithms(True)
            print(json.dumps(torch_device_metadata(device)));return 0
        if any(getattr(args,name) is None for name in ('input','metadata','output','result_metadata','checkpoint_cache')):parser.error('input, metadata, output, result-metadata and checkpoint-cache are required')
        return run(args)
    except Exception as error:
        print(json.dumps({'status':'failed','reason':type(error).__name__+': '+str(error)}),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
