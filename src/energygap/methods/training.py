"""Optimizer-step recovery of the actual PyPOTS SAITS model and original ORT+MIT loss.

Architecture is upstream, not reimplemented. The explicitly named protocol training
loop changes dynamic masking and orchestration, preserving native gradient losses.
CPU and a single CUDA device use the same native float32 losses and training schedule.
Exact recovery is tested on CPU; CUDA recovery tests require a CUDA host.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.metadata
from pathlib import Path
import random
import time
import numpy as np
from energygap.runtime import StateStore,CheckpointStore,Progress,StopRequested
from energygap.runtime.state import atomic_bytes,atomic_json
from energygap.runtime.training import capture_torch_state,restore_torch_state
from energygap.runtime.device import resolve_device,synchronize,torch_device_metadata
from energygap.runtime.threads import configure_torch_threads
from .neural import PyPOTSAdapter


def _mixed_mask(observed, rng):
    """Balanced point/block corruption; no natural missing point becomes supervision."""
    hidden=np.zeros_like(observed,dtype=bool)
    n,l,d=observed.shape
    for sample in range(n):
        if rng.random()<.5:
            hidden[sample]=(rng.random((l,d))<.2)&observed[sample]
        else:
            size=int(rng.choice([s for s in (1,3,12,36) if s<l] or [1]))
            # Half of blocks end just before the reconnection observation.
            end=l-1 if rng.random()<.5 else int(rng.integers(size,l))
            hidden[sample,end-size:end,:]=observed[sample,end-size:end,:]
        if not hidden[sample].any() and observed[sample].sum()>1:
            candidates=np.argwhere(observed[sample]);pos=candidates[int(rng.integers(len(candidates)))]
            hidden[(sample,*pos)]=True
        if (observed[sample]&~hidden[sample]).sum()==0:
            pos=np.argwhere(observed[sample])[0];hidden[(sample,*pos)]=False
    return hidden


def train_saits_steps(windows, run_dir, *, seed=11,max_steps=40,batch_size=2,
                      checkpoint_every_steps=5,checkpoint_every_seconds=60,
                      progress='none',resume='auto',stop_after_steps=None,
                      budget_seconds=3600,split='train',architecture=None,learning_rate=.001,device='cpu'):
    """Train or resume a real SAITS optimizer loop, returning the fitted adapter.

    ``stop_after_steps`` is a recovery-test/controlled interruption boundary, not a
    scientific stopping rule. The model remains in last state on resume; evaluation
    may explicitly load ``best_model``. No HPO or validation-selection claim is made:
    early-stopping diagnostics below use native training loss and are labelled as such.
    ``max_steps`` remains unchanged across recovery.
    """
    import torch
    if split!='train':raise ValueError('training-only input required')
    x=np.asarray(windows,dtype=np.float32)
    if x.ndim!=3 or x.shape[1]<2 or not np.isfinite(x).any() or np.isinf(x).any():raise ValueError('expected finite-support [N,L,D] windows')
    if (np.isfinite(x).sum(axis=(1,2))<2).any():raise ValueError('each training window requires at least two original observations')
    if max_steps<=0 or batch_size<=0 or checkpoint_every_steps<=0:raise ValueError('positive training limits required')
    device=resolve_device(device,allow_missing_torch=False)
    configure_torch_threads(torch);torch.use_deterministic_algorithms(True)
    architecture=architecture or {'n_layers':1,'d_model':16,'n_heads':2,'d_k':8,'d_v':8,'d_ffn':16,'dropout':.1}
    data_hash=hashlib.sha256(x.tobytes()).hexdigest()
    config={'kind':'saits_protocol_training_loop','upstream':'pypots','pypots_version':importlib.metadata.version('pypots'),
            'torch_version':str(torch.__version__),'numpy_version':np.__version__,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'data_sha256':data_hash,'seed':seed,'max_steps':max_steps,'batch_size':batch_size,
            'context':x.shape[1],'channels':x.shape[2],'architecture':architecture,'learning_rate':learning_rate,
            'masking':'balanced_sample_point20_or_block1-3-12-36;half_blocks_before_reconnection',
            'precision':'float32','device':device,'runtime':torch_device_metadata(device),'workers':0,'deterministic_algorithms':True,
            'validation_selection':'not_implemented_in_this_diagnostic_runner'}
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.startswith('cuda:'):torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    adapter=PyPOTSAdapter('SAITS',context=x.shape[1],n_features=x.shape[2],epochs=1,seed=seed,device=device,**architecture)
    native=adapter.model.model
    optimizer=torch.optim.Adam(native.parameters(),lr=learning_rate)
    mean=np.nanmean(x,axis=(0,1));scale=np.nanstd(x,axis=(0,1));scale[scale<1e-8]=1
    adapter.mean,adapter.scale=mean,scale
    normalized=(x-mean)/scale
    observed=np.isfinite(normalized)
    sampler={'permutation':rng.permutation(len(x)),'cursor':0}
    early={'best':float('inf'),'bad_epochs':0,'monitor':'native_training_loss_diagnostic','best_step':None}
    history=[];best_model=None;step=0;epoch=0
    path=Path(run_dir)
    with StateStore(path,config,resume=resume,budget_seconds=budget_seconds) as store:
        store.add_job('saits:train',max_steps,metadata={'phase':'train','variant':'saits_protocol_training_loop'})
        checkpoints=CheckpointStore(path,store.fingerprint,ledger=store)
        loaded=checkpoints.load('saits:train')
        if loaded:
            state,manifest=loaded
            recovered=restore_torch_state(state['torch_state'],native,optimizer,numpy_rng=rng)
            step=recovered['global_step'];epoch=recovered['epoch'];sampler=recovered['sampler'];early=recovered['early_stopping']
            extra=recovered['extra'];history=extra['history'];best_model=extra['best_model']
            adapter.mean=extra['mean'];adapter.scale=extra['scale']
            if not np.array_equal(adapter.mean,mean) or not np.array_equal(adapter.scale,scale):raise ValueError('train statistics differ after resume')
        adapter.supports_resume_training=True
        adapter.checkpoint_granularity='optimizer_step_safe_boundary'
        adapter.training_variant='saits_protocol_training_loop'
        initial=step
        if step<store.job('saits:train')['completed_units']:
            store.event('progress_corrected_after_checkpoint_fallback',job_id='saits:train',
                        result_units_confirmed=store.job('saits:train')['completed_units'],
                        optimizer_steps_restored=step,reason='last optimizer checkpoint fallback requires deterministic recomputation')
        if store.job('saits:train')['status']=='succeeded':
            return {'adapter':adapter,'state_dir':str(path),'initial':initial,'executed_steps':0,'completed':True,
                    'checkpoint':loaded[1] if loaded else None,'history':history}
        store.start_session()
        with store.claim('saits:train') as lease,StopRequested() as stop,Progress(max_steps,step,progress,'SAITS optimizer steps') as bar:
            def save():
                io_started=time.monotonic()
                blob=capture_torch_state(native,optimizer,epoch=epoch,global_step=step,
                            sampler_state=sampler,early_stopping=early,
                            trial_state={'trial_id':0,'config':config,'maximum_steps':max_steps},numpy_rng=rng,
                            extra={'history':history,'best_model':best_model,'mean':adapter.mean,'scale':adapter.scale})
                manifest=checkpoints.save('saits:train',{'torch_state':blob},step)
                confirmed=store.job('saits:train')['completed_units']
                if step>confirmed:
                    store.publish('saits:train',lease.token,history[confirmed:step],step)
                store.record_active(time.monotonic()-io_started,'saits:train')
                return manifest
            if step>store.job('saits:train')['completed_units']:save()
            last_checkpoint_time=time.monotonic();executed=0;last_manifest=loaded[1] if loaded else None
            while step<max_steps:
                store.check_budget()
                if sampler['cursor']==len(x):
                    sampler={'permutation':rng.permutation(len(x)),'cursor':0};epoch+=1
                indices=sampler['permutation'][sampler['cursor']:sampler['cursor']+batch_size]
                sampler['cursor']+=len(indices)
                mask=_mixed_mask(observed[indices],rng)
                admissible=observed[indices]&~mask
                original=np.where(observed[indices],normalized[indices],0)
                inputs=[torch.as_tensor(indices,device=device),
                        torch.as_tensor(np.where(admissible,original,0),dtype=torch.float32,device=device),
                        torch.as_tensor(admissible,dtype=torch.float32,device=device),
                        torch.as_tensor(original,dtype=torch.float32,device=device),
                        torch.as_tensor(mask,dtype=torch.float32,device=device)]
                reservation=store.begin_work('saits:train',lease.token);synchronize(device);started=time.monotonic()
                native.train();optimizer.zero_grad(set_to_none=True)
                assembled=adapter.model._assemble_input_for_training(inputs)
                result=native(assembled,calc_criterion=True)
                loss=result['loss'].sum()
                if not torch.isfinite(loss):raise ValueError('nonfinite native SAITS ORT+MIT loss')
                loss.backward();optimizer.step();step+=1;executed+=1
                scalar=float(loss.detach())
                if scalar<early['best']:
                    early['best']=scalar;early['bad_epochs']=0;early['best_step']=step
                    best_model=deepcopy(native.state_dict())
                else:early['bad_epochs']+=1
                history.append({'key':f'saits:train:optimizer_step:{step:08d}','step':step,'native_loss':scalar,
                                'batch_ids':indices.tolist(),'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
                                'n_supervised_hidden':int(mask.sum()),'n_observed_ORT':int(admissible.sum())})
                synchronize(device)
                store.finish_work(reservation,time.monotonic()-started)
                stop_now=stop.requested or (stop_after_steps is not None and step>=stop_after_steps)
                if step%checkpoint_every_steps==0 or time.monotonic()-last_checkpoint_time>=checkpoint_every_seconds or stop_now or step==max_steps:
                    last_manifest=save();last_checkpoint_time=time.monotonic()
                bar.update(step)
                if stop_now and step<max_steps:
                    store.interrupt('saits:train',lease.token,'controlled safe optimizer-step interruption')
                    store.end_session('interrupted')
                    return {'adapter':adapter,'state_dir':str(path),'initial':initial,'executed_steps':executed,
                            'completed':False,'checkpoint':last_manifest,'history':history}
            store.finish('saits:train',lease.token);store.complete()
        atomic_json(path/'training_audit.json',{'variant':'saits_protocol_training_loop','upstream':'pypots',
                    'architecture_and_gradient_loss':'upstream_SAITS_ORT_plus_MIT','completed_steps':step,
                    'initial':initial,'device':device,'precision':'float32','runtime':config['runtime'],'validation_selection':'not_claimed',
                    'resume_scope':'this_runner_only; native PyPOTSAdapter.fit_windows remains nonresumable',
                    'state_path':str(path/'checkpoints'),'last_checkpoint':last_manifest})
        return {'adapter':adapter,'state_dir':str(path),'initial':initial,'executed_steps':executed,
                'completed':True,'checkpoint':last_manifest,'history':history}


def get_resume_capabilities():
    return {'variant':'saits_protocol_training_loop','supports_training_resume':True,
            'validated_device':'cpu','num_workers':0,'checkpoint_granularity':'optimizer_step',
            'upstream_native_fit_resume':False,'gradient_loss':'native_ORT_plus_MIT',
            'validation_selection':'not_implemented_in_diagnostic_runner',
            'supported_devices':['cpu','cuda:0'],'cuda_recovery_validation':'requires_CUDA_host_test',
            'distributed_and_gpu':'single_CUDA_supported; distributed_not_supported'}
