"""Optional native frozen adapters. No generic regressor substitutes these models.

MOMENT uses the official masked reconstruction API. FIM targets the temporal block
extension and requires a locally licensed temporal-base + pointwise-base bundle.
FIM's upstream example helper is intentionally not called: it copies target truth.
"""
from __future__ import annotations
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from .base import UnsupportedTask, check_series
from energygap.runtime.device import resolve_device,synchronize,torch_device_metadata

MOMENT_CHECKPOINT='AutonLab/MOMENT-1-base'
MOMENT_REVISION='5e44b0ea26376a176360f87831124e018f876d96'
MOMENT_SHA256='1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825'
FIM_TEMPORAL_CHECKPOINT='FIM4Science/fim-imp-temporal-base'
FIM_TEMPORAL_REVISION='da43de66094b5f2e3813380cb4efc6bf8df9261e'
FIM_POINTWISE_CHECKPOINT='FIM4Science/fim-imp-pointwise-base'
FIM_POINTWISE_REVISION='e00537a9fc695c69e6f22bf8dce3903c3bff3af3'


def prepare_moment_input(values,observed_mask=None,padding_mask=None):
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=1 or len(x)!=512:raise ValueError('MOMENT native core requires L=512; no compression permitted')
    padding=np.zeros(len(x),bool) if padding_mask is None else np.asarray(padding_mask,bool)
    observed=np.isfinite(x) if observed_mask is None else np.asarray(observed_mask,bool)&np.isfinite(x)
    if padding.shape!=x.shape or observed.shape!=x.shape:raise ValueError('Mask shape mismatch')
    observed=observed&~padding
    if observed.sum()<2:raise ValueError('Insufficient admitted normalization support')
    # Placeholder values are zeroed BEFORE the upstream internal RevIN normalizer.
    safe=np.where(observed,x,0.)
    return {'x_enc':safe[None,None,:], 'input_mask':(~padding).astype(np.int64)[None,:], 'mask':observed.astype(np.int64)[None,:]}


class MOMENTFrozen:
    name='MOMENT-1-base'
    def __init__(self,cache_dir='artifacts/checkpoints/hf',device='cpu',allow_network=False):
        self.cache_dir=str(Path(cache_dir).resolve());self.device=resolve_device(device);self.allow_network=allow_network;self.model=None;self.load_seconds=0.

    def fit(self,values,times,*,split='train'):
        if split!='train':raise ValueError('Only train metadata permitted; frozen checkpoint has no local training')
        check_series(values,times)
        self._load();return self

    def _load(self):
        if self.model is not None:return
        from momentfm import MOMENTPipeline
        started=perf_counter()
        self.model=MOMENTPipeline.from_pretrained(MOMENT_CHECKPOINT,revision=MOMENT_REVISION,cache_dir=self.cache_dir,local_files_only=not self.allow_network,model_kwargs={'task_name':'reconstruction'})
        weight=Path(self.cache_dir)/'models--AutonLab--MOMENT-1-base'/'snapshots'/MOMENT_REVISION/'model.safetensors'
        if weight.exists():
            from energygap.data.provenance import sha256_file
            if sha256_file(weight)!=MOMENT_SHA256:
                self.model=None
                raise RuntimeError('MOMENT checkpoint SHA-256 mismatch; refusing scientifically incompatible reuse')
        self.model.init();self.model.to(self.device);self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        synchronize(self.device)
        self.runtime=torch_device_metadata(self.device)
        self.load_seconds=perf_counter()-started

    def impute(self,values,times,*,task='R',observed_mask=None,padding_mask=None):
        if task not in ('R','O'):raise UnsupportedTask('MOMENT bidirectional reconstruction not audited for ongoing task C')
        x,t=check_series(values,times)
        batch=prepare_moment_input(x,observed_mask,padding_mask)
        self._load()
        import torch
        tensor={k:torch.as_tensor(v,device=self.device) for k,v in batch.items()}
        with torch.inference_mode():
            result=self.model.reconstruction(**tensor).reconstruction[0,0].detach().cpu().numpy()
        observed=batch['mask'][0].astype(bool)
        result=np.asarray(result,dtype=float);result[observed]=x[observed]
        if padding_mask is not None:result[np.asarray(padding_mask,bool)]=np.nan
        if not np.isfinite(result[~np.asarray(padding_mask,bool) if padding_mask is not None else np.ones(len(x),bool)]).all():
            raise RuntimeError('MOMENT returned nonfinite nonpadding output')
        return result

    def impute_query(self,query):
        return self.impute(query.values,query.times,task=query.task,observed_mask=query.available_to_method,padding_mask=query.padding_mask)

    def impute_batch(self, values, times, *, task='R', observed_mask=None, padding_mask=None):
        """Reconstruct independent L512 windows in one official FP32 forward.

        RevIN statistics remain per row/channel, as in the scalar upstream API.
        Masks are prepared independently before stacking, including partial
        patches and padding; no context is compressed or shared between rows.
        The scalar method deliberately remains an independent parity reference.
        """
        if task not in ('R', 'O'):
            raise UnsupportedTask('MOMENT bidirectional reconstruction not audited for ongoing task C')
        x, t = np.asarray(values, float), np.asarray(times, float)
        if x.ndim != 2 or x.shape[1] != 512 or not len(x) or t.shape != x.shape:
            raise ValueError('MOMENT batches require matching nonempty [N,512] values/times')
        padding = np.zeros_like(x, dtype=bool) if padding_mask is None else np.asarray(padding_mask, bool)
        observed = np.isfinite(x) if observed_mask is None else np.asarray(observed_mask, bool)
        if padding.shape != x.shape or observed.shape != x.shape:
            raise ValueError('Mask shape mismatch')
        prepared = []
        for row, ticks, admitted, pad in zip(x, t, observed, padding):
            check_series(row, ticks)
            prepared.append(prepare_moment_input(row, admitted, pad))
        batch = {key: np.concatenate([row[key] for row in prepared], axis=0)
                 for key in prepared[0]}
        self._load()
        import torch
        with torch.inference_mode():
            tensor = {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}
            output = self.model.reconstruction(**tensor).reconstruction
            if tuple(output.shape) != (len(x), 1, 512):
                raise ValueError('MOMENT native batch output shape mismatch')
            result = output[:, 0].detach().cpu().numpy().astype(float)
        admitted = batch['mask'].astype(bool)
        result[admitted] = x[admitted]
        result[padding] = np.nan
        if not np.isfinite(result[~padding]).all():
            raise RuntimeError('MOMENT returned nonfinite nonpadding output')
        return result

    def get_resume_capabilities(self):
        return {'training':'not_applicable','inference':'query_shards','optimizer_state':'not_applicable','checkpoint':MOMENT_REVISION,'device':self.device,'precision':'float32','validation':'native_output_parity_test_required_for_environment'}


def prepare_fim_temporal_batch(values,times,target_gap_mask,observed_mask=None,padding_mask=None):
    x,t=check_series(values,times)
    target=np.asarray(target_gap_mask,bool)
    if target.shape!=x.shape or not target.any():raise ValueError('Explicit target episode required')
    indices=np.flatnonzero(target)
    if not np.array_equal(indices,np.arange(indices[0],indices[-1]+1)):raise ValueError('One contiguous target episode required')
    observed=np.isfinite(x) if observed_mask is None else np.asarray(observed_mask,bool)&np.isfinite(x)
    if padding_mask is not None:observed &= ~np.asarray(padding_mask,bool)
    observed &= ~target
    left,right=indices[0]-1,indices[-1]+1
    if left<0 or right>=len(x) or not observed[left] or not observed[right]:
        raise UnsupportedTask('FIM temporal block requires both admitted boundaries')
    # Use every admissible logical context position; tensor padding grants no extra history.
    contexts=(np.arange(0,indices[0]),np.arange(indices[-1]+1,len(x)))
    width=max(64,*[len(c) for c in contexts])
    v=np.zeros((1,2,width,1),dtype=np.float32);tt=np.zeros_like(v);missing=np.ones_like(v,dtype=bool)
    duration=t[-1]-t[0]
    if duration<=0:raise ValueError('Positive time range required')
    scaled=(t-t[0])/duration
    for window,positions in enumerate(contexts):
        offset=width-len(positions) if window==0 else 0
        selected=np.arange(offset,offset+len(positions))
        v[0,window,selected,0]=np.where(observed[positions],x[positions],0.)
        tt[0,window,selected,0]=scaled[positions]
        missing[0,window,selected,0]=~observed[positions]
    # Boundary times are included in the integration grid: initial conditions refer
    # to their own physical timestamps rather than to the first hidden timestamp.
    locations=np.arange(left,right+1)
    batch={'observation_values':v,'observation_times':tt,'observation_mask':missing,'location_times':scaled[locations][None,:,None].astype(np.float32),'linitial_conditions':np.asarray([[x[left]]],np.float32),'rinitial_conditions':np.asarray([[x[right]]],np.float32)}
    return batch,indices


class FIMTemporalFrozen:
    name='FIM-temporal-base-native'
    def __init__(self,bundle_dir,device='cpu',checkpoint_license_acknowledged=False):
        self.bundle_dir=Path(bundle_dir);self.device=resolve_device(device);self.model=None;self.license_acknowledged=checkpoint_license_acknowledged

    def fit(self,values,times,*,split='train'):
        if split!='train':raise ValueError('Only train metadata permitted; FIM checkpoint is frozen')
        check_series(values,times)
        self._load();return self

    def _load(self):
        if not self.license_acknowledged:
            raise RuntimeError('FIM checkpoint license unresolved: official model card says [More Information Needed]; code MIT does not establish weight license')
        temporal=self.bundle_dir/'temporal_base';pointwise=self.bundle_dir/'pointwise_base'
        if not all((p/'config.json').exists() and (p/'model.safetensors').exists() for p in (temporal,pointwise)):
            raise FileNotFoundError('FIM requires verified local temporal_base and pointwise_base checkpoints; no implicit remote nested downloads')
        from fim.models.imputation_temporal import FIMImpTempBase, FIMImpTempBaseConfig
        config=json.loads((temporal/'config.json').read_text());config['fim_base']=str(pointwise.resolve())
        self.model=FIMImpTempBase.from_pretrained(str(temporal),config=FIMImpTempBaseConfig(**config),local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        synchronize(self.device)
        self.runtime=torch_device_metadata(self.device)

    def impute(self,values,times,*,task='R',target_gap_mask=None,observed_mask=None,padding_mask=None):
        if task not in ('R','O'):raise UnsupportedTask('FIM temporal block extension requires a received right boundary')
        if target_gap_mask is None:raise ValueError('FIM requires explicit episode support, never evaluator truth')
        batch,indices=prepare_fim_temporal_batch(values,times,target_gap_mask,observed_mask,padding_mask)
        if self.model is None:self._load()
        import torch
        tensor={k:torch.as_tensor(v,device=self.device) for k,v in batch.items()}
        with torch.inference_mode():
            output=self.model(tensor)['visualizations']['imputation_window']['learnt'][0,:,0].detach().cpu().numpy()
        result=np.asarray(values,float).copy()
        admitted=np.isfinite(result) if observed_mask is None else np.asarray(observed_mask,bool)&np.isfinite(result)
        result[~admitted]=np.nan
        if padding_mask is not None:result[np.asarray(padding_mask,bool)]=np.nan
        result[indices]=output[1:-1]
        if not np.isfinite(result[indices]).all():raise RuntimeError('FIM returned nonfinite target output')
        return result

    def impute_query(self,query):
        return self.impute(query.values,query.times,task=query.task,target_gap_mask=query.target_gap_mask,observed_mask=query.available_to_method,padding_mask=query.padding_mask)

    def get_resume_capabilities(self):
        return {'training':'not_applicable','inference':'query_shards','validation':'not_upstream_validated','checkpoint':FIM_TEMPORAL_REVISION,'availability':'checkpoint_license_unresolved','device':self.device,'precision':'float32'}
