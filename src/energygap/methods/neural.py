"""Optional wrappers around real PyPOTS implementations; no architecture substitutes.

Native fit retains upstream losses. This adapter currently supports durable inference
and completed fitted models, not exact in-progress native-fit recovery.
"""
from __future__ import annotations
import importlib
import random
from time import perf_counter
import numpy as np
from energygap.runtime.device import resolve_device,synchronize,torch_device_metadata
from energygap.runtime.threads import configure_torch_threads


PARAMETERS={
    'SAITS':dict(n_layers=1,d_model=32,n_heads=2,d_k=16,d_v=16,d_ffn=32),
    'BRITS':dict(rnn_hidden_size=32),
    'CSDI':dict(n_layers=2,n_heads=2,n_channels=32,d_time_embedding=32,d_feature_embedding=8,d_diffusion_embedding=32,n_diffusion_steps=50,target_strategy='mix',schedule='quad'),
    'ImputeFormer':dict(n_layers=1,d_input_embed=16,d_learnable_embed=16,d_proj=8,d_ffn=32,n_temporal_heads=2),
}


class PyPOTSAdapter:
    supports_resume_training=False
    checkpoint_granularity='completed_native_fit_and_query'

    def __init__(self,name,context=512,n_features=1,epochs=100,seed=11,device='cpu',**kwargs):
        if name not in PARAMETERS:raise ValueError('No audited PyPOTS wrapper for '+name)
        module=importlib.import_module('pypots.imputation')
        import torch
        self.device=resolve_device(device,allow_missing_torch=False)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);configure_torch_threads(torch)
        if self.device.startswith('cuda:'):torch.cuda.manual_seed_all(seed)
        self.name=name;self.context=context;self.n_features=n_features;self.seed=seed
        self.parameters=dict(PARAMETERS[name]);self.parameters.update(kwargs)
        self.model=getattr(module,name)(n_steps=context,n_features=n_features,epochs=epochs,batch_size=8,patience=min(10,epochs),num_workers=0,device=self.device,verbose=False,**self.parameters)
        self.runtime=torch_device_metadata(self.device)
        self.timings={}

    def fit_windows(self,windows,*,split='train'):
        if split!='train':raise ValueError('Neural fit is training-only')
        x=np.asarray(windows,dtype=np.float32)
        if x.ndim!=3 or x.shape[1:]!=(self.context,self.n_features):raise ValueError('Expected [N,context,channels]')
        if np.isinf(x).any() or not np.isfinite(x).any():raise ValueError('Invalid training support')
        self.mean=np.nanmean(x,axis=(0,1));self.scale=np.nanstd(x,axis=(0,1));self.scale[self.scale<1e-8]=1
        z=(x-self.mean)/self.scale
        synchronize(self.device);tic=perf_counter();self.model.fit({'X':z});synchronize(self.device)
        self.timings['fit_seconds']=perf_counter()-tic
        return self

    def infer(self,values,*,task='R',samples=100,sample_seed=0):
        if task not in ('R','O'):raise ValueError('Bidirectional wrapper unsupported for C')
        x=np.asarray(values,dtype=np.float64)
        if x.ndim==1:x=x[:,None]
        if x.shape!=(self.context,self.n_features):raise ValueError('Context/channel mismatch, no compression or observed padding allowed')
        if np.isinf(x).any():raise ValueError('Invalid infinite input')
        import torch
        torch.manual_seed(sample_seed)
        if self.device.startswith('cuda:'):torch.cuda.manual_seed_all(sample_seed)
        z=((x-self.mean)/self.scale).astype(np.float32)[None]
        synchronize(self.device);tic=perf_counter()
        if self.name=='CSDI':
            output=self.model.predict({'X':z},n_sampling_times=samples)['imputation']
            if hasattr(output,'detach'):output=output.detach().cpu().numpy()
            draws=np.asarray(output)[0]
            if draws.shape!=(samples,self.context,self.n_features):raise ValueError('Unexpected CSDI trajectory shape '+str(draws.shape))
            draws=draws*self.scale+self.mean
            draws=np.where(np.isfinite(x)[None],x[None],draws)
            point=np.median(draws,axis=0)
        else:
            output=self.model.predict({'X':z})['imputation']
            if hasattr(output,'detach'):output=output.detach().cpu().numpy()
            point=np.asarray(output)[0]*self.scale+self.mean;draws=None
        point=np.where(np.isfinite(x),x,point)
        if not np.isfinite(point).all():raise ValueError('Native model returned nonfinite output')
        synchronize(self.device)
        return dict(point_estimate=point,samples=draws,quantiles=None,status='success',timing=dict(inference_seconds=perf_counter()-tic),model_metadata=dict(implementation='pypots',name=self.name,parameters=self.parameters,training_resume=False,device=self.device,precision='float32',runtime=self.runtime))


class TSICLAdapter:
    """Native impute API. Local checkpoint and separate noncommercial environment required."""
    supports_resume_training=False
    def __init__(self,checkpoint,context=512):
        from pathlib import Path
        if not Path(checkpoint).is_file():raise FileNotFoundError('Local TS-ICL checkpoint required; no implicit download')
        from tsicl import TSICL
        self.model=TSICL(model_path=str(checkpoint));self.context=context

    def infer(self,values):
        x=np.asarray(values,float)
        if x.ndim!=1 or len(x)!=self.context:raise ValueError('Univariate L=512 core required')
        levels=[.05,.1,.25,.5,.75,.9,.95]
        point,quantiles=self.model.impute(inputs=x.copy(),quantile_levels=levels,point_estimator='median',denormalize=True)
        # Returned quantiles are marginal, never claim joint paths.
        return dict(native_point=point,quantiles=quantiles,quantile_levels=levels,samples=None,capability='marginal_quantiles',validation_status='not_validated_without_checkpoint')
