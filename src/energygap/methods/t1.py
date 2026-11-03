"""Pinned official T1 source adapter; no reimplementation and no implicit download.

The official MIT source is fetched separately under artifacts/upstream/T1 and is not
relicensed as EnergyGapBench code. Architecture and mask-aware normalization execute
unchanged. Training uses the official MSE MIT/optional ORT objective on train-only
support; the bounded optimizer-step schedule is a documented protocol adaptation.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
from time import perf_counter
import numpy as np
from energygap.runtime.device import resolve_device,synchronize,torch_device_metadata
from energygap.runtime.threads import configure_torch_threads

COMMIT='823b76aab57a2e60eb5029caab06fdfec8b19256'
MODEL_SHA256='3f87aac66745029a88907b7ce65766d3ce5f785a6d4190f6343f9649d4ad3de8'
REPOSITORY='https://github.com/Oppenheimerdinger/T1'
DEFAULTS=dict(task_name='imputation',label_len=0,n_heads=128,patch_size=2,patch_stride=1,
              n_blocks=[2,2],kernel_size_large=[71,31],kernel_size_small=5,ffn_ratio=1.,
              downsample_ratio=2,positional_encoding=True,qkv_bias=True,cosine_attention=False,
              use_head_reconstruction=True,head_params_shared=True,imputation_use_mask_embedding=False,
              drop_attn=.1,drop_ffn=0.,drop_proj=0.,drop_path=.1,drop_head=0.,
              SACA=False,SACA_first_only=False,saca_hidden_dim_ratio=8,drop_saca=0.,
              sig_q_mode=1,sig_k_mode=1,sig_v_mode=1,mu_q_mode=1,mu_k_mode=1,mu_v_mode=1)


def load_official_t1(source_root='artifacts/upstream/T1'):
    source_root=Path(source_root);path=source_root/'models'/'T1.py'
    if not path.is_file():raise FileNotFoundError('T1 official source unavailable: run scripts/fetch_t1.py for pinned MIT upstream source')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:raise ValueError('T1 source checksum differs from pinned upstream commit')
    license_path=source_root/'LICENSE'
    if not license_path.is_file() or 'MIT License' not in license_path.read_text():raise ValueError('T1 original MIT license missing')
    spec=importlib.util.spec_from_file_location('energygap_pinned_upstream_T1',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class T1Adapter:
    supports_resume_training=False
    checkpoint_granularity='completed_fit_and_query_only'
    def __init__(self,context=512,n_features=1,seed=11,source_root='artifacts/upstream/T1',device='cpu',**architecture):
        import torch
        self.context,self.n_features,self.seed=context,n_features,seed
        self.device=resolve_device(device,allow_missing_torch=False)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);configure_torch_threads(torch)
        if self.device.startswith('cuda:'):torch.cuda.manual_seed_all(seed)
        config={**DEFAULTS,**architecture,'seq_len':context,'pred_len':context,'enc_in':n_features}
        self.config=SimpleNamespace(**config)
        self.upstream=load_official_t1(source_root)
        self.model=self.upstream.Model(self.config).float().to(self.device)
        self.mean=np.zeros(n_features);self.scale=np.ones(n_features)
        self.fit_history=[]
        self.metadata={'implementation':'official_source_adapter','repository':REPOSITORY,'commit':COMMIT,
                       'source_sha256':MODEL_SHA256,'code_license':'MIT','checkpoint':'trained_locally',
                       'pretraining':'not_applicable','supports_D1':'audited_with_native_forward_and_training',
                       'D1_limitation':'channel attention has one channel; multivariate dependence advantage cannot be tested at D1',
                       'training_resume':False,'training_schedule':'bounded_constant_lr_steps_protocol_adaptation',
                       'loss':'official MIT MSE + configurable observed ORT MSE; original defaults MIT1/ORT0',
                       'normalization':'train channel scaler followed by upstream mask-aware per-window normalization',
                       'device':self.device,'precision':'float32','runtime':torch_device_metadata(self.device)}

    def fit_windows(self,windows,*,split='train',steps=50,batch_size=4,learning_rate=.001,
                    ORT_weight=0.,MIT_weight=1.,fixed_hidden_mask=None):
        import torch
        from .training import _mixed_mask
        if split!='train':raise ValueError('T1 fit is train-only')
        x=np.asarray(windows,dtype=np.float32)
        if x.ndim!=3 or x.shape[1:]!=(self.context,self.n_features):raise ValueError('Expected [N,L,D] train windows')
        if np.isinf(x).any() or (np.isfinite(x).sum(axis=(1,2))<2).any():raise ValueError('Insufficient original training support')
        if steps<=0 or batch_size<=0 or MIT_weight<=0 or ORT_weight<0:raise ValueError('Invalid training/loss configuration')
        self.mean=np.nanmean(x,axis=(0,1));self.scale=np.nanstd(x,axis=(0,1));self.scale[self.scale<1e-8]=1
        normalized=(x-self.mean)/self.scale;observed=np.isfinite(normalized)
        rng=np.random.default_rng(self.seed);optimizer=torch.optim.Adam(self.model.parameters(),lr=learning_rate)
        self.fit_history=[];self.model.train();synchronize(self.device);tic=perf_counter()
        for step in range(steps):
            indices=rng.choice(len(x),size=min(batch_size,len(x)),replace=False)
            visible_original=observed[indices]
            hidden=_mixed_mask(visible_original,rng) if fixed_hidden_mask is None else np.asarray(fixed_hidden_mask,bool)[indices]&visible_original
            if not hidden.any():raise ValueError('No originally observed artificial training targets')
            admissible=visible_original&~hidden
            truth=np.where(visible_original,normalized[indices],0)
            inputs=torch.as_tensor(np.where(admissible,truth,0).astype(np.float32),device=self.device)
            mask=torch.as_tensor(admissible.astype(np.float32),device=self.device)
            target=torch.as_tensor(truth.astype(np.float32),device=self.device);target_mask=torch.as_tensor(hidden,device=self.device)
            optimizer.zero_grad(set_to_none=True)
            output=self.model(inputs,None,None,None,mask)
            mit=torch.nn.functional.mse_loss(output[target_mask],target[target_mask])
            loss=MIT_weight*mit
            if ORT_weight:
                if not admissible.any():raise ValueError('ORT requested without observed input support')
                valid=torch.as_tensor(admissible,device=self.device)
                loss=loss+ORT_weight*torch.nn.functional.mse_loss(output[valid],target[valid])
            if not torch.isfinite(loss):raise ValueError('Nonfinite native T1 loss')
            loss.backward();optimizer.step()
            self.fit_history.append(float(loss.detach()))
        synchronize(self.device)
        self.metadata.update(optimizer='torch.optim.Adam',learning_rate=learning_rate,steps=steps,
                             ORT_weight=ORT_weight,MIT_weight=MIT_weight,fit_seconds=perf_counter()-tic,
                             training_masks='fixed_diagnostic' if fixed_hidden_mask is not None else 'balanced_point_block',
                             upstream_deviations=['bounded constant-LR step schedule','balanced point/block masks instead of source tslib random points',
                                                  'no test set evaluated during training','epoch-level upstream LR schedule not used in diagnostic fit'])
        return self

    def infer(self,values,*,task='R'):
        import torch
        if task not in ('R','O'):raise ValueError('T1 bidirectional adapter unsupported for ongoing causal task C')
        x=np.asarray(values,float)
        if x.ndim==1:x=x[:,None]
        if x.shape!=(self.context,self.n_features) or np.isinf(x).any():raise ValueError('T1 context/channel mismatch or infinite input')
        visible=np.isfinite(x);normalized=np.where(visible,(x-self.mean)/self.scale,0).astype(np.float32)
        self.model.eval();synchronize(self.device);tic=perf_counter()
        with torch.no_grad():
            raw=self.model(torch.as_tensor(normalized[None],device=self.device),None,None,None,torch.as_tensor(visible[None].astype(np.float32),device=self.device)).cpu().numpy()[0]
        synchronize(self.device)
        prediction=raw*self.scale+self.mean;prediction=np.where(visible,x,prediction)
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite native T1 inference')
        return {'point_estimate':prediction,'samples':None,'quantiles':None,'status':'success',
                'timing':{'inference_seconds':perf_counter()-tic},'model_metadata':dict(self.metadata)}

    def save_fitted(self,path):
        """Save completed model only. This does not claim optimizer-step recovery."""
        from io import BytesIO
        import torch
        from energygap.runtime.state import atomic_bytes
        buffer=BytesIO()
        torch.save({'schema_version':1,'commit':COMMIT,'source_sha256':MODEL_SHA256,
                    'torch_version':str(torch.__version__),'context':self.context,'n_features':self.n_features,
                    'seed':self.seed,'architecture':vars(self.config),'model_state':self.model.state_dict(),
                    'mean':self.mean.tolist(),'scale':self.scale.tolist(),
                    'fit_history':self.fit_history,'metadata':self.metadata},buffer)
        digest=atomic_bytes(Path(path),buffer.getvalue())
        return {'path':str(path),'sha256':digest,'state_scope':'completed_fit_only','training_resume':False}

    @classmethod
    def load_fitted(cls,path,*,source_root='artifacts/upstream/T1',device='cpu'):
        import torch
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['schema_version']!=1 or state['commit']!=COMMIT or state['source_sha256']!=MODEL_SHA256:
            raise ValueError('T1 fitted checkpoint upstream identity differs')
        if state['torch_version']!=str(torch.__version__):raise ValueError('T1 PyTorch version differs; audit compatibility explicitly')
        adapter=cls(context=state['context'],n_features=state['n_features'],seed=state['seed'],source_root=source_root,device=device,**state['architecture'])
        adapter.model.load_state_dict(state['model_state']);adapter.mean=np.asarray(state['mean'],np.float32)
        adapter.scale=np.asarray(state['scale'],np.float32);adapter.fit_history=state['fit_history'];adapter.metadata=state['metadata']
        adapter.metadata.update(device=adapter.device,precision='float32',runtime=torch_device_metadata(adapter.device))
        return adapter
