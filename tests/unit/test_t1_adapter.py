"""Optional source-backed T1 tests; source is deliberately outside the repository."""
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('torch')
from energygap.methods.t1 import T1Adapter,load_official_t1,MODEL_SHA256,COMMIT

SOURCE=Path('artifacts/upstream/T1/models/T1.py')
pytestmark=pytest.mark.skipif(not SOURCE.exists(),reason='pinned optional T1 source not fetched; run scripts/fetch_t1.py')


def test_t1_official_source_hash_and_d1_l512_native_equivalence():
    import hashlib,torch
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==MODEL_SHA256
    adapter=T1Adapter(context=512,n_features=1,seed=11)
    x=np.sin(np.arange(512)/40)[:,None];x[470:482]=np.nan
    output=adapter.infer(x)['point_estimate']
    available=np.isfinite(x)
    normalized=np.where(available,(x-adapter.mean)/adapter.scale,0).astype(np.float32)
    with torch.no_grad():
        native=adapter.model(torch.from_numpy(normalized[None]),None,None,None,
                             torch.from_numpy(available[None].astype(np.float32))).numpy()[0]
    expected=np.where(available,x,native*adapter.scale+adapter.mean)
    assert np.array_equal(output,expected)
    assert np.isfinite(output).all() and output.shape==(512,1)
    assert np.array_equal(output[available],x[available])
    assert adapter.metadata['commit']==COMMIT
    assert not adapter.supports_resume_training
    with pytest.raises(ValueError):adapter.infer(x,task='C')


def test_t1_tiny_overfit_uses_original_mse_target_and_preserves_observed(tmp_path):
    x=np.sin(np.arange(16,dtype=np.float32)/4)[None,:,None]
    hidden=np.zeros_like(x,bool);hidden[:,5:9]=True
    adapter=T1Adapter(context=16,n_features=1,n_heads=8,n_blocks=[1],kernel_size_large=[7],
                      kernel_size_small=3,drop_attn=0.,drop_path=0.)
    adapter.fit_windows(x,steps=200,batch_size=1,learning_rate=.01,fixed_hidden_mask=hidden)
    assert adapter.fit_history[-1]<1e-7
    assert adapter.fit_history[-1]<adapter.fit_history[0]*1e-5
    query=x[0].astype(float);query[hidden[0]]=np.nan
    prediction=adapter.infer(query)['point_estimate']
    assert np.mean(np.abs(prediction[hidden[0]]-x[0][hidden[0]]))<1e-3
    assert np.array_equal(prediction[~hidden[0]],query[~hidden[0]])
    checkpoint=adapter.save_fitted(tmp_path/'fitted.pt')
    assert checkpoint['state_scope']=='completed_fit_only'
    restored=T1Adapter.load_fitted(tmp_path/'fitted.pt')
    assert np.array_equal(restored.infer(query)['point_estimate'],prediction)
    with pytest.raises(ValueError):adapter.fit_windows(x,split='test',steps=1)


def test_t1_checksum_mismatch_cannot_use_another_transformer(tmp_path):
    (tmp_path/'models').mkdir();(tmp_path/'models/T1.py').write_text('not official')
    (tmp_path/'LICENSE').write_text('MIT License')
    with pytest.raises(ValueError,match='checksum'):load_official_t1(tmp_path)


def test_t1_cuda_runs_native_training_and_loads_fitted_on_selected_device(tmp_path):
    import torch
    if not torch.cuda.is_available():pytest.skip('actual CUDA hardware required')
    adapter=T1Adapter(context=16,n_heads=8,n_blocks=[1],kernel_size_large=[7],kernel_size_small=3,device='cuda:0')
    x=np.sin(np.arange(16,dtype=np.float32)/4)[None,:,None]
    adapter.fit_windows(x,steps=2,batch_size=1)
    query=x[0].astype(float);query[5:8]=np.nan
    expected=adapter.infer(query)['point_estimate']
    assert next(adapter.model.parameters()).device.type=='cuda'
    adapter.save_fitted(tmp_path/'cuda-fitted.pt')
    restored=T1Adapter.load_fitted(tmp_path/'cuda-fitted.pt',device='cuda:0')
    assert next(restored.model.parameters()).device.type=='cuda'
    np.testing.assert_array_equal(restored.infer(query)['point_estimate'],expected)
    cpu=T1Adapter.load_fitted(tmp_path/'cuda-fitted.pt',device='cpu')
    assert next(cpu.model.parameters()).device.type=='cpu'
