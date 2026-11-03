"""Small actual CUDA test for the server; not a scientific benchmark result."""
from __future__ import annotations


def run_gpu_smoke(device='cuda:0'):
    from .device import cuda_probe,resolve_device,synchronize
    selected=resolve_device(device,allow_missing_torch=False)
    if not selected.startswith('cuda:'):raise ValueError('gpu-smoke requires cuda:N; it does not simulate CUDA on CPU or MPS')
    diagnostic=cuda_probe(selected)
    try:
        from energygap.methods.neural import PyPOTSAdapter
        import pypots
    except ImportError as exc:
        raise RuntimeError('gpu-smoke requires PyPOTS for the actual SAITS training test; activate the environment containing PyTorch CUDA and PyPOTS') from exc
    import numpy as np
    import torch
    from energygap.forecasting.ridge import DirectRidge
    values=np.sin(np.arange(140)/7)+np.arange(140)*.002
    times=np.arange(140)*300.
    reference=DirectRidge(history=20,device='cpu').fit(values,times)
    ridge=DirectRidge(history=20,device=selected).fit(values,times)
    if not all(tensor.device==torch.device(selected) for tensor in ridge._torch_state.values()):
        raise RuntimeError('Ridge tensors were not placed on the requested GPU')
    prediction=ridge.predict(values[-20:],times[-1])
    if not np.isclose(prediction,reference.predict(values[-20:],times[-1]),atol=1e-9,rtol=1e-9):
        raise RuntimeError('CUDA ridge disagrees with the unchanged CPU reference')
    train=np.sin(np.arange(8*16,dtype=np.float32).reshape(8,16,1)/9)
    model=PyPOTSAdapter('SAITS',context=16,epochs=1,seed=11,device=selected).fit_windows(train)
    if not all(parameter.device==torch.device(selected) for parameter in model.model.model.parameters()):
        raise RuntimeError('SAITS parameters were not placed on the requested GPU')
    query=train[0,:,0].astype(float);query[8:12]=np.nan
    output=model.infer(query)['point_estimate'][:,0]
    observed=np.isfinite(query)
    if not np.isfinite(output).all() or not np.array_equal(output[observed],query[observed]):
        raise RuntimeError('CUDA SAITS failed finite output or observed preservation')
    synchronize(selected)
    return {'status':'passed','scope':'synthetic_actual_cuda_device_test_not_scientific_evaluation',
            'device':selected,'diagnostic':diagnostic,'ridge':{'backend':'torch','precision':'float64','gpu_cpu_parity':True},
            'SAITS':{'implementation':'pypots','epochs':1,'training_device':selected,'inference_device':selected,
                     'precision':'float32','observed_preserved':True},'automatic_mixed_precision':False}
