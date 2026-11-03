"""Framework-native full-state serialization for adapters that expose their optimizer loop.

An adapter must opt in and pass its own recovery tests. This module alone does not
make third-party fit() routines resumable.
"""
from __future__ import annotations
from io import BytesIO
import random
import numpy as np
from .device import synchronize,torch_device_metadata


def _model_runtime(model):
    tensors=list(model.parameters())+list(model.buffers())
    devices={str(t.device) for t in tensors}
    if len(devices)>1:raise ValueError('training recovery requires one model device')
    device=next(iter(devices),'cpu')
    return {'execution':torch_device_metadata(device),
            'dtypes':sorted({str(t.dtype) for t in tensors if t.is_floating_point()})}


def _optimizer_to_parameter_devices(optimizer):
    """Keep moments with parameters; noncapturable Adam step counters remain on CPU."""
    import torch
    def move(value,device):
        if isinstance(value,torch.Tensor):return value.to(device)
        if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
        if isinstance(value,list):return [move(v,device) for v in value]
        if isinstance(value,tuple):return tuple(move(v,device) for v in value)
        return value
    for group in optimizer.param_groups:
        for parameter in group['params']:
            values=optimizer.state.get(parameter,{})
            for key,value in values.items():
                # PyTorch intentionally stores these scalar counters on the host
                # for ordinary (noncapturable, unfused) Adam/AdamW.
                device='cpu' if key=='step' and not (group.get('capturable') or group.get('fused')) else parameter.device
                values[key]=move(value,device)


def capture_torch_state(model, optimizer, *, scheduler=None, scaler=None,
                        epoch: int, global_step: int, sampler_state: dict,
                        early_stopping: dict, trial_state: dict, numpy_rng=None,
                        extra: dict | None = None) -> bytes:
    import torch
    runtime=_model_runtime(model);device=runtime['execution']['device'];synchronize(device)
    state={'schema_version':1,'torch_version':str(torch.__version__),'runtime':runtime,
           'model':model.state_dict(),'optimizer':optimizer.state_dict(),
           'scheduler':scheduler.state_dict() if scheduler else None,
           'scaler':scaler.state_dict() if scaler else None,
           'epoch':epoch,'global_step':global_step,'sampler':sampler_state,
           'early_stopping':early_stopping,'trial':trial_state,
           'rng_python':random.getstate(),'rng_numpy_legacy':np.random.get_state(),
           'rng_numpy_generator':numpy_rng.bit_generator.state if numpy_rng is not None else None,
           'rng_torch_cpu':torch.get_rng_state(),
           'rng_torch_cuda':torch.cuda.get_rng_state_all() if device.startswith('cuda:') else None,
           'extra':extra or {}}
    buffer=BytesIO();torch.save(state,buffer);return buffer.getvalue()


def restore_torch_state(data: bytes, model, optimizer, *, scheduler=None, scaler=None,
                        numpy_rng=None, strict_version=True) -> dict:
    """Restore before consuming another batch. Only use trusted local checkpoints."""
    import torch
    state=torch.load(BytesIO(data),map_location='cpu',weights_only=False)
    if state['schema_version']!=1:raise ValueError('training checkpoint schema unsupported')
    if strict_version and state['torch_version']!=str(torch.__version__):
        raise ValueError('PyTorch version differs; explicit compatibility audit required')
    runtime=_model_runtime(model)
    if 'runtime' in state:
        if state['runtime']!=runtime:raise ValueError('training checkpoint device, precision or runtime differs; start a separate run')
    elif runtime['execution']['device']!='cpu':
        raise ValueError('legacy checkpoint has no GPU runtime identity; start a separate run')
    if state['rng_torch_cuda'] is not None:
        if not torch.cuda.is_available():raise ValueError('GPU checkpoint requires compatible GPU')
        if len(state['rng_torch_cuda'])!=torch.cuda.device_count():raise ValueError('GPU checkpoint visible device count differs')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    _optimizer_to_parameter_devices(optimizer)
    for object_,key in ((scheduler,'scheduler'),(scaler,'scaler')):
        if (object_ is None)!=(state[key] is None):raise ValueError('missing or extra '+key)
        if object_ is not None:object_.load_state_dict(state[key])
    random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy_legacy'])
    if state['rng_numpy_generator'] is not None:
        if numpy_rng is None:raise ValueError('explicit NumPy generator required')
        numpy_rng.bit_generator.state=state['rng_numpy_generator']
    torch.set_rng_state(state['rng_torch_cpu'])
    if state['rng_torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(state['rng_torch_cuda'])
    return {k:state[k] for k in ('epoch','global_step','sampler','early_stopping','trial','extra')}
