"""Explicit single-device selection without a mandatory torch dependency on CPU.

Device/runtime identity is scientific metadata. An explicit CUDA request never
falls back to CPU, including when the installed binary cannot execute on Volta.
"""
from __future__ import annotations
import importlib
import os
import re


def _torch(*, required=True):
    try:
        return importlib.import_module('torch')
    except ImportError as exc:
        if required:
            raise RuntimeError('PyTorch is not installed in the active environment; activate your CUDA environment first') from exc
        return None


def resolve_device(requested='auto', *, allow_missing_torch=True):
    value=str(requested).lower().strip()
    if value not in ('auto','cpu','cuda') and not re.fullmatch(r'cuda:\d+',value):
        raise ValueError('device must be auto, cpu, cuda, or cuda:N')
    if value=='cpu':return 'cpu'
    # Set before CUDA is initialized, preserving an explicit user configuration.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch=_torch(required=value!='auto' or not allow_missing_torch)
    if torch is None:return 'cpu'
    if not torch.cuda.is_available():
        if value=='auto':return 'cpu'
        raise RuntimeError('CUDA requested but unavailable in the active PyTorch environment; check torch.cuda.is_available() and the NVIDIA driver')
    index=0 if value in ('auto','cuda') else int(value.split(':')[1])
    if index>=torch.cuda.device_count():
        raise ValueError(f'CUDA device {index} does not exist; visible device count is {torch.cuda.device_count()}')
    device=f'cuda:{index}'
    capability=tuple(torch.cuda.get_device_capability(index))
    arch_list=list(torch.cuda.get_arch_list())
    actual=capability[0]*10+capability[1]
    # Same-major SASS can run on a higher minor; older PTX can be JIT compiled.
    supported=any((a.startswith('sm_') and a[3:].isdigit() and int(a[3:])//10==actual//10 and int(a[3:])<=actual)
                  or (a.startswith('compute_') and a[8:].isdigit() and int(a[8:])<=actual)
                  for a in arch_list)
    if arch_list and not supported:
        raise RuntimeError(f'Installed PyTorch CUDA binary does not advertise support for {torch.cuda.get_device_name(index)} (sm_{actual}); compiled architectures: {arch_list}. Use a CUDA/PyTorch build compatible with this GPU; no CPU fallback was performed')
    return device


def synchronize(device):
    if str(device).startswith('cuda'):
        _torch().cuda.synchronize(device)


def torch_device_metadata(device):
    """Static fields only: safe for fingerprints and checkpoint compatibility."""
    selected=resolve_device(device)
    torch=_torch(required=selected!='cpu')
    if torch is None:return {'device':'cpu','backend':'numpy_scipy','torch_version':None}
    data={'device':selected,'backend':'pytorch','torch_version':str(torch.__version__),
          'cuda_runtime':torch.version.cuda if selected.startswith('cuda') else None,
          'deterministic_algorithms':bool(torch.are_deterministic_algorithms_enabled()),
          'precision_policy':'existing_float32_models_float64_ridge_no_automatic_mixed_precision'}
    if selected.startswith('cuda'):
        index=int(selected.split(':')[1]);properties=torch.cuda.get_device_properties(index)
        data.update(gpu_name=torch.cuda.get_device_name(index),compute_capability=list(torch.cuda.get_device_capability(index)),
                    total_memory_bytes=int(properties.total_memory),arch_list=list(torch.cuda.get_arch_list()),
                    cudnn_version=torch.backends.cudnn.version(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))
    return data


def cuda_probe(requested='cuda:0'):
    """Run a real float64 CUDA kernel/solve, rather than reporting driver presence."""
    device=resolve_device(requested,allow_missing_torch=False)
    if not device.startswith('cuda'):raise ValueError('CUDA probe requires a CUDA device')
    torch=_torch()
    a=torch.tensor([[3.,1.],[1.,2.]],device=device,dtype=torch.float64)
    b=torch.tensor([1.,2.],device=device,dtype=torch.float64)
    solution=torch.linalg.solve(a,b)
    synchronize(device)
    if not bool(torch.isfinite(solution).all()) or not bool(torch.allclose(a@solution,b,atol=1e-12,rtol=1e-12)):
        raise RuntimeError('CUDA float64 solve validation failed')
    return {**torch_device_metadata(device),'cuda_kernel_test':'passed','solution_device':str(solution.device),
            'solution':solution.detach().cpu().tolist()}
