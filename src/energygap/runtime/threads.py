"""Explicit CPU thread budgets, inherited by isolated model workers.

This module imports only the standard library. Call ``configure_cpu_threads``
before importing NumPy or Torch. A thread count is a per-process CPU pool limit,
not a count of independent experiments or CUDA kernels.
"""
from __future__ import annotations

import os
import sys

ENVIRONMENT_VARIABLES = (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
)
_pool_limiter = None


def _positive(value):
    if isinstance(value, bool):
        raise ValueError('threads must be a positive integer')
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('threads must be a positive integer') from exc
    if str(parsed) != str(value) or parsed < 1:
        raise ValueError('threads must be a positive integer')
    return parsed


def requested_threads(explicit=None, default=1):
    """Return the explicit/inherited budget; preserve historical neural default1."""
    value = explicit if explicit is not None else os.environ.get('ENERGYGAP_NUM_THREADS', default)
    return _positive(value)


def configure_cpu_threads(threads=None):
    """Set pool limits before scientific imports; update already-loaded BLAS pools.

    No explicit/inherited EnergyGap budget preserves existing environment values.
    Torch is never imported here, so the minimum CLI remains usable without it.
    """
    global _pool_limiter
    inherited = os.environ.get('ENERGYGAP_NUM_THREADS')
    count = requested_threads(threads) if threads is not None or inherited is not None else None
    result = {'requested': count, 'cpu_count': os.cpu_count(),
              'meaning': 'CPU threads per process; one GPU experiment at a time',
              'existing_pool_control': 'not_needed_before_import'}
    if hasattr(os, 'sched_getaffinity'):
        result['affinity_cpu_count'] = len(os.sched_getaffinity(0))
    if count is not None:
        os.environ['ENERGYGAP_NUM_THREADS'] = str(count)
        for name in ENVIRONMENT_VARIABLES:
            os.environ[name] = str(count)
        # An explicit budget must remain stable rather than silently shrinking.
        os.environ['OMP_DYNAMIC'] = 'FALSE'
        os.environ['MKL_DYNAMIC'] = 'FALSE'
        if any(name in sys.modules for name in ('numpy', 'scipy', 'torch')):
            try:
                from threadpoolctl import threadpool_limits
            except ImportError:
                result['existing_pool_control'] = 'threadpoolctl_unavailable_environment_applies_to_new_imports'
            else:
                _pool_limiter = threadpool_limits(limits=count)
                result['existing_pool_control'] = 'threadpoolctl'
        if 'torch' in sys.modules:
            result['torch'] = configure_torch_threads(sys.modules['torch'], count)
        if 'numexpr' in sys.modules:
            module = sys.modules['numexpr']
            maximum = getattr(module, 'MAX_THREADS', count)
            if count <= maximum:
                module.set_num_threads(count)
                result['numexpr_threads'] = int(module.get_num_threads())
            else:
                result['numexpr_pool_control'] = 'already_initialized_with_lower_maximum_restart_required'
    result['environment'] = {name: os.environ.get(name) for name in ENVIRONMENT_VARIABLES}
    return result


def configure_torch_threads(torch_module=None, threads=None):
    """Apply intra-op N and serial inter-op; never reset inter-op after work began."""
    if torch_module is None:
        import torch as torch_module
    count = requested_threads(threads)
    torch_module.set_num_threads(count)
    interop_status = 'unchanged'
    if hasattr(torch_module, 'get_num_interop_threads') and hasattr(torch_module, 'set_num_interop_threads'):
        if torch_module.get_num_interop_threads() != 1:
            try:
                torch_module.set_num_interop_threads(1)
            except RuntimeError:
                interop_status = 'already_initialized_retained'
            else:
                interop_status = 'serial'
        else:
            interop_status = 'serial'
    return {'intraop': int(torch_module.get_num_threads()),
            'interop': int(torch_module.get_num_interop_threads()) if hasattr(torch_module, 'get_num_interop_threads') else None,
            'interop_control': interop_status}
