"""Keep optional model interpreters independent of the caller's Python packages."""
from __future__ import annotations

import os
from pathlib import Path


def isolated_worker_environment(repository_root):
    environment=dict(os.environ)
    # Preserve CUDA/library/thread settings while excluding foreign Python paths.
    for name in ('PYTHONHOME','PYTHONUSERBASE'):
        environment.pop(name,None)
    environment['PYTHONPATH']=str(Path(repository_root).resolve()/'src')
    environment['PYTHONNOUSERSITE']='1'
    return environment
