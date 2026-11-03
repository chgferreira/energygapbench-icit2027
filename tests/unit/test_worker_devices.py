"""Dispatch contracts; actual CUDA execution is covered separately on a GPU host."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def load_worker(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class HostResult:
    def __init__(self,values):self.values=values
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.values


def test_tsicl_native_call_receives_cuda_and_times_synchronized(monkeypatch):
    module=load_worker('tsicl_worker')
    calls=[];sync=[]
    fake_torch=SimpleNamespace(device=lambda value:value,inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setattr(module,'synchronize',sync.append)
    def impute(**kwargs):
        calls.append(kwargs)
        return HostResult(np.full(512,2.)),HostResult(np.full((512,7),2.))
    model=module.NativeTSICLWorker.__new__(module.NativeTSICLWorker)
    model.device='cuda:0';model.runtime={'device':'cuda:0'};model.model=SimpleNamespace(impute=impute)
    values=np.ones(512);values[499:511]=np.nan
    result=model.infer(values)
    assert calls[0]['device']=='cuda:0'
    assert sync==['cuda:0','cuda:0']
    assert result['device']=='cuda:0' and result['inference_seconds']>=0
    np.testing.assert_array_equal(result['point_estimate'][:499],values[:499])
    assert np.all(result['point_estimate'][499:511]==2.)


def test_moment_bridge_passes_device_to_isolated_interpreter(monkeypatch):
    from energygap.methods.moment_subprocess import MOMENTSubprocess
    import energygap.methods.moment_subprocess as bridge
    def run(command,**kwargs):
        assert command[command.index('--device')+1]=='cuda:0'
        with np.load(command[command.index('--input')+1]) as source:
            safe=source['values'];assert np.isnan(safe[0,500:511]).all()
        np.savez(command[command.index('--output')+1],point_estimate=np.nan_to_num(safe,nan=2.))
        Path(command[command.index('--result-metadata')+1]).write_text(json.dumps({'environment':{'device':'cuda:0'}}))
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr(bridge.subprocess,'run',run)
    model=MOMENTSubprocess(repository_root=ROOT,device='cuda:0')
    values=np.ones(512);values[500:511]=np.nan
    result=model.impute(values,np.arange(512)*300.)
    assert model.last_metadata['environment']['device']=='cuda:0'
    assert np.all(result[500:511]==2.)
