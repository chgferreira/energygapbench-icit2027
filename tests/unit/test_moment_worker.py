from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest

_SPEC=importlib.util.spec_from_file_location('moment_worker_contract',Path(__file__).resolve().parents[2]/'scripts/moment_worker.py')
worker=importlib.util.module_from_spec(_SPEC);_SPEC.loader.exec_module(worker)


def case(tmp_path,extra=None,task='R',late=False):
    values=np.ones((1,512));obs=np.ones_like(values,bool);obs[:,497:511]=False;values[~obs]=1e20
    arrays=dict(values=values,times=(np.arange(512)*300.+1e9)[None,:],observed_mask=obs,padding_mask=np.zeros_like(obs),query_ids=np.asarray(['safe']))
    if extra:arrays.update(extra)
    np.savez_compressed(tmp_path/'input.npz',**arrays)
    issuance=arrays['times'][0,-1]-(300 if late else 0)
    (tmp_path/'input.json').write_text(json.dumps({'schema_version':1,'task':task,'issuance_times':[float(issuance)]}))
    return tmp_path/'input.npz',tmp_path/'input.json'


def test_worker_rejects_truth_and_causal_budget_violations(tmp_path):
    a,b=case(tmp_path,extra={'truth':np.zeros((1,512))})
    with pytest.raises(ValueError,match='truth'):worker.load_input(a,b)
    a,b=case(tmp_path,task='C')
    with pytest.raises(ValueError,match='R/O'):worker.load_input(a,b)
    a,b=case(tmp_path,late=True)
    with pytest.raises(ValueError,match='issuance'):worker.load_input(a,b)


def test_worker_removes_hidden_placeholders_before_model_import(tmp_path):
    a,b=case(tmp_path)
    meta,arrays=worker.load_input(a,b)
    assert np.isnan(arrays['values'][0,497:511]).all()
    assert np.isfinite(arrays['values'][arrays['observed_mask']]).all()


def test_audited_torch_pin_accepts_cuda_wheel_suffix(monkeypatch):
    packages={'momentfm':'0.1.4','numpy':'1.25.2','huggingface-hub':'0.24.0','transformers':'4.33.3','torch':'2.5.1+cu124'}
    monkeypatch.setattr(worker.importlib.metadata,'version',packages.__getitem__)
    monkeypatch.setattr(worker.sys,'version_info',(3,11,16))
    assert worker.check_environment()['torch']=='2.5.1+cu124'
    packages['torch']='2.6.0+cu124'
    with pytest.raises(RuntimeError,match='Incompatible'):worker.check_environment()


def test_explicit_cuda_selection_failure_never_starts_model(tmp_path,monkeypatch):
    from types import SimpleNamespace
    a,b=case(tmp_path)
    def no_cuda(requested,**kwargs):
        assert requested=='cuda:0'
        raise RuntimeError('CUDA requested but unavailable')
    monkeypatch.setattr(worker,'resolve_device',no_cuda)
    args=SimpleNamespace(input=a,metadata=b,output=tmp_path/'out.npz',result_metadata=tmp_path/'out.json',device='cuda:0',threads=1)
    with pytest.raises(RuntimeError,match='CUDA requested'):worker.run(args)
    assert not args.output.exists() and not args.result_metadata.exists()
