import json
import pytest
from energygap.cli import main


def test_cli_passes_cuda_zero_to_pipeline(monkeypatch,capsys):
    import energygap.pipeline as pipeline
    calls=[]
    monkeypatch.setattr(pipeline,'run_pipeline',lambda **kwargs:calls.append(kwargs) or {'status':'completed'})
    assert main(['run','--profile','smoke','--device','cuda:0','--progress','none'])==0
    assert calls[0]['device']=='cuda:0'
    assert json.loads(capsys.readouterr().out)['status']=='completed'


def test_doctor_explicit_cuda_runs_kernel_probe(monkeypatch,capsys):
    from energygap.runtime import device
    called=[]
    monkeypatch.setattr(device,'resolve_device',lambda value:'cuda:0')
    monkeypatch.setattr(device,'cuda_probe',lambda value:called.append(value) or {'device':value,'cuda_kernel_test':'passed'})
    assert main(['doctor','--device','cuda:0'])==0
    assert called==['cuda:0']
    assert json.loads(capsys.readouterr().out)['execution']['cuda_kernel_test']=='passed'


def test_unavailable_cuda_rejected_before_any_data_or_output(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    import energygap.pipeline as pipeline
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(pipeline,'prepare_series',lambda *args:pytest.fail('must validate CUDA before reading data'))
    with pytest.raises(RuntimeError,match='CUDA requested but unavailable'):
        pipeline.run_pipeline(root=tmp_path,device='cuda:0')
    assert not (tmp_path/'runs').exists()


def test_gpu_smoke_defaults_to_server_gpu_zero(monkeypatch,capsys):
    from energygap.runtime import gpu_smoke
    calls=[]
    monkeypatch.setattr(gpu_smoke,'run_gpu_smoke',lambda value:calls.append(value) or {'status':'passed','device':value})
    assert main(['gpu-smoke'])==0
    assert calls==['cuda:0']
    assert json.loads(capsys.readouterr().out)['device']=='cuda:0'


def test_gpu_smoke_never_simulates_cuda_on_cpu():
    from energygap.runtime.gpu_smoke import run_gpu_smoke
    with pytest.raises(ValueError,match='requires cuda'):
        run_gpu_smoke('cpu')
