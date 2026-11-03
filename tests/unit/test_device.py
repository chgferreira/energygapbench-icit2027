"""Device selection never disguises missing CUDA as successful GPU execution."""
from types import SimpleNamespace
import pytest
from energygap.runtime import device as devices


def fake_torch(*,available=True,architectures=('sm_70',),count=1):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:available,device_count=lambda:count,
        get_device_capability=lambda i:(7,0),get_arch_list=lambda:list(architectures),
        get_device_name=lambda i:'Tesla V100-PCIE-16GB'))


def test_explicit_cpu_does_not_import_or_probe_torch(monkeypatch):
    monkeypatch.setattr(devices,'_torch',lambda **kwargs:pytest.fail('CPU selection must remain lightweight'))
    assert devices.resolve_device('cpu')=='cpu'


def test_auto_accepts_minimal_environment_without_torch(monkeypatch):
    monkeypatch.setattr(devices,'_torch',lambda **kwargs:None)
    assert devices.resolve_device('auto')=='cpu'


def test_auto_cpu_on_mac_but_explicit_cuda_fails(monkeypatch):
    monkeypatch.setattr(devices,'_torch',lambda **kwargs:fake_torch(available=False))
    assert devices.resolve_device('auto')=='cpu'
    with pytest.raises(RuntimeError,match='CUDA requested but unavailable'):devices.resolve_device('cuda:0')


def test_single_v100_selection_and_workspace_before_cuda_probe(monkeypatch):
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    fake=fake_torch()
    def available():
        import os
        assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        return True
    fake.cuda.is_available=available
    monkeypatch.setattr(devices,'_torch',lambda **kwargs:fake)
    assert devices.resolve_device('auto')=='cuda:0'
    assert devices.resolve_device('cuda')=='cuda:0'
    assert devices.resolve_device('cuda:0')=='cuda:0'
    with pytest.raises(ValueError,match='does not exist'):devices.resolve_device('cuda:1')


def test_v100_rejects_binary_without_volta_support(monkeypatch):
    monkeypatch.setattr(devices,'_torch',lambda **kwargs:fake_torch(architectures=('sm_80','sm_90')))
    with pytest.raises(RuntimeError,match='sm_70'):devices.resolve_device('cuda:0')


def test_compatible_ptx_can_support_v100(monkeypatch):
    monkeypatch.setattr(devices,'_torch',lambda **kwargs:fake_torch(architectures=('compute_70','sm_80')))
    assert devices.resolve_device('cuda:0')=='cuda:0'


@pytest.mark.parametrize('name',['mps','cuda:-1','gpu','cuda:abc'])
def test_device_typos_are_not_silent_cpu_fallbacks(name):
    with pytest.raises(ValueError,match='device must'):devices.resolve_device(name)
