"""Execute on the V100 server; never pretend CUDA exists on the Mac."""
import pytest


def test_real_cuda_smoke_trains_saits_and_solves_ridge():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA hardware is on the server, not this local machine')
    pytest.importorskip('pypots')
    from energygap.runtime.gpu_smoke import run_gpu_smoke
    result=run_gpu_smoke('cuda:0')
    assert result['status']=='passed'
    assert result['device']==result['SAITS']['training_device']==result['SAITS']['inference_device']=='cuda:0'
    assert result['ridge']['backend']=='torch' and result['ridge']['gpu_cpu_parity']
    assert result['SAITS']['observed_preserved'] and not result['automatic_mixed_precision']
