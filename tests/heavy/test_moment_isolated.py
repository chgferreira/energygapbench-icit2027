"""Cross-interpreter native smoke; minimum/CI installs can skip this optional stack."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'.venv-moment/bin/python'
CACHE=ROOT/'artifacts/checkpoints/hf'
AVAILABLE=PYTHON.exists() and (CACHE/'models--AutonLab--MOMENT-1-base').exists()


@pytest.mark.skipif(not AVAILABLE,reason='optional isolated Python3.11 MOMENT environment/checkpoint unavailable')
@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_isolated_native_parity_poisoning_and_completed_resume(tmp_path,device):
    if device.startswith('cuda'):
        probe=subprocess.run([str(PYTHON),'-c','import torch; print(torch.cuda.is_available())'],capture_output=True,text=True,timeout=20)
        assert probe.returncode==0,probe.stderr
        assert probe.stdout.strip() in ('True','False'),probe.stdout
        if probe.stdout.strip()=='False':pytest.skip('actual CUDA unavailable in the isolated MOMENT interpreter')
    x=np.sin(np.arange(512)/30)+.2*np.cos(np.arange(512)/7)
    values=np.stack([x.copy(),x.copy()]);obs=np.ones_like(values,bool);obs[:,497:511]=False
    values[0,~obs[0]]=1e20;values[1,~obs[1]]=-1e20
    times=np.broadcast_to(np.arange(512)*300.+1e9,values.shape).copy()
    np.savez_compressed(tmp_path/'input.npz',values=values,times=times,observed_mask=obs,padding_mask=np.zeros_like(obs),query_ids=np.asarray(['synthetic-a','synthetic-hidden-poison']))
    (tmp_path/'input.json').write_text(json.dumps({'schema_version':1,'task':'R','issuance_times':times[:,-1].tolist()}))
    command=[str(PYTHON),str(ROOT/'scripts/moment_worker.py'),'--input',str(tmp_path/'input.npz'),'--metadata',str(tmp_path/'input.json'),'--output',str(tmp_path/'prediction.npz'),'--result-metadata',str(tmp_path/'result.json'),'--checkpoint-cache',str(CACHE),'--audit-native','--device',device]
    first=subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert first.returncode==0,first.stderr
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']=='success' and result['native_parity_max_abs_error']==0
    assert result['environment']['device']==device
    assert result['environment']['packages']['numpy']=='1.25.2'
    assert result['environment']['packages']['huggingface-hub']=='0.24.0'
    assert result['environment']['packages']['transformers']=='4.33.3'
    assert not result['environment']['dependency_override']
    with np.load(tmp_path/'prediction.npz',allow_pickle=False) as output:
        p=output['point_estimate']
        np.testing.assert_array_equal(p[0],p[1])
        np.testing.assert_array_equal(p[obs],values[obs])
        assert np.isfinite(p).all()
    digest=hashlib.sha256((tmp_path/'prediction.npz').read_bytes()).hexdigest()
    # A completed shard is reusable without importing torch or constructing a model.
    fake=tmp_path/'fake';fake.mkdir();(fake/'torch.py').write_text("raise RuntimeError('model import forbidden during completed resume')\n")
    env=dict(os.environ)
    if device=='cpu':env['PYTHONPATH']=str(fake)
    resumed=subprocess.run(command,capture_output=True,text=True,timeout=20,env=env)
    assert resumed.returncode==0,resumed.stderr
    assert json.loads(resumed.stdout)['status']=='already_complete'
    assert hashlib.sha256((tmp_path/'prediction.npz').read_bytes()).hexdigest()==digest


@pytest.mark.skipif(not AVAILABLE,reason='optional isolated MOMENT environment/checkpoint unavailable')
def test_primary_interpreter_bridge(tmp_path):
    from energygap.methods.moment_subprocess import MOMENTSubprocess
    x=np.sin(np.arange(512)/30);x[500:511]=np.nan;t=np.arange(512)*300.+1e9
    model=MOMENTSubprocess(repository_root=ROOT)
    output=model.impute(x,t)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[np.isfinite(x)],x[np.isfinite(x)])
    assert model.last_metadata['environment']['packages']['torch'].split('+')[0]=='2.5.1'
    assert model.last_metadata['subprocess_wall_seconds']>0
