"""Actual upstream SAITS architecture and native ORT+MIT gradient loss."""
import numpy as np
import pytest

pytest.importorskip('pypots')
torch=pytest.importorskip('torch')
from io import BytesIO
from energygap.methods.training import train_saits_steps,_mixed_mask
from energygap.runtime import StateStore,CheckpointStore


def _full_state(path):
    with StateStore(path,resume='require') as store:
        cp=CheckpointStore(path,store.fingerprint)
        blob=cp.load('saits:train')[0]['torch_state']
        return torch.load(BytesIO(blob),weights_only=False)


def _equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):_equal(x,y)
    else:assert a==b


def test_actual_saits_optimizer_resume_equal(tmp_path):
    x=np.sin(np.arange(6*16,dtype=np.float32).reshape(6,16,1)/9)
    x[0,1,0]=np.nan
    config={'seed':23,'max_steps':12,'batch_size':2,'checkpoint_every_steps':2,
            'architecture':{'n_layers':1,'d_model':16,'n_heads':2,'d_k':8,'d_v':8,'d_ffn':16,'dropout':.1}}
    continuous=train_saits_steps(x,tmp_path/'continuous',**config)
    stopped=train_saits_steps(x,tmp_path/'resumed',stop_after_steps=5,**config)
    assert not stopped['completed'] and stopped['checkpoint']['step']==5
    resumed=train_saits_steps(x,tmp_path/'resumed',**config)
    assert resumed['completed'] and resumed['initial']==5 and resumed['executed_steps']==7
    assert continuous['history']==resumed['history']
    expected=_full_state(tmp_path/'continuous');actual=_full_state(tmp_path/'resumed')
    _equal(expected,actual)
    assert expected['global_step']==12
    assert expected['optimizer']['state']
    assert expected['early_stopping']['best_step'] is not None
    assert expected['scheduler'] is None
    assert expected['sampler']['cursor']==6
    assert resumed['adapter'].supports_resume_training
    query=x[1,:,0].astype(float);query[5:8]=np.nan
    prediction=resumed['adapter'].infer(query)['point_estimate'][:,0]
    assert prediction.shape==(16,) and np.isfinite(prediction).all()
    assert np.array_equal(prediction[np.isfinite(query)],query[np.isfinite(query)])
    completed=train_saits_steps(x,tmp_path/'resumed',**config)
    assert completed['executed_steps']==0


def test_dynamic_mixed_masks_supervise_only_original_support():
    observed=np.ones((20,16,1),bool);observed[:,3:5]=False
    hidden=_mixed_mask(observed,np.random.default_rng(12))
    assert not hidden[:,3:5].any()
    assert (observed&~hidden).sum()>0
    assert hidden.any()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA hardware required')
def test_actual_saits_cuda_optimizer_steps_and_resume(tmp_path):
    x=np.sin(np.arange(6*16,dtype=np.float32).reshape(6,16,1)/9)
    config={'seed':23,'max_steps':4,'batch_size':2,'checkpoint_every_steps':1,'device':'cuda:0',
            'architecture':{'n_layers':1,'d_model':16,'n_heads':2,'d_k':8,'d_v':8,'d_ffn':16,'dropout':.1}}
    continuous=train_saits_steps(x,tmp_path/'continuous',**config)
    train_saits_steps(x,tmp_path/'resumed',stop_after_steps=2,**config)
    resumed=train_saits_steps(x,tmp_path/'resumed',**config)
    assert resumed['history']==continuous['history'] and resumed['initial']==2
    assert next(resumed['adapter'].model.model.parameters()).device.type=='cuda'
    expected=_full_state(tmp_path/'continuous');actual=_full_state(tmp_path/'resumed')
    assert expected['runtime']['execution']['device']=='cuda:0'
    assert expected['rng_torch_cuda'] is not None
    _equal(expected,actual)


@pytest.mark.parametrize('signal_name,expected_initial',[('SIGKILL',4),('SIGTERM',5)])
def test_actual_saits_real_process_kill(tmp_path,signal_name,expected_initial):
    import json,os,signal,subprocess,sys
    x=np.sin(np.arange(6*16,dtype=np.float32).reshape(6,16,1)/9)
    x[0,1,0]=np.nan
    config={'seed':23,'max_steps':12,'batch_size':2,'checkpoint_every_steps':2,
            'architecture':{'n_layers':1,'d_model':16,'n_heads':2,'d_k':8,'d_v':8,'d_ffn':16,'dropout':.1}}
    train_saits_steps(x,tmp_path/'continuous',**config)
    script=r'''
import json,sys,numpy as np
from energygap.methods.training import train_saits_steps
from energygap.runtime import StateStore
original=StateStore.finish_work
counter=0
def hook(self,reservation,seconds):
    global counter
    result=original(self,reservation,seconds);counter+=1
    if counter==5:
        print('RECOVERY_READY '+json.dumps({'step':5}),flush=True)
        sys.stdin.readline()
    return result
StateStore.finish_work=hook
x=np.sin(np.arange(6*16,dtype=np.float32).reshape(6,16,1)/9);x[0,1,0]=np.nan
train_saits_steps(x,sys.argv[1],seed=23,max_steps=12,batch_size=2,checkpoint_every_steps=2,
architecture={'n_layers':1,'d_model':16,'n_heads':2,'d_k':8,'d_v':8,'d_ffn':16,'dropout':.1})
'''
    proc=subprocess.Popen([sys.executable,'-c',script,str(tmp_path/'resumed')],stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    while True:
        line=proc.stdout.readline()
        if line.startswith('RECOVERY_READY '):break
        if not line:
            _,err=proc.communicate(timeout=10);pytest.fail('worker stopped before boundary: '+err[-2000:])
    proc.send_signal(getattr(signal,signal_name))
    if signal_name=='SIGTERM':proc.stdin.write('\n');proc.stdin.flush()
    proc.communicate(timeout=20)
    result=train_saits_steps(x,tmp_path/'resumed',**config)
    assert result['initial']==expected_initial
    _equal(_full_state(tmp_path/'continuous'),_full_state(tmp_path/'resumed'))
    with StateStore(tmp_path/'resumed',resume='require') as store:
        assert len(store.results())==12
        assert len({r['key'] for r in store.results()})==12
