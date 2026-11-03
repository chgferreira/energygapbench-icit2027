"""CPU optimizer-loop fixture; this does not certify any official catalogue adapter."""
import hashlib
import random
import numpy as np
import pytest
from energygap.runtime import StateStore,CheckpointStore
from energygap.runtime.training import capture_torch_state,restore_torch_state

torch=pytest.importorskip('torch')


def test_torch_training_full_state_exact_resume(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x=torch.arange(48,dtype=torch.float64).reshape(12,4)/40
    y=x@torch.tensor([[.3],[-.2],[.9],[.1]],dtype=torch.float64)+.2

    def initialize():
        random.seed(42);np.random.seed(19);torch.manual_seed(7)
        rng=np.random.default_rng(11)
        model=torch.nn.Linear(4,1).double()
        opt=torch.optim.Adam(model.parameters(),lr=.03)
        sch=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.99)
        return model,opt,sch,rng,{'permutation':rng.permutation(12),'cursor':0}, {'best':float('inf'),'bad_epochs':0},[]

    def advance(state,start,end):
        model,opt,sch,rng,sampler,early,history=state
        for step in range(start,end):
            if sampler['cursor']==12:
                sampler['permutation']=rng.permutation(12);sampler['cursor']=0
            batch=sampler['permutation'][sampler['cursor']:sampler['cursor']+4];sampler['cursor']+=4
            mask=torch.rand((4,4))>.05
            # Every RNG is exercised and restored; targets remain original observed data.
            dynamic=torch.where(mask,x[batch],torch.zeros_like(x[batch]))
            dynamic+=float(rng.random())*1e-7+float(np.random.random())*1e-7+random.random()*1e-7
            opt.zero_grad();loss=torch.mean((model(dynamic)-y[batch])**2);loss.backward();opt.step();sch.step()
            value=float(loss.detach());early['bad_epochs']=early['bad_epochs']+1 if value>=early['best'] else 0
            early['best']=min(value,early['best'])
            history.append((batch.tolist(),hashlib.sha256(mask.numpy().tobytes()).hexdigest(),value))
        return state

    full=advance(initialize(),0,30)
    expected_bytes=capture_torch_state(full[0],full[1],scheduler=full[2],epoch=10,global_step=30,
                                      sampler_state=full[4],early_stopping=full[5],trial_state={'id':0,'lr':.03},numpy_rng=full[3],extra={'history':full[6]})
    partial=advance(initialize(),0,13)
    blob=capture_torch_state(partial[0],partial[1],scheduler=partial[2],epoch=4,global_step=13,
                            sampler_state=partial[4],early_stopping=partial[5],trial_state={'id':0,'lr':.03},numpy_rng=partial[3],extra={'history':partial[6]})
    store=StateStore(tmp_path,{'kind':'torch_optimizer_fixture','seed':7})
    cp=CheckpointStore(tmp_path,store.fingerprint)
    cp.save('train',{'torch_state':blob},13)
    restored=initialize()
    metadata=restore_torch_state(cp.load('train')[0]['torch_state'],restored[0],restored[1],scheduler=restored[2],numpy_rng=restored[3])
    assert metadata['global_step']==13 and metadata['trial']=={'id':0,'lr':.03}
    resumed=advance((*restored[:4],metadata['sampler'],metadata['early_stopping'],metadata['extra']['history']),13,30)
    final_bytes=capture_torch_state(resumed[0],resumed[1],scheduler=resumed[2],epoch=10,global_step=30,
                                   sampler_state=resumed[4],early_stopping=resumed[5],trial_state={'id':0,'lr':.03},numpy_rng=resumed[3],extra={'history':resumed[6]})
    # Torch archive metadata may differ; compare every scientific state recursively.
    from io import BytesIO
    expected=torch.load(BytesIO(expected_bytes),weights_only=False)
    actual=torch.load(BytesIO(final_bytes),weights_only=False)
    def equal(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for aa,bb in zip(a,b):equal(aa,bb)
        else:assert a==b
    equal(expected,actual)
    assert resumed[6]==full[6]
    assert resumed[2].last_epoch==30


def test_tiny_torch_loop_can_overfit_and_serialize(tmp_path):
    torch.manual_seed(17);torch.set_num_threads(1)
    model=torch.nn.Linear(2,1).double();optimizer=torch.optim.Adam(model.parameters(),lr=.08)
    x=torch.tensor([[0.,0.],[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64)
    truth=x@torch.tensor([[2.],[-1.]],dtype=torch.float64)+.4
    for _ in range(250):
        optimizer.zero_grad();loss=torch.mean((model(x)-truth)**2);loss.backward();optimizer.step()
    assert float(loss.detach())<1e-9


def test_training_restore_rejects_runtime_and_precision_mismatch_before_mutation():
    from io import BytesIO
    model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters())
    blob=capture_torch_state(model,optimizer,epoch=0,global_step=0,sampler_state={},early_stopping={},trial_state={})
    state=torch.load(BytesIO(blob),weights_only=False)
    assert state['runtime']['execution']['device']=='cpu'
    assert state['runtime']['dtypes']==['torch.float32']
    before={key:value.clone() for key,value in model.state_dict().items()}
    for replacement in ({**state['runtime'],'dtypes':['torch.float64']},
                        {**state['runtime'],'execution':{**state['runtime']['execution'],'device':'cuda:0'}}):
        changed={**state,'runtime':replacement};buffer=BytesIO();torch.save(changed,buffer)
        with pytest.raises(ValueError,match='device, precision or runtime differs'):
            restore_torch_state(buffer.getvalue(),model,optimizer)
        for key,value in model.state_dict().items():assert torch.equal(value,before[key])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual CUDA hardware required')
def test_cuda_optimizer_and_rng_resume_remain_on_cuda():
    from energygap.runtime.device import resolve_device
    device=resolve_device('cuda:0',allow_missing_torch=False)
    torch.use_deterministic_algorithms(True)
    def initialize():
        torch.manual_seed(19);torch.cuda.manual_seed_all(19)
        model=torch.nn.Linear(4,1).to(device)
        return model,torch.optim.Adam(model.parameters(),lr=.03)
    def advance(model,optimizer,steps):
        for _ in range(steps):
            x=torch.rand((4,4),device=device)
            optimizer.zero_grad();loss=(model(x)-x.sum(1,keepdim=True)).square().mean()
            loss.backward();optimizer.step()
    model,optimizer=initialize();advance(model,optimizer,2)
    blob=capture_torch_state(model,optimizer,epoch=0,global_step=2,sampler_state={},early_stopping={},trial_state={})
    advance(model,optimizer,3)
    expected={key:value.clone() for key,value in model.state_dict().items()}
    restored,resumed_optimizer=initialize()
    restore_torch_state(blob,restored,resumed_optimizer)
    assert all(p.device.type=='cuda' for p in restored.parameters())
    for parameter,values in resumed_optimizer.state.items():
        for key,value in values.items():
            if isinstance(value,torch.Tensor) and key!='step':assert value.device==parameter.device
    advance(restored,resumed_optimizer,3)
    for key,value in restored.state_dict().items():assert torch.equal(value,expected[key])
