import numpy as np
import pytest
from energygap.methods.foundation import prepare_moment_input, prepare_fim_temporal_batch, FIMTemporalFrozen


def test_moment_never_normalizes_hidden_placeholder_or_padding():
    x=np.arange(512,dtype=float);mask=np.ones(512,bool);mask[250:253]=False;pad=np.zeros(512,bool);pad[:5]=True
    a=prepare_moment_input(x,mask,pad)
    x[~mask]=1e25;x[pad]=-1e30
    b=prepare_moment_input(x,mask,pad)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert np.all(a['x_enc'][0,0,250:253]==0)
    assert np.all(a['mask'][0,:5]==0) and np.all(a['input_mask'][0,:5]==0)
    with pytest.raises(ValueError):prepare_moment_input(np.ones(1024))


def test_fim_temporal_batch_excludes_truth_and_has_physical_endpoints():
    x=np.sin(np.arange(512)/30);t=np.arange(512)*300.+1e9
    target=np.zeros(512,bool);target[499:511]=True
    observed=~target
    a,indices=prepare_fim_temporal_batch(x,t,target,observed)
    x[target]=1e20
    b,other=prepare_fim_temporal_batch(x,t,target,observed)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    assert 'target_sample_path' not in a and 'initial_conditions' not in a
    np.testing.assert_allclose(a['location_times'][0,:,0],(t[498:512]-t[0])/(t[-1]-t[0]))
    assert a['observation_mask'][0,1,:,0].sum()==a['observation_mask'].shape[2]-1
    assert sum((~a['observation_mask']).ravel())==500
    assert len(indices)==12


def test_fim_unresolved_checkpoint_terms_are_explicit(tmp_path):
    model=FIMTemporalFrozen(tmp_path)
    with pytest.raises(RuntimeError,match='license unresolved'):model._load()
