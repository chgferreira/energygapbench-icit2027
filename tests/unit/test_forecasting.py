import numpy as np
import pytest
from energygap.forecasting.ridge import causal_fill, DirectRidge, match_target


def test_causal_fill_preserves_zeros():
    assert causal_fill([np.nan,0,np.nan,3,np.nan]).tolist()==[0,0,0,3,3]


def test_physical_matching_and_ties():
    assert match_target([3590,3610,10000],[True]*3,0,3600)==(0,-10.)
    assert match_target([3700],[True],0,3600)==(None,None)
    assert match_target([3600],[False],0,3600)==(None,None)


def test_ridge_fits_only_original_training_targets():
    t=np.arange(1000)*300.
    y=np.sin(t/86400*2*np.pi)
    y[800]=np.nan
    m=DirectRidge(history=20).fit(y,t)
    assert m.n_training_targets==968
    assert np.isfinite(m.predict(y[700:720],t[719]))
    with pytest.raises(ValueError,match='training'):
        DirectRidge().fit(y,t,split='test')
