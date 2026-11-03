import numpy as np
import pytest
from energygap.evaluation.energy import integrate_signed,signed_energy_gradient
from energygap.methods.correction import correct_energy,EnergyConstraint


@pytest.mark.parametrize('semantics,expected',[('interval_mean_start',3.),('interval_mean_end',5.)])
def test_mean_timestamp_placement_is_explicit(semantics,expected):
    p=np.array([1.,2.,3.]);t=np.array([0.,3600.,7200.])
    assert integrate_signed(p,t,semantics=semantics)['net_kwh']==expected
    gradient=signed_energy_gradient(p,t,semantics=semantics)['net_kwh']
    for i in range(3):
        altered=p.copy();altered[i]+=.0001
        numerical=(integrate_signed(altered,t,semantics=semantics)['net_kwh']-expected)/.0001
        assert gradient[i]==pytest.approx(numerical)


def test_interval_start_does_not_identify_gross_flows():
    result=correct_energy([1.,1.,1.],[0,300,600],[False,True,False],EnergyConstraint(import_kwh=1.,tau_import=.1,semantics='interval_mean_start'),scale_train=1.)
    assert result.status=='unsupported'
