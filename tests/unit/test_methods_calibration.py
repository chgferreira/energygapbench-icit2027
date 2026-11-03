from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from energygap.data.fixture import make_fixture
from energygap.protocols.splits import rolling_origin_splits
from energygap.methods.calibration import fit_energy_tolerances, select_correction_parameters


def test_energy_tolerance_calibration_excludes_every_nontrain_value():
    series=make_fixture(days=10)
    fold=rolling_origin_splits(series)[0]
    train=fold.indices(series,'train')
    expected=fit_energy_tolerances(series,train)
    poisoned=deepcopy(series)
    for i,r in enumerate(poisoned.records):
        if i not in train:
            poisoned.records[i]=replace(r,value_canonical=1e10,import_counter=-1e10,export_counter=1e10)
    assert expected==fit_energy_tolerances(poisoned,train)
    assert expected['directions']['import']['15']['tau_kwh']>=.001
    with pytest.raises(ValueError,match='train'):
        fit_energy_tolerances(series,train,split='test')


def test_correction_selection_cannot_use_test_or_silently_drop_duration():
    with pytest.raises(ValueError,match='validation'):
        select_correction_parameters({},[],{},1,partition='test')
    result=select_correction_parameters({'linear':object()},[],{},1)
    assert result['linear']['status']=='not_estimable'
    assert result['linear']['selected']['lambda_energy']==0
