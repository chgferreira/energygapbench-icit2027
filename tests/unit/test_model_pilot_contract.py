"""Train-window and completed-fit contracts for the separate optional real pilot."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from energygap.data.fixture import make_fixture
from energygap.protocols.splits import rolling_origin_splits


def pilot_module():
    path=Path(__file__).resolve().parents[2]/'scripts/pilot_models.py'
    spec=importlib.util.spec_from_file_location('energygap_test_model_pilot',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_optional_pilot_windows_train_only_and_cannot_see_future_poison():
    module=pilot_module();series=make_fixture(days=7)
    fold=rolling_origin_splits(series)[0]
    expected,manifest=module.training_windows(series,fold,16)
    assert expected.shape==(16,512,1)
    assert all(fold.start.isoformat()<=time<fold.train_end.isoformat() for time in manifest['end_times'])
    for i,record in enumerate(series.records):
        if fold.partition(record.grid_time)!='train':
            series.records[i]=replace(record,value_canonical=1e10)
    actual,second=module.training_windows(series,fold,16)
    np.testing.assert_array_equal(actual,expected)
    assert manifest==second


def test_optional_pilot_rejects_inadequate_L512_support():
    module=pilot_module();series=make_fixture(days=1)
    with pytest.raises(ValueError,match='L=512'):
        module.training_windows(series,rolling_origin_splits(series)[0],16)


def test_native_optional_completed_fit_roundtrip_preserves_inference():
    pytest.importorskip('pypots')
    module=pilot_module()
    train=np.tile(np.sin(np.arange(512)/40)[None,:,None],(16,1,1)).astype(np.float32)
    adapter,payload=module.native_fit('SAITS',train,epochs=1,seed=11)
    query=train[0,:,0].copy();query[500:511]=np.nan
    first=adapter.infer(query)['point_estimate']
    restored=module.load_native(payload)
    second=restored.infer(query)['point_estimate']
    np.testing.assert_array_equal(first,second)
    assert payload['supports_in_progress_training_resume'] is False
