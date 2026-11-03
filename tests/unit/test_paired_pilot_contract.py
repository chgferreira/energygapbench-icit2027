import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path=Path(__file__).resolve().parents[2]/'scripts/paired_integrity_pilot.py'
    spec=importlib.util.spec_from_file_location('energygap_test_paired_pilot',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_recovered_query_uses_no_unavailable_prediction_or_hidden_truth():
    paired=module()
    available=np.ones(512,bool);available[450:511]=False
    old=np.linspace(-2,2,512);poisoned=old.copy();poisoned[~available]=1e20
    a=paired.safe_query_input(old,available);b=paired.safe_query_input(poisoned,available)
    np.testing.assert_array_equal(a,b)
    assert np.isnan(a[~available]).all()
    np.testing.assert_array_equal(a[available],old[available])


def test_prediction_artifact_checksum_required_before_reuse(tmp_path):
    paired=module();path=tmp_path/'predictions.npz'
    np.savez(path,point=np.array([1.,2.]))
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(paired.verified_array(path,expected)['point'],[1,2])
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        paired.verified_array(path,expected)
