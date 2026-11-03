from energygap.statistics.paired import holm, paired_bootstrap, fixed_estimand, bootstrap_replicate
import pytest


def rows():
    return [dict(fold=1,day=f'2026-01-{i:02}',duration=d,gap_id=f'{i}-{d}',difference=0.) for i in range(1,29) for d in (15,60,180)]


def test_holm_preserves_fixed_family_and_missing():
    assert holm([.01,.04,.03],3)==pytest.approx([.03,.06,.06])
    assert holm([None,.001],12)==[None,.012]


def test_no_support_does_not_redistribute_weights():
    assert fixed_estimand(rows()) is None


def test_identical_methods_and_duplicate_seed_add_no_temporal_support():
    r=rows();calendar={1:[f'2026-01-{i:02}' for i in range(1,29)]}
    a=paired_bootstrap(r,calendar,replicates=20,folds=(1,))
    b=paired_bootstrap(r+r,calendar,replicates=20,folds=(1,))
    assert a==b
    assert a['estimate']==0 and a['ci95']==[0,0]
    assert a['p_raw'] is None and not a['confirmatory']


def test_bootstrap_deterministic_ids_and_empty_strata():
    r=rows();cal={1:[f'2026-01-{i:02}' for i in range(1,29)]}
    assert bootstrap_replicate(r,cal,17,folds=(1,))==bootstrap_replicate(r,cal,17,folds=(1,))
    assert bootstrap_replicate(r,cal,17,folds=(1,),durations=(720,)) is None
