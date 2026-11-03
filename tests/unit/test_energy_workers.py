"""Spawned CPU corrections retain all paths and coordinator ordering/isolation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from energygap.methods.correction import EnergyConstraint, correct_energy
from energygap.methods.probabilistic import correct_joint_samples, ProbabilisticCapabilities
from energygap.runtime.energy_workers import EnergyCorrectionPool, EnergyEpisodeTask, EnergyWorkCancelled, correct_episode


def task(samples=100):
    base=np.sin(np.arange(32)/5)+.3
    times=np.arange(32)*300.
    mask=np.zeros(32,bool);mask[13:16]=True
    seasonal=base.copy();seasonal[mask]+=.3
    draws=np.repeat(base[None],samples,axis=0)
    draws[:,mask]+=np.random.default_rng(11).normal(0,.2,(samples,mask.sum()))
    hypotheses=tuple((name,EnergyConstraint(net_kwh=.08,tau_net=.01,semantics=name),
                      dict(lambda_energy=1.,gamma=.01))
        for name in ('instantaneous','interval_mean_start','interval_mean_end'))
    return EnergyEpisodeTask(base,times,mask,seasonal,2.,hypotheses,draws)


def test_spawned_point_and_all_100_trajectories_match_original_api():
    case=task()
    with EnergyCorrectionPool(2,2) as pool:
        actual=next(pool.map_ordered([case]))
    for output,(hypothesis,constraint,parameters) in zip(actual,case.hypotheses):
        options=dict(scale_train=case.scale,seasonal_start=case.seasonal,**parameters)
        point=correct_energy(case.base,case.times,case.target_mask,constraint,**options)
        samples=correct_joint_samples(case.samples,case.times,case.target_mask,constraint,
            capabilities=ProbabilisticCapabilities(joint_samples=True),**options)
        assert output['hypothesis']==hypothesis
        np.testing.assert_allclose(output['point'].values,point.values,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(output['samples']['samples'],samples['samples'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(output['samples']['summary']['point'],samples['summary']['point'],rtol=1e-12,atol=1e-12)
        assert output['samples']['samples'].shape==(100,32)
        assert output['samples']['sample_statuses']==samples['sample_statuses']
        assert output['point'].status==point.status
        assert output['samples']['summary']['sample_count']==100
        np.testing.assert_array_equal(output['samples']['samples'][:,~case.target_mask],case.samples[:,~case.target_mask])


def test_spawned_workers_have_one_cpu_thread_and_never_import_torch():
    pytest.importorskip('threadpoolctl')
    with EnergyCorrectionPool(2,2) as pool:
        list(pool.map_ordered([task(samples=2)]))
        diagnostics=pool.diagnostics()
    # Accelerate on macOS is not exposed by threadpoolctl. Linux OpenBLAS/MKL
    # must expose real pools, whose live limits are checked rather than inferred.
    assert diagnostics and all(item['pools'] or item['numpy_blas']['name']=='accelerate' for item in diagnostics)
    assert all(not item['torch_imported'] for item in diagnostics)
    assert all(pool['num_threads']==1 for item in diagnostics for pool in item['pools'])
    assert all(set(item['environment'].values())=={'1'} for item in diagnostics)


def test_bounded_producer_returns_results_in_episode_order_and_cancel_releases_pool():
    case=task(samples=2)
    produced=[]
    def cases():
        for index in range(8):
            produced.append(index)
            yield replace(case,base=case.base+index,samples=None)
    stop=SimpleNamespace(requested=False)
    with EnergyCorrectionPool(2,3) as pool:
        results=pool.map_ordered(cases(),stop=stop)
        first=next(results)
        assert produced==[0,1,2]
        assert first[0]['point'].values[0]==case.base[0]
        second=next(results)
        assert second[0]['point'].values[0]==case.base[0]+1
        assert produced==[0,1,2,3]
        stop.requested=True
        with pytest.raises(EnergyWorkCancelled):
            next(results)
        assert pool._executor is None


def test_worker_error_does_not_turn_into_a_correction_fallback_or_leak_pool():
    case=replace(task(samples=2),base=np.zeros(3))
    with EnergyCorrectionPool(2,2) as pool:
        with pytest.raises(ValueError):
            list(pool.map_ordered([case]))
        assert pool._executor is None


def test_serial_joint_cancellation_checks_between_complete_paths():
    case=task(samples=100)
    calls=[]
    def cancel():
        calls.append(1)
        if len(calls)==4:
            raise EnergyWorkCancelled('stop between draws')
    with pytest.raises(EnergyWorkCancelled,match='between draws'):
        correct_episode(case,check=cancel)
    assert len(calls)==4


@pytest.mark.parametrize('workers',[None,'eight',0,True])
def test_invalid_worker_count_has_explicit_configuration_error(workers):
    from energygap.conference import load_settings
    config=dict(sites=['SYNTH'],folds=[1],conditions=['C3'],methods=['linear'],train_seeds=[11],
                validation_mask_seeds=[1001],test_mask_seeds=[2001],energy_cpu_workers=workers)
    with pytest.raises(ValueError,match='energy_cpu_workers'):
        load_settings('.',settings=config)
