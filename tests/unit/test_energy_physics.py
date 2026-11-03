import numpy as np
import pytest

from energygap.evaluation.energy import integrate_signed, counter_increment, signed_energy_gradient
from energygap.methods.correction import EnergyConstraint, correct_energy, calibrate_tolerance
from energygap.methods.probabilistic import ProbabilisticCapabilities, correct_joint_samples


def test_signed_crossing_exact_and_irregular_borders():
    got = integrate_signed([2, -1], [0, 3600])
    assert got["import_kwh"] == pytest.approx(2/3)
    assert got["export_kwh"] == pytest.approx(1/6)
    assert got["net_kwh"] == pytest.approx(.5)
    irregular = integrate_signed([0, 2, 0], [0, 900, 3600])
    assert irregular["import_kwh"] == pytest.approx(1)
    reverse = integrate_signed([0, -2, 0], [0, 900, 3600])
    assert reverse["export_kwh"] == pytest.approx(1)


@pytest.mark.parametrize("values", [[2., -1., 3., -2.], [1., 2., 3., 4.], [-1., -2., -3., -4.], [0., 0., 0., 0.]])
def test_exact_signed_energy_gradient_against_symmetric_finite_difference(values):
    times = [0., 300., 700., 1800.]
    gradient = signed_energy_gradient(values, times)
    epsilon = 1e-6
    for direction in ("import_kwh", "export_kwh", "net_kwh"):
        for j in range(len(values)):
            plus, minus = np.array(values), np.array(values)
            plus[j] += epsilon; minus[j] -= epsilon
            finite = (integrate_signed(plus, times)[direction]-integrate_signed(minus, times)[direction])/(2*epsilon)
            assert gradient[direction][j] == pytest.approx(finite, abs=1e-8)


def test_interval_mean_end_semantics():
    assert integrate_signed([100, 2, -1], [0, 1800, 3600], semantics="interval_mean") == {
        "import_kwh": 1., "export_kwh": .5, "net_kwh": .5}


def test_counters_never_abs_negative_and_constant_valid():
    assert counter_increment(10, 3)[0] is None
    assert counter_increment(10, 3, reset=True) == (None, "reset")
    assert counter_increment(10, 10) == (0., "valid")
    assert counter_increment(99, 2, rollover=100, rollover_confirmed=True) == (3., "known_rollover")
    assert counter_increment(99, 2, rollover=100)[0] is None


def test_tolerance_train_only_fallback_not_numeric_epsilon():
    result = calibrate_tolerance({15: [0.01]*40, 60: [0.04]}, resolution_kwh=.1)
    assert result["15"]["tau_kwh"] == .1
    assert result["60"]["source"] == "pooled_train_q95_fallback"
    assert calibrate_tolerance({15: [0.]}, resolution_kwh=None)["15"]["tau_kwh"] is None
    assert calibrate_tolerance({15: []}, resolution_kwh=.1)["15"]["source"] == "resolution_only_uncalibrated_fallback"
    with pytest.raises(ValueError):
        calibrate_tolerance({15: [1]}, resolution_kwh=.1, split="test")


def test_correction_target_only_observed_boundaries_and_uncertainty():
    t = np.arange(7.) * 300
    base = np.array([9., 2., .1, .1, .1, 2., 9.])
    mask = np.array([False, False, True, True, True, False, False])
    c = EnergyConstraint(import_kwh=.5, export_kwh=0, tau_import=.01, tau_export=.01)
    result = correct_energy(base, t, mask, c, scale_train=3, lambda_energy=10)
    assert result.status == "success"
    np.testing.assert_array_equal(result.values[~mask], base[~mask])
    assert result.diagnostics["objective"] <= result.diagnostics["objective_before"]
    before = integrate_signed(base[1:6], t[1:6])["import_kwh"]
    after = integrate_signed(result.values[1:6], t[1:6])["import_kwh"]
    assert abs(after-.5) < abs(before-.5)
    assert result.diagnostics["boundary_indices"] == [1, 5]


def test_disabled_or_unreliable_energy_is_identity():
    base, t, mask = [1., 1., 1.], [0, 300, 600], [False, True, False]
    for c, kw, expected in [(EnergyConstraint(import_kwh=1, tau_import=.1), {"lambda_energy": 0}, "disabled"),
                             (EnergyConstraint(reliable=False, reason="counter_delayed"), {}, "unavailable"),
                             (EnergyConstraint(import_kwh=-1, tau_import=.1), {}, "unavailable"),
                             (EnergyConstraint(import_kwh=1), {}, "unavailable")]:
        result = correct_energy(base, t, mask, c, scale_train=1, **kw)
        assert result.status == expected
        np.testing.assert_array_equal(result.values, base)
    result=correct_energy(base,t,mask,EnergyConstraint(import_kwh=1,tau_import=.1),scale_train=None)
    assert result.status=='unavailable' and result.diagnostics['reason']=='nonpositive_training_scale'


def test_interval_mean_does_not_identify_gross_bidirectional():
    result = correct_energy([1, 0, 1], [0, 300, 600], [False, True, False],
                            EnergyConstraint(import_kwh=.2, export_kwh=.2, tau_import=.01,
                                             tau_export=.01, semantics="interval_mean"), scale_train=1)
    assert result.status == "unsupported"
    assert "identify" in result.diagnostics["reason"]


def test_correction_deterministic_multistart_and_no_truth_parameter():
    c = EnergyConstraint(import_kwh=.4, export_kwh=.2, tau_import=.02, tau_export=.02)
    args = ([1, .1, -.1, -.5], [0, 300, 1000, 1800], [False, True, True, False], c)
    a = correct_energy(*args, scale_train=2)
    b = correct_energy(*args, scale_train=2)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.diagnostics["objective"] == b.diagnostics["objective"]


def test_all_solver_failures_retain_exact_base(monkeypatch):
    import energygap.methods.correction as correction
    from types import SimpleNamespace
    def fail_solver(fun,start,**kwargs):
        return SimpleNamespace(success=False,x=np.array(start),fun=12.,nit=3,message='controlled failure')
    monkeypatch.setattr(correction,'minimize',fail_solver)
    base=np.array([1., .1, 1.])
    result=correct_energy(base,[0,300,600],[False,True,False],
                          EnergyConstraint(import_kwh=.4,tau_import=.01),scale_train=2)
    assert result.status=='failed'
    np.testing.assert_array_equal(result.values,base)
    assert all(not trial['converged'] for trial in result.diagnostics['trials'])


def test_probabilistic_quantiles_crps_recomputed_after_sample_correction():
    c = EnergyConstraint(import_kwh=.3, tau_import=.01)
    samples = np.array([[1, .1, 1], [1, .2, 1], [1, .3, 1]])
    result = correct_joint_samples(samples, [0, 300, 600], [False, True, False], c,
        capabilities=ProbabilisticCapabilities(joint_samples=True), scale_train=1,
        lambda_energy=10)
    from energygap.evaluation.probabilistic import sample_crps, summarize_samples
    scored=summarize_samples(result['samples'],[1.,2.,1.],[False,True,False],scale_train=1)
    np.testing.assert_allclose(result["summary"]["point"], np.median(result["samples"], axis=0))
    assert scored['crps'] == sample_crps(result["samples"], [1, 2, 1], [False, True, False])
    assert 'crps' not in result['summary']
    assert not np.array_equal(result["summary"]["quantiles"]["0.5"], np.median(samples, axis=0))
    with pytest.raises(ValueError, match="joint"):
        correct_joint_samples(samples, [0, 300, 600], [False, True, False], c,
            capabilities=ProbabilisticCapabilities(marginal_quantiles=True), scale_train=1)
    with pytest.raises(ValueError,match='outside'):
        correct_joint_samples(samples,[0,300,600],[False,True,False],c,
            capabilities=ProbabilisticCapabilities(joint_samples=True),scale_train=1,truth=[1,2,1])
