"""Compare the convex specialization with the unchanged general solver and J."""
from dataclasses import replace

import numpy as np
import pytest

from energygap.evaluation.energy import integrate_signed, signed_energy_gradient
import energygap.methods.correction as correction


def objective(values, base, times, constraint, scale, weight, gamma):
    delta = (np.asarray(values) - np.asarray(base)) / scale
    energy = integrate_signed(values, times, semantics=constraint.semantics)['net_kwh']
    return (np.mean(delta[1:-1] ** 2) + gamma * np.mean(np.diff(delta) ** 2)
            + weight * max(0., abs(energy - constraint.net_kwh) / constraint.tau_net - 1.) ** 2)


@pytest.mark.parametrize('semantics', ['interval_mean_start', 'interval_mean_end'])
@pytest.mark.parametrize('gamma', [0., .01])
@pytest.mark.parametrize('weight', [1., 100.])
@pytest.mark.parametrize('tau', [.02, .5])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_exact_specialization_matches_existing_multistart_and_finite_gradient(
        monkeypatch, semantics, gamma, weight, tau, sign):
    base = np.array([2., -1., .3, -.8, 1.1, -.5])
    times = np.array([0., 90., 390., 1000., 1400., 2000.])
    mask = np.array([False, True, True, True, True, False])
    energy = integrate_signed(base, times, semantics=semantics)['net_kwh']
    constraint = correction.EnergyConstraint(net_kwh=energy + sign * (tau + .4), tau_net=tau,
                                              semantics=semantics)
    options = dict(scale_train=1.8, lambda_energy=weight, gamma=gamma,
                   seasonal_start=base * .7)
    exact = correction.correct_energy(base, times, mask, constraint, **options)
    monkeypatch.setattr(correction, '_exact_net_interval_mean', lambda *args, **kwargs: None)
    legacy = correction.correct_energy(base, times, mask, constraint, **options)
    assert exact.status == legacy.status == 'success'
    assert exact.diagnostics['solver_variant'] == 'exact_convex_net_interval_mean'
    assert exact.diagnostics['global_optimum_guaranteed'] is True
    assert exact.diagnostics['iterations'] == 0
    np.testing.assert_array_equal(exact.values[~mask], base[~mask])
    np.testing.assert_allclose(exact.values, legacy.values, atol=1e-6, rtol=1e-6)
    expected = objective(exact.values, base, times, constraint, 1.8, weight, gamma)
    assert exact.diagnostics['objective'] == pytest.approx(expected, abs=1e-13)
    assert expected <= legacy.diagnostics['objective'] + 1e-10
    assert expected <= exact.diagnostics['objective_before']
    # An independent finite difference of the full objective is stationary.
    for position in np.flatnonzero(mask):
        plus, minus = exact.values.copy(), exact.values.copy()
        plus[position] += 1e-6
        minus[position] -= 1e-6
        derivative = (objective(plus, base, times, constraint, 1.8, weight, gamma)
                      - objective(minus, base, times, constraint, 1.8, weight, gamma)) / 2e-6
        assert derivative == pytest.approx(0., abs=2e-7)


@pytest.mark.parametrize('semantics', ['interval_mean_start', 'interval_mean_end'])
@pytest.mark.parametrize('offset', [0., -.1, .1])
def test_deadzone_is_bitwise_identity_without_iterative_solver(monkeypatch, semantics, offset):
    base = np.array([3., -2., .7, -.3, 2.])
    times = np.array([0., 300., 700., 900., 1800.])
    energy = integrate_signed(base, times, semantics=semantics)['net_kwh']
    constraint = correction.EnergyConstraint(net_kwh=energy + offset, tau_net=.11, semantics=semantics)
    monkeypatch.setattr(correction, 'minimize', lambda *args, **kwargs: pytest.fail('iterative solver was called'))
    result = correction.correct_energy(base, times, [False, True, True, True, False], constraint, scale_train=2.)
    np.testing.assert_array_equal(result.values, base)
    assert result.diagnostics['objective'] == result.diagnostics['objective_before'] == 0.
    assert result.diagnostics['selected_start'] == 'base_identity'
    assert result.diagnostics['global_optimum_guaranteed'] is True


@pytest.mark.parametrize('semantics', ['interval_mean_start', 'interval_mean_end'])
@pytest.mark.parametrize('count', [1, 12, 96])
def test_irregular_weights_and_dirichlet_smoothing_match_dense_stationarity(semantics, count):
    rng = np.random.default_rng(718 + count)
    times = np.r_[0., np.cumsum(rng.uniform(10., 600., count + 1))]
    base = rng.normal(size=count + 2)
    mask = np.r_[False, np.ones(count, bool), False]
    scale, gamma, weight, tau = 7., .3, 100., .01
    initial = integrate_signed(base, times, semantics=semantics)['net_kwh']
    constraint = correction.EnergyConstraint(net_kwh=initial - .5, tau_net=tau, semantics=semantics)
    result = correction.correct_energy(base, times, mask, constraint,
        scale_train=scale, lambda_energy=weight, gamma=gamma)
    a = signed_energy_gradient(base, times, semantics=semantics)['net_kwh'][1:-1]
    laplacian = 2 * np.eye(count) - np.eye(count, k=1) - np.eye(count, k=-1)
    q = (np.eye(count) / count + gamma * laplacian / (count + 1)) / scale ** 2
    residual = integrate_signed(result.values, times, semantics=semantics)['net_kwh'] - constraint.net_kwh
    displacement = result.values[1:-1] - base[1:-1]
    gradient = 2 * q @ displacement + 2 * weight * (residual - tau) * a / tau ** 2
    np.testing.assert_allclose(gradient, 0., atol=2e-9)
    assert residual >= tau - 1e-13  # stationary point belongs to the active branch used
    np.testing.assert_array_equal(result.values[~mask], base[~mask])


@pytest.mark.parametrize('constraint', [
    correction.EnergyConstraint(net_kwh=.2, tau_net=.01, semantics='instantaneous'),
    correction.EnergyConstraint(net_kwh=.2, tau_net=.01, semantics='interval_mean'),
    correction.EnergyConstraint(import_kwh=.2, tau_import=.01, semantics='interval_mean_end', no_within_interval_reversal=True),
    correction.EnergyConstraint(import_kwh=.2, tau_import=.01, net_kwh=.2, tau_net=.01,
                               semantics='interval_mean_start', no_within_interval_reversal=True),
])
def test_other_semantics_and_constraints_retain_general_solver(monkeypatch, constraint):
    original = correction.minimize
    calls = []
    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(correction, 'minimize', record)
    result = correction.correct_energy([1., -.1, 1.], [0., 300., 600.], [False, True, False],
                                       constraint, scale_train=1.)
    assert calls and result.status == 'success'
    assert result.diagnostics['global_optimum_guaranteed'] is False
    assert 'solver_variant' not in result.diagnostics


def test_existing_validation_is_not_bypassed_and_factor_cache_reuses_only_geometry():
    constraint = correction.EnergyConstraint(net_kwh=.3, tau_net=.01, semantics='interval_mean_end')
    with pytest.raises(ValueError, match='Seasonal multistart shape'):
        correction.correct_energy([1., 1., 1.], [0., 300., 600.], [False, True, False],
                                  constraint, scale_train=1., seasonal_start=[1.])
    correction._interval_mean_quadratic_factor.cache_clear()
    for counter in [.3, -.4, .1]:
        correction.correct_energy([1., 1., 1.], [0., 300., 600.], [False, True, False],
                                  replace(constraint, net_kwh=counter), scale_train=2., gamma=.01)
    cache = correction._interval_mean_quadratic_factor.cache_info()
    assert cache.misses == 1 and cache.hits == 2
