"""Compare vectorized execution with the former scalar scientific contract."""
import numpy as np
import pytest
from scipy.linalg import solve

from energygap.forecasting.ridge import DirectRidge, causal_fill


def _scalar_fill(values, initial=0.):
    result = np.asarray(values, float).copy()
    previous = float(initial)
    for index, value in enumerate(result):
        if np.isfinite(value):
            previous = value
        else:
            result[index] = previous
    return result


def _scalar_features(values, issuance, history, initial=0.):
    values = np.asarray(values, float)
    if len(values) < history:
        values = np.pad(values, (history - len(values), 0), constant_values=initial)
    values = _scalar_fill(values[-history:], initial)
    phase = float(issuance) / 86400 * 2 * np.pi
    return np.r_[values, np.sin(phase), np.cos(phase), np.sin(phase / 7), np.cos(phase / 7)]


@pytest.mark.parametrize("initial", [0., 2.5, -0., np.nan, np.inf, -np.inf])
def test_vectorized_causal_fill_preserves_scalar_nan_inf_and_initial_behavior(initial):
    rng = np.random.default_rng(47)
    for size in (0, 1, 288, 1000):
        values = rng.normal(size=size)
        values[rng.random(size) < .2] = np.nan
        values[rng.random(size) < .1] = np.inf
        values[rng.random(size) < .1] = -np.inf
        values[:min(size, 3)] = np.nan
        if size > 4:
            values[4] = -0.
        original = values.copy()
        actual, expected = causal_fill(values, initial), _scalar_fill(values, initial)
        assert np.array_equal(actual, expected, equal_nan=True)
        assert np.array_equal(np.signbit(actual), np.signbit(expected))
        assert np.array_equal(values, original, equal_nan=True)


def test_features_many_matches_scalar_with_ragged_dense_empty_and_truncated_histories():
    model = DirectRidge(history=8)
    model.initial = 1.25
    # The 99 lies outside the retained tail: it must not fill that tail's gap.
    histories = [np.empty(0), np.array([np.nan, 0., np.inf, -np.inf, 3.]),
                 np.r_[99., np.full(8, np.nan)], np.arange(18, dtype=float)]
    times = np.asarray([1700000000., 1700000300., 1700000600., 1700000900.])
    expected = np.asarray([_scalar_features(h, t, 8, model.initial) for h, t in zip(histories, times)])
    np.testing.assert_allclose(model.features_many(histories, times), expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(model.features_many(iter(histories), iter(times)), expected, rtol=0, atol=1e-15)
    assert np.array_equal(model.features_many(histories, times)[2, :8], np.full(8, model.initial))
    for width in (0, 3, 8, 15):
        dense = np.arange(4 * width, dtype=float).reshape(4, width)
        dense[:, 1::3] = np.nan
        original = dense.copy()
        expected = np.asarray([_scalar_features(h, t, 8, model.initial) for h, t in zip(dense, times)])
        np.testing.assert_allclose(model.features_many(dense, times), expected, rtol=0, atol=1e-15)
        assert np.array_equal(dense, original, equal_nan=True)
    assert model.features_many([], []).shape == (0, 12)
    with pytest.raises(ValueError, match="issuance"):
        model.features_many(histories, times[:-1])


def _scalar_fit(values, times, history, steps, alpha, bounds):
    filled = _scalar_fill(values)
    rows, labels, target_indices = [], [], []
    for i in range(history - 1, len(values) - steps):
        j = i + steps
        if not np.isfinite(values[j]):
            continue
        if bounds and not bounds[0] <= times[j] < bounds[1]:
            continue
        if abs(times[j] - times[i] - steps * 300) > 90:
            continue
        rows.append(_scalar_features(filled[i - history + 1:i + 1], times[i], history))
        labels.append(values[j])
        target_indices.append(j)
    x, labels = np.asarray(rows), np.asarray(labels)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1
    z, intercept = (x - mean) / scale, float(labels.mean())
    coef = solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (labels - intercept), assume_a="pos")
    return mean, scale, intercept, coef, target_indices


@pytest.mark.parametrize("steps", [6, 12, 36, 72])
def test_vectorized_fit_keeps_original_labels_row_order_bounds_and_physical_horizons(steps):
    rng = np.random.default_rng(47)
    values = rng.normal(size=220)
    values[:8] = np.nan
    values[[30, 80, 100, 190]] = [np.nan, np.inf, -np.inf, 0.]
    times = np.arange(220, dtype=float) * 300
    times[[40, 70, 140, 170]] += [90, -90, 90.001, -90.001]
    bounds, history, alpha = (times[25], times[210]), 13, .1
    expected = _scalar_fit(values, times, history, steps, alpha, bounds)
    model = DirectRidge(history=history, horizon_steps=steps, alpha=alpha).fit(values, times, target_bounds=bounds)
    assert model.n_training_targets == len(expected[-1])
    assert all(np.isfinite(values[j]) and bounds[0] <= times[j] < bounds[1] for j in expected[-1])
    np.testing.assert_allclose(model.mean, expected[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(model.scale, expected[1], rtol=1e-13, atol=1e-13)
    assert model.intercept == expected[2]
    np.testing.assert_allclose(model.coef, expected[3], rtol=1e-10, atol=1e-11)
    history_values, issuance = values[110:150], times[149]
    x = _scalar_features(history_values, issuance, history)
    prediction = float((x - expected[0]) / expected[1] @ expected[3] + expected[2])
    assert model.predict(history_values, issuance) == pytest.approx(prediction, abs=1e-10)


def test_exact_90_second_horizon_boundary_is_inclusive_and_filled_targets_are_not_labels():
    values, times = np.arange(80, dtype=float), np.arange(80, dtype=float) * 300
    times[20] -= 90
    times[30] += 90
    times[40] += 90.000001
    values[50], values[60] = np.nan, np.inf
    model = DirectRidge(history=5, horizon_steps=6).fit(values, times, target_bounds=(times[25], times[70]))
    # 45 bounded target indices: reject two >90s pairs and two nonoriginal labels.
    assert model.n_training_targets == 41
