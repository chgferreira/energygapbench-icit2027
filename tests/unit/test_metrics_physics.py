import numpy as np
import pytest

from energygap.evaluation.reconstruction import reconstruction_metrics, evaluation_mask, training_scale, skill
from energygap.evaluation.aggregation import gap_macro, equal_weight_estimand
from energygap.evaluation.probabilistic import sample_crps, interval_metrics, pinball
from energygap.evaluation.events import event_metrics


def test_natural_missing_and_padding_not_evaluation_truth():
    mask = evaluation_mask([True, False, True, True], [True]*4, [False, False, True, False])
    np.testing.assert_array_equal(mask, [True, False, False, True])
    m = reconstruction_metrics([1, np.nan, 10, 3], [2, 1000, -999, 5], mask, scale_train=2)
    assert m["mae_kw"] == 1.5 and m["nmae"] == .75 and m["n"] == 2
    with pytest.raises(ValueError, match="Natural"):
        reconstruction_metrics([np.nan], [1], [True])


def test_no_support_constant_scale_and_skill_are_na_not_zero():
    assert reconstruction_metrics([1], [1], [False])["mae_kw"] is None
    assert training_scale([2, 2, np.nan])[0] is None
    assert skill(0, 0)[0] is None
    assert skill(1, 2)[0] == .5


def test_macro_and_micro_analytic_example():
    result = gap_macro([[10], [0, 0, 0]])
    assert result["gap_macro_mae"] == 5
    assert result["point_micro_mae"] == 2.5


def test_fixed_weights_missing_stratum_and_seed_average():
    rows = [{"fold": 1, "duration": 15, "episode_id": "a", "nmae": x} for x in [1, 3]]
    rows += [{"fold": 1, "duration": 60, "episode_id": "b", "nmae": 8}]
    assert equal_weight_estimand(rows, folds=(1,), durations=(15, 60))["estimate"] == 5
    assert equal_weight_estimand(rows)["estimate"] is None


def test_crps_exact_pairwise_formula_and_primary_median():
    s = np.array([[0., 1.], [2., 3.], [10., 5.]])
    y = np.array([2., 2.])
    exact = np.mean(np.mean(np.abs(s-y), axis=0) - .5*np.mean(np.abs(s[:, None, :]-s[None, :, :]), axis=(0, 1)))
    assert sample_crps(s, y) == pytest.approx(exact)
    assert sample_crps(s, y, [False, False]) is None
    assert pinball([2], [3], .5) == .5


def test_interval_score_coverage_and_width():
    r = interval_metrics([0, 0], [2, 2], [1, 3], coverage=.8, scale_train=2)
    assert r["coverage"] == .5 and r["width"] == 2 and r["normalized_width"] == 1
    assert r["interval_score"] == pytest.approx(7)


def test_events_use_physical_ramp_time_and_earliest_peak_tie():
    got = event_metrics([0, 2, 2], [0, 1, 3], [0, 60, 180], [True]*3)
    assert got["import_peak_time_error_seconds"] == 120
    assert got["ramp_mae_kw_per_minute"] == 1
