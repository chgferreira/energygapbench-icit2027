import numpy as np
import pytest

from energygap.methods.base import UnsupportedTask
from energygap.methods.baselines import create_baseline, SeasonalImputer, DailyKalmanImputer


@pytest.mark.parametrize("name", ["median", "locf", "linear", "seasonal", "kalman_filter", "kalman_smoother"])
def test_baselines_preserve_observed_and_never_fit_test(name):
    t = np.arange(1000.) * 300
    train = 3 + np.sin(t * 2*np.pi/86400)
    method = create_baseline(name).fit(train, t)
    x = train[-20:].copy(); x[4:8] = np.nan
    got = method.impute(x, t[-20:])
    np.testing.assert_array_equal(got[np.isfinite(x)], x[np.isfinite(x)])
    assert np.isfinite(got).all()
    with pytest.raises(ValueError, match="training"):
        method.fit(train, t, split="test")


def test_linear_physical_time_and_unbracketed_missing():
    method = create_baseline("linear").fit([1, 2], [0, 1])
    result = method.impute([np.nan, 0, np.nan, 12, np.nan], [0, 10, 20, 50, 60])
    assert result[2] == 3
    assert np.isnan(result[[0, 4]]).all()
    with pytest.raises(UnsupportedTask):
        method.impute([0, np.nan], [0, 1], task="C")


def test_locf_age_does_not_expose_the_right_border():
    method = create_baseline("locf").fit([1, 3], [0, 1])
    assert method.impute([2, np.nan, 100], [0, 1, 2])[1] == 2


def test_seasonal_minimum_support_and_hierarchy():
    # Twenty Mondays at precisely the same slot support weekday+slot profile.
    t = 4*86400 + np.arange(20) * 7*86400.  # Monday 1970-01-05 UTC
    model = SeasonalImputer(timezone="UTC").fit(np.full(20, 7.), t)
    assert model.impute([np.nan], [t[-1]+7*86400])[0] == 7
    assert model.counts_[0][(0, 0)] == 20
    model2 = SeasonalImputer(timezone="UTC").fit([1., 3., 10.], [0, 300, 600])
    assert model2.impute([np.nan], [100000])[0] == 3
    assert not model2.profiles_[0]


def test_daily_kalman_filter_and_smoother_are_distinct():
    t = np.arange(1500.) * 300
    train = 5 + 2*np.sin(t * 2*np.pi/86400) + .3*np.sin(t/1200)
    filter_model = DailyKalmanImputer().fit(train, t)
    smoother = DailyKalmanImputer(smooth=True).fit(train, t)
    qt = t[-20:]; x = train[-20:].copy(); x[5:15] = np.nan
    poisoned = x.copy(); poisoned[-1] += 20
    filtered = filter_model.impute(x, qt)
    np.testing.assert_array_equal(filtered[5:15], filter_model.impute(poisoned, qt)[5:15])
    assert not np.allclose(smoother.impute(x, qt)[5:15], smoother.impute(poisoned, qt)[5:15])
    with pytest.raises(UnsupportedTask):
        smoother.impute(x, qt, task="C")
