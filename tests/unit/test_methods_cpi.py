import numpy as np
import pytest

from energygap.methods.cpi import CPIEnergyDonorAdapted


def bank_model():
    t = np.arange(2*288+1.)*300
    increments = np.tile(1+np.sin(np.arange(288)*2*np.pi/288), 2)*.1
    counter = np.r_[0, np.cumsum(increments)]
    return CPIEnergyDonorAdapted(timezone="UTC").fit(counter, t)


def test_cpi_adapted_energy_conserved_and_not_instantaneous():
    model = bank_model()
    assert model.fit_diagnostics_["complete_train_days"] == 2
    times = 3*86400 + np.arange(5.)*300
    mask = [False, True, True, True, False]
    result = model.impute_cumulative([100, np.nan, np.nan, np.nan, 102], times, mask, issuance_time=times[-1])
    assert result.status == "success"
    assert result.cumulative_kwh[[0, -1]].tolist() == [100, 102]
    assert np.isfinite(result.cumulative_kwh).all()
    assert np.sum(result.interval_mean_kw[1:] * np.diff(times)/3600) == pytest.approx(2)
    assert "not_instantaneous" in result.diagnostics["output_semantics"]


def test_cpi_delay_counter_reset_and_poisoning():
    model = bank_model(); times = 3*86400 + np.arange(5.)*300
    mask = [False, True, True, True, False]
    delayed = model.impute_cumulative([100, np.nan, np.nan, np.nan, 102], times, mask,
        issuance_time=times[-1], availability_times=[*times[:-1], times[-1]+1])
    assert delayed.status == "unavailable"
    assert delayed.diagnostics["reason"] == "missing_endpoint"
    reset = model.impute_cumulative([100, np.nan, np.nan, np.nan, 2], times, mask, issuance_time=times[-1])
    assert reset.status == "unavailable"
    future_times = np.r_[times, times[-1]+300]
    args = ([False, True, True, True, False, False],)
    a = model.impute_cumulative([100, np.nan, np.nan, np.nan, 102, 10000], future_times, *args, issuance_time=times[-1])
    b = model.impute_cumulative([100, np.nan, np.nan, np.nan, 102, -999], future_times, *args, issuance_time=times[-1])
    np.testing.assert_array_equal(a.cumulative_kwh, b.cumulative_kwh)


def test_cpi_train_only_no_complete_donor_and_overlap_rejected():
    with pytest.raises(ValueError):
        CPIEnergyDonorAdapted().fit([0, 1], [0, 300], split="test")
    model = CPIEnergyDonorAdapted(timezone="UTC").fit([0, 1], [0, 300])
    result = model.impute_cumulative([1, np.nan, 2], [900, 1200, 1500], [False, True, False], issuance_time=1500)
    assert result.diagnostics["reason"] == "no_complete_training_donor_day"
    model = bank_model()
    result = model.impute_cumulative([1, np.nan, 2], [0, 300, 600], [False, True, False], issuance_time=600)
    assert result.status == "unsupported"
