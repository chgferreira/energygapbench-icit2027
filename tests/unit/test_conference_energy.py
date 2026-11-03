import json
from types import SimpleNamespace

import numpy as np
import pytest

from energygap.conference_energy import conditional_energy_metrics, cpi_energy_records
from energygap.methods.cpi import CPIEnergyDonorAdapted


def query_case():
    query = SimpleNamespace(
        site_id='H1', query_id='query', target_gap_id='gap',
        times=tuple(3*86400+np.arange(5.)*300), issuance_time=3*86400+1200,
        values=(2., np.nan, np.nan, np.nan, 2.),
        available_to_method=(True, False, False, False, True),
        target_gap_mask=(False, True, True, True, False),
        energy_intervals=({'left_index': 0, 'right_index': 4, 'import_kwh': 2/3, 'export_kwh': 0.},),
    )
    truth = SimpleNamespace(values=(np.nan, 2., 2., 2., np.nan),
                            evaluation_mask=(False, True, True, True, False))
    return query, truth


def donors():
    times = np.arange(2*288+1.)*300
    return {direction: CPIEnergyDonorAdapted(timezone='UTC').fit(np.arange(len(times))*increment, times)
            for direction, increment in (('import', 1/6), ('export', 0.))}


def test_conditional_integral_error_is_separate_from_input_residual():
    query, truth = query_case()
    query.energy_intervals[0]['import_kwh'] = 1.
    got = conditional_energy_metrics(query, truth, [2.]*5, 'instantaneous')
    assert got['net_power_integral_error_kwh'] == pytest.approx(0.)
    assert got['net_input_counter_residual_kwh'] == pytest.approx(-1/3)
    assert got['confirmatory'] is False
    assert got['independent_meter_calibration'] is False
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize('hypothesis', ['interval_mean_start', 'interval_mean_end'])
def test_means_never_claim_gross_power_energy(hypothesis):
    query, truth = query_case()
    got = conditional_energy_metrics(query, truth, [2.]*5, hypothesis)
    assert got['net_power_integral_error_kwh'] == pytest.approx(0.)
    assert got['import_power_integral_error_kwh'] is None
    assert got['export_input_counter_residual_kwh'] is None
    assert 'not_identifiable' in got['gross_energy_reason']


def test_incomplete_truth_and_changed_edges_are_unavailable():
    query, truth = query_case()
    truth.values = (np.nan, 2., np.nan, 2., np.nan)
    assert conditional_energy_metrics(query, truth, [2.]*5, 'instantaneous')['energy_metrics_status'] == 'unavailable'
    query, truth = query_case()
    assert conditional_energy_metrics(query, truth, [9., 2., 2., 2., 2.], 'instantaneous')['energy_metrics_reason'] == 'prediction_changed_admitted_boundary_power'


def test_cpi_stays_in_mean_track_and_constraint_residual_is_labelled():
    query, truth = query_case()
    rows = cpi_energy_records(query, truth, donors(), 2.)
    assert {row['physical_hypothesis'] for row in rows} == {'interval_mean_start', 'interval_mean_end'}
    for row in rows:
        assert row['phase'] == 'energy_track'
        assert row['status'] == 'success'
        assert row['conditional_power_nmae'] == pytest.approx(0.)
        assert row['net_input_counter_residual_kwh'] == pytest.approx(0.)
        assert 'tautological' in row['counter_residual_kind']
        assert row['import_power_integral_error_kwh'] is None
        assert row['confirmatory'] is False
    json.dumps(rows, allow_nan=False)


def test_cpi_receives_only_endpoint_deltas_and_never_hidden_power():
    query, truth = query_case()
    observed_inputs = []
    models = donors()
    for model in models.values():
        native = model.impute_cumulative
        def spy(values, times, mask, *, original=native, **kwargs):
            observed_inputs.append(np.asarray(values).copy())
            return original(values, times, mask, **kwargs)
        model.impute_cumulative = spy
    first = cpi_energy_records(query, truth, models, 2.)
    truth.values = (np.nan, 999., -999., 12345., np.nan)
    second = cpi_energy_records(query, truth, models, 2.)
    assert len(observed_inputs) == 4
    assert all(np.isnan(x[1:-1]).all() and x[0] == 0 for x in observed_inputs)
    np.testing.assert_array_equal(observed_inputs[0], observed_inputs[2])
    np.testing.assert_array_equal(observed_inputs[1], observed_inputs[3])
    assert first[0]['net_input_counter_residual_kwh'] == second[0]['net_input_counter_residual_kwh']
    assert first[0]['conditional_power_nmae'] != second[0]['conditional_power_nmae']


def test_missing_direction_is_na_instead_of_zero_export():
    query, truth = query_case()
    query.energy_intervals[0]['export_kwh'] = None
    for row in cpi_energy_records(query, truth, donors(), 2.):
        assert row['status'] == 'unavailable'
        assert row['conditional_power_nmae'] is None
        assert row['net_input_counter_residual_kwh'] is None
