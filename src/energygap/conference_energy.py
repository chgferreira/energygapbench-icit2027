"""Exploratory energy diagnostics and the separate CPI interval-mean track.

Original power is opened only by evaluation. Meter endpoint residuals describe
agreement with an input constraint, never independent physical accuracy.
"""
from __future__ import annotations

import numpy as np

from energygap.evaluation.energy import integrate_signed
from energygap.evaluation.reconstruction import reconstruction_metrics

HYPOTHESES = ('instantaneous', 'interval_mean_start', 'interval_mean_end')


def _bounds(query):
    target = np.asarray(query.target_gap_mask, bool)
    positions = np.flatnonzero(target)
    if not len(positions) or not np.array_equal(positions, np.arange(positions[0], positions[-1]+1)):
        raise ValueError('One nonempty contiguous target gap is required')
    left, right = int(positions[0]-1), int(positions[-1]+1)
    if left < 0 or right >= len(target):
        raise ValueError('Both boundary positions are required')
    available = np.asarray(query.available_to_method, bool)
    values = np.asarray(query.values, float)
    if not available[[left, right]].all() or not np.isfinite(values[[left, right]]).all():
        raise ValueError('Both power boundaries must be admitted original observations')
    times = np.asarray(query.times, float)
    if times.shape != values.shape or not np.isfinite(times).all() or not np.all(np.diff(times)>0):
        raise ValueError('Finite, increasing physical timestamps are required')
    return left, right, positions, values, times


def _endpoints(query, left, right):
    for item in getattr(query, 'energy_intervals', ()):
        if item.get('left_index') == left and item.get('right_index') == right:
            result = {}
            for direction in ('import', 'export'):
                value = item.get(direction+'_kwh')
                # The query API has already enforced endpoint availability/reset rules.
                result[direction] = float(value) if value is not None and np.isfinite(value) and value >= 0 else None
            return result
    return {'import': None, 'export': None}


def _empty_metrics(hypothesis, reason):
    result = {'physical_hypothesis': hypothesis, 'energy_metrics_status': 'unavailable',
              'energy_metrics_reason': reason, 'exploratory': True, 'confirmatory': False,
              'energy_reference_kind': 'original_power_integral_under_hypothesis_not_independent_meter',
              'counter_residual_kind': 'consistency_with_input_constraint_not_independent_accuracy',
              'independent_meter_calibration': False}
    for direction in ('net', 'import', 'export'):
        for suffix in ('power_integral_error_kwh', 'input_counter_residual_kwh',
                       'reference_power_integral_kwh', 'estimated_power_integral_kwh'):
            result[direction+'_'+suffix] = None
    return result


def conditional_energy_metrics(query, truth, prediction, hypothesis):
    """Boundary-inclusive error on only target truth and received original edges.

    Net interval means do not identify gross import/export with possible reversal,
    so their gross metrics remain absent even when both counters were provided.
    """
    if hypothesis not in HYPOTHESES:
        raise ValueError('Unknown physical hypothesis: '+str(hypothesis))
    result = _empty_metrics(hypothesis, None)
    try:
        left, right, positions, admitted, times = _bounds(query)
    except ValueError as error:
        result['energy_metrics_reason'] = str(error)
        return result
    prediction = np.asarray(prediction, float).reshape(-1)
    original = np.asarray(truth.values, float).reshape(-1)
    evaluated = np.asarray(truth.evaluation_mask, bool)
    if prediction.shape != admitted.shape or original.shape != admitted.shape or evaluated.shape != admitted.shape:
        result['energy_metrics_reason'] = 'prediction_or_truth_shape_mismatch'
        return result
    if not evaluated[positions].all() or not np.isfinite(original[positions]).all():
        result['energy_metrics_reason'] = 'incomplete_original_target_power_integral_support'
        return result
    if not np.isfinite(prediction[left:right+1]).all():
        result['energy_metrics_reason'] = 'nonfinite_prediction_integral_support'
        return result
    if not np.array_equal(prediction[[left, right]], admitted[[left, right]]):
        result['energy_metrics_reason'] = 'prediction_changed_admitted_boundary_power'
        return result
    # Never inspect other hidden/natural-missing context values as truth.
    reference = admitted[left:right+1].copy()
    reference[positions-left] = original[positions]
    estimated = integrate_signed(prediction[left:right+1], times[left:right+1], semantics=hypothesis)
    actual = integrate_signed(reference, times[left:right+1], semantics=hypothesis)
    counters = _endpoints(query, left, right)
    net_counter = counters['import']-counters['export'] if all(v is not None for v in counters.values()) else None
    directions = ('net', 'import', 'export') if hypothesis == 'instantaneous' else ('net',)
    for direction in directions:
        result[direction+'_reference_power_integral_kwh'] = actual[direction+'_kwh']
        result[direction+'_estimated_power_integral_kwh'] = estimated[direction+'_kwh']
        result[direction+'_power_integral_error_kwh'] = estimated[direction+'_kwh']-actual[direction+'_kwh']
        counter = net_counter if direction == 'net' else counters[direction]
        if counter is not None:
            result[direction+'_input_counter_residual_kwh'] = estimated[direction+'_kwh']-counter
    result.update(energy_metrics_status='success', energy_metrics_reason=None,
                  integral_boundary_span_seconds=float(times[right]-times[left]),
                  original_target_points=int(len(positions)),
                  gross_energy_reason=None if hypothesis == 'instantaneous' else 'not_identifiable_from_net_interval_means_with_possible_within_interval_reversal')
    return result


def cpi_energy_records(query, truth, donor_models, scale):
    """Use admitted counter *deltas* only; evaluate CPI means in a separate track.

    ``donor_models`` maps import/export to train-fitted CPIEnergyDonorAdapted.
    Counter origins are arbitrary zero, not claimed actual meter readings.
    Neither target power nor intermediate meter values enter donor selection.
    """
    base = {'phase': 'energy_track', 'method': 'cpi_energy_donor_adapted',
            'variant': 'train_energy_donor_interval_mean_adapted',
            'site': query.site_id, 'query_id': query.query_id,
            'gap_id': query.target_gap_id, 'information_tier': 'E',
            'exploratory': True, 'confirmatory': False,
            'output_semantics': 'cumulative_energy_and_net_interval_means_not_instantaneous',
            'official_output_parity_claimed': False, 'independent_meter_calibration': False,
            'counter_residual_kind': 'tautological_input_constraint_residual_after_donor_scaling_not_independent_accuracy',
            'counter_origin': 'arbitrary_zero_for_admissible_increment_not_actual_meter_level',
            'input_policy': 'only_admissible_left_right_counter_deltas; all_intermediate_counter_inputs_NaN'}
    fields = {'conditional_power_mae_kw': None, 'conditional_power_nmae': None,
              'net_input_counter_residual_kwh': None, 'net_power_integral_error_kwh': None,
              'import_power_integral_error_kwh': None, 'export_power_integral_error_kwh': None,
              'import_scaling_constraint_residual_kwh': None, 'export_scaling_constraint_residual_kwh': None}
    def unavailable(reason, status='unavailable', diagnostics=None):
        return [{**base, **fields, 'physical_hypothesis': h, 'status': status,
                 'reason': reason, 'donor_diagnostics': diagnostics or {}}
                for h in ('interval_mean_start', 'interval_mean_end')]
    try:
        left, right, positions, admitted, all_times = _bounds(query)
    except ValueError as error:
        return unavailable(str(error))
    counters = _endpoints(query, left, right)
    if any(value is None for value in counters.values()):
        return unavailable('both_admissible_import_and_export_endpoint_deltas_required_for_net_power')
    if any(direction not in donor_models or donor_models[direction] is None for direction in ('import', 'export')):
        return unavailable('both_train_fitted_directional_donor_models_required')
    times = all_times[left:right+1]
    gap = np.zeros(len(times), bool);gap[1:-1] = True
    issuance = query.issuance_time.timestamp() if hasattr(query.issuance_time, 'timestamp') else float(query.issuance_time)
    inferred = {}
    diagnostics = {}
    # Evaluation truth is deliberately untouched until both inference calls finish.
    for direction in ('import', 'export'):
        cumulative = np.full(len(times), np.nan)
        cumulative[0] = 0.;cumulative[-1] = counters[direction]
        try:
            output = donor_models[direction].impute_cumulative(cumulative, times, gap,
                issuance_time=issuance, availability_times=times)
        except Exception as error:
            return unavailable(direction+':'+type(error).__name__+':'+str(error), status='failed', diagnostics=diagnostics)
        diagnostics[direction] = output.diagnostics
        if output.status != 'success':
            return unavailable(direction+':'+str(output.diagnostics.get('reason', output.status)),
                               status=output.status, diagnostics=diagnostics)
        means = np.asarray(output.interval_mean_kw, float)
        restored = np.asarray(output.cumulative_kwh, float)
        if means.shape != times.shape or restored.shape != times.shape or not np.isfinite(means[1:]).all() or not np.isfinite(restored).all():
            return unavailable(direction+':nonfinite_or_invalid_native_donor_output', status='failed', diagnostics=diagnostics)
        inferred[direction] = means
    net = inferred['import']-inferred['export']
    dt = np.diff(times)/3600.
    energy = float(np.sum(net[1:]*dt))
    net_counter = counters['import']-counters['export']
    constraints = {d+'_scaling_constraint_residual_kwh': float(np.sum(inferred[d][1:]*dt)-counters[d]) for d in ('import', 'export')}
    original = np.asarray(truth.values, float)
    evaluated = np.asarray(truth.evaluation_mask, bool)
    if original.shape != admitted.shape or evaluated.shape != admitted.shape:
        return unavailable('truth_shape_mismatch')
    evaluated = evaluated & np.asarray(query.target_gap_mask, bool)
    rows = []
    for hypothesis in ('interval_mean_start', 'interval_mean_end'):
        # Native CPI means[j] describe (time[j-1],time[j]]. Under start placement,
        # a power target at position j instead corresponds to native means[j+1].
        target_means = net[2:] if hypothesis == 'interval_mean_start' else net[1:-1]
        prediction = np.full(len(admitted), np.nan)
        prediction[positions] = target_means
        point = reconstruction_metrics(original, prediction, evaluated, scale_train=scale)
        reference = admitted[left:right+1].copy()
        support = evaluated[positions].all() and np.isfinite(original[positions]).all()
        reference[positions-left] = original[positions]
        reference_energy = integrate_signed(reference, times, semantics=hypothesis)['net_kwh'] if support else None
        rows.append({**base, **fields, **constraints, 'physical_hypothesis': hypothesis,
            'status': 'success', 'reason': None, 'donor_diagnostics': diagnostics,
            'conditional_power_mae_kw': point['mae_kw'], 'conditional_power_nmae': point['nmae'],
            'conditional_power_evaluated_points': point['n'],
            'conditional_power_comparison': 'original_power_only_if_declared_interval_mean_timestamp_hypothesis_holds',
            'net_input_counter_residual_kwh': energy-net_counter,
            'net_power_integral_error_kwh': energy-reference_energy if reference_energy is not None else None,
            'energy_reference_kind': 'original_power_integral_under_hypothesis_not_independent_meter',
            'energy_metrics_reason': None if support else 'incomplete_original_target_power_integral_support',
            'gross_energy_reason': 'not_identifiable_from_net_interval_means_with_possible_within_interval_reversal'})
    return rows
