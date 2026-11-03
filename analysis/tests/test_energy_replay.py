"""Integrity and numerical checks for the supplemental read-only energy replay."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from energy_replay_common import output_directory
from diagnose_energy_constraints import recorded_diagnostics, residual_audit
from replay_linear_energy_controls import scalar_control
import replay_linear_energy_controls as control_module
from energygap.conference import _conditional_constraint
from energygap.data import SeriesData
from energygap.data.fixture import make_fixture
from energygap.evaluation.energy import integrate_signed
from energygap.methods.calibration import fit_energy_tolerances
from energygap.methods.correction import EnergyConstraint
from energygap.runtime.state import StateStore


def test_scalar_positive_direction_matches_analytic_solution_and_preserves_boundaries():
    base = np.full(5, 10.)
    times = np.arange(5) * 300.
    # The linear segment's integrated energy changes by a*c, a=3*(5/60).
    constraint = EnergyConstraint(import_kwh=4., tau_import=.2)
    scale, weight, a = 2., 1., .25
    r = integrate_signed(base, times)["import_kwh"] - 4.
    for deadzone in (False, True):
        excess = r - np.sign(r) * .2 if deadzone else r
        expected = -weight * a * excess / (.2**2 / scale**2 + weight * a**2)
        result, status, _ = scalar_control(base, times, constraint, scale=scale,
            lambda_energy=weight, tau_multiplier=1., deadzone=deadzone)
        assert status == "success"
        np.testing.assert_allclose(result[1:-1] - base[1:-1], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(result[[0, -1]], base[[0, -1]])
    np.testing.assert_array_equal(base, np.full(5, 10.))


def test_both_counter_directions_affect_zero_crossing_control():
    base = np.array([2., 1., -.4, -1., -2.])
    times = np.arange(5) * 300.
    first = EnergyConstraint(import_kwh=.4, export_kwh=.1, tau_import=.05, tau_export=.05)
    second = replace(first, export_kwh=.8)
    a = scalar_control(base, times, first, scale=2., lambda_energy=1., tau_multiplier=1., deadzone=False)[0]
    b = scalar_control(base, times, second, scale=2., lambda_energy=1., tau_multiplier=1., deadzone=False)[0]
    assert not np.allclose(a[1:-1], b[1:-1])
    np.testing.assert_array_equal(a[[0, -1]], base[[0, -1]])
    np.testing.assert_allclose(np.diff(a[1:-1] - base[1:-1]), 0., atol=1e-14)


def test_tolerance_identity_and_lambda_zero():
    base = np.full(5, 1.)
    times = np.arange(5) * 300.
    constraint = EnergyConstraint(import_kwh=.34, tau_import=.1)
    result, status, reason = scalar_control(base, times, constraint, scale=1.,
        lambda_energy=10., tau_multiplier=1., deadzone=True)
    np.testing.assert_array_equal(result, base)
    assert (status, reason) == ("success", "within_directional_tolerance")
    result, status, reason = scalar_control(base, times, constraint, scale=1.,
        lambda_energy=0., tau_multiplier=1., deadzone=False)
    np.testing.assert_array_equal(result, base)
    assert status == "disabled"


def test_failed_scalar_searches_keep_identity_and_expose_failure(monkeypatch):
    monkeypatch.setattr(control_module, "minimize_scalar", lambda *a, **k: SimpleNamespace(success=False))
    base = np.full(5, 1.)
    result, status, reason = scalar_control(base, np.arange(5) * 300.,
        EnergyConstraint(import_kwh=2., tau_import=.1), scale=1.,
        lambda_energy=1., tau_multiplier=1., deadzone=False)
    np.testing.assert_array_equal(result, base)
    assert status == "failed" and reason == "all_scalar_searches_failed_identity_fallback"


def test_zero_export_with_ample_support_blocks_interval_mean_only():
    fixture = make_fixture(days=5, natural_missing=False)
    # Positive constant power, flat export counter, deliberately discordant import
    # counter. These are synthetic calibration intervals, not production data.
    rows = [replace(row, value_original=1., value_canonical=1.,
                    import_counter=i * .10, export_counter=0.) for i, row in enumerate(fixture.records)]
    series = SeriesData(rows, "SYNTH", 300, {"power_semantics": "instantaneous"})
    indices = list(range(len(rows) - 1))
    cutoff = rows[-1].grid_time
    calibrated = fit_energy_tolerances(series, indices, cutoff_time=cutoff)
    audited = residual_audit(series, indices, cutoff, "instantaneous")
    assert all(len(values) >= 30 and np.max(values) == 0 for values in audited["export"].values())
    for duration in (15, 60, 180):
        assert calibrated["directions"]["import"][str(duration)]["tau_kwh"] > 0
        export = calibrated["directions"]["export"][str(duration)]
        assert export["n_used"] >= 30 and export["tau_kwh"] is None
        assert export["source"] == "insufficient_support_unknown_resolution"
    class Query:
        energy_intervals = ({"import_kwh": 1., "export_kwh": 0.},)
    instantaneous = _conditional_constraint(Query(), calibrated, 15, "instantaneous")
    means = _conditional_constraint(Query(), calibrated, 15, "interval_mean_start")
    assert instantaneous.reliable and instantaneous.tau_import > 0
    assert instantaneous.tau_export is None
    assert not means.reliable and means.tau_net is None


def test_complete_sqlite_is_read_only_and_preserves_recorded_selection(tmp_path):
    run = tmp_path / "run"
    with StateStore(run, config={"synthetic": True}) as store:
        store.add_job("selection", 1, metadata={"site": "I1", "fold": 1, "method": "linear"})
        with store.claim("selection") as lease:
            store.publish("selection", lease.token, [
                {"key": "trial0", "phase": "correction_selection", "hypothesis": "interval_mean_start",
                 "trial": 0, "parameters": {"lambda_energy": 0.}, "loss": .2, "failures": 10},
                {"key": "trial1", "phase": "correction_selection", "hypothesis": "interval_mean_start",
                 "trial": 1, "parameters": {"lambda_energy": 1.}, "loss": .2, "failures": 10}], 1)
            store.finish("selection", lease.token)
        store.add_job("inference", 1)
        with store.claim("inference") as lease:
            store.publish("inference", lease.token, [
                {"key": "result", "phase": "inference", "site": "I1", "fold": 1,
                 "method": "linear+energy_consistent", "variant": "E:interval_mean_start",
                 "status": "unavailable", "reason": "conditional_net_mean_train_internal_residual_tolerance",
                 "diagnostics": {"selected_start": None}}], 1)
            store.finish("inference", lease.token)
    database = run / "state.sqlite"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    selected, statuses, source = recorded_diagnostics(run)
    assert selected[0]["trial"] == 0 and selected[0]["recorded_trials"] == 2
    assert selected[0]["site"] == "I1"
    assert statuses[0]["status"] == "unavailable" and statuses[0]["count"] == 1
    assert source == "read_only_sqlite_recorded_statuses"
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert recorded_diagnostics(tmp_path / "compact") == ([], [], "not_in_compact_bundle")


def test_refuse_output_inside_immutable_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="outside"):
        output_directory(run / "reports" / "new", run)
    assert output_directory(tmp_path / "supplemental", run).exists()
