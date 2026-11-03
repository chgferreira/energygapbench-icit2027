"""Independent checks for weighting, pairing, missing targets and day resampling."""
from datetime import date, timedelta
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "forecast_common_target_v2.py"
SPEC = importlib.util.spec_from_file_location("forecast_v2", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class ForecastRevisionTests(unittest.TestCase):
    def fixture(self, sparse=False):
        rows, values, calendars = [], [], {}
        for fold in mod.FOLDS:
            first = date(2020, fold, 1)
            days = [(first + timedelta(days=i)).isoformat() for i in range(8)]
            calendars[fold] = days
            for duration in mod.DURATIONS:
                # Unequal episode counts across durations make a pooled-episode
                # implementation observably different from the fixed estimand.
                for j in range(1 if sparse else mod.DURATIONS.index(duration) + 1):
                    day = days[0] if sparse else days[2 * j]
                    value = fold * 10 + duration / 60 + j
                    rows.append({"fold": fold, "duration": duration, "day": day})
                    values.append([value, -2 * value, 0.0])
        return pd.DataFrame(rows), np.asarray(values), calendars

    def test_fixed_stratum_weights_and_pair_identity(self):
        rows, values, calendars = self.fixture()
        arrays = mod.day_arrays(rows, values, calendars)
        got = mod.fixed_estimate(arrays)
        expected = []
        for fold in mod.FOLDS:
            for duration in mod.DURATIONS:
                mask = rows.fold.eq(fold) & rows.duration.eq(duration)
                expected.append(values[mask].mean(axis=0))
        np.testing.assert_allclose(got, np.mean(expected, axis=0), rtol=0, atol=1e-14)
        self.assertNotAlmostEqual(got[0], values[:, 0].mean())
        draws = mod.bootstrap_draws(arrays, 2, attempts=40)
        np.testing.assert_allclose(draws[:, 1], -2 * draws[:, 0], equal_nan=True)
        np.testing.assert_equal(draws[:, 2][np.isfinite(draws[:, 2])], 0)

    def test_bootstrap_matches_independent_episode_resampling(self):
        rows, values, calendars = self.fixture()
        arrays = mod.day_arrays(rows, values, calendars)
        actual = mod.bootstrap_draws(arrays, 3, attempts=30)
        reference = []
        for replicate in range(30):
            rng = np.random.default_rng(mod.stable_seed(mod.SEED, replicate))
            cells = []
            valid = True
            for fold in mod.FOLDS:
                days = calendars[fold]
                sampled = []
                while len(sampled) < len(days):
                    start = int(rng.integers(0, len(days) - 3 + 1))
                    sampled.extend(days[start:start + 3])
                indices = [i for day in sampled[:len(days)] for i, row in rows.iterrows() if row.fold == fold and row.day == day]
                for duration in mod.DURATIONS:
                    matching = [i for i in indices if rows.loc[i, "duration"] == duration]
                    if not matching:
                        valid = False
                        break
                    cells.append(values[matching].mean(axis=0))
                if not valid:
                    break
            reference.append(np.mean(cells, axis=0) if valid else [np.nan] * 3)
        np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-14, equal_nan=True)

    def test_empty_days_invalid_draws_not_redrawn_and_support_fail(self):
        rows, values, calendars = self.fixture(sparse=True)
        arrays = mod.day_arrays(rows, values, calendars)
        self.assertEqual(int((arrays[1][1].sum(axis=1) == 0).sum()), 7)
        draws = mod.bootstrap_draws(arrays, 1, attempts=100)
        invalid = int(np.isnan(draws[:, 0]).sum())
        self.assertGreater(invalid, 0)
        self.assertLess(invalid, 100)
        summary = mod.summarize_bootstrap(arrays, draws, 1)[0]
        self.assertEqual(summary["attempted"], 100)
        self.assertEqual(summary["invalid"], invalid)
        self.assertFalse(summary["temporal_support_sufficient"])
        self.assertIsNone(summary["p_exploratory"])
        self.assertFalse(summary["confirmatory"])
        self.assertEqual(summary["multiplicity_control"], "none")

    def test_common_target_filter_preserves_all_policies_and_rejects_asymmetry(self):
        records = []
        for gap in range(2):
            for method, variant in mod.POLICIES:
                records.append({"site": "H1", "fold": 1, "condition": "C3", "duration": 15,
                                "mask_seed": 2001, "gap_id": str(gap), "day": "2023-01-01",
                                "partition": "test", "method": method, "variant": variant,
                                "nmae": .1, "forecast_nmae": .2 if gap == 0 else np.nan,
                                "forecast_mae_kw": .3 if gap == 0 else np.nan,
                                "seed_evaluations": 3 if method.startswith("SAITS") else 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.parquet"
            frame = pd.DataFrame(records)
            frame.to_parquet(path, index=False)
            wide, observed = mod.load_wide(path)
            self.assertEqual(wide["nmae"].shape, (2, 32))
            np.testing.assert_array_equal(observed, [True, False])
            frame.loc[0, ["forecast_nmae", "forecast_mae_kw"]] = np.nan
            frame.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "availability differs"):
                mod.load_wide(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
