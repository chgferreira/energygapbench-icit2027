import numpy as np
import pytest

from energygap.forecasting.batched_evaluation import ForecastQueue, HISTORY_POINTS
from energygap.forecasting.ridge import DirectRidge


class CountedForecaster:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def predict_many(self, histories, issuances):
        self.calls.append((len(histories), [len(history) for history in histories]))
        return self.model.predict_many(histories, issuances)


def test_four_horizons_match_scalar_evaluation_with_missing_targets_and_histories():
    times = np.arange(650) * 300.
    values = np.sin(times / 86400 * 2 * np.pi) + np.arange(len(times)) * .0001
    values[30:34] = np.nan
    horizons = (1800, 3600, 10800, 21600)
    forecasts = {horizon: CountedForecaster(DirectRidge(history=20, horizon_steps=horizon // 300).fit(values, times))
                 for horizon in horizons}
    queue = ForecastQueue(forecasts)
    histories = [np.asarray([np.nan, 0., np.nan, 2.]), values[:500].copy(), np.full(12, np.nan)]
    originals = [history.copy() for history in histories]
    rows, expected = [], []
    for index, history in enumerate(histories):
        issuance = 0. if index == 0 else times[500 + index]
        scale = (None, 0., 1.3)[index]
        cells = {str(h): dict(mae_kw=None, nmae=None, target_offset_seconds=7., exclusion_reason=None) for h in horizons}
        result = dict(forecasts=cells, forecast_mae_kw=None, forecast_nmae=None)
        scalar = {}
        for horizon, forecaster in forecasts.items():
            target = None if index == 2 and horizon == 21600 else (float(index), 7.)
            if target is None:
                result['forecasts'][str(horizon)]['exclusion_reason'] = 'unavailable original target'
            else:
                error = abs(forecaster.model.predict(history[-HISTORY_POINTS:], issuance) - target[0])
                scalar[horizon] = (error, error / scale if scale else None)
            queue.add(horizon, history, issuance, target, scale, result)
        rows.append(result)
        expected.append(scalar)
    queue.flush()
    for index, (result, scalar) in enumerate(zip(rows, expected)):
        for horizon in horizons:
            cell = result['forecasts'][str(horizon)]
            assert cell['target_offset_seconds'] == 7.
            if horizon in scalar:
                assert cell['mae_kw'] == pytest.approx(scalar[horizon][0], abs=1e-12)
                if scalar[horizon][1] is None:
                    assert cell['nmae'] is None
                else:
                    assert cell['nmae'] == pytest.approx(scalar[horizon][1])
            else:
                assert cell['mae_kw'] is None
                assert cell['exclusion_reason'] == 'unavailable original target'
        assert result['forecast_mae_kw'] == result['forecasts']['3600']['mae_kw']
        assert result['forecast_nmae'] == result['forecasts']['3600']['nmae']
        np.testing.assert_array_equal(histories[index], originals[index])
    assert forecasts[21600].calls == [(2, [4, HISTORY_POINTS])]
    assert all(model.calls == [(3, [4, HISTORY_POINTS, 12])] for horizon, model in forecasts.items() if horizon != 21600)
    queue.flush()
    assert all(len(model.calls) == 1 for model in forecasts.values())


def test_queue_copies_history_and_flushes_large_groups_in_bounded_chunks():
    class Tail:
        def __init__(self): self.sizes = []
        def predict_many(self, histories, issuances):
            self.sizes.append(len(histories))
            return np.asarray([history[-1] + issuance for history, issuance in zip(histories, issuances)])
    model = Tail()
    queue = ForecastQueue({3600: model}, max_batch_size=2)
    first = np.arange(500.)
    rows = [dict(forecasts={'3600': {}}) for _ in range(5)]
    queue.add(3600, first, 0., 0., 1., rows[0])
    first[:] = -100  # changing a later variant cannot change the admitted copy
    for i in range(1, 5):
        queue.add(3600, np.asarray([float(i)]), 0., 0., 1., rows[i])
    queue.flush()
    assert model.sizes == [2, 2, 1]
    assert rows[0]['forecast_mae_kw'] == 499.
    assert [row['forecast_mae_kw'] for row in rows[1:]] == [1., 2., 3., 4.]
    assert all(not pending for pending in queue._pending.values())


def test_unavailable_targets_never_call_forecaster_or_mutate_exclusions():
    class Never:
        def predict_many(self, *_): pytest.fail('unavailable target must not be predicted')
    queue = ForecastQueue({3600: Never()})
    row = dict(forecasts={'3600': dict(mae_kw=None, nmae=None, exclusion_reason='outside test fold')})
    queue.add(3600, [np.nan], 0., None, None, row)
    queue.flush()
    assert row['forecasts']['3600']['exclusion_reason'] == 'outside test fold'
    assert row['forecasts']['3600']['mae_kw'] is None


def test_mismatched_batch_output_is_rejected_before_mutating_rows():
    class Broken:
        def predict_many(self, *_): return []
    queue = ForecastQueue({3600: Broken()})
    row = dict(forecasts={'3600': dict(mae_kw=None, nmae=None)})
    queue.add(3600, [0.], 0., 1., 1., row)
    with pytest.raises(ValueError, match='output count'):
        queue.flush()
    assert row['forecasts']['3600']['mae_kw'] is None
