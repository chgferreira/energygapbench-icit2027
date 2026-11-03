"""Public statistics checks use synthetic aggregate draws, never private data."""
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from reproduce_public import verify_draw_summaries


def example(supported):
    draws = np.array([-.04, -.03, -.02, -.01, 0.]).reshape(-1, 1)
    estimate = -.02
    row = dict(site="H2", block_days=7, contrast="energy", attempted=5, valid=5,
               invalid=0, ci95=np.quantile(draws[:, 0], [.025, .975]).tolist(),
               estimate=estimate, temporal_support_sufficient=supported,
               p_exploratory=3 / 6 if supported else None, confirmatory=False,
               physical_confirmatory=False, p_holm=None)
    return row, {"H2_block7": draws}, [{"contrast": "energy"}]


@pytest.mark.parametrize("supported", [True, False])
def test_mean_draw_checks_preserve_support(supported):
    row, arrays, columns = example(supported)
    result = verify_draw_summaries([row], arrays, columns)
    assert result[0]["temporal_support_sufficient"] is supported
    assert result[0]["ci_max_absolute_error"] == 0


def test_changed_p_value_and_interval_are_rejected():
    row, arrays, columns = example(True)
    row["p_exploratory"] = .01
    with pytest.raises(ValueError, match="p-value"):
        verify_draw_summaries([row], arrays, columns)
    row, arrays, columns = example(True)
    row["ci95"][0] += .001
    with pytest.raises(ValueError, match="interval"):
        verify_draw_summaries([row], arrays, columns)
