"""Fixed-weight downstream direct ridge forecasting on physical time."""
from .ridge import DirectRidge, causal_fill, match_target

__all__ = ["DirectRidge", "causal_fill", "match_target"]
