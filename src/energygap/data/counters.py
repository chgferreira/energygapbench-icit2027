"""Counter differences never turn a negative/reset into positive energy by abs."""
import math


def counter_increment(left, right, *, known_rollover=None, left_available=True, right_available=True):
    if not (left_available and right_available):
        return None, 'endpoint_unavailable'
    if left is None or right is None or not (math.isfinite(left) and math.isfinite(right)):
        return None, 'endpoint_missing'
    delta = right-left
    if delta < 0:
        if known_rollover is None:
            return None, 'reset_or_unknown_rollover'
        if not (0 <= left <= known_rollover and 0 <= right <= known_rollover):
            return None, 'invalid_rollover_bounds'
        return known_rollover-left+right, 'known_rollover'
    return delta, 'ok'  # Constant is a valid zero increment.
