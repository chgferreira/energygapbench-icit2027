"""Explicit units and sign conventions. No conversion is inferred from magnitude."""
from .schema import AmbiguousDataError


def power_to_kw(value: float, unit: str, import_sign: str) -> float:
    factors = {'W': .001, 'kW': 1.0, 'MW': 1000.0}
    if unit not in factors or import_sign not in ('positive','negative'):
        raise AmbiguousDataError('Explicit supported power unit and import_sign required')
    return value * factors[unit] * (1 if import_sign == 'positive' else -1)


def energy_to_kwh(value: float, unit: str) -> float:
    factors = {'Wh': .001, 'kWh': 1.0, 'MWh': 1000.0}
    if unit not in factors:
        raise AmbiguousDataError('Explicit supported energy unit required')
    return value * factors[unit]
