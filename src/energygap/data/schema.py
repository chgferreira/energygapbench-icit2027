"""Canonical observations; truth is deliberately absent from inference queries."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping
import math


class AmbiguousDataError(ValueError):
    """The evidence does not establish the physical/temporal data contract."""


@dataclass(frozen=True)
class CanonicalRecord:
    site_id: str
    source_file_id: str
    source_record_id: str
    measurement_time: datetime
    availability_time: datetime
    grid_time: datetime
    value_original: float
    value_canonical: float
    unit_original: str = 'kW'
    observed_original: bool = True
    natural_missing: bool = False
    artificial_hidden: bool = False
    available_to_method: bool = True
    padding_mask: bool = False
    imputed: bool = False
    quality_flag: str = 'ok'
    gap_id: str | None = None
    fold_id: str | None = None
    mask_id: str | None = None
    elapsed_since_observed: float = math.inf
    auxiliaries: Mapping[str, float] = field(default_factory=dict)
    import_counter: float | None = None
    export_counter: float | None = None
    counter_availability_time: datetime | None = None


@dataclass
class SeriesData:
    records: list[CanonicalRecord]
    site_id: str
    nominal_seconds: int = 300
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.nominal_seconds <= 0:
            raise ValueError('nominal_seconds must be positive')
        if any(r.site_id != self.site_id for r in self.records):
            raise ValueError('Sites cannot be combined as concurrent channels')
        if any(a.grid_time > b.grid_time for a, b in zip(self.records, self.records[1:])):
            raise ValueError('records must be sorted chronologically')

    @property
    def times(self):
        return [r.grid_time.timestamp() for r in self.records]

    @property
    def values(self):
        return [r.value_canonical for r in self.records]


@dataclass(frozen=True)
class DatasetConfig:
    data: dict

    @classmethod
    def load(cls, path):
        import yaml
        with open(path, encoding='utf8') as f:
            return cls(yaml.safe_load(f))

    def unresolved(self, purpose="physical") -> list[str]:
        required = [('timestamp', 'column'), ('timestamp', 'timezone'),
                    ('timestamp', 'semantics'), ('power', 'column'),
                    ('power', 'unit'), ('power', 'semantics'), ('power', 'import_sign')]
        if purpose not in ('physical','statistical'):
            raise ValueError('purpose must be physical or statistical')
        if purpose=='statistical':
            required=[item for item in required if item[1]!='semantics']
        missing = [f'{a}.{b}' for a,b in required if self.data.get(a, {}).get(b) in (None, '', 'unconfirmed')]
        confirmations = self.data.get('confirmed', {})
        missing += [f'confirmation:{a}.{b}' for a,b in required if not confirmations.get(f'{a}.{b}', False)]
        return sorted(set(missing))

    def validate(self, purpose="physical"):
        missing = self.unresolved(purpose)
        if missing:
            raise AmbiguousDataError('Dependent physical experiments blocked; unresolved: '+', '.join(missing))
        if purpose=='physical' and self.data['power']['semantics'] not in ('instantaneous', 'interval_mean'):
            raise AmbiguousDataError('power.semantics must be instantaneous or interval_mean')
        if self.data['power']['import_sign'] not in ('positive', 'negative'):
            raise AmbiguousDataError('power.import_sign must be positive or negative')
        return self
