"""Alignment preserves measurement/arrival times and explicitly flags conflicts."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
from .schema import CanonicalRecord


def nearest_slot(time: datetime, nominal_seconds=300, tolerance_seconds=90):
    if time.tzinfo is None:
        raise ValueError('Alignment requires an evidenced timezone')
    seconds = time.timestamp()
    slot = math.floor(seconds / nominal_seconds + .5) * nominal_seconds
    if abs(slot-seconds) > tolerance_seconds:
        return None
    return datetime.fromtimestamp(slot, timezone.utc).astimezone(time.tzinfo)


def align_records(records: list[CanonicalRecord], nominal_seconds=300, tolerance_seconds=90):
    groups, rejected = {}, []
    for r in records:
        slot = nearest_slot(r.measurement_time, nominal_seconds, tolerance_seconds)
        if slot is None:
            rejected.append(replace(r, available_to_method=False, quality_flag='outside_alignment_tolerance'))
        else:
            groups.setdefault(slot, []).append(replace(r, grid_time=slot))
    accepted, collisions = [], []
    for slot, rows in sorted(groups.items()):
        if len(rows) == 1:
            accepted.append(rows[0])
        elif len({repr((r.measurement_time,r.availability_time,r.value_original,r.value_canonical,sorted(r.auxiliaries.items()),r.import_counter,r.export_counter,r.counter_availability_time)) for r in rows}) == 1:
            # Exact target/arrival duplicates retain a deterministic first record and audit all IDs.
            first = min(rows, key=lambda r: (r.source_file_id,r.source_record_id))
            accepted.append(replace(first, quality_flag='duplicate_identical'))
            collisions.append({'grid_time':slot.isoformat(), 'kind':'identical','record_ids':[r.source_record_id for r in rows]})
        else:
            # No conflicting value is chosen as usable ground truth.
            first = min(rows, key=lambda r: (r.source_file_id,r.source_record_id))
            accepted.append(replace(first, value_canonical=math.nan, observed_original=False, natural_missing=True, available_to_method=False, quality_flag='conflicting_collision'))
            collisions.append({'grid_time':slot.isoformat(), 'kind':'conflicting','record_ids':[r.source_record_id for r in rows]})
    return accepted, {'collisions':collisions, 'rejected':rejected}


def regularize(records, site_id, nominal_seconds=300):
    """Add explicit absent slots; this does not interpolate or manufacture truth."""
    if not records:
        return []
    by_time = {r.grid_time:r for r in records}
    current, end = min(by_time), max(by_time)
    result=[]
    while current <= end:
        result.append(by_time.get(current) or CanonicalRecord(site_id,'missing',f'missing:{current.isoformat()}',current,current,current,math.nan,math.nan,unit_original=records[0].unit_original,observed_original=False,natural_missing=True,available_to_method=False,quality_flag='natural_missing'))
        current += timedelta(seconds=nominal_seconds)
    return result
