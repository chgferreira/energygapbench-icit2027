from .availability import Query, EvaluationTruth, build_query


def match_future_target(series, issuance_time, horizon_seconds, tolerance_seconds=90, fold=None):
    """Nearest original measurement, deterministic tie, no interpolation."""
    from datetime import timedelta
    import math
    target=issuance_time+timedelta(seconds=horizon_seconds)
    if horizon_seconds<=0: raise ValueError('Forecast horizon must be strictly positive')
    if fold is not None and fold.partition(target)!='test':
        return None,'target_outside_test'
    candidates=[]
    for i,r in enumerate(series.records):
        offset=(r.measurement_time-target).total_seconds()
        if r.measurement_time>issuance_time and abs(offset)<=tolerance_seconds and r.observed_original and not r.natural_missing and math.isfinite(r.value_canonical):
            if fold is None or fold.partition(r.measurement_time)=='test':
                candidates.append((abs(offset),r.measurement_time,r.source_record_id,i,offset))
    if not candidates: return None,'no_original_target_in_tolerance'
    _,_,_,index,offset=min(candidates)
    return {'index':index,'target_time':target,'measurement_time':series.records[index].measurement_time,'offset_seconds':offset,'value':series.records[index].value_canonical},None
