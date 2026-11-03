"""Query API is a one-way information boundary: truth is returned separately."""
from dataclasses import dataclass
from datetime import timedelta
import math
from ..data.counters import counter_increment


@dataclass(frozen=True)
class Query:
    query_id: str
    site_id: str
    issuance_time: object
    task: str
    information_tier: str
    target_gap_id: str
    times: tuple[float,...]
    values: tuple[float,...]
    source_record_ids: tuple[str,...]
    available_to_method: tuple[bool,...]
    input_missing_mask: tuple[bool,...]
    target_gap_mask: tuple[bool,...]
    evaluation_mask: tuple[bool,...]
    padding_mask: tuple[bool,...]
    elapsed_since_observed: tuple[float,...]
    auxiliaries: dict
    energy_intervals_with_endpoint_ids: tuple[dict,...]

    @property
    def energy_intervals(self):
        return self.energy_intervals_with_endpoint_ids


@dataclass(frozen=True)
class EvaluationTruth:
    query_id: str
    values: tuple[float,...]
    evaluation_mask: tuple[bool,...]
    original_record_ids: tuple[str,...]


def build_query(series, episode, issuance_time=None, context_length=512, task='R', information_tier='P', artificial_hidden=None, lookahead_seconds=0, blackout=True, **kwargs):
    if 'info' in kwargs: information_tier=kwargs.pop('info')
    if kwargs: raise TypeError(f'Unknown query arguments: {list(kwargs)}')
    if task not in ('R','O','C') or information_tier not in ('P','A','E','AE'):
        raise ValueError('Unknown task/information tier')
    if context_length<=0: raise ValueError('context_length must be positive')
    rows=series.records
    target=set(episode.indices)
    hidden=(artificial_hidden if isinstance(artificial_hidden,(set,frozenset))
            else set(artificial_hidden if artificial_hidden is not None else episode.indices))
    if not target or min(target)<=0 or max(target)>=len(rows)-1:
        raise ValueError('Target episode needs valid left/right boundaries')
    right=episode.right_index
    issuance_time=issuance_time or rows[episode.end_index if task=='C' else right].availability_time
    if task=='R':
        end=right
        if rows[right].availability_time>issuance_time:
            raise ValueError('Reconnection boundary not yet available for R')
    elif task=='C':
        end=max(target)
    else:
        if lookahead_seconds<=0: raise ValueError('Offline O requires explicit positive look-ahead')
        cutoff=issuance_time+timedelta(seconds=lookahead_seconds)
        end=max((i for i,r in enumerate(rows) if r.grid_time<=cutoff),default=right)
    effective_cutoff=issuance_time+timedelta(seconds=lookahead_seconds) if task=='O' else issuance_time
    start=end-context_length+1
    if min(target)<max(0,start) or max(target)>end:
        raise ValueError('Target episode does not fit entirely in the declared context')
    if task=='C' and rows[max(target)].measurement_time>effective_cutoff:
        raise ValueError('Ongoing-failure targets must not exceed issuance time')
    # Apply replica-wide hide BEFORE extracting the window and computing age.
    def admissible(i):
        r=rows[i]
        return (i not in hidden and r.observed_original and not r.natural_missing and
                r.available_to_method and math.isfinite(r.value_canonical) and
                r.availability_time<=effective_cutoff and r.measurement_time<=effective_cutoff)
    # Only the window and the last preceding admissible observation determine
    # its inputs/ages. Avoid rescanning months of history for every query.
    first=max(0,start); last=None
    for i in range(first-1,-1,-1):
        if admissible(i):
            last=rows[i].grid_time.timestamp();break
    eligible={}; ages={}
    for i in range(first,end+1):
        r=rows[i]
        ok=(i not in hidden and r.observed_original and not r.natural_missing and r.available_to_method and math.isfinite(r.value_canonical) and r.availability_time<=effective_cutoff and r.measurement_time<=effective_cutoff)
        eligible[i]=ok
        if ok: last=r.grid_time.timestamp()
        ages[i]=math.inf if last is None else max(0.,r.grid_time.timestamp()-last)
    values=[]; times=[]; ids=[]; available=[]; missing=[]; targets=[]; evals=[]; padding=[]; elapsed=[]; truth=[]
    aux_names=sorted({k for r in rows for k in r.auxiliaries}) if information_tier in ('A','AE') else []
    aux={k:[] for k in aux_names}
    for i in range(start,end+1):
        if i<0:
            times.append(rows[0].grid_time.timestamp()+i*series.nominal_seconds)
            ids.append('padding');values.append(math.nan);available.append(False);missing.append(True);targets.append(False);evals.append(False);padding.append(True);elapsed.append(math.inf);truth.append(math.nan)
            for k in aux: aux[k].append(math.nan)
            continue
        r=rows[i]; ok=eligible[i]; is_target=i in target
        evaluate=is_target and i in hidden and r.observed_original and not r.natural_missing and math.isfinite(r.value_canonical)
        times.append(r.grid_time.timestamp());ids.append(r.source_record_id);values.append(r.value_canonical if ok else math.nan);available.append(ok);missing.append(not ok);targets.append(is_target);evals.append(evaluate);padding.append(False);elapsed.append(ages[i]);truth.append(r.value_canonical if evaluate else math.nan)
        for k in aux:
            # All channels disappear under a communication blackout, even when target power was naturally missing.
            auxok=r.availability_time<=effective_cutoff and not (blackout and i in hidden)
            aux[k].append(r.auxiliaries.get(k,math.nan) if auxok else math.nan)
    energy=[]
    if information_tier in ('E','AE') and task!='C':
        left=episode.left_index
        a,b=rows[left],rows[right]
        endpoints_ok=all(r.observed_original and r.available_to_method and (r.counter_availability_time or r.availability_time)<=effective_cutoff and r.measurement_time<=effective_cutoff for r in (a,b)) and left not in hidden and right not in hidden
        interval={'left_endpoint_id':a.source_record_id,'right_endpoint_id':b.source_record_id,'left_time':a.grid_time.timestamp(),'right_time':b.grid_time.timestamp(),'left_index':left-start,'right_index':right-start}
        for direction in ('import','export'):
            lv=getattr(a,direction+'_counter');rv=getattr(b,direction+'_counter')
            delta,status=counter_increment(lv,rv,left_available=endpoints_ok,right_available=endpoints_ok)
            interval[direction+'_kwh']=delta
            interval[direction+'_status']=status
        energy.append(interval)
    query_id=f'{series.site_id}:{episode.gap_id}:{task}:{information_tier}'
    q=Query(query_id,series.site_id,issuance_time,task,information_tier,episode.gap_id,tuple(times),tuple(values),tuple(ids),tuple(available),tuple(missing),tuple(targets),tuple(evals),tuple(padding),tuple(elapsed),{k:tuple(v) for k,v in aux.items()},tuple(energy))
    return q,EvaluationTruth(query_id,tuple(truth),tuple(evals),tuple(ids))
