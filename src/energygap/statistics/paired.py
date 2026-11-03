from __future__ import annotations
from datetime import date, timedelta
import hashlib
import numpy as np


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def holm(pvalues, family_size=12):
    """Missing hypotheses remain absent in output, but count as p=1 in family."""
    if len(pvalues) > family_size:
        raise ValueError("Family size smaller than supplied hypotheses")
    numeric = [1.0 if p is None else float(p) for p in pvalues] + [1.]*(family_size-len(pvalues))
    if any(p<0 or p>1 or not np.isfinite(p) for p in numeric):
        raise ValueError("Invalid p value")
    order = sorted(range(family_size), key=lambda i: numeric[i])
    adjusted, previous = [1.]*family_size, 0.
    for rank,i in enumerate(order):
        previous = max(previous, min(1., (family_size-rank)*numeric[i]))
        adjusted[i] = previous
    return [None if p is None else adjusted[i] for i,p in enumerate(pvalues)]


def benjamini_yekutieli(pvalues):
    m = len(pvalues)
    if not m:
        return []
    harmonic = sum(1/i for i in range(1,m+1))
    order = np.argsort([1. if p is None else p for p in pvalues])
    adjusted, previous = np.ones(m), 1.
    for rank in range(m-1,-1,-1):
        i = order[rank]
        previous = min(previous, (1. if pvalues[i] is None else pvalues[i])*m*harmonic/(rank+1))
        adjusted[i] = min(1., previous)
    return [None if p is None else float(adjusted[i]) for i,p in enumerate(pvalues)]


def collapse_replicates(rows):
    """Average losses, not reconstructions; identical seed copies add no days."""
    groups = {}
    for r in rows:
        key = (int(r["fold"]), str(r["day"]), int(r["duration"]), str(r["gap_id"]))
        groups.setdefault(key, []).append(float(r["difference"]))
    return [dict(fold=k[0],day=k[1],duration=k[2],gap_id=k[3],difference=float(np.mean(v))) for k,v in groups.items()]


def fixed_estimand(rows, folds=(1,2,3), durations=(15,60,180)):
    losses=[]
    for fold in folds:
        for duration in durations:
            values=[r["difference"] for r in rows if int(r["fold"])==fold and int(r["duration"])==duration]
            if not values:
                return None
            losses.append(float(np.mean(values)))
    return float(np.mean(losses))


def bootstrap_replicate(rows, calendars, replicate_id, *, block_days=7, seed=20261031, folds=(1,2,3), durations=(15,60,180)):
    rng=np.random.default_rng(stable_seed(seed,replicate_id))
    sampled=[]
    for fold in folds:
        days=list(calendars[fold])  # Includes empty days, never bridges folds.
        if not days:
            return None
        if len(days)>1 and any((date.fromisoformat(str(b))-date.fromisoformat(str(a))).days != 1 for a,b in zip(days,days[1:])):
            raise ValueError("Calendars must include contiguous empty days")
        length=min(block_days,len(days))
        chosen=[]
        while len(chosen)<len(days):
            start=int(rng.integers(0,len(days)-length+1))
            chosen.extend(days[start:start+length])
        for day in chosen[:len(days)]:
            sampled.extend(r for r in rows if int(r["fold"])==fold and str(r["day"])==str(day))
    return fixed_estimand(sampled,folds,durations)


def paired_bootstrap(rows, calendars, *, replicates=10000,block_days=7,seed=20261031,folds=(1,2,3),durations=(15,60,180)):
    rows=collapse_replicates(rows)
    estimate=fixed_estimand(rows,folds,durations)
    support_blocks=sum(len(calendars[f])//block_days for f in folds)
    minimum_by_fold=all(len(calendars[f])//block_days>=4 for f in folds)
    duration_days={d:len({(r['fold'],r['day']) for r in rows if r['duration']==d}) for d in durations}
    support=bool(support_blocks>=20 and minimum_by_fold and all(v>=20 for v in duration_days.values()))
    values=[bootstrap_replicate(rows,calendars,i,block_days=block_days,seed=seed,folds=folds,durations=durations) for i in range(replicates)]
    valid=np.array([v for v in values if v is not None],float)
    invalid=replicates-len(valid)
    confirmatory=support and invalid==0 and estimate is not None
    interval=list(map(float,np.quantile(valid,[.025,.975]))) if len(valid) else None
    p=(1+int(np.sum(np.abs(valid-estimate)>=abs(estimate))))/(replicates+1) if confirmatory else None
    return dict(estimate=estimate,ci95=interval,p_raw=p,confirmatory=confirmatory,interval_kind="pointwise" if confirmatory else "exploratory_conditional_valid",attempted=replicates,invalid=invalid,support_blocks=support_blocks,duration_days=duration_days,replicates=values,reason=None if confirmatory else "insufficient temporal support, empty strata, or invalid bootstrap replicates")
