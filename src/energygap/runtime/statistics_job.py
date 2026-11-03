"""Persist attempted paired-bootstrap IDs; invalid replicates are never redrawn."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .state import StateStore,canonical
from .scheduler import run_units
from energygap.statistics.paired import (bootstrap_replicate,collapse_replicates,
                                        fixed_estimand,stable_seed)


def persisted_paired_bootstrap(store_or_path, rows, calendars, *, contrast_id: str,
                              replicates=10000, block_days=7, seed=20261031,
                              folds=(1,2,3), durations=(15,60,180),
                              shard_size=100, progress='none') -> dict:
    """Same estimator as statistics.paired, with immutable per-replicate ledger rows.

    Input/config hashes form the stage ID. A changed input creates a separate stage,
    and cannot reuse old attempted draws. Scientific protocol changes must still be
    declared by the caller as a derived experiment, before test evaluation.
    """
    if replicates<=0 or block_days<=0:raise ValueError('positive attempted replicates/block days required')
    rows=collapse_replicates(rows)
    identity={'contrast_id':contrast_id,'rows':rows,'calendars':{str(k):list(v) for k,v in calendars.items()},
              'replicates':replicates,'block_days':block_days,'seed':seed,
              'folds':list(folds),'durations':list(durations),'algorithm':'paired-moving-block-v1'}
    digest=hashlib.sha256(canonical(identity)).hexdigest()[:20]
    job_id=f'bootstrap:{contrast_id}:{digest}'
    owns=not isinstance(store_or_path,StateStore)
    store=StateStore(store_or_path,config=None,resume='require') if owns else store_or_path
    try:
        store.add_job(job_id,replicates,metadata={'phase':'bootstrap','contrast_id':contrast_id,
                      'input_sha256':digest,'seed':seed,'block_days':block_days,'replicates':replicates})
        if store.job(job_id)['status']!='succeeded':
            if owns and store.get_meta('status') in {'completed','completed_with_failures'}:
                store.set_meta('status','planned')
            def compute(index,unused_seed):
                value=bootstrap_replicate(rows,calendars,index,block_days=block_days,seed=seed,folds=folds,durations=durations)
                return {'key':f'{job_id}:replicate:{index:08d}','phase':'bootstrap',
                        'contrast_id':contrast_id,'replicate_id':index,'sample_seed':stable_seed(seed,index),
                        'status':'invalid' if value is None else 'valid','value':value,
                        'reason':'required fold-duration stratum empty' if value is None else None}
            run_units(store,job_id,compute,shard_size=shard_size,progress=progress,
                      signature=f'bootstrap:days{block_days}:rows{len(rows)}')
        records=sorted(store.results(job_id),key=lambda r:r['replicate_id'])
        if len(records)!=replicates:raise RuntimeError('bootstrap incomplete: no provisional inferential summary')
        attempted_ids=[r['replicate_id'] for r in records]
        if attempted_ids!=list(range(replicates)):raise RuntimeError('bootstrap IDs missing/duplicated')
        values=[r['value'] for r in records]
        estimate=fixed_estimand(rows,folds,durations)
        valid=np.asarray([v for v in values if v is not None],float)
        invalid=replicates-len(valid)
        support_blocks=sum(len(calendars[f])//block_days for f in folds)
        duration_days={d:len({(r['fold'],r['day']) for r in rows if r['duration']==d}) for d in durations}
        support=bool(support_blocks>=20 and all(len(calendars[f])//block_days>=4 for f in folds)
                     and all(n>=20 for n in duration_days.values()))
        confirmatory=support and invalid==0 and estimate is not None
        interval=list(map(float,np.quantile(valid,[.025,.975]))) if len(valid) else None
        p=(1+int(np.sum(np.abs(valid-estimate)>=abs(estimate))))/(replicates+1) if confirmatory else None
        result={'estimate':estimate,'ci95':interval,'p_raw':p,'confirmatory':confirmatory,
                'interval_kind':'pointwise' if confirmatory else 'exploratory_conditional_valid',
                'attempted':replicates,'invalid':invalid,'support_blocks':support_blocks,
                'duration_days':duration_days,'replicates':values,'job_id':job_id,
                'reason':None if confirmatory else 'insufficient temporal support, empty strata, or invalid bootstrap replicates'}
        if owns:store.complete()
        return result
    finally:
        if owns:store.close()
