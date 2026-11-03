"""User-authorized conditional metrology sensitivity; never confirmatory physics."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import time
import numpy as np
import pandas as pd
from energygap.data import DatasetConfig,ingest,SeriesData
from energygap.protocols.splits import rolling_origin_splits
from energygap.protocols.availability import build_query
from energygap.masking.generators import generate_mask
from energygap.methods.baselines import create_baseline
from energygap.methods.calibration import fit_energy_tolerances,constraint_from_query
from energygap.methods.correction import EnergyConstraint,correct_energy
from energygap.evaluation.reconstruction import training_scale,reconstruction_metrics
from energygap.runtime.state import StateStore,fingerprint,atomic_json
from energygap.pipeline import code_hash


def run(root=Path('.'),budget_seconds=360):
    root=Path(root).resolve()
    hypotheses=('instantaneous','interval_mean_start','interval_mean_end')
    configs={s:DatasetConfig.load(root/'configs/datasets'/f'{s}.yaml') for s in ('H1','H2','I1')}
    cfg=dict(profile='exploratory_energy',code_hash=code_hash(),sites={s:c.data for s,c in configs.items()},hypotheses=hypotheses,folds=[1],conditions=['C3','C4','C5'],max_episodes=1,mask_seed=2001,methods=['linear','seasonal'],solver=dict(lambda_energy=1.,gamma=.01),classification='exploratory_conditional_on_unconfirmed_semantics',independent_calibration=False)
    identifier='energy-hypotheses-'+fingerprint(cfg)[:16]
    path=root/'runs'/identifier
    with StateStore(path,cfg,budget_seconds=budget_seconds) as store:
        if store.get_meta('status') in ('completed','completed_with_failures'):return store.status()
        atomic_json(path/'protocol.lock.json',cfg)
        # Plan fixed before any held-out evaluation; no physical hypothesis selected by outcome.
        for site in configs:
            for hypothesis in hypotheses:store.add_job(site+':'+hypothesis,3,metadata={'phase':'conditional_energy','site':site,'hypothesis':hypothesis})
        store.start_session()
        for site,config in configs.items():
            original=ingest(config,root,purpose='statistical')
            fold=rolling_origin_splits(original)[0];train=fold.indices(original,'train')
            values=np.asarray(original.values);times=np.asarray(original.times)
            scale,_=training_scale(values[train])
            models={m:create_baseline(m).fit(values[train],times[train]) for m in ('linear','seasonal')}
            masks={c:generate_mask(original,fold,condition=c,physical=True,max_episodes=1,seed=2001) for c in ('C3','C4','C5')}
            for hypothesis in hypotheses:
                job=site+':'+hypothesis
                if store.job(job)['status']=='succeeded':continue
                conditional=SeriesData(original.records,site,original.nominal_seconds,{**original.metadata,'power_semantics':hypothesis,'hypothesis_only':True})
                calibration=fit_energy_tolerances(conditional,train)
                atomic_json(path/'calibration'/f'{job}.json',calibration)
                start=store.job(job)['completed_units']
                with store.claim(job) as lease:
                    for index,condition in enumerate(('C3','C4','C5')):
                        if index<start:continue
                        store.check_budget();tic=time.perf_counter();records=[]
                        manifest=masks[condition]
                        atomic_json(path/'manifests'/f'{site}-{condition}.json',manifest.to_dict())
                        for episode in manifest.episodes:
                            query,truth=build_query(conditional,episode,information_tier='E',artificial_hidden=manifest.hidden_indices)
                            constraint=constraint_from_query(query,calibration,episode.duration_minutes,metadata=conditional.metadata)
                            if hypothesis!='instantaneous':
                                interval=query.energy_intervals[0]
                                # Net interval means do not identify gross bidirectional flows.
                                # Sum of separate train tolerances is conservative internal residual tolerance.
                                imp=constraint.tau_import;exp=constraint.tau_export
                                valid=all(v is not None for v in (imp,exp,interval.get('import_kwh'),interval.get('export_kwh')))
                                constraint=EnergyConstraint(net_kwh=interval['import_kwh']-interval['export_kwh'] if valid else None,tau_net=imp+exp if valid else None,semantics=hypothesis,reliable=valid,reason='conditional_net_mean_internal_train_residual_tolerance')
                            for method,model in models.items():
                                base=model.impute(query.values,query.times)
                                corrected=correct_energy(base,query.times,query.target_gap_mask,constraint,scale_train=scale,lambda_energy=1.,gamma=.01)
                                before=reconstruction_metrics(truth.values,base,truth.evaluation_mask,scale_train=scale)
                                after=reconstruction_metrics(truth.values,corrected.values,truth.evaluation_mask,scale_train=scale)
                                records.append(dict(key=f'{job}:{episode.gap_id}:{method}',site=site,fold=1,hypothesis=hypothesis,condition=condition,gap_id=episode.gap_id,method=method,information_tier='E',base_mae_kw=before['mae_kw'],corrected_mae_kw=after['mae_kw'],status=corrected.status,reason=corrected.diagnostics.get('reason'),classification='exploratory_conditional',confirmatory=False,independent_energy_measurement=False))
                        if not records:records=[dict(key=job+':'+condition,status='no_support',reason='no_counter_eligible_episode',hypothesis=hypothesis,classification='exploratory_conditional',confirmatory=False)]
                        store.publish(job,lease.token,records,index+1);store.record_active(time.perf_counter()-tic,job)
                    store.finish(job,lease.token)
        store.complete();store.end_session()
        records=store.results();pd.DataFrame(records).to_csv(path/'conditional_results.csv',index=False)
        summary=dict(experiment_id=identifier,status=store.status()['status'],n_results=len(records),classification='exploratory_conditional_on_unconfirmed_semantics',hypothesis_selected_by_test=False,physical_confirmatory_pending=True,output=str(path/'conditional_results.csv'))
        atomic_json(root/'artifacts/reports/exploratory_energy_summary.json',summary)
        print(summary);return summary


if __name__=='__main__':run()
