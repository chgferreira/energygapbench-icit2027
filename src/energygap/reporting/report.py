from __future__ import annotations
from pathlib import Path
import json
import os
import sqlite3
import numpy as np
import pandas as pd
from energygap.statistics.paired import paired_bootstrap,holm
from energygap.runtime.state import atomic_json


def generate_report(run_dir):
    path=Path(run_dir)
    db=sqlite3.connect(f'file:{path/"state.sqlite"}?mode=ro',uri=True)
    rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM results ORDER BY result_key')]
    meta={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM meta')}
    jobs=[dict(zip(['job_id','status','reason'],r)) for r in db.execute('SELECT job_id,status,reason FROM jobs')]
    db.close()
    out=path/'reports';out.mkdir(exist_ok=True)
    df=pd.DataFrame([r for r in rows if 'method' in r])
    status=dict(scientific_experiment_complete=False,synthetic=meta['config'].get('profile')=='smoke',runtime_status=meta['status'],n_confirmed_results=len(df),blocked_jobs=[j for j in jobs if j['status'] in ('blocked','unavailable','unsupported')],limitations=['Synthetic validation is not scientific evaluation on H1/H2/I1.','Conference profile remains incomplete until metadata, upstream integrations and recovery requirements pass.'])
    if df.empty:
        atomic_json(out/'summary.json',status)
        (out/'report.md').write_text('# EnergyGapBench execution report\n\nNo scientific result is available.\n\n'+ '\n'.join(f"- {j['job_id']}: {j['reason']}" for j in status['blocked_jobs'])+'\n')
        return status
    df.drop(columns=['prediction'],errors='ignore').to_csv(out/'episodes.csv',index=False)
    df.drop(columns=['prediction'],errors='ignore').to_parquet(out/'episodes.parquet',index=False)
    # Explicit long schema; undefined metrics remain null rather than zero.
    identifiers=['run_id','site','fold','task','information_tier','method','variant','context','gap_id','issuance_time','duration','condition','mask_seed','train_seed','sample_seed','status']
    long=df.melt(id_vars=identifiers,value_vars=['mae_kw','rmse_kw','nmae','forecast_mae_kw','forecast_nmae'],var_name='metric',value_name='value')
    long['unit']=long['metric'].map(lambda m:'kW' if m.endswith('_kw') else '1')
    long.to_csv(out/'metrics_long.csv',index=False)
    primary=df[df.partition=='test'].copy()
    summary=primary.groupby(['site','fold','duration','method','information_tier'],dropna=False).agg(gap_macro_nmae=('nmae','mean'),gap_macro_mae_kw=('mae_kw','mean'),forecast_nmae=('forecast_nmae','mean'),episodes=('gap_id','count'),latency_p50_seconds=('latency_seconds','median'),latency_p95_seconds=('latency_seconds',lambda x:x.quantile(.95))).reset_index()
    summary.to_csv(out/'main_table.csv',index=False)
    # Select B solely on validation: equal weight C3/C4/C5, stable tie order.
    validation=df[(df.partition=='validation') & ~df.method.str.contains('+',regex=False)]
    chosen={}
    for (site,fold),part in validation.groupby(['site','fold']):
        candidates=[]
        for method,g in part.groupby('method'):
            cells=g.groupby('duration').nmae.mean()
            if {15,60,180}<=set(cells.index) and cells.notna().all():
                candidates.append((float(cells.reindex([15,60,180]).mean()),float(g.latency_seconds.median()),method))
        if candidates:chosen[(site,int(fold))]=min(candidates)[2]
    contrasts=[]
    for site in primary.site.unique():
        for contrast in ('contrast_energy_gain','contrast_complexity_gain'):
            for metric in ('nmae','forecast_nmae'):
                paired=[];calendars={}
                from datetime import date,timedelta
                for fold in (1,2,3):
                    b=chosen.get((site,fold))
                    if not b:continue
                    proposal=b+'+energy_consistent'
                    comparison=b if contrast=='contrast_energy_gain' else 'linear+energy_consistent'
                    a=primary[(primary.site==site)&(primary.fold==fold)&(primary.method==proposal)]
                    reference=primary[(primary.site==site)&(primary.fold==fold)&(primary.method==comparison)]
                    merged=a.merge(reference,on=['gap_id','duration','day','mask_seed'],suffixes=('_a','_b'))
                    for _,r in merged.iterrows():
                        if pd.notna(r[metric+'_a']) and pd.notna(r[metric+'_b']):
                            paired.append(dict(fold=fold,day=r.day,duration=int(r.duration),gap_id=r.gap_id,difference=float(r[metric+'_a']-r[metric+'_b'])))
                    # Use full TEST calendar from split manifest, including empty days.
                    manifest=json.loads((path/'manifests'/f'{site}:fold{fold}-split.json').read_text())
                    first=date.fromisoformat(manifest['validation_end_exclusive'][:10]);last=date.fromisoformat(manifest['test_end'][:10])
                    calendars[fold]=[(first+timedelta(days=i)).isoformat() for i in range((last-first).days+1)]
                if set(calendars)=={1,2,3} and paired:
                    from energygap.runtime.statistics_job import persisted_paired_bootstrap
                    result=persisted_paired_bootstrap(path,paired,calendars,contrast_id=f'{site}:{contrast}:{metric}',replicates=100 if status['synthetic'] else 10000)
                    result.pop('replicates')
                else:
                    result=dict(estimate=None,ci95=None,p_raw=None,confirmatory=False,reason='selected B correction not integrated or missing fold/duration support')
                contrasts.append(dict(site=site,contrast=contrast,endpoint=metric,**result))
    adjusted=holm([r['p_raw'] for r in contrasts],family_size=max(12,len(contrasts)))
    for row,p in zip(contrasts,adjusted):row['p_holm']=p
    atomic_json(out/'contrasts.json',contrasts)
    status['selected_B_validation']={f'{s}:fold{f}':m for (s,f),m in chosen.items()}
    status['n_confirmatory_tests']=sum(r['confirmatory'] for r in contrasts)
    atomic_json(out/'summary.json',status)
    # Standard exportable figures. Plot only measured outputs.
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/energygap-mpl')
    os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS','1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ('error_duration','pareto','forecast_utility'):
        fig,ax=plt.subplots(figsize=(7,4))
        if kind=='error_duration':
            for method,g in primary.groupby('method'):
                means=g.groupby('duration').nmae.mean();ax.plot(means.index,means.values,marker='o',label=method)
            ax.set(xlabel='Hidden duration (min)',ylabel='Gap-macro nMAE (1)')
            ax.legend(fontsize=6,ncol=2)
        elif kind=='pareto':
            for method,g in primary.groupby('method'):
                ax.scatter(g.latency_seconds.median(),g.nmae.mean(),label=method)
            ax.set(xlabel='Median inference latency (s)',ylabel='Gap-macro nMAE (1)');ax.legend(fontsize=6)
        else:
            g=primary.dropna(subset=['forecast_mae_kw','mae_kw'])
            ax.scatter(g.mae_kw,g.forecast_mae_kw,s=8,alpha=.5)
            ax.set(xlabel='Reconstruction MAE (kW)',ylabel='Forecast absolute error at 1 h (kW)')
        ax.set_title('SYNTHETIC VALIDATION' if status['synthetic'] else 'RESTRICTED PILOT')
        fig.tight_layout()
        for ext in ('png','svg','pdf'):fig.savefig(out/f'{kind}.{ext}',dpi=150)
        plt.close(fig)
    (out/'report.md').write_text('# EnergyGapBench execution report\n\n'+('Synthetic fixture; no H1/H2/I1 scientific claim.\n\n' if status['synthetic'] else 'Restricted pilot; conference incomplete.\n\n')+f"Confirmed episode-method results: {len(df)}. Confirmatory tests supported: {status['n_confirmatory_tests']}.\n\nSee main_table.csv, episodes.parquet and contrasts.json. Undefined contrasts remain NA.\n")
    return status
