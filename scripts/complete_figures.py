"""Additional deterministic figures from confirmed local outputs only."""
from pathlib import Path
import argparse
import json
import hashlib
import os
os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS','1')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/energygap-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,path):
    fig.tight_layout()
    for extension in ('png','svg','pdf'):fig.savefig(path.with_suffix('.'+extension),dpi=160)
    plt.close(fig)


def complete(path):
    path=Path(path).resolve();out=path/'reports';status={}
    df=pd.read_csv(out/'episodes.csv')
    primary=df[df.partition=='test'].copy()
    fig,ax=plt.subplots(figsize=(7,4))
    for name,g in primary.groupby('method'):
        ax.scatter(g.realized_rate,g.nmae,s=10,alpha=.6,label=name)
    ax.set(xlabel='Realized artificial missing fraction of eligible support',ylabel='Episode nMAE (1)',title='Integrity pilot: capped masks, descriptive only')
    ax.legend(fontsize=6,ncol=2);save(fig,out/'error_missing_fraction')
    reference=primary[primary.method=='locf'][['site','fold','gap_id','mask_seed','mae_kw','forecast_mae_kw']]
    paired=primary.merge(reference,on=['site','fold','gap_id','mask_seed'],suffixes=('','_locf'))
    fig,ax=plt.subplots(figsize=(6,4))
    for site,g in paired.groupby('site'):
        ax.scatter(g.mae_kw_locf-g.mae_kw,g.forecast_mae_kw_locf-g.forecast_mae_kw,s=10,alpha=.6,label=site)
    ax.axhline(0,color='gray',linewidth=.5);ax.axvline(0,color='gray',linewidth=.5)
    ax.set(xlabel='Reconstruction improvement over LOCF (kW)',ylabel='Forecast 1 h improvement over LOCF (kW)',title='Matched episodes; repeated methods share temporal support')
    ax.legend();save(fig,out/'paired_forecast_utility')
    contrasts=json.loads((out/'contrasts.json').read_text())
    fig,ax=plt.subplots(figsize=(8,max(3,len(contrasts)*.28)))
    labels=[]
    for i,c in enumerate(contrasts):
        labels.append(f"{c['site']} {c['contrast']} {c['endpoint']}")
        if c.get('ci95') is not None and c.get('estimate') is not None:
            lo,hi=c['ci95'];estimate=c['estimate'];ax.plot([lo,hi],[i,i],color='navy');ax.plot(estimate,i,'o',color='navy')
        else:ax.text(.02,i,'NA: no admissible physical/support contrast',fontsize=7,va='center',transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(labels)),labels,fontsize=6);ax.axvline(0,color='gray',linewidth=.5)
    ax.set(xlabel='Proposed loss minus reference loss (normalized)',title='Pointwise intervals; confirmatory p values absent when unsupported')
    save(fig,out/'paired_difference_forest')
    # Fixed example selection: locf median and p90 of test episode error, stable ties.
    import sqlite3
    db=sqlite3.connect(path/'state.sqlite')
    records=[json.loads(r[0]) for r in db.execute("SELECT payload FROM results WHERE payload LIKE '%\"prediction\"%'")];db.close()
    ref=primary[primary.method=='locf'].sort_values(['mae_kw','key'],kind='stable')
    for quantile in (.5,.9):
        if ref.empty:break
        episode=ref.iloc[int(round((len(ref)-1)*quantile))]
        truthfile=path.parent.parent/'data/evaluation_truth'/path.name/(episode.query_id.replace(':','_')+'.json')
        if not truthfile.exists():status[f'example_p{int(100*quantile)}']='truth file unavailable; not fabricated';continue
        truth=json.loads(truthfile.read_text());mask=np.asarray(truth['evaluation_mask'],bool)
        y=np.asarray([np.nan if v is None else v for v in truth['values']],float)[mask]
        x=np.arange(len(y))*5
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,y,'k.-',label='Original observed truth (evaluation only)')
        for r in records:
            if r.get('gap_id')==episode.gap_id and r.get('method') in ('locf','linear','seasonal','linear+energy_consistent'):
                ax.plot(x,r['prediction'],'.-',label=r['method']+' ['+r['status']+']')
        ax.set(xlabel='Time within hidden target (min)',ylabel='Canonical power (kW)',title=f'Fixed LOCF error p{int(100*quantile)} example: {episode.site}, fold {episode.fold}')
        ax.legend(fontsize=7);save(fig,out/f'example_p{int(100*quantile)}')
    status['probabilistic_figures']='Not applicable to deterministic baseline run; use optional model pilot report.'
    status['confirmatory_inference']='Pending measurement semantics and temporal support; no fabricated p values.'
    (out/'figure_status.json').write_text(json.dumps(status,indent=2))
    return status


def model_figures(path):
    import sqlite3
    path=Path(path).resolve();out=path/'reports';out.mkdir(exist_ok=True)
    db=sqlite3.connect(path/'state.sqlite')
    records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM results')];db.close()
    rows=[]
    for r in records:
        if r.get('phase')!='inference' or 'probabilistic' not in r:continue
        for nominal,interval in r['probabilistic']['intervals'].items():
            rows.append(dict(site=r['site'],gap_id=r['gap_id'],nominal=float(nominal),crps=r['probabilistic']['crps'],**interval))
    df=pd.DataFrame(rows)
    if df.empty:return {'status':'no_valid_probabilistic_support'}
    df.to_csv(out/'probabilistic_metrics.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for site,g in df.groupby('site'):
        group=g.groupby('nominal').mean(numeric_only=True)
        axes[0].plot(group.index,group.coverage,'o-',label=site)
        width='normalized_width' if 'normalized_width' in group else 'width'
        axes[1].plot(group[width],group.coverage,'o-',label=site)
    axes[0].plot([0,1],[0,1],'k--',linewidth=.5)
    axes[0].set(xlabel='Nominal marginal coverage',ylabel='Observed marginal coverage')
    axes[1].set(xlabel='Mean interval width (normalized if available)',ylabel='Observed marginal coverage')
    for ax in axes:ax.legend()
    fig.suptitle('CSDI integrity pilot, 10 samples; no calibration claim');save(fig,out/'coverage_width')
    return {'status':'generated_from_confirmed_CSDI_samples','n_interval_rows':len(df)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--models',action='store_true');args=parser.parse_args()
    print(model_figures(args.run) if args.models else complete(args.run))
