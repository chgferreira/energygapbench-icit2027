"""Independent read-only audit of the supplied ICIT compact results.

Example:
    python analysis/scripts/audit_results.py --run-dir results/RUN_ID \
        --source-root . --output-dir analysis/audit

Only audit outputs are written. Source checks are optional; numerical and
scientific-fingerprint checks need only the compact result directory.
"""
from pathlib import Path
import argparse
import hashlib, json
from datetime import datetime
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True, help='Compact run directory containing protocol.lock.json and reports/')
parser.add_argument('--source-root', type=Path, help='Optional matching experiment source root with src/ and configs/')
parser.add_argument('--output-dir', type=Path, default=Path('analysis/audit'), help='Directory for audit JSON and CSV outputs')
args = parser.parse_args()
RUN = args.run_dir.resolve()
OUT = args.output_dir.resolve()
SOURCE_ROOT = args.source_root.resolve() if args.source_root is not None else None
if not (RUN/'protocol.lock.json').is_file():
    parser.error('--run-dir must contain protocol.lock.json')
if OUT == RUN or OUT.is_relative_to(RUN):
    parser.error('--output-dir must be outside the immutable source run directory')
OUT.mkdir(parents=True, exist_ok=True)

def read(name): return json.loads((RUN/name).read_text())
def hashfile(path): return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
def plain(x):
    if isinstance(x, dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

lock=read('protocol.lock.json'); summary=read('reports/summary.json'); progress=read('progress.json')
loss=pd.read_parquet(RUN/'reports/episode_losses.parquet')
main=pd.read_csv(RUN/'reports/main_table.csv'); core=pd.read_csv(RUN/'reports/core_aggregate.csv')
fits=pd.read_csv(RUN/'reports/fits.csv'); metadata=pd.DataFrame(fits.training_metadata.map(json.loads).tolist())
fit=pd.concat([fits.drop(columns=['training_metadata']), metadata[['best_epoch','completed_epochs','fit_seconds','train_windows']]],axis=1)
contrasts=pd.DataFrame(read('reports/contrasts.json'))
sens=read('reports/contrasts_block_sensitivity.json')
selected=summary['selected_B_validation']
keys=['site','fold','partition','condition','duration','mask_seed','method','variant','gap_id','day']
assert not loss.duplicated(keys).any()
assert loss.nmae.notna().all()
test=loss[loss.partition.eq('test')].copy()
test['base_method']=test.method.str.replace('+energy_consistent','',regex=False)
identity=['site','fold','condition','duration','mask_seed','gap_id','day']
assert test.groupby(identity).size().eq(32).all()
assert test[test.base_method.eq('SAITS')].seed_evaluations.eq(3).all()
assert test[~test.base_method.eq('SAITS')].seed_evaluations.eq(1).all()
cells=test.groupby(['site','fold','condition','duration','method','variant'],as_index=False)[['nmae','mae_kw']].mean()
chk=cells.merge(main,on=['site','fold','condition','duration','method','variant'],suffixes=('_audit','_reported'),validate='one_to_one')
assert len(chk)==864
main_max={m:float(abs(chk[m+'_audit']-chk[m+'_reported']).max()) for m in ['nmae','mae_kw']}
agg=cells.groupby(['site','method','variant'],as_index=False)[['nmae','mae_kw']].mean()
c=agg.merge(core,on=['site','method','variant'],suffixes=('_audit','_reported'),validate='one_to_one')
core_max={m:float(abs(c[m+'_audit']-c[m+'_reported']).max()) for m in ['nmae','mae_kw']}
assert max(main_max.values())<1e-11 and max(core_max.values())<1e-11

# Every base/variant shares exactly the same observed one-hour target support.
avail=test.groupby(identity).forecast_nmae.agg(['count','size'])
assert avail['count'].isin([0,32]).all()
base=test[test.method.eq('linear') & test.variant.eq('P')]
coverage=[]
for site,g in base.groupby('site'):
 coverage.append(dict(site=site,episodes=len(g),forecast_available=int(g.forecast_nmae.notna().sum()),forecast_unavailable=int(g.forecast_nmae.isna().sum()),forecast_unavailable_percent=float(100*g.forecast_nmae.isna().mean()),distinct_reconnection_days=g.day.nunique()))

# Assemble validation-selected B using fold-specific choices, with no reselection.
selection_key=test.site.astype(str)+':fold'+test.fold.astype(str)
B=test[test.base_method.eq(selection_key.map(selected))]
B_cells=B.groupby(['site','fold','duration','variant'],as_index=False).nmae.mean()
B_means=B_cells.groupby(['site','variant']).nmae.mean().unstack()
B_results=[]
for site,row in B_means.iterrows():
 for variant in ['E:instantaneous','E:interval_mean_start','E:interval_mean_end']:
  B_results.append(dict(site=site,variant=variant,base_nmae=row['P'],corrected_nmae=row[variant],difference=row[variant]-row['P'],relative_reduction_percent=100*(row['P']-row[variant])/row['P']))
B_results=pd.DataFrame(B_results)
for _,r in B_results.iterrows():
 exported=contrasts[(contrasts.site==r.site)&(contrasts.hypothesis==r.variant[2:])&(contrasts.contrast=='contrast_energy_gain')&(contrasts.endpoint=='nmae')].iloc[0]
 assert abs(r.difference-exported.estimate)<1e-12

# Equality of I1 interval-mean losses is observable; solver/selection cause is absent.
i=test[test.site.eq('I1')].pivot(index=identity+['base_method'],columns='variant',values='nmae')
identical={v:bool((i[v]==i['P']).all()) for v in ['E:interval_mean_start','E:interval_mean_end']}

# Verify supplied canonical scientific identity separately from file-byte hashes.
# This reproduces the frozen scientific identity algorithm without importing
# the experiment package (which can require GPU/model dependencies).
PRESENTATION = {'progress', 'output_dir', 'state_dir', 'checkpoint_every_steps',
                'checkpoint_every_seconds', 'checkpoint_optimizer_steps',
                'checkpoint_active_seconds', 'progress_refresh_seconds',
                'progress_persist_seconds', 'status_snapshot_seconds', 'heartbeat_seconds',
                'resume', 'budget_total_hours', 'budget_seconds', 'timezone', 'retry_failed'}
def scientific_config(config):
    return {k: scientific_config(v) if isinstance(v, dict) else v
            for k, v in config.items() if k not in PRESENTATION}
def fingerprint(config):
    canonical = json.dumps(scientific_config(config), sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()
assert fingerprint(lock)==(RUN/'protocol.lock.sha256').read_text().strip()
assert read('reports/report_complete.json')['fingerprint']==fingerprint(lock)==progress['fingerprint']
source_mismatches = []
registry_checks = []
checked_source_files = 0
source_check_status = 'not_performed_no_source_root'
if SOURCE_ROOT is not None:
    registry=SOURCE_ROOT/'configs/runtime_patches/reporting_performance_v1.json'
    reg=json.loads(registry.read_text())
    expected_files=reg['target_source_files']
    actual_files={str(path.relative_to(SOURCE_ROOT)):hashfile(path)
                  for path in (SOURCE_ROOT/'src/energygap').rglob('*.py')}
    source_mismatches=sorted(name for name in actual_files.keys() | expected_files.keys()
                             if actual_files.get(name)!=expected_files.get(name))
    assert not source_mismatches, f'Experiment source differs from recorded release: {source_mismatches}'
    checked_source_files=len(actual_files)
    source_check_status='passed_exact_source_manifest'
    for item in summary['runtime_execution_history']:
        patch=item.get('patch',{})
        if not patch:continue
        filename={'cpu-worker-scaling-v1':'cpu_worker_scaling_v1.json',
                  'reporting-performance-v1':'reporting_performance_v1.json'}[patch['patch_id']]
        actual=hashfile(SOURCE_ROOT/'configs/runtime_patches'/filename)
        assert actual==patch['manifest_sha256']
        registry_checks.append(dict(session_id=item['session_id'],patch_id=patch['patch_id'],
            manifest_sha256_matches=True,scientific_fingerprint_matches=item['scientific_fingerprint']==fingerprint(lock)))

runtime=dict(active_accounted_hours=progress['active_seconds']/3600,first_started_at=progress['first_started_at'],ended_at=progress['ended_at'],elapsed_calendar_hours=(datetime.fromisoformat(progress['ended_at'])-datetime.fromisoformat(progress['first_started_at'])).total_seconds()/3600,resume_count=progress['resume_count'],jobs=progress['total_jobs'],succeeded=progress['counts']['succeeded'])
fit_summary=fit.groupby(['site','training_role']).agg(fits=('key','size'),best_epoch_min=('best_epoch','min'),best_epoch_max=('best_epoch','max'),completed_epoch_min=('completed_epochs','min'),completed_epoch_max=('completed_epochs','max'),recorded_fit_seconds=('fit_seconds','sum')).reset_index()
fit[['site','fold','train_seed','training_role','status','max_epochs','patience','best_epoch','completed_epochs','fit_seconds','train_windows']].to_csv(OUT/'fit_audit.csv',index=False)
B_results.to_csv(OUT/'selected_B_reconstruction.csv',index=False)
pd.DataFrame(coverage).to_csv(OUT/'target_support.csv',index=False)
contrasts.to_csv(OUT/'reported_contrasts_audit.csv',index=False)

audit=dict(run_id=RUN.name,scientific_fingerprint=fingerprint(lock),protocol_file_sha256=hashfile(RUN/'protocol.lock.json'),canonical_fingerprint_not_file_checksum=True,source_check_status=source_check_status,checked_current_source_files=checked_source_files,source_mismatches=source_mismatches,registry_checks=registry_checks,operational_status=summary['runtime_status'],execution_complete=summary['execution_complete'],scientific_complete=summary['scientific_complete'],unresolved_jobs=summary['unresolved_jobs'],counts={k:summary[k] for k in summary if k.startswith('n_')},fallback_fraction_of_all_inference=summary['n_fallback_records']/summary['n_confirmed_inference_records'],hpo_epochs_actual=lock['experiment_settings']['hpo_max_epochs'],selected_B=selected,coverage=coverage,unique_test_masked_episode_units=len(base),loss_rows=len(loss),test_loss_rows=len(test),duplicate_loss_keys=0,loss_support_rows_per_episode=32,recomputed_main_max_abs_error=main_max,recomputed_core_max_abs_error=core_max,forecast_missing_same_across_all_methods=True,forecast_primary_available=False,forecast_main_finite_rows=int(main.forecast_nmae.notna().sum()),forecast_main_total_rows=len(main),forecast_core_finite_rows=int(core.forecast_nmae.notna().sum()),forecast_unavailable_cause='Executed reporter requires every episode loss to be finite and refuses any unmatched/unscorable pair; this is stricter than retaining matched available-target episodes with all required strata.',I1_interval_mean_episode_losses_identical_to_P=identical,I1_identity_cause='Not established by compact archive. Calibration, selected correction parameters and per-record solver diagnostics are absent.',B_reconstruction=B_results.to_dict('records'),fit_summary=fit_summary.to_dict('records'),runtime=runtime,evidence_limitations=['No state.sqlite, model artifacts, mask manifests, correction-selection records, raw forecasts_long table, or detailed inference/fallback records supplied.','Canonical protocol identity is checked. Experiment-source checks are performed only when --source-root is supplied and are reported separately. Compact report tables have no supplied per-file hash manifest to independently validate against the original ledger.','Current table/loss agreement is an internal numerical consistency check, not independent physical validation.'])
(OUT/'results_audit.json').write_text(json.dumps(plain(audit),indent=2,allow_nan=False)+'\n')
print(json.dumps(plain({k:audit[k] for k in ['recomputed_main_max_abs_error','recomputed_core_max_abs_error','coverage','B_reconstruction','runtime']}),indent=2))
