"""Report declared scope separately from measured coverage and missing methods."""
from __future__ import annotations

import json
from pathlib import Path


CATALOGUE_METHODS = ('median','locf','linear','seasonal','kalman_filter','kalman_smoother',
    'BRITS','SAITS','CSDI','ImputeFormer','T1','MOMENT-1-base','TS-ICL','FIM','TabPFN-TS','MoTM','CBiT')
CONDITIONS = ('C1','C2','C3','C4','C5','C6')


def committed_coverage(db):
    """Scalar counts only; inference arrays never enter the scope summary."""
    inference = {job: int(count) for job,count in db.execute(
        "SELECT job_id,COUNT(*) FROM results WHERE json_extract(payload,'$.phase')='inference' "
        "AND json_extract(payload,'$.variant')='P' GROUP BY job_id")}
    hpo = {job: int(count) for job,count in db.execute(
        "SELECT job_id,COUNT(*) FROM results WHERE json_extract(payload,'$.phase')='hpo_score' GROUP BY job_id")}
    return dict(inference=inference,hpo=hpo)


def study_design(path,config,fit_records,jobs,coverage):
    settings = config.get('experiment_settings',{})
    profile = settings.get('study_profile',config.get('study_profile','full_available'))
    requested = list(config.get('methods',settings.get('methods',())))
    conditions = list(config.get('conditions',settings.get('conditions',())))
    availability = config.get('availability',{})
    omitted_methods = [name for name in dict.fromkeys((*CATALOGUE_METHODS,*availability)) if name not in requested]
    omitted_conditions = [name for name in CONDITIONS if name not in conditions] if conditions else []
    unavailable = {name:availability[name] for name in requested if name in availability and
                   availability[name].get('status') in ('unavailable','unsupported','blocked')}
    all_available = bool(requested) and all(availability.get(name,{}).get('status')=='available' for name in requested)
    panels = []
    for file in sorted((Path(path)/'manifests').glob('*-epoch-validation-panel.json')):
        panel = json.loads(file.read_text())
        panels.append(dict(manifest=str(file.relative_to(path)),**{key:panel.get(key) for key in (
            'policy','site','fold','partition','selection_seed','max_per_stratum','candidate_count',
            'selected_count','covers_all_candidates','candidate_identity_sha256','selected_identity_sha256',
            'panel_sha256','strata','missing_required_strata','scope','ranking','stratification',
            'selection_values_used','selection_metrics_used','coverage_status')}))
    budgets = []
    for record in fit_records:
        metadata = record.get('training_metadata',{})
        value = lambda key:record.get(key,metadata.get(key))
        budgets.append(dict(job_id=record.get('key'),method=record.get('method'),site=record.get('site'),
            fold=record.get('fold'),train_seed=record.get('train_seed'),status=record.get('status'),
            training_role=value('training_role'),max_epochs=value('max_epochs'),patience=value('patience'),
            epoch_validation_policy=value('epoch_validation_policy'),
            epoch_validation_candidates=value('epoch_validation_candidates'),
            epoch_validation_selected=value('epoch_validation_selected'),
            epoch_validation_panel_sha256=value('epoch_validation_panel_sha256'),
            epoch_validation_manifest=value('epoch_validation_manifest')))
    indexed = [(job,json.loads(job['metadata'])) for job in jobs]
    dimensions = {key:config.get(key,settings.get(key)) for key in
                  ('sites','folds','conditions','train_seeds','validation_mask_seeds','test_mask_seeds')}
    trainable = {'BRITS','SAITS','CSDI','ImputeFormer','T1'}
    eligible = [name for name in requested if availability.get(name,{}).get('status')=='available']
    cache = {}
    def manifest_count(site,fold,partition,condition,seed):
        key = site,fold,partition,condition,seed
        if key not in cache:
            file = Path(path)/'manifests'/f'{site}:fold{fold}-{partition}-{condition}-{seed}.json'
            cache[key] = len(json.loads(file.read_text())['episodes']) if file.is_file() else None
        return cache[key]
    def evaluation(partition):
        selected = [(job,meta) for job,meta in indexed if meta.get('phase')=='inference' and meta.get('partition')==partition]
        expected_keys = {(site,fold,method,seed,condition,mask_seed)
            for site in dimensions['sites'] or () for fold in dimensions['folds'] or ()
            for method in eligible for seed in (dimensions['train_seeds'] or ()) if method in trainable
            for condition in dimensions['conditions'] or () for mask_seed in dimensions[partition+'_mask_seeds'] or ()}
        expected_keys.update((site,fold,method,None,condition,mask_seed)
            for site in dimensions['sites'] or () for fold in dimensions['folds'] or ()
            for method in eligible if method not in trainable
            for condition in dimensions['conditions'] or () for mask_seed in dimensions[partition+'_mask_seeds'] or ())
        actual_keys = {(meta.get('site'),meta.get('fold'),meta.get('method'),meta.get('train_seed'),
                       meta.get('condition'),meta.get('mask_seed')) for job,meta in selected}
        plan_verified = bool(expected_keys) and actual_keys==expected_keys
        matched = []
        for job,meta in selected:
            expected = manifest_count(meta.get('site'),meta.get('fold'),partition,meta.get('condition'),meta.get('mask_seed'))
            matched.append(expected is not None and expected==job['total_units'])
        verified = bool(selected) and all(matched)
        complete = (bool(selected) and all(job['status']=='succeeded' and
            job['completed_units']==job['total_units']==coverage['inference'].get(job['job_id'],0)
            for job,meta in selected))
        available_complete = complete and verified and plan_verified if selected else None
        return dict(reference_scope='available requested methods with configured conditions, sites, folds and seeds',
            planned_jobs=len(selected),planned_queries=sum(job['total_units'] for job,meta in selected),
            committed_queries=sum(coverage['inference'].get(job['job_id'],0) for job,meta in selected),
            full_manifest_counts_verified=verified if selected else None,
            declared_plan_dimensions_verified=plan_verified,
            requested_methods_all_available=all_available,
            full_available_manifest_evaluation_completed=available_complete,
            full_declared_manifest_evaluation_completed=available_complete and all_available if selected else None,
            source='immutable mask manifests, job totals/status and committed P records',
            configured_episode_cap=settings.get('max_episodes'),
            episode_cap_explicitly_declared='max_episodes' in settings)
    trial_indices = settings.get('hpo_trial_indices')
    if trial_indices is None and 'hpo_trials' in settings:
        trial_indices = list(range(settings['hpo_trials']))
    hpo_jobs = [(job,meta) for job,meta in indexed if meta.get('phase')=='hpo']
    expected_hpo = {(site,method,trial) for site in dimensions['sites'] or ()
                    for method in eligible if method in trainable for trial in trial_indices or ()}
    actual_hpo = {(meta.get('site'),meta.get('method'),meta.get('trial')) for job,meta in hpo_jobs}
    hpo_plan_verified = bool(expected_hpo) and actual_hpo==expected_hpo
    hpo_verified = []
    folds = config.get('folds',settings.get('folds',()))
    seeds = config.get('validation_mask_seeds',settings.get('validation_mask_seeds',()))
    for job,meta in hpo_jobs:
        counts = [manifest_count(meta.get('site'),min(folds),'validation',condition,seed)
                  for condition in conditions if condition in ('C3','C4','C5') for seed in seeds] if folds else []
        hpo_verified.append(bool(counts) and None not in counts and sum(counts)==job['total_units']-1)
    hpo_completed = bool(hpo_jobs) and all(job['status']=='succeeded' and
        job['completed_units']==job['total_units'] and coverage['hpo'].get(job['job_id'],0)==job['total_units']-1
        for job,meta in hpo_jobs)
    reduced = bool(omitted_methods or omitted_conditions or settings.get('epoch_validation_max_per_stratum') is not None or
                   (trial_indices is not None and trial_indices!=list(range(6))))
    return dict(study_profile=profile,scope_reduced_by_design=reduced,full_catalogue_reproduced=False,
        requested_methods=requested,omitted_methods_by_design=omitted_methods,
        requested_methods_all_available=all_available,
        requested_conditions=conditions,omitted_conditions_by_design=omitted_conditions,
        omission_reasons=settings.get('omission_reasons',{}),
        condition_omission_reasons=settings.get('condition_omission_reasons',{}),
        requested_methods_unavailable=unavailable,
        declaration_source='immutable experiment configuration; omissions are not runtime failures',
        declared_training_budgets=dict(hpo_trial_indices=trial_indices,
            hpo_max_epochs=settings.get('hpo_max_epochs',settings.get('max_epochs')),
            hpo_patience=settings.get('hpo_patience',settings.get('patience')),
            final_max_epochs=settings.get('max_epochs'),final_patience=settings.get('patience')),
        actual_fit_budgets=budgets,epoch_validation_panels=panels,
        epoch_validation_policy=dict(max_per_stratum=settings.get('epoch_validation_max_per_stratum'),
            selection_seed=settings.get('epoch_validation_seed'),selection_partition='validation',
            panel_evidence_source='saved panel manifests and committed fit metadata'),
        hpo_scoring=dict(planned_jobs=len(hpo_jobs),actual_trial_indices=sorted({meta['trial'] for job,meta in hpo_jobs if 'trial' in meta}),
            reference_scope='configured HPO trial identities for available requested trainable methods',
            declared_trial_matrix_verified=hpo_plan_verified,
            planned_queries=sum(max(0,job['total_units']-1) for job,meta in hpo_jobs),
            committed_queries=sum(coverage['hpo'].get(job['job_id'],0) for job,meta in hpo_jobs),
            full_manifest_counts_verified=all(hpo_verified) if hpo_verified else None,
            full_declared_validation_scoring_completed=hpo_completed and all(hpo_verified) and hpo_plan_verified if hpo_jobs else None),
        final_validation=evaluation('validation'),test_evaluation=evaluation('test'),
        budget_interpretation='cumulative operational ceiling; profile name and configuration do not guarantee completion within 24 hours')
