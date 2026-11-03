"""Explicit scientific DAG. Unknown cardinalities/costs never become zero work.

This planner enumerates the full requested conference catalogue even when an adapter
is blocked. The existing restricted runner is not claimed to execute this DAG.
"""
from __future__ import annotations
import csv
from collections import Counter,defaultdict,deque
from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,statistics
from energygap.methods.registry import load_registry,REQUIRED
from energygap.data import DatasetConfig

BASELINES=('median','locf','linear','seasonal','kalman_filter','kalman_smoother')
CONDITIONS=('C1','C2','C3','C4','C5','C6')
FOLDS=(1,2,3)
TRAIN_SEEDS=(11,23,47)
VAL_SEEDS=tuple(range(1001,1006))
TEST_SEEDS=tuple(range(2001,2011))
PHYSICAL_CONDITIONS=('C3','C4','C5')


def validate_dag(jobs):
    """Validate identities/dependencies with Kahn's algorithm, without heavy libraries."""
    ids=[j['job_id'] for j in jobs]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate logical job IDs')
    index={j['job_id']:j for j in jobs};indegree={key:0 for key in index};children=defaultdict(list)
    for job in jobs:
        if len(job['dependencies'])!=len(set(job['dependencies'])):raise ValueError('duplicate dependency: '+job['job_id'])
        for parent in job['dependencies']:
            if parent not in index:raise ValueError('missing dependency '+parent+' for '+job['job_id'])
            indegree[job['job_id']]+=1;children[parent].append(job['job_id'])
    queue=deque(key for key,n in indegree.items() if n==0);ordered=[]
    while queue:
        key=queue.popleft();ordered.append(key)
        for child in children[key]:
            indegree[child]-=1
            if indegree[child]==0:queue.append(child)
    if len(ordered)!=len(jobs):raise ValueError('cycle in scientific DAG')
    return ordered


def _calibration(root):
    """Reuse measured pilot latencies; never extrapolate tiny neural fits to L512."""
    result={};sources=[]
    pilot=root/'runs'/'pilot-083764a78be2ffec'/'reports'/'episodes.csv'
    if pilot.is_file():
        observations=defaultdict(list)
        with pilot.open(newline='') as stream:
            for row in csv.DictReader(stream):
                try:value=float(row['latency_seconds']);context=int(row['context'])
                except (ValueError,KeyError):continue
                if value>=0 and context==512 and row.get('method') in BASELINES:
                    observations[row['method']].append(value)
        for name,values in observations.items():
            result[name]={'query_p50_seconds':statistics.median(values),'query_mean_seconds':statistics.mean(values),
                          'observations':len(values),'context':512,'source':str(pilot.relative_to(root)),
                          'scope':'restricted real pilot; no extrapolation to training, other hardware or query cardinality'}
        sources.append(str(pilot.relative_to(root)))
    audit=root/'artifacts/reports/foundation_audit.json'
    if audit.is_file():
        try:
            moment=json.loads(audit.read_text()).get('MOMENT-1-base',{})
            if moment.get('context')==512 and moment.get('device')=='cpu':
                result['MOMENT-1-base']={k:moment[k] for k in ('query_p50_seconds','query_p95_seconds','load_seconds_cached','precision','device','threads') if k in moment}
                result['MOMENT-1-base'].update(source=str(audit.relative_to(root)),context=512,
                        scope='synthetic CPU native L512 audit; provisional timing on same signature only')
                sources.append(str(audit.relative_to(root)))
        except (ValueError,KeyError,TypeError):pass
    return result,sources


def plan(root=Path('.'),profile='conference',budget_total_hours=1.):
    root=Path(root)
    if profile not in ('smoke','pilot','conference','extended'):raise ValueError('unknown profile')
    if budget_total_hours<=0:raise ValueError('budget must be positive')
    registry=load_registry(root);neural=[m for m in registry['methods'] if m['name'] in REQUIRED]
    catalogue=[*BASELINES,*[m['name'] for m in neural]]
    full=profile in ('conference','extended')
    sites=['SYNTH'] if profile=='smoke' else ['H1','H2','I1']
    conditions=CONDITIONS if full else PHYSICAL_CONDITIONS
    validation_seeds=VAL_SEEDS if full else (1001,)
    test_seeds=TEST_SEEDS if full else (2001,)
    training_seeds=TRAIN_SEEDS if full else (11,)
    planned_methods=neural if full else []
    methods=[{'name':name,'local_training':'baseline_fit','conference_eligible':True} for name in BASELINES]+planned_methods
    method_map={m['name']:m for m in methods}
    local=[m for m in planned_methods if m['local_training']=='required']
    frozen=[m for m in planned_methods if m['local_training']=='not_applicable']
    unresolved_training=[m for m in planned_methods if m['local_training'] not in ('required','not_applicable')]
    calibration,calibration_sources=_calibration(root)
    unresolved={};physical_unresolved={};contracts={}
    for site in sites:
        if site=='SYNTH':unresolved[site]=[];physical_unresolved[site]=[];continue
        try:
            cfg=DatasetConfig.load(root/'configs/datasets'/f'{site}.yaml')
            unresolved[site]=cfg.unresolved(purpose='statistical')
            physical_unresolved[site]=cfg.unresolved(purpose='physical')
            contracts[site]=hashlib.sha256((root/'configs/datasets'/f'{site}.yaml').read_bytes()).hexdigest()
        except (FileNotFoundError,ValueError,TypeError) as error:
            unresolved[site]=['dataset_config_absent_or_invalid: '+str(error)]
            physical_unresolved[site]=list(unresolved[site])
    jobs=[];blocked=[];by_id={}
    def add(job_id,phase,dependencies=(),*,reason=None,status=None,units=1,unit='jobs',resource='cpu',**metadata):
        if job_id in by_id:raise ValueError('planner attempted duplicate ID: '+job_id)
        row=dict(job_id=job_id,phase=phase,dependencies=list(dict.fromkeys(dependencies)),
                 status=status or ('blocked' if reason else 'pending'),reason=reason,
                 resource=resource,units=units,unit=unit,estimated_seconds=None,**metadata)
        if phase in ('inference_manifest','correction','evaluate','forecast_evaluate','forecast_reference'):
            row['n_queries']=None
        if phase=='model_load' and metadata.get('method')=='MOMENT-1-base':
            row['estimated_seconds']=calibration.get('MOMENT-1-base',{}).get('load_seconds_cached')
            row['cost_calibration_scope']='cached checkpoint load; no cold-download estimate'
        jobs.append(row);by_id[job_id]=row
        return job_id
    def dataset_reason(site,physical=False):
        fields=physical_unresolved[site] if physical else unresolved[site]
        return ('unconfirmed '+('physical' if physical else 'statistical')+' data contract: '+', '.join(fields)) if fields else None
    def method_reason(site,name):
        method=method_map[name]
        reasons=[]
        if dataset_reason(site):reasons.append(dataset_reason(site))
        if method['local_training'] not in ('required','not_applicable','baseline_fit'):
            reasons.append('implementation unavailable; training/pretraining semantics unknown_not_reproduced; no verified frozen checkpoint')
        if not method.get('conference_eligible',False):
            reasons.append(method.get('conference_blocker') or method.get('availability_reason') or 'conference integration not validated')
        return '; '.join(reasons) or None
    def variants(name):return training_seeds if method_map[name]['local_training']=='required' else (None,)
    def seedlabel(seed):return 'not_applicable' if seed is None else str(seed)
    def model_id(site,fold,name,seed):
        if name in BASELINES:return f'baseline_fit:{site}:{fold}:{name}'
        if method_map[name]['local_training']=='required':return f'train:{site}:{name}:{fold}:{seed}'
        if method_map[name]['local_training']=='not_applicable':return f'frozen_prepare:{site}:{fold}:{name}'
        return f'implementation_availability:{name}'
    def cell(site,fold,partition,seed,condition):return f'{site}:{fold}:{partition}:{seed}:{condition}'
    def inference_id(site,fold,partition,seed,condition,name,train_seed):
        label=seedlabel(train_seed) if method_map[name]['local_training'] in ('required','not_applicable','baseline_fit') else 'unknown_not_reproduced'
        return f'infer:{cell(site,fold,partition,seed,condition)}:{name}:{label}'

    catalogue_audit=add('catalogue_audit','catalogue_audit',methods=catalogue,required_candidates=sorted(REQUIRED),source='model_registry.yaml')
    for method in frozen:
        name=method['name'];reason=method.get('conference_blocker') if not method.get('conference_eligible') else None
        add(f'model_load:{name}','model_load',[catalogue_audit],reason=reason,method=name,local_training='not_applicable',train_seed=None,
            external_pretraining=method.get('external_pretraining','unknown'),checkpoint=method.get('checkpoint'),
            timing=calibration.get(name))

    for method in unresolved_training:
        add(f'implementation_availability:{method["name"]}','implementation_availability',[catalogue_audit],
            reason='implementation and training semantics unknown_not_reproduced; no verified architecture, weights or frozen role',
            method=method['name'],local_training=method['local_training'],training_semantics='unknown_not_reproduced',
            external_pretraining='unknown_not_applicable_to_verified_method',checkpoint=None,
            planned_training_jobs=None,planned_training_seeds=None,availability_reason=method.get('availability_reason'))

    cells={};all_eval=[];all_forecast=[];validation_selection={};physical_selection={};ridge_ids={}
    for site in sites:
        dr=dataset_reason(site);pr=dataset_reason(site,True)
        discover=add(f'discover:{site}','discovery',site=site,reason=dr,raw_files_never_modified=True)
        prepare=add(f'prepare:{site}','prepare',[discover],site=site,reason=dr,observations_original_preserved=True)
        physics=add(f'physical_contract:{site}','physical_contract',[discover],site=site,reason=pr,information_tier='E')
        if dr:blocked.append(dict(site=site,method=None,scope='P',status='blocked',reason=dr))
        if pr:blocked.append(dict(site=site,method=None,scope='E',status='blocked',reason=pr))
        for method in planned_methods:
            reason=method_reason(site,method['name'])
            if reason:blocked.append(dict(site=site,method=method['name'],scope='R/P',status='blocked',reason=reason))
        for fold in FOLDS:
            split=add(f'split:{site}:{fold}','split',[prepare],reason=dr,site=site,fold=fold,rule='elapsed_time_fractions',test_disjoint=True)
            scaler=add(f'scaler:{site}:{fold}','train_statistics',[split],reason=dr,site=site,fold=fold,fit_partition='train')
            train_windows=add(f'training_windows:{site}:{fold}','training_windows',[split,scaler],reason=dr,site=site,fold=fold,fit_partition='train',dynamic_masks=True)
            gapmodel=add(f'gap_model:{site}:{fold}','gap_model_fit',[split],reason=dr,site=site,fold=fold,fit_partition='train',used_by='C6 replay only')
            counter_calibration=add(f'counter_calibration:{site}:{fold}','energy_calibration',[split,physics],reason=pr,site=site,fold=fold,fit_partition='train',directions=['import','export'])
            future_targets=add(f'forecast_targets:{site}:{fold}','forecast_targets',[split],reason=dr,site=site,fold=fold,horizons_seconds=[1800,3600,10800,21600],matching_tolerance_seconds=90,targets_original_only=True)
            ridge_hpo=f'ridge_hpo:{site}'
            if fold==1:add(ridge_hpo,'forecast_hpo',[train_windows,split],reason=dr,site=site,fold=1,fit_partition='train',selection_partition='validation',reuse_later_folds=True)
            ridge=add(f'ridge_fit:{site}:{fold}','forecast_fit',[train_windows,ridge_hpo],reason=dr,site=site,fold=fold,weights_shared_across_imputations=True,history=288)
            ridge_ids[(site,fold)]=ridge
            for name in BASELINES:
                add(model_id(site,fold,name,None),'baseline_fit',[train_windows,scaler],reason=dr,site=site,fold=fold,method=name,train_seed=None,fit_partition='train')
            for method in frozen:
                name=method['name']
                add(model_id(site,fold,name,None),'frozen_prepare',[f'model_load:{name}',scaler],reason=method_reason(site,name),site=site,fold=fold,method=name,local_training='not_applicable',train_seed=None)
            for partition,seeds in (('validation',validation_seeds),('test',test_seeds)):
                for mask_seed in seeds:
                    for condition in conditions:
                        key=cell(site,fold,partition,mask_seed,condition)
                        mask=add(f'mask:{key}','mask_manifest',[split]+([gapmodel] if condition=='C6' else []),reason=dr,site=site,fold=fold,partition=partition,mask_seed=mask_seed,condition=condition,shared_by_all_methods=True,immutable=True)
                        query=add(f'queries:{key}:P','query_build',[mask,prepare],reason=dr,site=site,fold=fold,partition=partition,mask_seed=mask_seed,condition=condition,information_tier='P',context=512,availability_cut='reconnection',truth_delivered=False)
                        truth=add(f'truth:{key}','evaluation_truth',[mask,prepare],reason=dr,site=site,fold=fold,partition=partition,mask_seed=mask_seed,condition=condition,path='data/evaluation_truth',not_an_imputer_dependency=True)
                        physical=None
                        if condition in PHYSICAL_CONDITIONS:
                            physical=add(f'energy_inputs:{key}','energy_eligibility',[mask,counter_calibration,physics],reason=pr,site=site,fold=fold,partition=partition,mask_seed=mask_seed,condition=condition,information_tier='E',endpoint_counters_only=True,eligibility_defined_ex_ante=True)
                        cells[(site,fold,partition,mask_seed,condition)]={'key':key,'query':query,'truth':truth,'physical':physical,'inferences':[],'available_inferences':[]}
        # HPO uses first train/validation only; architecture frozen across later folds.
        for method in local:
            name=method['name'];reason=method_reason(site,name);trials=[]
            val_masks=[cells[(site,1,'validation',seed,condition)]['query'] for seed in validation_seeds for condition in PHYSICAL_CONDITIONS]
            val_truth=[cells[(site,1,'validation',seed,condition)]['truth'] for seed in validation_seeds for condition in PHYSICAL_CONDITIONS]
            for trial in range(6):
                trials.append(add(f'hpo:{site}:{name}:{trial}','hpo',[f'training_windows:{site}:1',*val_masks,*val_truth],reason=reason,site=site,method=name,fold=1,trial_id=trial,trial_seed=11,configuration_list_fixed=True,max_epochs=100,patience=10,selection='gap_macro_nMAE_R_P_C3_C5_equal_weights',validation_mask_seeds=list(validation_seeds),training_mask_distribution='balanced_points_blocks',fit_partition='train',selection_partition='validation'))
            selection=add(f'hpo_selection:{site}:{name}','hpo_selection',trials,reason=reason,site=site,method=name,source_fold=1,frozen_later_folds=True)
            for fold in FOLDS:
                for seed in training_seeds:
                    add(model_id(site,fold,name,seed),'train',[f'training_windows:{site}:{fold}',selection],reason=reason,site=site,fold=fold,method=name,train_seed=seed,fit_partition='train',max_epochs=100,patience=10,training_mask_distribution='balanced_points_blocks',context=512)

    # All inference variants use exactly the same query and immutable full-flow mask.
    for (site,fold,partition,mask_seed,condition),item in cells.items():
        for method in methods:
            name=method['name'];reason=method_reason(site,name)
            for train_seed in variants(name):
                deps=[item['query'],model_id(site,fold,name,train_seed)]
                if partition=='test':deps.append('protocol_lock:P')
                identifier=add(inference_id(site,fold,partition,mask_seed,condition,name,train_seed),'inference_manifest',deps,reason=reason,
                           units=None,unit='queries',site=site,fold=fold,partition=partition,mask_seed=mask_seed,condition=condition,method=name,train_seed=train_seed,information_tier='P',task='R',context=512,
                           samples=100 if name=='CSDI' else None,local_training=method['local_training'],
                           train_seed_semantics='independent_local_training' if method['local_training']=='required' else ('not_applicable' if method['local_training'] in ('not_applicable','baseline_fit') else 'unknown_not_reproduced'),
                           query_timing_calibration=calibration.get(name),n_queries_reason='eligible immutable manifests have not been exhaustively generated; raw row count is not query count')
                item['inferences'].append(identifier)
                if not reason:item['available_inferences'].append(identifier)
        # Evaluation is isolated from inference; only evaluator depends on truth.
        metric=add(f'evaluate:{item["key"]}:P','evaluate',[*item['inferences'],item['truth']],reason=dataset_reason(site),units=None,unit='query_method_pairs',site=site,fold=fold,partition=partition,condition=condition,information_tier='P',preserve_unavailable_status=True)
        forecast=add(f'forecast_evaluate:{item["key"]}:P','forecast_evaluate',[*item['available_inferences'],ridge_ids[(site,fold)],f'forecast_targets:{site}:{fold}'],reason=dataset_reason(site),units=None,unit='query_method_pairs',site=site,fold=fold,partition=partition,condition=condition,information_tier='P',same_weights_all_methods=True,primary_horizon_seconds=3600)
        all_eval.append(metric);all_forecast.append(forecast)
        reference=add(f'forecast_reference:{item["key"]}','forecast_reference',[item['query'],ridge_ids[(site,fold)],f'forecast_targets:{site}:{fold}'],reason=dataset_reason(site),units=None,unit='queries',site=site,fold=fold,partition=partition,condition=condition,reference='history_before_artificial_mask_with_causal_natural_fill')
        all_forecast.append(reference)

    for site in sites:
        for fold in FOLDS:
            available_validation=[];physical_validation=[]
            for seed in validation_seeds:
                for condition in PHYSICAL_CONDITIONS:
                    item=cells[(site,fold,'validation',seed,condition)]
                    available_validation.extend(item['available_inferences'])
                    available_validation.append(item['truth'])
                    physical_validation.append(item['physical'])
            selection=add(f'validation_selection:{site}:{fold}','validation_selection',[catalogue_audit,*available_validation],reason=dataset_reason(site),site=site,fold=fold,conditions=list(PHYSICAL_CONDITIONS),mask_seeds=list(validation_seeds),duration_weights='equal_1/3',train_seed_loss_aggregation='mean_per_episode',candidate_eligibility='documented_available_validation_candidates',ties=['latency','stable_name'],test_access=False)
            validation_selection[(site,fold)]=selection
            physical_selection[(site,fold)]=add(f'physical_policy_selection:{site}:{fold}','physical_policy_selection',[selection,*physical_validation],reason=dataset_reason(site,True),site=site,fold=fold,selection_support='same_counter_eligible_validation_episodes',test_access=False)
    # Freeze the declared available candidate list before any test inference.
    add('protocol_lock:P','protocol_lock',[catalogue_audit,*validation_selection.values(),*[f'ridge_hpo:{site}' for site in sites]],scope='P',candidate_exclusions_recorded=True,hash_before_first_test=True,conference_incomplete_if_any_required_candidate_blocked=True)
    add('protocol_lock:E','protocol_lock',['protocol_lock:P',*physical_selection.values()],reason='; '.join(filter(None,[dataset_reason(s,True) for s in sites])) or None,scope='E',hash_before_first_physical_test=True)

    for (site,fold,partition,mask_seed,condition),item in cells.items():
        if condition not in PHYSICAL_CONDITIONS:continue
        deps=[*item['available_inferences'],item['physical']]
        if partition=='test':deps.append('protocol_lock:E')
        correction=add(f'correction:{item["key"]}','correction',deps,reason=dataset_reason(site,True),units=None,unit='query_method_pairs',site=site,fold=fold,partition=partition,condition=condition,information_tier='E',requested_base_variants=len(item['inferences']),available_base_variants=len(item['available_inferences']),same_wrapper_all_compatible_methods=True,scope='target_gap_only',directions=['import','export'],tau_fit='train_only')
        metric=add(f'evaluate:{item["key"]}:E','evaluate',[correction,item['truth']],reason=dataset_reason(site,True),units=None,unit='query_method_pairs',site=site,fold=fold,partition=partition,condition=condition,information_tier='E')
        forecast=add(f'forecast_evaluate:{item["key"]}:E','forecast_evaluate',[correction,ridge_ids[(site,fold)],f'forecast_targets:{site}:{fold}'],reason=dataset_reason(site,True),units=None,unit='query_method_pairs',site=site,fold=fold,partition=partition,condition=condition,information_tier='E',same_weights_all_methods=True)
        all_eval.append(metric);all_forecast.append(forecast)
    # CPI is a separate compatible-energy reference, never an alias for linear correction.
    for site in sites:
        for fold in FOLDS:
            add(f'CPI_energy_reference:{site}:{fold}','energy_reference',[f'counter_calibration:{site}:{fold}',f'training_windows:{site}:{fold}'],reason='official compatible-energy CPI adapter not integrated; power semantics require validation',site=site,fold=fold,method='CPI',information_tier='E',implementation_status='not_reproduced',fallback_method=None)
    aggregate=add('aggregate_metrics','aggregation',[*all_eval,*all_forecast],units=None,unit='episode_method_records',fixed_fold_duration_weights=True,no_support='NA_with_reason',unavailable_rows_preserved=True)
    statistics_jobs=[]
    # Fixed family always has twelve hypotheses, including a reduced/synthetic plan.
    for site in ('H1','H2','I1'):
        for contrast in ('contrast_energy_gain','contrast_complexity_gain'):
            for endpoint in ('reconstruction_nmae','forecast_1h_nmae'):
                reason=dataset_reason(site,True) if site in sites else 'site absent from this reduced profile'
                stats=add(f'statistics:{site}:{contrast}:{endpoint}','bootstrap',[aggregate],reason=reason,units=10000,unit='attempted_replicates',site=site,contrast=contrast,endpoint=endpoint,bootstrap_seed=20261031,block_days=7,invalid_draws='persist_no_redraw',family_size=12,support_gate='20_nonoverlapping_blocks;4_per_fold;20_days_each_duration')
                statistics_jobs.append(stats)
    holm=add('holm:fixed_family_12','multiplicity',statistics_jobs,units=12,unit='hypotheses',family_size=12,unsupported_p='missing_in_results;1_only_for_family_adjustment')
    execution=add('execution_audit','execution_audit',[aggregate,holm],records=['sessions','checkpoints','recomputation','budget','ETA_predictions','actual_timestamps','failures'])
    report=add('report','report',[aggregate,holm,execution,catalogue_audit],units=1,unit='reports',exports=['CSV','Parquet','PDF','SVG','PNG'],scientific_complete=False)
    ordering=validate_dag(jobs)
    counts=dict(sorted(Counter(j['phase'] for j in jobs).items()))
    now=datetime.now(timezone.utc);unknown=sum(j['estimated_seconds'] is None for j in jobs)
    known=sum(j['estimated_seconds'] or 0 for j in jobs)
    reason='Full costs remain unknown until eligible query cardinalities and HPO/training/correction/forecast/statistics timings are calibrated. Tiny neural fits are not extrapolated to L512.'
    coverage={name:{'requested':True,'local_training':method_map[name]['local_training'],
                    'train_seeds':list(training_seeds) if method_map[name]['local_training']=='required' else [],
                    'train_seed_semantics':'independent_local_training' if method_map[name]['local_training']=='required' else ('not_applicable' if method_map[name]['local_training'] in ('not_applicable','baseline_fit') else 'unknown_not_reproduced'),
                    'training_job_count':len(sites)*len(FOLDS)*len(training_seeds) if method_map[name]['local_training']=='required' else (0 if method_map[name]['local_training'] in ('not_applicable','baseline_fit') else None),
                    'verified_frozen_role':method_map[name]['local_training']=='not_applicable',
                    'planned_inference_jobs':sum(j['phase']=='inference_manifest' and j.get('method')==name for j in jobs),
                    'conference_eligible':method_map[name].get('conference_eligible',False)} for name in catalogue if name in method_map}
    return dict(schema_version=3,profile=profile,complete=False,conference_complete=False,
                execution_supported=False,execution_scope='planning_only_full_DAG; restricted runner is separate',
                budget_total_hours=budget_total_hours,budget_seconds=budget_total_hours*3600,
                can_start_full_matrix=False,budget_decision='blocked_pending_cost_calibration',
                counts=counts,total_jobs=len(jobs),jobs=jobs,blocked=blocked,
                dataset_unresolved=unresolved,physical_dataset_unresolved=physical_unresolved,dataset_config_hashes=contracts,
                start_meaning='when the run command is executed; no resource reservation',estimated_start=now.isoformat(),
                estimated_finish=None,estimated_seconds=None,known_cost_lower_bound_seconds=known,unknown_cost_jobs=unknown,
                eta_reason=reason,cost_calibration=calibration,calibration_sources=calibration_sources,
                budget_policy='Do not start uncalibrated long training; do not shrink seeds/context/samples on restart or OOM.',
                catalogue=catalogue,candidate_catalogue=[m['name'] for m in neural],baseline_catalogue=list(BASELINES),method_coverage=coverage,
                training_counts={'M_required_local':len(local),'final_trainings':len(local)*len(sites)*len(FOLDS)*len(training_seeds),
                                 'HPO_trials':6*len(local)*len(sites),'baseline_fits':len(BASELINES)*len(sites)*len(FOLDS),
                                 'frozen_checkpoint_loads':len(frozen),'frozen_training_replicates':0,
                                 'unresolved_training_methods':[m['name'] for m in unresolved_training],
                                 'unresolved_additional_trainings':None if unresolved_training else 0,
                                 'count_scope':'verified local architectures and declared frozen checkpoints only; unresolved methods do not imply zero training'},
                masks={'validation_seeds':list(validation_seeds),'test_seeds':list(test_seeds),'conditions':list(conditions),
                       'shared_manifests':counts.get('mask_manifest',0),'applied_before_windowing':True},
                dag_validation={'valid':True,'unique_jobs':len(ordering),'missing_dependencies':0,'cycles':0},
                reduced_plan={'profile':'smoke','scope':'synthetic implementation validation','sites':['SYNTH'],
                              'conditions':list(PHYSICAL_CONDITIONS),'mask_seeds':[2001],'max_episodes_per_cell':2,
                              'not_conference_substitute':True},
                remaining_requirements=['full-matrix eligible episode enumeration','method-specific cost calibration at protocol shapes',
                    'blocked official model/checkpoint integrations','physical timestamp/power/counter semantics',
                    'conference executor for this DAG; current restricted runner does not satisfy it'])
