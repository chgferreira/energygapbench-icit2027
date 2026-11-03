"""Materialize catalogue from recorded upstream inspection, never invented commits."""
from pathlib import Path
import json
import yaml

ROOT=Path(__file__).resolve().parents[1]


def build():
    audit=json.loads((ROOT/'artifacts/reports/upstream_audit.json').read_text())
    tiny_path=ROOT/'artifacts/reports/learned_adapter_audit.json'
    tiny={r['method']:r for r in json.loads(tiny_path.read_text())} if tiny_path.exists() else {}
    foundation_path=ROOT/'artifacts/reports/foundation_audit.json'
    foundation=json.loads(foundation_path.read_text()) if foundation_path.exists() else {}
    t1_path=ROOT/'artifacts/reports/t1_integration_audit.json'
    t1=json.loads(t1_path.read_text()) if t1_path.exists() else {}
    tsicl_path=ROOT/'artifacts/reports/tsicl_audit.json'
    tsicl=json.loads(tsicl_path.read_text()) if tsicl_path.exists() else {}
    records=[]
    definitions=[
        ('BRITS','consolidated',2018,'https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html','caow13/BRITS','pypots'),
        ('SAITS','consolidated',2023,'https://doi.org/10.1016/j.eswa.2023.119619','WenjieDu/SAITS','pypots'),
        ('CSDI','consolidated_probabilistic',2021,'https://proceedings.neurips.cc/paper_files/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html','ermongroup/CSDI','pypots'),
        ('ImputeFormer','multivariate_dependencies',2024,'https://doi.org/10.1145/3637528.3671751','tongnie/ImputeFormer','pypots'),
        ('T1','multivariate_dependencies',2026,'https://proceedings.iclr.cc/paper_files/paper/2026/hash/41ff2b921a875902fad0c1840593c19a-Abstract-Conference.html','Oppenheimerdinger/T1','official_source_not_integrated'),
        ('MOMENT-1-base','general_foundation',2024,'https://proceedings.mlr.press/v235/goswami24a.html','moment-timeseries-foundation-model/moment','checkpoint_not_available'),
        ('FIM','time_indexed_foundation_priority',2025,'https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d03791377a31cb3e7357014ba58eb80-Abstract-Conference.html','FIM4Science/OpenFIM','checkpoint_variant_not_validated'),
        ('TabPFN-TS','time_indexed_foundation_priority',2026,'https://arxiv.org/abs/2511.05980','taharnbl/tsfm_imputation','R24_source_audited_not_integrated'),
        ('MoTM','recent_foundation',2026,'https://arxiv.org/abs/2511.05980','taharnbl/tsfm_imputation','R24_source_audited_not_integrated'),
        ('TS-ICL','recent_foundation',2026,'https://arxiv.org/abs/2606.05878','EDF-Lab/ts-icl','native_impute_wrapper_not_validated'),
        ('CBiT','compressed_history',2026,'https://proceedings.mlr.press/v306/fan26h.html',None,'not_reproduced')]
    reasons={
        'T1':'Official repo verified, no PyPOTS T1 class in installed 1.5; native integration and D=1 audit not completed.',
        'MOMENT-1-base':'No local AutonLab/MOMENT-1-base checkpoint; masked normalization/partial-patch audit not completed. No implicit weight download.',
        'FIM':'OpenFIM source inspected: temporal and pointwise variants separate; no locally verified appropriate gap checkpoint; FIM-l is not substituted.',
        'TabPFN-TS':'R24 FourierTokenizer/TabPFN v2 wrapper inspected. Repository has no asserted open license; exact checkpoint and redistribution terms unresolved. Generic TabPFN is not substituted.',
        'MoTM':'R24 expects model directories with ckpt/weights.pt and .hydra/config.yaml for INR mixture; those artifacts absent locally, no verified weight license.',
        'TS-ICL':'Native API inspected and wrapper written; no local checkpoint or validation. Code/weights noncommercial and nonproduction, separate environment required.',
        'CBiT':'Paper landing page confirms ICML2026 and compressed-history method; no usable author-linked implementation verified, no own_reimplementation claimed.'}
    for name,role,year,paper,repo,implementation in definitions:
        upstream=audit.get(repo,{})
        verified=tiny.get(name,{})
        integrated=verified.get('implementation_status')=='integrated'
        r=dict(name=name,scientific_role=role,paper=paper,doi_url=paper,year=year,publication_status='preprint' if name=='TS-ICL' else 'see_primary_source',repo='https://github.com/'+repo if repo else None,commit=upstream.get('commit'),code_license=(upstream.get('license') or {}).get('spdx_id'),checkpoint_license=None,implementation=implementation,implementation_status='integrated' if integrated else 'unavailable',validation_status=verified.get('validation_status','not_validated'),availability_reason=None if integrated else reasons.get(name,verified.get('reason','optional dependency unavailable')),capabilities=dict(tasks=['R','O'],univariate_D1=True if integrated else 'not_audited',multivariate=True,counters_native=False,irregular_native=False,joint_samples=name=='CSDI',marginal_quantiles=name=='TS-ICL'),context=dict(core=512,validated_tiny=verified.get('context'),maximum=None),normalization='training_channel_mean_std' if integrated else 'requires_upstream_audit',local_training='required' if implementation=='pypots' else 'not_applicable',local_cost_seconds=verified.get('fit_seconds'),external_pretraining='not_applicable' if implementation=='pypots' else 'unknown',pretraining_overlap='unknown',supports_resume=False,checkpoint_granularity='completed_fit/query_only' if integrated else 'not_validated',environment='environments/neural.lock.txt' if implementation=='pypots' else 'separate_optional_environment',verified_at='2026-10-31',scientific_benchmark_run=False,conference_eligible=False,conference_blocker='exact native training recovery and protocol training-mask integration not validated' if integrated else reasons.get(name))
        if name=='TS-ICL':r['code_license']='TS-ICL Non-Commercial License v1.0';r['checkpoint_license']='same_noncommercial_terms'
        if integrated:r['wrapper_source']='src/energygap/methods/neural.py';r['wrapper_version']='0.1.0';r['upstream_wrapper_version']='pypots==1.5';r['wrapper_repo_commit']=audit['WenjieDu/PyPOTS']['commit'];r['wrapper_code_license']='BSD-3-Clause'
        if name=='MOMENT-1-base' and foundation.get(name,{}).get('validation_status')=='upstream_parity_passed_synthetic':
            a=foundation[name]
            r.update(implementation='official_momentfm_native_reconstruction',implementation_status='integrated',validation_status=a['validation_status'],availability_reason=None,checkpoint_license=a['checkpoint_license'],checkpoint=a['checkpoint'],checkpoint_revision=a['checkpoint_revision'],checkpoint_sha256=a['checkpoint_sha256'],normalization=a['normalization'],partial_patch_policy=a['partial_patch_policy'],local_training='not_applicable',supports_resume={'training':'not_applicable','inference':'query_shards'},checkpoint_granularity='query',environment='environments/moment.lock.txt',local_cost_seconds=a['load_seconds_cached'],query_p50_seconds=a['query_p50_seconds'],peak_process_rss_bytes=a['peak_process_rss_bytes'],conference_eligible=True,conference_blocker=None)
            r['capabilities']['univariate_D1']=True;r['context']['validated_tiny']=512
        if name=='FIM' and 'FIM' in foundation:
            r.update(implementation='native_temporal_block_adapter_contract_only',availability_reason='Official temporal/pointwise checkpoint cards have unresolved weight license ([More Information Needed]); native-weight inference not run.',validation_status='contract_only_not_upstream_validated',checkpoint_license='unresolved_model_card',checkpoint=foundation['FIM'].get('checkpoint'))
            r['conference_blocker']=r['availability_reason']
        if name=='T1' and t1:
            r.update(implementation='pinned_official_model_protocol_training_loop',implementation_status='integrated',validation_status='D1_L512_native_parity_tiny_overfit_and_fitted_reload',availability_reason=None,local_training='required',normalization='training_channel_scaler_plus_upstream_mask_aware_window',supports_resume=False,checkpoint_granularity='completed_fit_and_query',environment='environments/neural.lock.txt',conference_eligible=False,conference_blocker='full training schedule/optimizer-step recovery/protocol HPO not validated')
            r['capabilities']['univariate_D1']=True;r['context']['validated_tiny']=512
        if name=='SAITS':
            r['additional_variant']=dict(name='saits_protocol_training_loop',source='src/energygap/methods/training.py',supports_resume='deterministic_cpu_optimizer_step_validated',dynamic_mask='balanced_points_blocks',scheduler='constant_lr',validation_selection='not_yet_integrated_in_conference_runner',audit='artifacts/reports/saits_recovery_comparison.json')
        if name=='TS-ICL' and tsicl:
            r.update(implementation='native_impute_isolated_noncommercial_worker',implementation_status='integrated',validation_status='native_parity_availability_and_hidden_poisoning_validated',availability_reason=None,wrapper_source='scripts/tsicl_worker.py',environment='environments/tsicl.lock.txt',local_training='not_applicable',checkpoint='taharnbl/TS-ICL:tsicl-v1.ckpt',checkpoint_revision='19c94031439fb31f36ce395088ee50a6762d3774',checkpoint_sha256='a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431',normalization='native masked context observed only',primary_point='native median 0.5',supports_resume={'training':'not_applicable','inference':'validated_query_shards'},checkpoint_granularity='query',conference_eligible=False,conference_blocker='separate noncommercial worker not integrated into full conference orchestration; contextual cost and all matrix requirements remain pending')
            r['capabilities']['univariate_D1']=True;r['capabilities']['joint_samples']=False;r['capabilities']['marginal_quantiles']=True;r['context']['validated_tiny']=512
            r['commit']=tsicl['source_commit'];r['inspected_head_commit']=upstream.get('commit');r['package_version']=tsicl['package_version']
        if name=='MOMENT-1-base' and (ROOT/'artifacts/reports/moment_real_isolated_audit.json').exists():
            r.update(validation_status='native_parity_synthetic_and_18_real_queries_official_isolated_environment',wrapper_source='src/energygap/methods/moment_subprocess.py',dependency_override=False,real_pilot_audit='artifacts/reports/moment_real_isolated_audit.json')
        if integrated and (ROOT/'artifacts/reports/models_pilot.json').exists():
            r['real_integration_pilot']='L512_3sites_2epochs_16train_windows_no_convergence_claim'
        if name=='CBiT':
            r.update(local_training='unknown_not_reproduced',external_pretraining='unknown',checkpoint_granularity='not_reproduced',normalization='not_audited')
        records.append(r)
        (ROOT/'configs/methods'/f'{name.lower().replace("-","_")}.yaml').write_text(yaml.safe_dump(r,sort_keys=False))
    data=dict(schema_version=1,selection='Consolidated references and recent competitors; not universal best eleven',methods=records,conference_complete=False)
    (ROOT/'model_registry.yaml').write_text(yaml.safe_dump(data,sort_keys=False,allow_unicode=True))
    return data


if __name__=='__main__':build()
