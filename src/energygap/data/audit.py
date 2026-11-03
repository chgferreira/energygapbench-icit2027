"""Read-only structural inspection is valid even while physical metadata is unknown."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import csv
import json
import math
from statistics import median
from .discovery import discover_files
from .ingest import parse_number, read_table
from .provenance import sha256_file


def _quantiles(values):
    values=sorted(v for v in values if math.isfinite(v))
    if not values: return None
    def q(f):
        p=(len(values)-1)*f;i=int(p);a=p-i
        return values[i]*(1-a)+values[min(i+1,len(values)-1)]*a
    return {str(f):q(f) for f in (0,.01,.05,.25,.5,.75,.95,.99,1)}


def _stats(rows,nominal=300):
    if not rows: return {'n_records':0,'status':'no_support'}
    times=[r['time'] for r in rows]
    unique=sorted(set(times));intervals=[(b-a).total_seconds() for a,b in zip(unique,unique[1:])]
    vals=[r['power'] for r in rows];finite=[v for v in vals if math.isfinite(v)]
    dt=(max(times)-min(times)).total_seconds()
    expected=int(dt/nominal)+1
    # Clock-face jitter is structural; no timezone is assigned to naive source timestamps.
    clock_seconds=[(t-t.replace(hour=0,minute=0,second=0,microsecond=0)).total_seconds() for t in times]
    jitter=[abs(t-math.floor(t/nominal+.5)*nominal) for t in clock_seconds]
    gaps=[{'start':a.isoformat(),'end':b.isoformat(),'boundary_minutes':(b-a).total_seconds()/60,'absent_nominal_positions':max(0,round((b-a).total_seconds()/nominal)-1)} for a,b in zip(unique,unique[1:]) if (b-a).total_seconds()>nominal*1.5]
    loss_calendar=Counter(datetime.fromisoformat(g['start']).strftime('%A %H') for g in gaps)
    counter_reports={}
    for direction in ('import','export'):
        pairs=[(a[direction],b[direction]) for a,b in zip(rows,rows[1:]) if math.isfinite(a[direction]) and math.isfinite(b[direction])]
        counter_reports[direction]={'available':sum(math.isfinite(r[direction]) for r in rows),'negative_increments':sum(b<a for a,b in pairs),'constant_increments':sum(b==a for a,b in pairs),'rollover_status':'unknown; negative increments not converted using abs','resolution_status':'unconfirmed; printed decimal precision is not device accuracy'}
    eligible={}
    by_time={r['time']:r for r in rows}
    counts=Counter(times)
    duplicates={t for t,count in counts.items() if count>1}
    duplicate_values={t:set() for t in duplicates}
    for r in rows:
        if r['time'] in duplicates: duplicate_values[r['time']].add(r['power'])
    conflicts={t for t,values in duplicate_values.items() if len(values)>1}
    ordered=[by_time[t] for t in unique]
    bad=[int(t in conflicts or not all(math.isfinite(by_time[t][key]) for key in ('power','import','export'))) for t in unique]
    bad_prefix=[0]
    for x in bad: bad_prefix.append(bad_prefix[-1]+x)
    edge_prefix=[0]
    for a,b in zip(ordered,ordered[1:]):
        edge_bad=(b['time']-a['time']).total_seconds()!=nominal or b['import']<a['import'] or b['export']<a['export']
        edge_prefix.append(edge_prefix[-1]+int(edge_bad))
    for minutes in (15,60,180):
        n=minutes*60//nominal;count=0
        for j in range(1,len(unique)-n):
            left,right=j-1,j+n
            if bad_prefix[right+1]==bad_prefix[left] and edge_prefix[right]==edge_prefix[left]:count+=1
        eligible[str(minutes)]={'candidate_starts':count,'selected_nonoverlap_episodes':None,'physical_status':'structurally eligible only; blocked pending units/timestamp/counter update confirmation'}
    channel_names=sorted({k for r in rows for k in r['aux_missing']})
    return {'n_records':len(rows),'n_unique_timestamps':len(unique),'start':min(times).isoformat(),'end':max(times).isoformat(),'elapsed_days':dt/86400,'nominal_coverage':sum(math.isfinite(by_time[t]['power']) for t in unique)/expected if expected else None,'n_expected_grid_points':expected,'power_observed':len(finite),'zeros':sum(v==0 for v in finite),'negative':sum(v<0 for v in finite),'positive':sum(v>0 for v in finite),'original_unit_quantiles':_quantiles(vals),'interval_seconds_distribution':dict(Counter(intervals).most_common(20)),'jitter_seconds':_quantiles(jitter),'outside_90s_tolerance':sum(j>90 for j in jitter),'duplicate_records':len(rows)-len(unique),'conflicting_timestamp_count':len(conflicts),'gap_count':len(gaps),'gap_absent_positions_quantiles':_quantiles([g['absent_nominal_positions'] for g in gaps]),'gap_boundary_minutes_quantiles':_quantiles([g['boundary_minutes'] for g in gaps]),'loss_calendar_start_weekday_hour':dict(loss_calendar),'sign_inversions':sum(a*b<0 for a,b in zip(finite,finite[1:])),'auxiliary_missing_counts':{name:sum(r['aux_missing'].get(name,True) for r in rows) for name in channel_names},'counters':counter_reports,'energy_integrated_vs_registered':{'status':'blocked','reason':'Target power unit/sign/instantaneous-vs-average and timestamp semantics/timezone unresolved; no calibrated energy comparison performed'},'eligible_episode_candidates':eligible,'natural_gaps':gaps}


def inspect_data(root='raw_data',output_dir='artifacts/reports',config_dir='configs/datasets',write_configs=True):
    root=Path(root);discovered=discover_files(root)
    sites={};unmapped=[]
    for entry in discovered:
        if entry['site_id'] is None:unmapped.append(entry);continue
        sites.setdefault(entry['site_id'],[]).append(entry)
    report={'status':'structural_audit_only','root':str(root),'unmapped_files':unmapped,'sites':{},'privacy':'Aggregate diagnostics only; real measurements are not redistributable without permission','assumptions':['No arrival log: optimistic availability_time=measurement_time only after timestamp contract confirmation','Five-minute nominal grid proposed from modal deltas; alignment 90s requires freeze before test','Power W/negative-import manuscript statement is contextual evidence, not confirmed instrument metadata'],'real_benchmark_status':'blocked_pending_material_metadata'}
    for site,entries in sorted(sites.items()):
        files=[];rows=[];headers=[]
        for e in entries:
            path=Path(e['path'])
            header=next(iter(read_table(path)),{})
            columns=list(header)
            headers+=columns
            tc=next((x for x in columns if x=='atualizadoEm'),None)
            pc=next((x for x in columns if x=='potenciaAtivaTotal'),None)
            ic=next((x for x in columns if x=='energia ativa consumida da rede (kwh)'),None)
            ec=next((x for x in columns if x=='energia ativa injetada na rede (kwh)'),None)
            files.append({'path':str(path),'sha256':sha256_file(path),'encoding':'utf-8-sig','delimiter':';','decimal':',','thousands':'.','evidence':'BOM/header and complete successful delimiter/locale parse','site_mapping_evidence':e['mapping_evidence']})
            if tc is None or pc is None:
                raise ValueError(f'No unambiguous mapping for {path}; supply explicit config')
            aux=[x for x in columns if x not in (tc,pc,ic,ec)]
            for lineno,row in enumerate(read_table(path),2):
                try:
                    rows.append({'time':datetime.fromisoformat(row[tc]),'power':parse_number(row[pc]),'import':parse_number(row.get(ic)), 'export':parse_number(row.get(ec)),'aux_missing':{x:not math.isfinite(parse_number(row[x])) for x in aux}})
                except (ValueError,TypeError) as exc:
                    raise ValueError(f'Parse error at {path}:{lineno}: {exc}') from exc
        rows.sort(key=lambda r:r['time'])
        summary=_stats(rows)
        first,last=rows[0]['time'],rows[-1]['time'];span=last-first
        splits={}
        for k,(a,b,c) in enumerate(((.4,.5,.65),(.55,.65,.8),(.7,.8,1.)),1):
            boundaries=(first,first+span*a,first+span*b,first+span*c)
            splits[str(k)]={}
            for name,lo,hi in zip(('train','validation','test'),boundaries,boundaries[1:]):
                subset=[r for r in rows if lo<=r['time']<hi or name=='test' and c==1. and r['time']==hi]
                splits[str(k)][name]=_stats(subset)
        report['sites'][site]={'files':files,'overall':summary,'splits':splits,'unresolved':['timestamp.timezone','timestamp.semantics','power.unit','power.semantics','power.import_sign','counter resolution/update/lag/reset semantics'],'confirmed_from_headers':{'timestamp_column':tc,'power_column':pc,'import_counter_column':ic,'export_counter_column':ec,'counter_label_unit':'kwh'}}
        if write_configs:
            import yaml
            config={'schema_version':1,'site_id':site,'files':files,'timestamp':{'column':tc,'timezone':None,'date_format':'ISO 8601 YYYY-MM-DD HH:MM:SS','date_ambiguity':'none in file lexical format','semantics':None,'availability_column':None,'availability_assumption':'optimistic: measurement_time after temporal contract confirmation'},'power':{'column':pc,'unit':None,'semantics':None,'import_sign':None,'canonical_unit':'kW','canonical_import_sign':'positive','contextual_hypothesis':{'unit':'W','import_sign':'negative','source':'user-provided prior-manuscript description; requires instrument/config confirmation'}},'counters':{'import':{'column':ic,'unit':'kWh','resolution':None,'update_period_seconds':None,'rollover':None,'evidence':'column label only; metrology/update not confirmed'},'export':{'column':ec,'unit':'kWh','resolution':None,'update_period_seconds':None,'rollover':None,'evidence':'column label only; metrology/update not confirmed'}},'auxiliaries':{x:{'column':x,'unit':x[x.rfind('(')+1:-1] if '(' in x and x.endswith(')') else None,'evidence':'column label'} for x in sorted(set(headers)) if x not in (tc,pc,ic,ec)},'alignment':{'nominal_seconds':300,'tolerance_seconds':90,'status':'proposed; modal interval audited','duplicate_policy':'identical target/arrival retain deterministic first and provenance; conflicts unavailable','conflict_policy':'flag; no value selected for evaluation'},'evidence':{'file_mapping':'actual path discovered recursively; originals untouched','encoding':'UTF-8 BOM','locale':'semicolon delimiters, comma decimals, dot thousands confirmed by full parse','power_metadata':'unconfirmed; magnitude/sign counts do not establish unit or direction','timestamp_metadata':'naive timestamps; timezone and updated-at-vs-measured-at unresolved'},'confirmed':{'timestamp.column':True,'timestamp.timezone':False,'timestamp.semantics':False,'power.column':True,'power.unit':False,'power.semantics':False,'power.import_sign':False},'unconfirmed':report['sites'][site]['unresolved'],'experiment_status':'blocked_pending_material_metadata'}
            cp=Path(config_dir)/f'{site}.yaml';cp.parent.mkdir(parents=True,exist_ok=True)
            if cp.exists():
                # Inspection must never erase manually confirmed metadata.
                existing=yaml.safe_load(cp.read_text())
                if any(existing.get('confirmed',{}).get(k) for k in ('power.unit','power.import_sign','timestamp.timezone','power.semantics','timestamp.semantics')):
                    config=existing
            cp.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf8')
            from .schema import DatasetConfig
            report['sites'][site]['unresolved']=DatasetConfig(config).unresolved('physical')+['counter resolution/update/lag/reset semantics']
            report['sites'][site]['confirmed_metadata']={'power_unit':config['power'].get('unit'),'import_sign':config['power'].get('import_sign'),'timezone':config['timestamp'].get('timezone'),'evidence':config.get('evidence',{})}
            if not DatasetConfig(config).unresolved('statistical'):
                report['sites'][site]['statistical_P_status']='allowed under optimistic reported-timestamp availability assumption'
                for stats in [summary]+[v for fold in splits.values() for v in fold.values()]:
                    stats['energy_integrated_vs_registered']['reason']='Power instantaneous/interval-mean and timestamp measurement/update/start/end semantics remain unresolved; any operator sensitivity is exploratory, not independent calibration.'
    if report['sites'] and all(site.get('statistical_P_status') for site in report['sites'].values()):
        report['real_benchmark_status']='statistical_P_pilot_allowed; physical_E_confirmatory_blocked'
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'data_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    text=['# Read-only structural data audit','', 'Physical confirmatory experiments require all material timestamp and power semantics. Statistical P pilots may proceed only when unit, sign and timezone are explicitly confirmed. No unit or import sign is inferred from magnitude or majority sign.','', '| Site | Records | Start | End | Coverage | Zeros | Natural gaps |','|---|---:|---|---|---:|---:|---:|']
    for site,item in report['sites'].items():
        s=item['overall'];text.append(f"| {site} | {s['n_records']} | {s['start']} | {s['end']} | {s['nominal_coverage']:.4f} | {s['zeros']} | {s['gap_count']} |")
    text+=['','Per-fold/split interval, jitter, original-unit quantile, sign, counter reset, missing-channel, gap-calendar and structurally eligible episode diagnostics are in data_audit.json. Candidate start counts overlap and are not a count of independent episodes.','', 'Counter units are suggested by headers, but resolution, update cadence and lag remain unresolved. Negative increments remain resets/unknown rollovers, never absolute-valued. Integrated-versus-counter energy is blocked pending semantics.','', 'Zeros and finite extremes were preserved; this audit makes no causal diagnosis of shutdown/failure. The three sites are independent periods, not simultaneous channels.']
    (out/'data_audit.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    return report
