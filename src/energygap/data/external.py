"""Optional UCI ElectricityLoadDiagrams adapter, independent consumers at 15 min.

The 249 MB archive is downloaded only when download_uci is explicitly called.
No cumulative counters are constructed from power, observed or hidden.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from urllib.request import urlopen
from .ingest import parse_number
from .provenance import sha256_file, canonical_hash, write_immutable_json
from .schema import CanonicalRecord, SeriesData
from .timebase import regularize

UCI_URL='https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip'
UCI_PAGE='https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014'
UCI_DOI='10.24432/C58C86'
UCI_LICENSE='CC-BY-4.0'
PORTUGUESE_ZONE=ZoneInfo('Europe/Lisbon')


def download_uci(destination, *, expected_sha256=None, maximum_bytes=300_000_000):
    """Stream official archive, hash actual bytes, never overwrite an existing file.

    A locally computed SHA is recorded as such; it is not advertised as an
    upstream-published reference checksum. expected_sha256 can pin later runs.
    """
    destination=Path(destination);manifest_path=destination.with_suffix(destination.suffix+'.download.json')
    if destination.exists():
        actual=sha256_file(destination)
        if expected_sha256 and actual!=expected_sha256:raise ValueError('UCI existing archive checksum mismatch')
        if not manifest_path.exists():raise FileExistsError('Existing UCI archive lacks download provenance; explicit local adoption required')
        manifest=json.loads(manifest_path.read_text())
        if manifest['sha256']!=actual:raise ValueError('UCI recorded archive checksum mismatch')
        return manifest
    destination.parent.mkdir(parents=True,exist_ok=True)
    descriptor,tmp=tempfile.mkstemp(prefix=destination.name+'.part-',dir=destination.parent)
    digest=hashlib.sha256();size=0
    try:
        with os.fdopen(descriptor,'wb') as out, urlopen(UCI_URL,timeout=60) as response:
            for block in iter(lambda:response.read(1024*1024),b''):
                size+=len(block)
                if size>maximum_bytes:raise ValueError('UCI download exceeds explicit byte budget')
                digest.update(block);out.write(block)
            out.flush();os.fsync(out.fileno())
        actual=digest.hexdigest()
        if expected_sha256 and actual!=expected_sha256:raise ValueError('Downloaded UCI checksum mismatch')
        # Exclusive hard-link publication prevents racing writers from replacing data.
        os.link(tmp,destination);os.unlink(tmp)
        manifest={'schema_version':1,'url':UCI_URL,'dataset_page':UCI_PAGE,'doi':UCI_DOI,'license':UCI_LICENSE,'sha256':actual,'checksum_source':'computed from actual downloaded bytes; not an upstream reference checksum','bytes':size,'downloaded_at':datetime.now(timezone.utc).isoformat()}
        write_immutable_json(manifest_path,manifest)
        return manifest
    except BaseException:
        if os.path.exists(tmp):os.unlink(tmp)
        raise


@contextmanager
def _table(source):
    source=Path(source)
    if source.suffix.lower()=='.zip':
        with zipfile.ZipFile(source) as archive:
            candidates=[n for n in archive.namelist() if Path(n).name=='LD2011_2014.txt' and not n.startswith('__MACOSX/')]
            if len(candidates)!=1:raise ValueError('UCI archive must identify one LD2011_2014.txt')
            # Read directly; no archive paths are ever extracted into the filesystem.
            with archive.open(candidates[0]) as binary, io.TextIOWrapper(binary,encoding='utf-8-sig',newline='') as text:
                yield csv.DictReader(text,delimiter=';')
    else:
        with source.open(encoding='utf-8-sig',newline='') as text:
            yield csv.DictReader(text,delimiter=';')


def _label(value):
    result=datetime.fromisoformat(str(value).strip())
    if result.tzinfo is not None:raise ValueError('Expected original UCI naive Portuguese clock labels')
    return result


def _transition_day(label):
    # Dataset spans 2011-2014: mainland Portugal follows last Sundays March/October.
    return label.month in (3,10) and label.weekday()==6 and (label+timedelta(days=7)).month!=label.month


def _time_column(reader):
    if not reader.fieldnames or len(reader.fieldnames)<2:raise ValueError('UCI table requires clock + independent consumer columns')
    return reader.fieldnames[0]


def _local_cutoff(value):
    value=datetime.fromisoformat(value) if isinstance(value,str) else value
    return value.astimezone(PORTUGUESE_ZONE).replace(tzinfo=None) if value.tzinfo else value


def freeze_uci_selection(source,train_end,*,count=30,min_train_observations=96*30,min_train_coverage=.9,manifest_path=None,require_count=True):
    """Freeze hash-selected IDs after train-only activation/coverage eligibility.

    train_end is exclusive in the reported Portuguese clock. Dataset SHA covers
    the complete file for provenance; train_fingerprint hashes only training rows.
    """
    if count<=0 or min_train_observations<1 or not 0<=min_train_coverage<=1:raise ValueError('Invalid external eligibility configuration')
    cutoff=_local_cutoff(train_end);stats={};train_hash=hashlib.sha256();last_train=None;excluded_days=set()
    with _table(source) as reader:
        tc=_time_column(reader);consumers=[c for c in reader.fieldnames if c!=tc]
        if len(set(consumers))!=len(consumers):raise ValueError('Duplicated consumer identifiers')
        stats={c:{'first_active':None,'observed_after_activation':0,'rows_after_activation':0,'zeros_after_activation':0} for c in consumers}
        previous=None
        for row in reader:
            label=_label(row[tc])
            if previous is not None and label<=previous:raise ValueError('UCI clock labels must strictly increase')
            previous=label
            if label>=cutoff:break  # Values after train cutoff are never parsed or used.
            train_hash.update(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
            if _transition_day(label):excluded_days.add(label.date().isoformat());continue
            last_train=label
            for c in consumers:
                value=parse_number(row[c],decimal=',',thousands='')
                state=stats[c]
                # UCI documents zeros before client creation; first positive train
                # value is an explicit conservative activation proxy, not a causal diagnosis.
                if state['first_active'] is None and math.isfinite(value) and value>0:state['first_active']=label
                if state['first_active'] is not None:
                    state['rows_after_activation']+=1
                    if math.isfinite(value):state['observed_after_activation']+=1;state['zeros_after_activation']+=int(value==0)
    eligible=[]
    for c,state in stats.items():
        active=state['first_active']
        if active is None:state['eligible']=False;state['reason']='no_positive_activation_observed_in_train';state['coverage']=None
        else:
            expected=round((last_train-active).total_seconds()/900)+1
            for date in excluded_days:
                start=datetime.fromisoformat(date)
                lo,hi=max(active,start),min(last_train+timedelta(minutes=15),start+timedelta(days=1))
                expected-=max(0,round((hi-lo).total_seconds()/900))
            state['coverage']=state['observed_after_activation']/expected if expected>0 else None
            state['eligible']=state['observed_after_activation']>=min_train_observations and state['coverage'] is not None and state['coverage']>=min_train_coverage
            state['reason']=None if state['eligible'] else 'insufficient_train_activation_support_or_coverage'
            state['first_active']=active.isoformat()
        if state['eligible']:eligible.append(c)
    ordered=sorted(eligible,key=lambda c:(hashlib.sha256(c.encode()).hexdigest(),c))
    if require_count and len(ordered)<count:raise ValueError(f'Only {len(ordered)} train-eligible consumers; requested {count}; do not silently reduce external study')
    selected=ordered[:count]
    manifest={'schema_version':1,'dataset':'UCI ElectricityLoadDiagrams20112014','source_sha256':sha256_file(source),'train_fingerprint':train_hash.hexdigest(),'train_end_exclusive_portuguese_clock':cutoff.isoformat(),'selection_rule':'ascending SHA-256 of exact consumer ID, eligibility fitted only on train','requested_consumers':count,'selected_consumer_ids':selected,'complete_requested_support':len(selected)==count,'eligibility':{'min_train_observations':min_train_observations,'min_train_coverage':min_train_coverage,'activation_proxy':'first positive value observed in train; leading zeros excluded from truth','fit_partition':'train'},'consumer_statistics':stats,'excluded_dst_dates_train':sorted(excluded_days),'dst_policy':'exclude entire last-Sunday transition dates in March/October; no nonexistent/ambiguous timezone localization','nominal_seconds':900,'license':UCI_LICENSE,'doi':UCI_DOI,'dataset_page':UCI_PAGE,'counter_channels':None,'physical_claims':'no bidirectional independent-energy validation; no counters synthesized'}
    manifest['selection_id']=canonical_hash(manifest)
    if manifest_path is not None:write_immutable_json(manifest_path,manifest)
    return manifest


def load_uci_consumers(source,selection):
    """Return one SeriesData per selected consumer, never a fictitious device graph."""
    if isinstance(selection,(str,Path)):selection=json.loads(Path(selection).read_text())
    if sha256_file(source)!=selection['source_sha256']:raise ValueError('UCI source changed after consumer selection freeze')
    content={k:v for k,v in selection.items() if k!='selection_id'}
    if canonical_hash(content)!=selection['selection_id']:raise ValueError('UCI selection manifest fingerprint mismatch')
    selected=selection['selected_consumer_ids'];rows={c:[] for c in selected};excluded={};source_hash=selection['source_sha256']
    activation={c:datetime.fromisoformat(selection['consumer_statistics'][c]['first_active']) for c in selected}
    with _table(source) as reader:
        tc=_time_column(reader)
        if any(c not in reader.fieldnames for c in selected):raise ValueError('Frozen consumer column absent')
        previous=None
        for number,row in enumerate(reader,2):
            label=_label(row[tc])
            if previous is not None and label<=previous:raise ValueError('UCI source clock labels must strictly increase')
            previous=label
            if _transition_day(label):
                excluded[label.date().isoformat()]='DST March placeholders/October two-hour aggregates: complete transition day excluded'
                continue
            aware=label.replace(tzinfo=PORTUGUESE_ZONE)
            if aware.utcoffset()!=label.replace(tzinfo=PORTUGUESE_ZONE,fold=1).utcoffset():raise ValueError('Unresolved timezone ambiguity outside excluded transition day')
            grid=aware.astimezone(timezone.utc)
            for c in selected:
                original=parse_number(row[c],decimal=',',thousands='')
                before=label<activation[c]
                placeholder=before and original==0
                observed=math.isfinite(original) and not placeholder
                rows[c].append(CanonicalRecord('UCI:'+c,source_hash,f'{source_hash}:{number}:{c}',aware,aware,grid,original,original if observed else math.nan,unit_original='kW',observed_original=observed,natural_missing=not observed,available_to_method=observed,quality_flag='preactivation_placeholder' if placeholder else 'nonfinite_source' if not observed else 'ok'))
    output=[]
    for c in selected:
        site='UCI:'+c
        series=regularize(rows[c],site,900)
        metadata={'external_dataset':'UCI ElectricityLoadDiagrams20112014','consumer_id':c,'consumer_selection_id':selection['selection_id'],'source_sha256':source_hash,'license':UCI_LICENSE,'doi':UCI_DOI,'power_unit':'kW','power_semantics':'15min interval aggregate; source timestamp start/end semantics not established','timestamp_assumption':'Europe/Lisbon mainland Portuguese clock outside entire DST transition days; explicit benchmark interpretation, not receipt logs','availability_assumption':'optimistic availability=reported timestamp; actual arrival logs absent','grid_timezone':'UTC after conversion of unambiguous labels','dst_excluded_dates':excluded,'activation_proxy_train':activation[c].isoformat(),'zeros_after_activation':'preserved as observed','physical_correction_allowed':False,'counter_channels':None,'pretraining_overlap':'public data; external foundation model pretraining overlap unknown','requested_selection_complete':selection['complete_requested_support']}
        output.append(SeriesData(series,site,900,metadata))
    return output
