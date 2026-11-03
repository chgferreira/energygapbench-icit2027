"""Ingest only an evidenced data contract; inspection works before confirmation."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import csv
import math
from .schema import CanonicalRecord, SeriesData, DatasetConfig, AmbiguousDataError
from .units import power_to_kw, energy_to_kwh
from .provenance import sha256_file
from .timebase import align_records, regularize


def parse_number(value, decimal=',', thousands='.'):
    if isinstance(value, (int,float)):
        return float(value)
    if value is None or str(value).strip() == '':
        return math.nan
    value = str(value).strip()
    if thousands:
        value = value.replace(thousands, '')
    if decimal != '.':
        value = value.replace(decimal, '.')
    return float(value)


def read_table(path, *, encoding='utf-8-sig', delimiter=';', decimal=',', thousands='.'):
    path = Path(path)
    if path.suffix.lower() in ('.csv','.txt'):
        with path.open(encoding=encoding, newline='') as f:
            yield from csv.DictReader(f, delimiter=delimiter)
    elif path.suffix.lower() == '.parquet':
        try:
            import pandas as pd
            table = pd.read_parquet(path)
        except ImportError as e:
            raise RuntimeError('Parquet support unavailable: install parquet extra (pandas/pyarrow)') from e
        yield from table.to_dict('records')
    elif path.suffix.lower() in ('.xlsx','.xls'):
        try:
            import pandas as pd
            table = pd.read_excel(path)
        except ImportError as e:
            raise RuntimeError('Excel file found; install excel extra (pandas/openpyxl)') from e
        yield from table.to_dict('records')
    else:
        raise ValueError(f'Unsupported file type: {path.suffix}')


def ingest(config: DatasetConfig | str, root='.', purpose='physical') -> SeriesData:
    config = DatasetConfig.load(config) if isinstance(config,(str,Path)) else config
    config.validate(purpose=purpose)
    c = config.data
    site, records = c['site_id'], []
    for source in c['files']:
        path = Path(root)/source['path']
        actual = sha256_file(path)
        if actual != source['sha256']:
            raise AmbiguousDataError(f'Source checksum changed: {path}')
        for line,row in enumerate(read_table(path, encoding=source.get('encoding','utf-8-sig'), delimiter=source.get('delimiter',';')),start=2):
            raw_time = str(row[c['timestamp']['column']])
            time = datetime.fromisoformat(raw_time)
            zone = ZoneInfo(c['timestamp']['timezone'])
            if time.tzinfo is None:
                # Ambiguous/nonexistent local times require an explicit policy, never guessed.
                a,b = time.replace(tzinfo=zone,fold=0),time.replace(tzinfo=zone,fold=1)
                if a.utcoffset()!=b.utcoffset():
                    raise AmbiguousDataError(f'DST ambiguity/nonexistence at {raw_time}; explicit mapping required')
                time = a
            value = parse_number(row[c['power']['column']], source.get('decimal',','),source.get('thousands','.'))
            finite = math.isfinite(value)
            arrival_col = c['timestamp'].get('availability_column')
            arrival = datetime.fromisoformat(str(row[arrival_col])) if arrival_col else time
            if arrival.tzinfo is None:
                arrival = arrival.replace(tzinfo=zone)
            counters={}
            for direction in ('import','export'):
                counter=c.get('counters',{}).get(direction,{})
                v=parse_number(row.get(counter.get('column','')),source.get('decimal',','),source.get('thousands','.'))
                counters[direction]=energy_to_kwh(v,counter['unit']) if math.isfinite(v) and counter.get('unit') else None
            auxiliaries={name:parse_number(row.get(meta['column']),source.get('decimal',','),source.get('thousands','.')) for name,meta in c.get('auxiliaries',{}).items()}
            records.append(CanonicalRecord(site,actual,f'{actual}:{line}',time,arrival,time,value,power_to_kw(value,c['power']['unit'],c['power']['import_sign']),unit_original=c['power']['unit'],observed_original=finite,natural_missing=not finite,available_to_method=finite,auxiliaries=auxiliaries,import_counter=counters['import'],export_counter=counters['export'],counter_availability_time=arrival))
    step=int(c.get('alignment',{}).get('nominal_seconds',300))
    aligned, audit = align_records(records,step,c.get('alignment',{}).get('tolerance_seconds',90))
    return SeriesData(regularize(aligned,site,step),site,step,{'config':c,'purpose':purpose,'power_semantics':c['power'].get('semantics'),'timestamp_semantics':c['timestamp'].get('semantics'),'physical_correction_allowed':not config.unresolved('physical'),'alignment_audit':{'collisions':audit['collisions'],'n_rejected':len(audit['rejected']),'rejected_record_ids':[r.source_record_id for r in audit['rejected']]},'availability_assumption':'arrival log' if c['timestamp'].get('availability_column') else 'optimistic: availability_time=measurement_time'})
