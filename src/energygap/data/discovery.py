"""Recursively discover sources, retaining explicit site mapping."""
from pathlib import Path


def discover_files(root='raw_data', mapping=None):
    root = Path(root)
    mapping = mapping or {}
    supported = {'.csv','.txt','.parquet','.xls','.xlsx'}
    results = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in supported:
            site = mapping.get(str(p)) or mapping.get(p.name)
            if site is None:
                candidates = [s for s in ('H1','H2','I1') if s in p.relative_to(root).parts[:-1] or p.stem == s]
                site = candidates[0] if len(candidates) == 1 else None
            results.append({'path':str(p), 'site_id':site, 'mapping_evidence':'explicit mapping' if str(p) in mapping or p.name in mapping else 'directory/site-ID filename; explicit map persisted in config'})
    return results
