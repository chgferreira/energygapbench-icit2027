#!/usr/bin/env python3
"""Package an exact, audited report-performance continuation without run data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REGISTRY='configs/runtime_patches/reporting_performance_v1.json'
ALLOWED_SOURCE={
    'src/energygap/cli.py','src/energygap/conference.py',
    'src/energygap/reporting/conference.py','src/energygap/reporting/bootstrap_execution.py',
    'src/energygap/runtime/conference_progress.py','src/energygap/runtime/execution_patch.py',
    'src/energygap/runtime/state.py',
}


def build(root=ROOT):
    root=Path(root).resolve()
    manifest=json.loads((root/REGISTRY).read_text())
    if manifest['patch_id']!='reporting-performance-v1' or set(manifest['changed_source_files'])!=ALLOWED_SOURCE:
        raise ValueError('Reporting patch source allowlist differs from the reviewed implementation')
    for name,digest in manifest['target_source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source differs from the audited reporting target: '+name)
    names={REGISTRY,'configs/runtime_patches/cpu_worker_scaling_v1.json',
           'scripts/server.sh','scripts/package_reporting_patch.py','README.md','SERVER_GUIDE.md',*ALLOWED_SOURCE}
    names.update(p.relative_to(root).as_posix() for p in (root/'docs').rglob('*')
                 if p.is_file() and p.name!='.DS_Store' and '__pycache__' not in p.parts)
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}
    destination=root/'dist';destination.mkdir(exist_ok=True)
    path=destination/'energygapbench-reporting-patch.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in hashes:archive.write(root/name,name)
        archive.writestr('REPORTING_PATCH_MANIFEST.sha256',''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise RuntimeError('Reporting patch ZIP CRC failed')
        for name,sha in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha:raise RuntimeError('Reporting patch checksum failed')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (destination/(path.name+'.sha256')).write_text(f'{digest}  {path.name}\n')
    result=dict(zip=str(path.relative_to(root)),sha256=digest,bytes=path.stat().st_size,files=len(hashes),
                patch_id=manifest['patch_id'],accepted_source_code_hashes=manifest['accepted_source_code_hashes'],
                target_code_hash=manifest['target_code_hash'],
                excludes=['experiment YAMLs','data','runs','weights','environments','numerical model and correction implementations'])
    (destination/'reporting_patch_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':build()
