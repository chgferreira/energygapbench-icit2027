#!/usr/bin/env python3
"""Package only the audited CPU-worker continuation and English documentation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
REGISTRY='configs/runtime_patches/cpu_worker_scaling_v1.json'
ALLOWED_SOURCE={'src/energygap/cli.py','src/energygap/conference.py',
                'src/energygap/reporting/conference.py','src/energygap/runtime/execution_patch.py'}


def build(root=ROOT):
    root=Path(root).resolve()
    manifest=json.loads((root/REGISTRY).read_text())
    if set(manifest['changed_source_files'])!=ALLOWED_SOURCE:
        raise ValueError('Patch source allowlist changed; a new audit is required')
    for name,digest in manifest['target_source_files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source differs from the audited runtime target: '+name)
    names={REGISTRY,'README.md','SERVER_GUIDE.md','THIRD_PARTY_NOTICES.md',
           'scripts/server.sh','scripts/resource_telemetry.py','scripts/gpu_telemetry.py',*ALLOWED_SOURCE}
    names.update(p.relative_to(root).as_posix() for p in (root/'docs').rglob('*')
                 if p.is_file() and p.name!='.DS_Store' and '__pycache__' not in p.parts)
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}
    destination=root/'dist';destination.mkdir(exist_ok=True)
    path=destination/'energygapbench-cpu-worker-patch.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in hashes:archive.write(root/name,name)
        archive.writestr('RUNTIME_WORKER_PATCH_MANIFEST.sha256',''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise RuntimeError('Runtime patch ZIP CRC failed')
        for name,sha in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha:raise RuntimeError('Runtime patch checksum failed')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (destination/(path.name+'.sha256')).write_text(f'{digest}  {path.name}\n')
    result=dict(zip=str(path.relative_to(root)),sha256=digest,bytes=path.stat().st_size,files=len(hashes),
                patch_id=manifest['patch_id'],source_code_hash=manifest['source_code_hash'],
                target_code_hash=manifest['target_code_hash'],
                excludes=['experiment YAMLs','data','runs','weights','environments','numerical model/correction implementations'])
    (destination/'runtime_worker_patch_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':build()
