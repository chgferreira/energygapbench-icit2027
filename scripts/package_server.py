"""Build the server ZIP with both presets; never bundle data, weights or analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from package_release import DIRECTORIES, TOP

ROOT=Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root=Path(root).resolve();destination=root/'dist';destination.mkdir(exist_ok=True)
    files=[root/name for name in TOP if (root/name).is_file()]
    for name in DIRECTORIES:
        files.extend(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc','.pyo') and p.name!='.DS_Store')
    files=sorted(set(files))
    required={'requirements-server.txt','constraints-server.txt'}
    if not required.issubset({p.relative_to(root).as_posix() for p in files}):
        raise RuntimeError('Server package is missing the active-environment requirements/constraints')
    hashes={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=''.join(f'{sha}  {name}\n' for name,sha in hashes.items())
    path=destination/'energygapbench-server-full.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for file in files:archive.write(file,file.relative_to(root).as_posix())
        archive.writestr('SERVER_FULL_MANIFEST.sha256',manifest)
    with zipfile.ZipFile(path) as archive:
        invalid=archive.testzip()
        if invalid:raise RuntimeError('ZIP CRC check failed: '+invalid)
        if any(name.split('/')[0] in ('raw_data','data','runs','artifacts','dist') for name in archive.namelist()):
            raise RuntimeError('Non-source directory entered server package')
        for name,digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:raise RuntimeError('Archive manifest mismatch: '+name)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (destination/(path.name+'.sha256')).write_text(f'{digest}  {path.name}\n')
    result={'zip':str(path.relative_to(root)),'sha256':digest,'bytes':path.stat().st_size,
            'files':len(files),'layout':'flat repository root; extract inside existing project',
            'default_preset':'focused','presets':['focused','full'],
            'excluded':['raw_data','data','runs','artifacts','dist','virtual environments','upstream weights/source'],
            'published':False}
    (destination/'server_full_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':build()
