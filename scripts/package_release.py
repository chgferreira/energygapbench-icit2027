"""Create a local, explicit-allowlist source package without private telemetry."""
from pathlib import Path
import hashlib
import io
import json
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=('src','tests','configs','docs','scripts','.github','environments','env')
TOP=('README.md','SERVER_GUIDE.md','pyproject.toml','uv.lock','requirements.lock.txt',
     'requirements-server.txt','constraints-server.txt','Makefile','CITATION.cff',
     'LICENSE','THIRD_PARTY_NOTICES.md','.gitignore','model_registry.yaml')


def build(root=ROOT):
    root=Path(root);destination=root/'dist';destination.mkdir(exist_ok=True)
    paths=[root/p for p in TOP if (root/p).is_file()]
    for directory in DIRECTORIES:
        paths.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'))
    # Dataset configs contain local hashes/evidence; real timestamps/telemetry/audit outputs are excluded.
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    archive=destination/'energygapbench-0.1.0-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname='energygapbench/'+str(p.relative_to(root)),recursive=False)
        content=''.join(f'{v}  {k}\n' for k,v in manifest.items()).encode()
        info=tarfile.TarInfo('energygapbench/MANIFEST.sha256');info.size=len(content);info.mtime=0;tar.addfile(info,io.BytesIO(content))
    zipped=destination/'energygapbench-0.1.0-source.zip'
    with zipfile.ZipFile(zipped,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths):z.write(p,arcname='energygapbench/'+str(p.relative_to(root)))
        z.writestr('energygapbench/MANIFEST.sha256',content)
    summary=dict(archive=str(archive.relative_to(root)),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),zip=str(zipped.relative_to(root)),zip_sha256=hashlib.sha256(zipped.read_bytes()).hexdigest(),zip_bytes=zipped.stat().st_size,n_files=len(manifest),excluded=['raw_data','data','runs','artifacts','checkpoints','secrets'],published=False)
    (destination/'release_manifest.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':build()
