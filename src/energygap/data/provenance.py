import hashlib
import json
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',',':'), allow_nan=False).encode()).hexdigest()


def write_immutable_json(path, value):
    """Never overwrite a scientific manifest with different content."""
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf8') != payload:
            raise FileExistsError(f'Immutable scientific manifest differs: {path}')
        return
    with path.open('x', encoding='utf8') as f:
        f.write(payload)
