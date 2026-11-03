#!/usr/bin/env python3
"""Explicitly download the pinned official MOMENT base cache; no user-data upload."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REVISION = '5e44b0ea26376a176360f87831124e018f876d96'
SHA256 = '1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825'
BYTES = 453940120


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/checkpoints/hf'))
    args = parser.parse_args(argv)
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo_id='AutonLab/MOMENT-1-base', revision=REVISION,
        cache_dir=str(args.output), allow_patterns=['config.json', 'model.safetensors', 'README.md', 'LICENSE*']))
    weight = snapshot/'model.safetensors'
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    if digest != SHA256 or weight.stat().st_size != BYTES:
        raise ValueError('Pinned MOMENT checkpoint hash/size mismatch; refusing scientific reuse')
    provenance = {'repo': 'AutonLab/MOMENT-1-base', 'revision': REVISION,
                  'file': str(weight), 'sha256': digest, 'bytes': BYTES,
                  'download_policy': 'explicit public checkpoint download; no user-data upload'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'moment_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'status': 'verified', **provenance}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
