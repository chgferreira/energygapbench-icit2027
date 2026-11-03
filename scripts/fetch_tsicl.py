#!/usr/bin/env python3
"""Retrieve the official pinned TS-ICL checkpoint for local noncommercial research.

Preserves the original license, verifies SHA-256/size, never downloads user data,
never uploads data, and keeps the optional model outside the MIT release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REVISION='19c94031439fb31f36ce395088ee50a6762d3774'
EXPECTED_SHA256='a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431'
EXPECTED_BYTES=219150987


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/checkpoints/tsicl'));args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    base=f'https://huggingface.co/taharnbl/TS-ICL/resolve/{REVISION}/'
    for name in ('LICENSE','README.md','config.json'):
        data=urllib.request.urlopen(base+name,timeout=30).read();path=args.output/name
        if path.exists() and path.read_bytes()!=data:raise FileExistsError('Existing metadata differs: '+str(path))
        path.write_bytes(data)
    if 'TS-ICL NON-COMMERCIAL LICENSE' not in (args.output/'LICENSE').read_text():raise ValueError('Unexpected upstream terms; audit before use')
    checkpoint=args.output/'tsicl-v1.ckpt'
    if checkpoint.exists():
        if checkpoint.stat().st_size!=EXPECTED_BYTES or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=EXPECTED_SHA256:
            raise ValueError('Existing checkpoint differs; refusing overwrite')
        print(json.dumps({'status':'already_complete','bytes':EXPECTED_BYTES,'sha256':EXPECTED_SHA256}));return
    temporary=args.output/'tsicl-v1.ckpt.part';digest=hashlib.sha256();total=0
    with urllib.request.urlopen(base+'tsicl-v1.ckpt',timeout=60) as response,temporary.open('wb') as stream:
        while True:
            block=response.read(8*1024*1024)
            if not block:break
            total+=len(block)
            if total>EXPECTED_BYTES:raise ValueError('Remote model exceeds verified authorized checkpoint size')
            digest.update(block);stream.write(block)
        stream.flush();os.fsync(stream.fileno())
    if total!=EXPECTED_BYTES or digest.hexdigest()!=EXPECTED_SHA256:raise ValueError('Incomplete checkpoint or SHA-256 mismatch; partial file is not success')
    os.replace(temporary,checkpoint)
    provenance={'repo':'taharnbl/TS-ICL','revision':REVISION,'file':checkpoint.name,'bytes':total,'sha256':EXPECTED_SHA256,
                'license':'TS-ICL NON-COMMERCIAL LICENSE v1.0','usage_scope':'local noncommercial research; not redistributed'}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'status':'downloaded','bytes':total,'sha256':EXPECTED_SHA256}))
if __name__=='__main__':main()
