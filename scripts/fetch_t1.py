#!/usr/bin/env python3
"""Fetch only pinned official T1 source/license, checking exact SHA-256.

No data, weights, or user files are sent. Requires public GitHub network access.
"""
from pathlib import Path
import argparse,hashlib,json,urllib.request
COMMIT='823b76aab57a2e60eb5029caab06fdfec8b19256'
FILES={'models/T1.py':'3f87aac66745029a88907b7ce65766d3ce5f785a6d4190f6343f9649d4ad3de8'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('artifacts/upstream/T1'));args=parser.parse_args()
    provenance={'repository':'https://github.com/Oppenheimerdinger/T1','commit':COMMIT,'license':'MIT','files':[]}
    for name in ['models/T1.py','LICENSE','README.md','requirements.txt','run.py','exp/exp_imputation.py','scripts/imputation/T1_ETTh1.sh']:
        url=f'https://raw.githubusercontent.com/Oppenheimerdinger/T1/{COMMIT}/{name}'
        data=urllib.request.urlopen(url,timeout=30).read();digest=hashlib.sha256(data).hexdigest()
        if name in FILES and digest!=FILES[name]:raise ValueError('Upstream source checksum differs: '+name)
        path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=data:raise FileExistsError('Existing upstream file differs; refusing overwrite: '+str(path))
        path.write_bytes(data);provenance['files'].append({'path':name,'sha256':digest,'bytes':len(data),'source':url})
    (args.output/'source_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'status':'downloaded','source':str(args.output),'commit':COMMIT}))
if __name__=='__main__':main()
