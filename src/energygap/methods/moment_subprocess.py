"""Lightweight parent-process bridge to the separately locked MOMENT worker."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
from time import perf_counter
import numpy as np
from .base import check_series,UnsupportedTask
from energygap.runtime.threads import requested_threads


class MOMENTSubprocess:
    name='MOMENT-1-base-isolated'
    def __init__(self,python=None,cache_dir=None,repository_root=None,timeout_seconds=120,threads=None,device='auto'):
        self.root=Path(repository_root or Path(__file__).resolve().parents[3]).resolve()
        self.python=Path(python or self.root/'.venv-moment/bin/python')
        self.cache_dir=Path(cache_dir or self.root/'artifacts/checkpoints/hf')
        self.worker=self.root/'scripts/moment_worker.py'
        self.timeout_seconds=timeout_seconds;self.threads=requested_threads(threads);self.last_metadata=None
        self.device=str(device)

    def fit(self,values,times,*,split='train'):
        if split!='train':raise ValueError('Frozen model accepts no validation/test fitting')
        check_series(values,times)
        if not self.python.exists():raise FileNotFoundError('Isolated MOMENT interpreter unavailable; construct environments/moment.lock.txt environment')
        return self

    def impute(self,values,times,*,task='R',observed_mask=None,padding_mask=None,issuance_time=None,query_id='query',lookahead_seconds=0):
        if task not in ('R','O'):raise UnsupportedTask('MOMENT isolated worker is audited for R/O only')
        x,t=check_series(values,times)
        observed=np.isfinite(x) if observed_mask is None else np.asarray(observed_mask,bool)&np.isfinite(x)
        padding=np.zeros(len(x),bool) if padding_mask is None else np.asarray(padding_mask,bool)
        if observed.shape!=x.shape or padding.shape!=x.shape:raise ValueError('Mask shape mismatch')
        observed &= ~padding
        # Hidden finite placeholders never leave the parent process.
        safe=np.where(observed,x,np.nan)
        issuance=float(t[-1] if issuance_time is None else issuance_time)
        with tempfile.TemporaryDirectory(prefix='energygap-moment-') as directory:
            d=Path(directory)
            np.savez_compressed(d/'input.npz',values=safe[None,:],times=t[None,:],observed_mask=observed[None,:],padding_mask=padding[None,:],query_ids=np.asarray([str(query_id)]))
            (d/'input.json').write_text(json.dumps({'schema_version':1,'task':task,'information_tier':'P','issuance_times':[issuance],'lookahead_seconds':lookahead_seconds}))
            command=[str(self.python),str(self.worker),'--input',str(d/'input.npz'),'--metadata',str(d/'input.json'),'--output',str(d/'output.npz'),'--result-metadata',str(d/'result.json'),'--checkpoint-cache',str(self.cache_dir),'--threads',str(self.threads),'--device',self.device]
            started=perf_counter()
            from .worker_environment import isolated_worker_environment
            result=subprocess.run(command,capture_output=True,text=True,timeout=self.timeout_seconds,cwd=self.root,
                                  env=isolated_worker_environment(self.root))
            if result.returncode:raise RuntimeError('Isolated MOMENT worker failed: '+result.stderr[-2000:])
            self.last_metadata=json.loads((d/'result.json').read_text())
            self.last_metadata['subprocess_wall_seconds']=perf_counter()-started
            with np.load(d/'output.npz',allow_pickle=False) as output:
                prediction=output['point_estimate'][0].copy()
        return prediction

    def impute_query(self,query):
        return self.impute(query.values,query.times,task=query.task,observed_mask=query.available_to_method,padding_mask=query.padding_mask,issuance_time=query.issuance_time.timestamp(),query_id=query.query_id)

    def get_resume_capabilities(self):
        return {'training':'not_applicable','inference':'query_shards','environment':'environments/moment.lock.txt','worker_sha256':hashlib.sha256(self.worker.read_bytes()).hexdigest(),'training_resume_validated':False,'frozen_checkpoint':True,'device_requested':self.device}
