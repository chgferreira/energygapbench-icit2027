"""Automated real-process recovery audit of the actual upstream SAITS loss/model."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .state import StateStore,atomic_json,canonical,utcnow
from .checkpoints import CheckpointStore


def _windows():
    import numpy as np
    x=np.sin(np.arange(6*16,dtype=np.float32).reshape(6,16,1)/9);x[0,1,0]=np.nan
    return x


def _train(path,**kwargs):
    from energygap.methods.training import train_saits_steps
    return train_saits_steps(_windows(),path,seed=23,max_steps=12,batch_size=2,checkpoint_every_steps=2,
                            architecture={'n_layers':1,'d_model':16,'n_heads':2,'d_k':8,'d_v':8,'d_ffn':16,'dropout':.1},**kwargs)


def _science_digest(path):
    from io import BytesIO
    import numpy as np
    import torch
    def convert(value):
        if isinstance(value,torch.Tensor):return {'dtype':str(value.dtype),'shape':list(value.shape),'data':value.cpu().tolist()}
        if isinstance(value,np.ndarray):return {'dtype':str(value.dtype),'shape':list(value.shape),'data':value.tolist()}
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {str(k):convert(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [convert(v) for v in value]
        return value
    with StateStore(path,resume='require') as ledger:
        blob=CheckpointStore(path,ledger.fingerprint).load('saits:train')[0]['torch_state']
        state=torch.load(BytesIO(blob),weights_only=False)
        return {'full_state_sha256':hashlib.sha256(canonical(convert(state))).hexdigest(),
                'results_sha256':hashlib.sha256(canonical(ledger.results())).hexdigest(),
                'n_results':len(ledger.results()),'status':ledger.status()['status'],
                'accounted_active_seconds':ledger.status()['active_seconds'],
                'uncertain_reservations':ledger.status()['crash_time_reservations']}


def demonstrate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any((output/n).exists() for n in ('continuous','SIGKILL','SIGTERM')):raise FileExistsError('choose a new output directory')
    started=time.monotonic();_train(output/'continuous');continuous_seconds=time.monotonic()-started
    expected=_science_digest(output/'continuous')
    report={'created_at':utcnow(),'implementation':'upstream PyPOTS SAITS architecture + original ORT/MIT loss',
            'variant':'saits_protocol_training_loop','context':16,'channels':1,'windows':6,
            'maximum_optimizer_steps':12,'continuous':expected,'continuous_seconds':continuous_seconds,'cases':[],
            'scope':'deterministic CPU num_workers=0 diagnostic; no validation-selected scientific experiment',
            'verified_components':['model','Adam optimizer','scheduler(None/native constant LR)','Python/NumPy/Torch RNG',
                                   'sampler permutation/cursor','next batch','dynamic point/block masks','early stopping diagnostic','best model','normalization','trial','steps']}
    for signal_ in (signal.SIGKILL,signal.SIGTERM):
        case_started=time.monotonic();path=output/signal_.name
        process=subprocess.Popen([sys.executable,'-m','energygap.runtime.saits_demo','--worker','--output',str(path)],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        while True:
            line=process.stdout.readline()
            if line.startswith('RECOVERY_READY '):break
            if not line:
                _,err=process.communicate(timeout=10);raise RuntimeError('worker failed before checkpoint boundary: '+err[-1000:])
        process.send_signal(signal_)
        if signal_==signal.SIGTERM:process.stdin.write('\n');process.stdin.flush()
        process.communicate(timeout=20)
        resumed=_train(path,progress='tqdm');actual=_science_digest(path)
        elapsed=time.monotonic()-case_started
        report['cases'].append({'signal':signal_.name,'interrupted_returncode':process.returncode,
                                'initial_optimizer_steps':resumed['initial'],'tqdm_total':12,
                                'recomputed_unconfirmed_steps':5-resumed['initial'],
                                'full_state_equal':actual['full_state_sha256']==expected['full_state_sha256'],
                                'results_equal':actual['results_sha256']==expected['results_sha256'],
                                'result':actual,'wall_seconds':elapsed,
                                'additional_wall_seconds_vs_continuous':elapsed-continuous_seconds,
                                'cost_note':'tiny trial includes subprocess/import/startup and conservative crash budget accounting, not an inference latency benchmark'})
    report['passed']=all(r['full_state_equal'] and r['results_equal'] for r in report['cases'])
    report['limitations']=['CPU determinism only; GPU/distributed unvalidated.',
                           'Diagnostic runner uses native training loss for early-stop bookkeeping, not validation selection.',
                           'Native PyPOTS fit_windows remains nonresumable; capability applies only to this named runner.',
                           'Context16 test does not validate full L512 scientific accuracy or convergence.',
                           'Cumulative budget includes unfinished work reservations; without enforced timeout upper crash-time uncertainty is unbounded.']
    atomic_json(output/'comparison.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        original=StateStore.finish_work;counter=0
        def hook(self,reservation,seconds):
            nonlocal counter
            result=original(self,reservation,seconds);counter+=1
            if counter==5:
                print('RECOVERY_READY '+json.dumps({'step':5}),flush=True);sys.stdin.readline()
            return result
        StateStore.finish_work=hook;_train(args.output)
    else:print(json.dumps(demonstrate(args.output)))
if __name__=='__main__':main()
