"""Real-process restart demonstration, independent of all scientific model claims.

Run ``python -m energygap.runtime.demo --output artifacts/reports/recovery-demo``.
The tiny momentum trainer exercises full state, but is not a catalogue method.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .state import StateStore, atomic_json, canonical, stable_seed, utcnow
from .checkpoints import CheckpointStore
from .progress import Progress
from .signals import StopRequested
from .scheduler import run_units

DEMO_CONFIG = {'kind':'runtime_fixture_momentum','data_hash':'deterministic-v1','model':'linear4',
               'seed':11023,'steps':40,'mask_rng':'numpy-PCG64','batch_size':8,'precision':'float64'}

def _initial_state():
    rng = np.random.default_rng(11023)
    return {'weight':rng.normal(0,.1,4),'bias':0.,'momentum':np.zeros(4),'bias_momentum':0.,
            'scheduler':{'lr':.04,'decay':.985},'step':0,'epoch':0,
            'sampler':{'permutation':rng.permutation(48),'cursor':0},
            'rng_numpy':rng.bit_generator.state,'rng_python':random.Random(23).getstate(),
            'early_stopping':{'best':float('inf'),'bad_epochs':0},'history':[],'trial':{'id':0,'params':{'lr':.04},'max_steps':40}}

def _step(state):
    data_rng = np.random.default_rng(9)
    x = data_rng.normal(size=(48,4)); y = x @ np.array([.8,-.2,.5,1.2]) + .3
    rng = np.random.default_rng(); rng.bit_generator.state = state['rng_numpy']
    pyrng = random.Random(); pyrng.setstate(state['rng_python'])
    sampler = state['sampler']
    if sampler['cursor'] == 48:
        sampler['permutation'] = rng.permutation(48); sampler['cursor'] = 0; state['epoch'] += 1
    ix = sampler['permutation'][sampler['cursor']:sampler['cursor']+8]; sampler['cursor'] += len(ix)
    dynamic_mask = rng.random((len(ix),4)) < .05
    xb = np.where(dynamic_mask,0,x[ix]); residual = xb @ state['weight'] + state['bias'] - y[ix]
    gradient = xb.T @ residual / len(ix); bg = float(np.mean(residual))
    # Python RNG is also part of the exact continuation, without changing batching.
    jitter = (pyrng.random()-.5)*1e-5
    state['momentum'] = .8*state['momentum']+gradient
    state['bias_momentum'] = .8*state['bias_momentum']+bg
    state['weight'] -= state['scheduler']['lr'] * state['momentum']
    state['bias'] -= state['scheduler']['lr'] * state['bias_momentum'] + jitter
    state['scheduler']['lr'] *= state['scheduler']['decay']
    state['step'] += 1
    metric = float(np.mean((x @ state['weight']+state['bias']-y)**2))
    es = state['early_stopping']
    if metric < es['best']: es['best'],es['bad_epochs'] = metric,0
    else: es['bad_epochs'] += 1
    state['rng_numpy'] = rng.bit_generator.state; state['rng_python'] = pyrng.getstate()
    state['history'].append({'key':f'train-step:{state["step"]:03d}', 'step':state['step'], 'metric':metric,
                             'batch_ids':ix.tolist(),'mask_sha256':hashlib.sha256(dynamic_mask.tobytes()).hexdigest()})

def train_fixture(path: Path, progress='none', ready_step: int | None = None,
                  ready_boundary='unconfirmed', waiting=False):
    store = StateStore(path, DEMO_CONFIG)
    store.add_job('train:trial-0',40,metadata={'phase':'training','kind':'runtime_fixture_only'})
    if store.job('train:trial-0')['status'] == 'succeeded':
        result = {'executed_steps':0,'initial':40,'complete':True}; store.close(); return result
    store.start_session(); checkpoints = CheckpointStore(path,store.fingerprint,ledger=store)
    restored = checkpoints.load('train:trial-0')
    state = restored[0] if restored else _initial_state()
    initial = state['step']; executed=0; start=time.monotonic()
    with store.claim('train:trial-0') as lease, StopRequested() as stop, Progress(40,initial,progress,'train fixture') as display:
        # A checkpoint may have published just before the result transaction.
        confirmed = store.job('train:trial-0')['completed_units']
        if initial > confirmed:
            store.publish('train:trial-0',lease.token,state['history'][confirmed:initial],initial)
        if initial < confirmed:
            store.event('progress_corrected_after_checkpoint_fallback', job_id='train:trial-0',
                        result_units_confirmed=confirmed,optimizer_steps_restored=initial,
                        recompute_steps=confirmed-initial)
        while state['step'] < 40:
            _step(state); executed += 1
            checkpoint_now = state['step'] % 5 == 0 or stop.requested
            if checkpoint_now:
                checkpoints.save('train:trial-0',state,state['step'])
                current = store.job('train:trial-0')['completed_units']
                store.publish('train:trial-0',lease.token,
                              state['history'][max(0,current-5):state['step']],max(current,state['step']))
                store.record_active(time.monotonic()-start,'train:trial-0'); start=time.monotonic()
            display.update(state['step'])
            if ready_step == state['step']:
                if ready_boundary == 'checkpoint' and not checkpoint_now:
                    checkpoints.save('train:trial-0',state,state['step'])
                    current=store.job('train:trial-0')['completed_units']
                    store.publish('train:trial-0',lease.token,state['history'][current:state['step']],state['step'])
                print(json.dumps({'ready':True,'step':state['step'],'checkpoint_step':(state['step']//5)*5,
                                  'initial':initial,'pid':os.getpid()}),flush=True)
                if waiting:
                    # Block on stdin: caller delivers SIGKILL or SIGTERM, then a byte.
                    sys.stdin.readline()
            if stop.requested:
                checkpoints.save('train:trial-0',state,state['step'])
                current=store.job('train:trial-0')['completed_units']
                store.publish('train:trial-0',lease.token,state['history'][current:state['step']],max(current,state['step']))
                store.interrupt('train:trial-0',lease.token,'SIGTERM safe boundary')
                store.end_session('interrupted'); store.close()
                return {'executed_steps':executed,'initial':initial,'complete':False}
        checkpoints.save('train:trial-0',state,40)
        store.finish('train:trial-0',lease.token)
    for phase,total in [('inference',13),('bootstrap',17)]:
        run_units(store,phase,lambda i,s,p=phase:{'key':f'{p}:{i}','id':i,'seed':s,'status':'invalid' if p=='bootstrap' and i%7==0 else 'valid','value':None if p=='bootstrap' and i%7==0 else stable_seed(s,p)%101},
                  total_units=total,shard_size=4,progress=progress)
    store.complete()
    atomic_json(path/'training_summary.json', {'weight':state['weight'].tolist(),'bias':state['bias'],
                'step':state['step'],'scheduler':state['scheduler'],'next_batch':state['sampler']['permutation'][state['sampler']['cursor']:].tolist(),
                'history_sha256':hashlib.sha256(canonical(state['history'])).hexdigest(),
                'checkpoint_state_sha256':_state_digest(state),'initial':initial,'executed_steps':executed})
    result={'executed_steps':executed,'initial':initial,'complete':True}; store.close(); return result

def _state_digest(state):
    def conv(v):
        if isinstance(v,np.ndarray): return {'dtype':str(v.dtype),'shape':v.shape,'data':v.tolist()}
        if isinstance(v,dict): return {k:conv(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [conv(x) for x in v]
        if isinstance(v,np.generic): return v.item()
        return v
    return hashlib.sha256(canonical(conv(state))).hexdigest()

def demonstrate(output: Path) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ('continuous','resumed-kill','resumed-term')):
        raise FileExistsError('demonstration output already exists; choose another directory')
    continuous=output/'continuous';before_continuous=time.monotonic();train_fixture(continuous)
    continuous_seconds=time.monotonic()-before_continuous
    report={'scope':'deterministic CPU NumPy momentum fixture; not validation of neural catalogue adapters',
            'cases':[],'created_at':utcnow(),'continuous_seconds':continuous_seconds}
    with StateStore(continuous,DEMO_CONFIG) as ledger:
        expected=ledger.results()
    baseline=json.loads((continuous/'training_summary.json').read_text())
    for name,signum in [('resumed-kill',signal.SIGKILL),('resumed-term',signal.SIGTERM)]:
        path=output/name
        case_started=time.monotonic()
        process=subprocess.Popen([sys.executable,'-m','energygap.runtime.demo','--worker','--output',str(path),
                                  '--ready-step','7','--wait'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        ready=json.loads(process.stdout.readline())
        process.send_signal(signum)
        if signum == signal.SIGTERM: process.stdin.write('\n');process.stdin.flush()
        _,stderr=process.communicate(timeout=15)
        before=time.monotonic(); result=train_fixture(path,progress='tqdm'); resume_seconds=time.monotonic()-before
        with StateStore(path,DEMO_CONFIG) as ledger:
            records=ledger.results(); status=ledger.status()
        summary=json.loads((path/'training_summary.json').read_text())
        same=records==expected and summary['checkpoint_state_sha256']==baseline['checkpoint_state_sha256']
        case={'case':name,'signal':signum.name,'interrupted_returncode':process.returncode,
              'saved_progress':result['initial'],'tqdm_initial':result['initial'],'tqdm_total':40,
              'executed_on_resume':result['executed_steps'],'repeated_unconfirmed_steps':7-result['initial'],
              'results_equal':records==expected,'full_training_state_equal':summary['checkpoint_state_sha256']==baseline['checkpoint_state_sha256'],
              'result_keys_unique':len({r['key'] for r in records})==len(records),
              'result_sha256':hashlib.sha256(canonical(records)).hexdigest(),
              'checkpoint_state_sha256':summary['checkpoint_state_sha256'],'resume_seconds':resume_seconds,
              'active_seconds':status['active_seconds'],'status':status['status'],'passed':same,
              'end_to_end_wall_seconds':time.monotonic()-case_started,
              'additional_wall_seconds_vs_continuous':time.monotonic()-case_started-continuous_seconds,
              'cost_note':'single tiny trial wall-clock difference includes subprocess startup, restart, I/O and unconfirmed recomputation; noisy, not model latency benchmark'}
        report['cases'].append(case)
    report['passed']=all(case['passed'] for case in report['cases'])
    report['limitations']=['Local POSIX filesystem only; NFS/distributed execution unsupported.',
                           'No claim of catalogue neural training resume until backend-specific tests pass.',
                           'Crash active time after last durable accounting is uncertain; offline interval is not throughput.',
                           'Checkpoint files are trusted local pickle, never load from untrusted sources.']
    atomic_json(output/'comparison.json',report)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--ready-step',type=int)
    parser.add_argument('--wait',action='store_true');parser.add_argument('--progress',default='none')
    args=parser.parse_args()
    result=train_fixture(args.output,args.progress,args.ready_step,waiting=args.wait) if args.worker else demonstrate(args.output)
    print(json.dumps(result,default=str))
if __name__=='__main__': main()
