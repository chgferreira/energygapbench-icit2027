"""Fixed evaluation masks with mechanisms distinct from temporal geometry."""
from dataclasses import dataclass, asdict
from datetime import timedelta
import math
import random
from ..data.provenance import canonical_hash, write_immutable_json

CONDITIONS={'C1':{'geometry':'points','mechanism':'MCAR','rate':.05,'minutes':5},'C2':{'geometry':'points','mechanism':'MCAR','rate':.20,'minutes':5},'C3':{'geometry':'block','mechanism':'MCAR','rate':.10,'minutes':15},'C4':{'geometry':'block','mechanism':'MCAR','rate':.10,'minutes':60},'C5':{'geometry':'block','mechanism':'MCAR','rate':.10,'minutes':180},'C6':{'geometry':'replay','mechanism':'train_fitted','rate':None,'minutes':None}}


@dataclass(frozen=True)
class Episode:
    gap_id: str
    indices: tuple[int,...]
    left_index: int
    right_index: int
    duration_minutes: float
    boundary_distance_minutes: float
    condition: str
    mask_seed: int
    fold_id: int
    partition: str

    @property
    def start_index(self): return self.indices[0]
    @property
    def end_index(self): return self.indices[-1]


@dataclass(frozen=True)
class MaskManifest:
    mask_id: str
    site_id: str
    fold_id: int
    partition: str
    condition: str
    seed: int
    episodes: tuple[Episode,...]
    hidden_indices: tuple[int,...]
    requested_rate: float
    realized_rate: float
    eligible_count: int
    rejections: dict
    replay_metadata: dict

    def to_dict(self): return asdict(self)

    def write(self,path):
        write_immutable_json(path,self.to_dict())


def _valid(r):
    return r.observed_original and not r.natural_missing and r.available_to_method and math.isfinite(r.value_canonical)


def fit_replay(series,fold):
    train=fold.indices(series,'train')
    gaps=[]; current=[]
    for i in train:
        if not _valid(series.records[i]): current.append(i)
        elif current:
            gaps.append(current);current=[]
    if current: gaps.append(current)
    incidence=sum(len(g) for g in gaps)/max(1,len(train))
    if len(gaps)<5:
        return {'status':'fallback','reason':'fewer_than_5_train_episodes','n_train_episodes':len(gaps),'lengths':[max(1,round(sum(map(len,gaps))/len(gaps))) if gaps else 3],'hours':[],'rate':incidence if incidence>0 else .01,'estimation_partition':'train','uncertainty':'sparse; marginal duration fallback'}
    return {'status':'empirical','n_train_episodes':len(gaps),'lengths':[len(g) for g in gaps],'hours':[series.records[g[0]].grid_time.hour for g in gaps],'rate':incidence,'estimation_partition':'train','uncertainty':'empirical finite train episodes'}


def generate_mask(series,fold,partition='test',condition='C3',seed=2001,max_episodes=None,physical=False,min_spacing=1,replay=None,max_gap_positions=None):
    if partition not in ('train','validation','test'): raise ValueError('Unknown partition')
    if condition not in CONDITIONS: raise ValueError('Unknown condition')
    if min_spacing<1: raise ValueError('At least one observed position must separate target gaps')
    if max_gap_positions is not None and max_gap_positions<1:raise ValueError('Positive context gap capacity required')
    spec=CONDITIONS[condition]
    rng=random.Random(seed)
    indices=fold.indices(series,partition)
    domain=set(indices)
    eligible=[i for i in indices if _valid(series.records[i])]
    replay=replay or (fit_replay(series,fold) if condition=='C6' else {})
    requested=spec['rate'] if spec['rate'] is not None else replay['rate']
    target_count=round(requested*len(eligible))
    candidates=eligible[:]
    rng.shuffle(candidates)
    blocked=set(); hidden=set(); episodes=[]
    rejections={'insufficient_original_truth':0,'boundary':0,'overlap_or_spacing':0,'counter_ineligible':0,'physical_duration_mismatch':0}
    if max_gap_positions is not None:rejections['context_capacity']=0
    for start in candidates:
        if len(hidden)>=target_count or max_episodes is not None and len(episodes)>=max_episodes: break
        length=rng.choice(replay['lengths']) if condition=='C6' else max(1,round(spec['minutes']*60/series.nominal_seconds))
        if max_gap_positions is not None and length>max_gap_positions:
            rejections['context_capacity']+=1;continue
        if condition=='C6' and replay.get('hours') and series.records[start].grid_time.hour not in replay['hours']: continue
        gap=tuple(range(start,start+length));left=start-1;right=start+length
        if any(i not in domain for i in gap) or left<0 or right>=len(series.records):
            rejections['boundary']+=1;continue
        if not all(_valid(series.records[i]) for i in (*gap,left,right)):
            rejections['insufficient_original_truth']+=1;continue
        actual=(series.records[right].grid_time-series.records[left].grid_time).total_seconds()
        if abs(actual-(length+1)*series.nominal_seconds)>1e-6:
            rejections['physical_duration_mismatch']+=1;continue
        if any(i in blocked for i in (*gap,left,right)):
            rejections['overlap_or_spacing']+=1;continue
        if physical:
            rows=[series.records[i] for i in range(left,right+1)]
            if any(r.import_counter is None or r.export_counter is None for r in rows) or any(b.import_counter<a.import_counter or b.export_counter<a.export_counter for a,b in zip(rows,rows[1:])):
                rejections['counter_ineligible']+=1;continue
        identity={'site':series.site_id,'fold':fold.fold_id,'partition':partition,'condition':condition,'seed':seed,'record_ids':[series.records[i].source_record_id for i in gap]}
        gid=canonical_hash(identity)[:24]
        episodes.append(Episode(gid,gap,left,right,length*series.nominal_seconds/60,actual/60,condition,seed,fold.fold_id,partition))
        hidden.update(gap)
        blocked.update(range(max(0,left-min_spacing+1),min(len(series.records),right+min_spacing)))
    episodes.sort(key=lambda e:e.start_index)
    identity={'site':series.site_id,'fold':fold.manifest(),'partition':partition,'condition':condition,'seed':seed,'hidden':sorted(hidden),'physical':physical,'min_spacing':min_spacing,'replay':replay}
    if max_gap_positions is not None:identity['max_gap_positions']=max_gap_positions
    mid=canonical_hash(identity)
    return MaskManifest(mid,series.site_id,fold.fold_id,partition,condition,seed,tuple(episodes),tuple(sorted(hidden)),requested,len(hidden)/len(eligible) if eligible else 0.,len(eligible),rejections,replay)
