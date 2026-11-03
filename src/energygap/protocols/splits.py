"""Rolling origins use elapsed physical time, never row-count quantiles."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from ..data.provenance import canonical_hash, write_immutable_json

FRACTIONS=((0,.40,.50,.65),(0,.55,.65,.80),(0,.70,.80,1.0))


@dataclass(frozen=True)
class Fold:
    fold_id: int
    start: datetime
    train_end: datetime
    validation_end: datetime
    test_end: datetime
    inclusive_test_end: bool = False

    def partition(self,time):
        if self.start<=time<self.train_end: return 'train'
        if self.train_end<=time<self.validation_end: return 'validation'
        if self.validation_end<=time<self.test_end or self.inclusive_test_end and time==self.test_end: return 'test'
        return None

    def indices(self, series, partition):
        return [i for i,r in enumerate(series.records) if self.partition(r.grid_time)==partition]

    def manifest(self):
        return {'fold_id':self.fold_id,'start':self.start.isoformat(),'train_end_exclusive':self.train_end.isoformat(),'validation_end_exclusive':self.validation_end.isoformat(),'test_end':self.test_end.isoformat(),'inclusive_test_end':self.inclusive_test_end}


def rolling_origin_splits(series):
    if len(series.records)<2: raise ValueError('At least two timepoints required')
    lo,hi=series.records[0].grid_time,series.records[-1].grid_time
    span=hi-lo
    if not span.total_seconds()>0: raise ValueError('Positive elapsed duration required')
    return [Fold(i+1,lo,lo+span*a,lo+span*b,lo+span*c,c==1.) for i,(_,a,b,c) in enumerate(FRACTIONS)]


def write_splits(series,path):
    content={'site_id':series.site_id,'basis':'elapsed_time','folds':[f.manifest() for f in rolling_origin_splits(series)]}
    content['sha256']=canonical_hash(content)
    write_immutable_json(path,content)
    return content


def require_train_indices(series, indices, fold):
    if any(fold.partition(series.records[i].grid_time)!='train' for i in indices):
        raise ValueError('fit/scaler attempted to access validation/test')


def future_target_within_fold(issuance_time,horizon_seconds,fold):
    target=issuance_time+timedelta(seconds=horizon_seconds)
    return target>issuance_time and fold.partition(target)=='test'
