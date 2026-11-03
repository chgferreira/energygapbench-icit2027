"""Batch failure isolation and throughput metadata use real evaluator boundaries."""
from types import SimpleNamespace

from energygap.conference import _query_outputs
from energygap.runtime.signals import StopRequested
from energygap.pipeline import GracefulStop
import pytest


def test_failed_batch_retries_each_query_and_only_marks_the_invalid_one():
    calls=[]
    class Adapter:
        def infer_queries(self,queries,**kwargs):
            calls.append(('batch',list(queries),kwargs))
            raise ValueError('one corrupt item')
        def infer_query(self,query,**kwargs):
            calls.append(('scalar',query,kwargs))
            if query=='bad':raise ValueError('invalid query')
            return dict(point_estimate=[query])
    outputs=_query_outputs(Adapter(),['a','bad','b'],[11,23,47],
                           {'samples':100,'inference_batch_size':32},StopRequested())
    assert outputs[0]['point_estimate']==['a'] and outputs[2]['point_estimate']==['b']
    assert outputs[0]['timing']['batch_retry_scalar'] is True
    assert 'invalid query' in outputs[1]['error']
    assert [call[2]['sample_seed'] for call in calls if call[0]=='scalar']==[11,23,47]


def test_signal_during_batch_never_becomes_an_operational_failure():
    stop=StopRequested()
    class Adapter:
        def infer_queries(self,queries,**kwargs):
            stop.requested=True
            raise RuntimeError('interrupted native call')
        def infer_query(self,*args,**kwargs):
            pytest.fail('requested stop cannot retry')
    with pytest.raises(GracefulStop):
        _query_outputs(Adapter(),[object(),object()],[1,2],
                       {'samples':100,'inference_batch_size':32},stop)
