"""Exercise real optional model wrappers on a labeled CPU fixture; writes truthful status."""
from pathlib import Path
import json
import traceback
import numpy as np
from energygap.methods.neural import PyPOTSAdapter


def audit(output=Path('artifacts/reports/learned_adapter_audit.json')):
    rng=np.random.default_rng(3)
    n=24
    train=np.tile(np.sin(np.arange(n)/5)[None,:,None],(8,1,1)).astype(np.float32)+rng.normal(0,.01,(8,n,1))
    query=train[0].copy();query[10:13]=np.nan
    records=[]
    for name in ('SAITS','BRITS','CSDI','ImputeFormer'):
        try:
            model=PyPOTSAdapter(name,context=n,epochs=2).fit_windows(train)
            result=model.infer(query,samples=10,sample_seed=19)
            direct=model.model.predict({'X':((query-model.mean)/model.scale)[None]},**({'n_sampling_times':10} if name=='CSDI' else {}))['imputation']
            # For deterministic methods compare to direct upstream on exactly the same masked input.
            parity=None
            if name!='CSDI':
                if hasattr(direct,'detach'):direct=direct.detach().cpu().numpy()
                native=np.asarray(direct)[0]*model.scale+model.mean
                parity=bool(np.allclose(native[10:13],result['point_estimate'][10:13]))
                assert parity
            assert np.array_equal(result['point_estimate'][np.isfinite(query)],query[np.isfinite(query)])
            records.append(dict(method=name,implementation_status='integrated',validation_status='tiny_cpu_mask_shape_observed_preservation_and_upstream_parity',D1=True,context=n,epochs=2,samples=10 if name=='CSDI' else None,joint_samples=result['samples'] is not None,upstream_parity=parity,fit_seconds=model.timings['fit_seconds'],training_resume=False,scientific_benchmark_run=False))
        except Exception as exc:
            records.append(dict(method=name,implementation_status='unavailable',validation_status='failed',reason=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),scientific_benchmark_run=False))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))
    return records


if __name__=='__main__':audit()
