"""Native upstream parity test, skipped on minimal/CI installs without cached weights."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest

_CACHE=Path('artifacts/checkpoints/hf/models--AutonLab--MOMENT-1-base')
_HAS=bool(importlib.util.find_spec('momentfm')) and _CACHE.exists()


@pytest.mark.skipif(not _HAS,reason='optional momentfm + verified local official checkpoint unavailable')
def test_moment_native_masked_output_parity_and_partial_patches():
    import torch
    from momentfm.utils.masking import Masking
    from energygap.methods.foundation import MOMENTFrozen,prepare_moment_input
    torch.set_num_threads(1)
    x=np.sin(np.arange(512)/30)+.2*np.cos(np.arange(512)/7)
    x[497:511]=np.nan;t=np.arange(512)*300.+1e9
    model=MOMENTFrozen();actual=model.impute(x,t)
    batch=prepare_moment_input(x)
    with torch.inference_mode():
        native=model.model.reconstruction(**{k:torch.as_tensor(v) for k,v in batch.items()}).reconstruction[0,0].cpu().numpy()
    missing=~np.isfinite(x)
    np.testing.assert_array_equal(actual[~missing],x[~missing])
    np.testing.assert_allclose(actual[missing],native[missing],atol=0,rtol=0)
    assert np.isfinite(actual).all()
    # Patch 62 includes admitted index496 and hidden497:503: whole patch is masked.
    patch=Masking.convert_seq_to_patch_view(torch.as_tensor(batch['mask']))
    assert patch[0,62].item()==0 and patch[0,61].item()==1
    assert all(not parameter.requires_grad for parameter in model.model.parameters())
