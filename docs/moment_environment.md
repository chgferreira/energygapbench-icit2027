# Historical isolated MOMENT audit

**Historical environment evidence.** The current focused/full conference paths
use official MOMENT source 0.1.5 in the active `cuda_env`. See
[single environment](single_cuda_env.md) and the [server guide](../SERVER_GUIDE.md).
This page preserves the older 0.1.4 audit and does not prescribe current installation.

MOMENT 0.1.4 declared exact NumPy 1.25.2, Hugging Face Hub 0.24.0, and Transformers
4.33.3 requirements, conflicting with the original NumPy 2/Python 3.12 diagnostic
stack. A separate Python 3.11.16 environment with PyTorch 2.5.1 was tested against
`environments/moment.lock.txt` (24 recorded packages), with a passing dependency
check. Earlier shared-environment override results were not retroactively certified.

The historical worker used the same locally pinned base checkpoint: revision
`5e44b0ea26376a176360f87831124e018f876d96`, SHA-256
`1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825`.
Weights, interpreters, and environments were local, excluded artifacts. Sharing
cached checkpoint bytes did not merge package environments.

`scripts/moment_worker.py` accepted a versioned NPZ/JSON contract with values,
times, observed/padding masks, query IDs, optional arrivals, and issuance/look-ahead
metadata. It required 512-position five-minute contexts and rejected extra truth
fields, future information, invalid cutoffs, and incompatible grids. Hidden values
were removed before model import; padding was not an observation or output target.

Outputs contained point estimates, imputation masks/IDs, hashes, runtime/checkpoint
provenance, and load/inference/RSS costs. Completion metadata was written last;
verified completed shards were reusable without loading torch. No evaluator truth
crossed the bridge. Parent subprocess wall time remained separate from child
inference time, including startup/model loading costs.

Tests compared native output, hidden-placeholder poisoning, observed preservation,
and compatible reuse. They established synthetic integration under the recorded
historical environment, not real-site superiority or V100 performance. Current
common-process batching and dependency compatibility have their own checks and
identities. Remaining scientific/server validation is tracked in
[development status](development_status.md).
