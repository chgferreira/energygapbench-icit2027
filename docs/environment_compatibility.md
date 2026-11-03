# Historical environment compatibility audit

**Historical evidence, not the current server setup.** Both conference profiles now
use a compatible shared active `cuda_env`; see [single environment](single_cuda_env.md).
Earlier outputs retain their original package versions and qualifications.

A minimum Python 3.12 environment was reproduced from `requirements.lock.txt` in a
clean environment. It exercised data/protocol/baseline/physics/statistics/recovery
behavior without requiring optional neural/foundation packages. Prerequisite skips
were explicit rather than replaced with simulated model outputs.

The earlier shared diagnostic `.venv` contained PyTorch 2.15.0, PyPOTS 1.5, and
MOMENT 0.1.4. Native checks passed in that stack, but MOMENT's exact declared NumPy,
Hub, and Transformers pins differed. Those results remain marked dependency overrides;
later successful environments do not retrospectively change their provenance.

Separate historical MOMENT/TS-ICL workers were subsequently checked under their
recorded environments with a no-truth file contract. The current conference path
instead uses official compatible MOMENT 0.1.5 source and TS-ICL 0.2.0 in one active
interpreter. Do not install a historical diagnostic lock over a working server
PyTorch/CUDA stack. Real-checkpoint CPU tests and actual V100 checks are distinct
levels of evidence. See [development status](development_status.md).

Optional checkpoint/source/environment directories remain local exclusions.
Component terms do not inherit the project's MIT license. Any package first-import
configuration is separate from scientific results; lightweight status/syntax checks
should not be mistaken for native training validation.
