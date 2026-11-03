# T1 integration: current and historical paths

T1 is selected in full and deferred in focused. The model is loaded unchanged from
[official T1](https://github.com/Oppenheimerdinger/T1), commit
`823b76aab57a2e60eb5029caab06fdfec8b19256`, model Git blob
`3e7af0da16f03b2a6fb0afbac78c1b2079b10585`, SHA-256
`3f87aac66745029a88907b7ce65766d3ce5f785a6d4190f6343f9649d4ad3de8`.
The original MIT notice stays beside local downloaded source. There is no supplied
external pretrained checkpoint or fabricated pretraining claim.

`scripts/fetch_t1.py` retrieves pinned public model/license files and verifies the
model digest, refusing divergent replacements. Downloaded source is local/excluded
and must be prepared for a clean full installation. Missing/incorrect files produce
explicit availability errors; they do not trigger another imputer under T1's name.

## Current conference path

The conference adapter uses the original architecture and native mask-aware
normalization, with training-only channel scaling and unavailable placeholders
removed before upstream calls. Received observations are restored. L=512/D=1 is
supported by local native-parity checks, but a single channel cannot test the
claimed multivariate-dependency advantage. Sites are not invented as simultaneous
channels. Causal task C remains unsupported.

The native loss uses artificially hidden-point MSE (MIT) with optional observed
MSE (ORT); natural gaps are not targets. Protocol orchestration deliberately adapts
masking/validation/training schedule rather than reproducing every source-paper
experiment. The current conference epoch loop has completed-epoch optimizer/RNG/
best-state recovery tests on CPU. Its full-profile HPO and final-fit budgets follow
[profiles](experiment_profiles.md). CUDA recovery and real convergence remain TODO.

## Historical standalone adapter

The earlier `T1Adapter` diagnostic used step-limited Adam with constant learning
rate and mixed masks; it did not reproduce the upstream schedule/test-logging
procedure. Small context-16 overfit checks demonstrated learning/native parity,
observation preservation, training isolation, and completed model save/load, not
scientific superiority. `save_fitted` stored weights/scaler/config/source provenance.
That standalone path had `supports_resume_training=False` and completed-fit/query
granularity. Its limit must not be confused with the later conference epoch loop.

The historical evidence is under `artifacts/reports/t1_integration_audit.json`, not
bundled benchmark results. See [recovery](resume_and_recovery.md),
[integration status](model_integration_status.md), and [development TODO](development_status.md).
