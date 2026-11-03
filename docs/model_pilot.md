# Historical model integration pilot

**Historical evidence, not a current-profile execution or setup guide.** These
small pilots preceded the focused/full designs and common `cuda_env`. Use the
[server guide](../SERVER_GUIDE.md) for current operation and
[development status](development_status.md) for pending evidence.

`scripts/pilot_models.py` produced separate integration supplements under
`runs/real-models-pilot-v2/` and `runs/real-models-pilot-recovery/`. BRITS, SAITS,
CSDI, ImputeFormer, official T1, and frozen MOMENT received the same L=512 R/P
queries on three sites, first fold, C3/C4/C5, one mask, and at most one episode per
validation/test cell: 108 queries per execution. This was not the conference matrix.

PyPOTS used 16 chronological training windows and two epochs. Historical T1 used
50 explicitly adapted steps. CSDI retained 50 diffusion steps but used ten joint
samples here, not the current full profile's 100. MOMENT had one shared frozen
load, without invented local training seeds. No energy correction, downstream
forecast, convergence, or confirmatory comparison was established.

The plan contained 15 local fits, one frozen load, 108 queries, and up to 198
optimizer steps. A 360-second cumulative budget was operational accounting, not
an imputation latency estimate. Native fits were indivisible: an interrupted fit
restarted, while completed fits/query commits could be reused. Stored final
optimizer/RNG information was not a certification of exact internal native-fit
resume, particularly when best weights and final optimizer differed.

A real SIGTERM test during H1/BRITS inference resumed with its displayed query
cursor at 2/6, then completed 124 durable units across two sessions without duplicate
keys. Earlier artifact/checkpoint hashes remained intact; measured cumulative work
was 45.065 seconds. A provisional reservation was replaced by measured duration
on clean completion. The local audit is
`artifacts/reports/model_pilot_recovery.json`; it is not included in source packages.

The original shared `.venv` MOMENT execution diverged from upstream NumPy,
Transformers, and Hugging Face Hub pins. Those outputs retain a dependency-override
qualification. A separate [isolated audit](moment_environment.md) later validated
that historical version under its declared pins. The current common-environment
integration is a further, separately identified implementation, not a retrospective
relabeling of the pilot. Private telemetry, truth, weights, and outputs remain local.
