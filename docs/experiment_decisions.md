# Experiment decisions and provenance

This log records the rationale and status of major design changes. Run protocol
locks and manifests retain actual hashes/settings; this page does not replace
those artifacts or claim external preregistration. Historical outputs are not
relabelled as results of later source versions.

## 31 October 2026: provenance and initial constraints

Original telemetry bytes and explicit site mapping were preserved. Units W, negative
import, and America/Sao_Paulo timezone were accepted from supplied internal evidence,
not manufacturer calibration. Instantaneous/interval-mean and timestamp-placement
semantics remained unknown, so confirmatory physical claims were blocked. Small
P pilots and explicitly exploratory energy hypotheses were kept separate.

The initial synthetic/real diagnostic subset used capped episodes/windows and short
native fits. It tested integration, pairing, and selected recovery paths rather
than scientific convergence. Historical shared MOMENT outputs had dependency
exceptions; separate isolated MOMENT/TS-ICL audits later checked their own official
versions. Those environment qualifications remain attached to earlier results.

Analytic signed-energy gradients and deterministic multi-start optimization replaced
slow finite-difference evaluation while retaining the stated general objective.
The separately named CPI interval-energy adaptation was not presented as exact
upstream reproduction. Known zeros/constant targets and negative outcomes were
retained.

## 31 October 2026: expanded server executor

A new `server_full.yaml` identity declared uncapped C1–C6, three folds/sites,
local training seeds 11/23/47, five validation and ten test masks, six fixed HPO
candidates per trainable model/site on the first fold, full core validation epoch
selection, and four ridge horizons. Five local architectures received completed-
epoch optimizer/RNG/best-state recovery; frozen models received no fictitious
training seeds. T1 training orchestration was explicitly adapted.

Correction selection averaged seed losses before episode/duration aggregation.
CSDI point and trajectory-corrected variants received distinct identities. The
four unavailable catalogue candidates stayed explicit. Earlier pilots were not
promoted into the new study. Serial `cuda:0`, requested CPU threads, source/settings,
model artifacts, and physical hypotheses were recorded. CPU tests did not certify
CUDA or predict expanded-matrix runtime.

## 1 November 2026: compatible common environment and execution optimization

The supported server path adopted one active `cuda_env`, official compatible
MOMENT 0.1.5 source and TS-ICL 0.2.0, without rewriting package metadata or replacing
the working CUDA PyTorch/NVIDIA runtime. The audited native inference code and
pinned checkpoint identities remain explicit. Historical isolated paths are
records rather than current setup instructions.

Deterministic inference requests 256-query batches with recorded OOM halving;
training stays batch 32. BRITS sanitized-input caching and named vectorized FP32
execution, MOMENT batching, bounded CPU correction workers, ridge/forecast batching,
and the exact single-net interval-mean shortcut reduce repeated work. Score reuse
requires the exact checkpoint and complete validation coverage. SQLite durability
and scientific commit boundaries remain intact. These changes required new source/
execution identities, not reuse of incompatible prior training state.

## 1 November 2026: focused design under a constrained compute budget

Observed recurrent-model cost and a requested 24-hour operational budget motivated
a separately declared `server_focused.yaml` design. It selects six baselines,
SAITS, and frozen MOMENT on C3–C5. HPO uses indices 0/1 with 20-epoch/patience-5
ceilings; final fits retain 100/10 and three seeds. A fixed metadata-only panel of
at most 20 episodes per duration/mask seed (300 per site/fold) selects epochs.
Full HPO ranking, correction selection, and final validation/test remain complete
within the selected scope.

BRITS, CSDI, ImputeFormer, T1, TS-ICL, and C1/C2/C6 were deferred by task relevance
and compute cost, not rejected by final test accuracy. Existing unavailable-model
blockers stayed distinct. Sites/folds, evaluation repetitions, context, uncapped
windows/episodes, horizons, hypotheses, and bootstrap settings were retained.
Reduced HPO/panel selection is a scientific tradeoff; sensitivity and architecture
transfer remain open. The focused default does not convert existing full runs.

The audited counts changed from 225 to 33 fits and 14,179,485 to 905,480 confirmable
queries. These are plans, not speedup or completion evidence. The shell defaults
to focused/24 hours; explicit full retains its YAML and 168-hour ceiling. Custom
YAMLs remain available. Both require honest scope and coverage reporting.

## 1 November 2026: public documentation organization

Documentation was consolidated in English around the two profiles, with explicit
work-in-progress status, historical audit labels, current operating instructions,
and [prioritized TODO acceptance criteria](development_status.md). The initial
requirements document became an English scope/traceability summary; its source
reference inventory remains available in [references](references.md). This editorial
refresh does not change Python implementation, experiment YAMLs, scientific results,
or runtime identities.

For future scientific changes, record the reason, affected cells, prior/new run
identities/settings/hashes, timing relative to test inspection, and exploratory versus
confirmatory status. Editing a saved protocol lock in place is not a valid continuation.

## 1 November 2026: audited CPU scheduling continuation

Spare memory and long CPU post-processing motivated explicit per-session correction
worker/queue controls. The numerical models/correction routines remain unchanged.
An exact predecessor/target source manifest authorizes a restricted continuation
that retains the original configuration, fingerprint/RNG identity, records, and
models. It checks remaining fields, holds the run lock, backs up SQLite, and records
new executable/session lineage. Arbitrary mismatches are still rejected. Correction
rates are recalibrated; actual CPU/GPU telemetry is separate from scientific timing.
The CPU predecessor-to-target acceptance test preserves committed payloads, future
scientific outputs, and future native fit weights. Server throughput remains to be
measured; memory occupancy alone is not evidence for a speedup.

## 2 November 2026: report execution and progress continuation

Long final aggregation/bootstrap work motivated an execution-only reporting update.
It uses bounded CPU bootstrap workers and ordered parent publication, larger
operational publication batches, streamed export progress, and current-stage
timing estimates. Replica IDs, per-replica RNG, numerical reductions, invalid
attempts, contrast definitions, and declared support are retained. Existing
confirmed model results and partial bootstrap draws are reusable.

The update also removes repeated shard reads during recovery, streams shared
artifact checksum verification once per unchanged version within a pass, and
identifies recovery progress. It does not bypass integrity verification. Its
explicit source/target registry, lock, SQLite backup, and runtime history preserve
the original frozen identity. New execution settings require new timing
calibration, not a new scientific design. Target-server timing and scientific
review remain required; see [report execution](reporting_performance.md).
