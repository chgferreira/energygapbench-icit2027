# Model integration status

**Work in progress.** Implementation, environment compatibility, numerical checks,
and scientific completion are different evidence levels. Native conference models
share one active `cuda_env`; selected profile and preflight determine which run.

| Method | Focused | Full | Implementation and current evidence |
| --- | --- | --- | --- |
| Six conventional baselines | Selected | Selected | Training-only fitted references; protocol/physics tests |
| SAITS | Selected | Selected | PyPOTS 1.5 native loss/architecture, CPU completed-epoch recovery and small-fit checks |
| BRITS | Deferred | Selected | PyPOTS native reference plus named FP32 vectorized path; local parity/loss/gradient/update and epoch-recovery tests |
| CSDI | Deferred | Selected | Native diffusion and joint samples; local sampling/recovery checks |
| ImputeFormer | Deferred | Selected | Native PyPOTS architecture/loss and CPU epoch-recovery checks |
| T1 | Deferred | Selected | Pinned untouched upstream model with conference epoch orchestration/recovery |
| MOMENT-1-base | Selected, frozen | Selected, frozen | Official source 0.1.5 and pinned checkpoint; CPU native/poisoning/cutoff/reuse/batch checks |
| TS-ICL | Deferred | Selected, frozen | Official 0.2.0 and pinned checkpoint; CPU native median/marginal-quantile and masking checks |
| FIM | Outside scope | Unavailable | Temporal adapter contract; checkpoint license unresolved |
| TabPFN-TS | Outside scope | Unavailable | R24 wrapper inspected; exact licensed implementation/checkpoint unresolved |
| MoTM | Outside scope | Unavailable | Required weight/config bundle and terms not verified |
| CBiT | Outside scope | Unavailable | Usable author-linked implementation not verified |

The authoritative component metadata are in `model_registry.yaml`. Deferred
integrated methods are not failed or inferior methods; unavailable catalogue
candidates have separate blockers and no fabricated substitutes. Full declares 17
entries, of which 13 are executable subject to preflight; focused selects eight.

Conference local training uses balanced point/block masks and native losses. Full
uses six HPO candidates and full core epoch validation; focused uses two shorter
HPO candidates and a fixed metadata-only epoch panel. Final HPO ranking and final
evaluation retain full declared support. Model/optimizer/RNG/best-state checkpoints
are written at completed epochs for each of the five supported local models.
An interrupted partial epoch can repeat; arbitrary internal-step recovery is not
claimed. Frozen models have no local training seed and use committed query shards.

MOMENT uses masked reconstruction at L=512 with observed-only normalization;
hidden placeholders are removed before upstream code. TS-ICL uses the checkpoint's
imputer, not its forecaster; quantiles are marginal rather than joint trajectories.
The common-environment source/package audit and license qualifications are in
[single environment](single_cuda_env.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).

Historical [model](model_pilot.md), [MOMENT](moment_environment.md), and
[TS-ICL](tsicl_integration.md) pilots retain their earlier package/settings and
recovery limits. They are not full-profile results. V100 parity/recovery, whole-run
coverage, convergence, fallbacks, and scientific effects remain subject to the
[TODO roadmap](development_status.md).
