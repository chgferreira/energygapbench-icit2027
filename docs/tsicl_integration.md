# Historical isolated TS-ICL audit

**Historical record.** The current full conference profile uses official TS-ICL
0.2.0 in the shared active `cuda_env`; focused defers TS-ICL. The audited imputation
pipeline is unchanged from 0.2.1, but actual package versions are recorded correctly.
Use [single environment](single_cuda_env.md) and the [server guide](../SERVER_GUIDE.md)
for current operation. This page is not a second installation procedure.

The historical isolated integration used `tsicl==0.2.1`, commit
`61d1e891a4aeab4b8aa4833c57480025c225b357`, and pipeline Git blob
`acd8c53f627e765429ba812d4ce14791c407c14d`. The pinned checkpoint revision is
`19c94031439fb31f36ce395088ee50a6762d3774`, file `tsicl-v1.ckpt`, SHA-256
`a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431`.
Original code/checkpoint noncommercial terms remain attached; common-environment
execution does not change them. Consult [third-party notices](../THIRD_PARTY_NOTICES.md).

The historical Python >=3.12 isolated runtime used PyTorch 2.9.1 and its recorded
`env/tsicl.lock`. Its worker accepted only values, measurement/arrival times, and
issuance metadata; extra/truth arrays and implicit downloads were rejected.
Queries were univariate, L=512, regular five-minute contexts cut at reconnection.
The checkpoint's **imputer**, not its forecaster, performed reconstruction.

Native `point_estimator='median'` returned quantile .5; the requested marginal grid
was .05/.10/.25/.50/.75/.90/.95. Observed values were preserved and normalization
used admitted context only. These are not joint trajectories and do not support
trajectory-wise energy correction. The truncated `2*trapezoid(pinball)` CRPS
approximation omits tails; it is not full or sample CRPS. Intervals 50/80/90% were
available; 95% remained NA because .025/.975 endpoints were not requested.

A separate real supplement reused the 18 small integration-pilot episodes: three
sites, fold 1, validation/test, C3–C5, one mask/episode per cell. It reconstructed
inputs from verified **observed** positions, turning all other values into NaN;
MOMENT imputed values were never observations for TS-ICL. Truth was opened by the
evaluator only after outputs were committed. The evaluator recomputed training-only
nMAE scales; those were not passed to the frozen imputer. No local training seeds
were invented.

The private audit `artifacts/reports/tsicl_real_pilot.json` records 18/18 successful
inferences/evaluations, costs, descriptive metrics, and hashes. It does not establish
calibration, general superiority, forecasting, or energy-correction performance.
The isolated worker's committed-shard evidence does not alone certify the later
conference coordinator. Current integration and outstanding tests are described in
[model status](model_integration_status.md) and [development TODO](development_status.md).
