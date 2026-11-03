# Full experiment profile

**Work in progress.** Full is the expanded available-method design, explicitly
selected with `--preset full`. Its YAML is
[`server_full.yaml`](../configs/experiments/server_full.yaml). The server entry
point defaults to focused; merely setting a larger budget does not select full.

## Declared scope

| Dimension | Full configuration |
| --- | --- |
| Sites / chronological folds | H1, H2, I1 / 1, 2, 3 |
| Conditions | C1–C6 |
| Conventional baselines | median, locf, linear, seasonal, kalman_filter, kalman_smoother |
| Locally trained models | BRITS, SAITS, CSDI, ImputeFormer, T1 |
| Frozen models | MOMENT-1-base, TS-ICL |
| Training seeds | 11, 23, 47 for each locally trained model |
| Mask replicas | Five validation; ten test |
| HPO | Six fixed candidates per trainable model/site on the first fold |
| HPO and final training | At most 100 epochs; patience 10; batch 32 |
| Epoch selection | All declared C3–C5 validation episodes |
| Context / training stride | 512 / 288 |
| Episode / training-window cap | None |
| CSDI | 100 joint trajectories per query; 50 native diffusion steps |
| Ridge forecasting | 30 min, 1 h, 3 h, 6 h |
| Energy on C3–C5 | Three explicitly exploratory measurement hypotheses |
| Bootstrap | 10,000 attempts; 7-day blocks; 2/14-day sensitivities |

The YAML declares 17 catalogue methods: 13 executable methods if all required
integrations pass preflight, and four unavailable candidates (FIM, TabPFN-TS, MoTM,
CBiT). Their unresolved statuses remain visible. An available-method run does not
constitute reproduction of all 17 entries.

## Execute or plan

```bash
conda activate cuda_env
bash scripts/server.sh setup --preset full --threads 32
bash scripts/server.sh run --preset full --threads 32 --budget-hours 168
```

For a plan without training, pass the YAML explicitly:

```bash
PYTHONPATH=src python -m energygap plan --profile conference \
  --config configs/experiments/server_full.yaml --device cuda:0 \
  --threads 32 --budget-total-hours 168
```

Planning reads and verifies data/artifacts, generates manifests, and writes a run
plan; it is not a zero-cost syntax check. The audited private-data plan contains
90 HPO fits, 135 final fits, and 14,179,485 confirmable main/HPO queries. Corrected
variants and multi-horizon exports add output and disk requirements. Reporting is
disk-backed, but local free space and actual full-run scaling still need measurement.

The 168-hour default is a cumulative operational ceiling, not an ETA or a guarantee.
One GPU job runs at a time. CPU thread pools and correction workers do not turn this
into distributed HPO. Follow the [server guide](../SERVER_GUIDE.md) for monitoring
and compatible resume; `scripts/run_server_full.sh` retains explicit full behavior.

## Interpretation and pending validation

Full broadens evidence across architectures and missingness patterns. It still
requires convergence/selection review, temporal support checks, fallback audits,
CUDA parity/recovery tests, and complete result analysis. CSDI's transformed samples
do not guarantee calibration; TS-ICL marginal quantiles are not joint trajectories.
Unknown instrument semantics keep energy branches exploratory. Unsupported results
remain NA, and absent methods must not appear as reproduced algorithms.

Read the [profile comparison](experiment_profiles.md), [limitations](limitations.md),
and [development TODO](development_status.md) before using this scope in a manuscript.
