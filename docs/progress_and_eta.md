# Progress, status, and ETA

Progress is reconstructed from committed state, not estimated from files appearing
on disk. `runs/ID/state.sqlite` is authoritative. `progress.json`, protocol locks,
checkpoints, and event logs provide supplementary audit information. Status does
not construct models or require GPU inference.

## Monitor either profile

```bash
conda activate cuda_env
PYTHONPATH=src python -m energygap status --experiment-id ID --watch
PYTHONPATH=src python -m energygap status --experiment-id ID --json
tail -f artifacts/analyses/logs/ID.log
```

The launch prints ID and paths. The main display separates:

1. **Confirmed jobs:** resolved durable work, with failures/unavailability distinguished.
2. **Committed queries:** main inference and final HPO scoring records published at safe boundaries.
3. **Current activity:** model, site, fold, training seed, HPO ordinal, epoch, batch,
   and validation progress, including a fixed-panel indication when relevant.

During reporting, the current-activity line names the export, aggregation,
bootstrap, or publication stage. Bootstrap details include the site, measurement
hypothesis, paired contrast, endpoint, block length, committed replica count, and
completed bootstrap tasks when supplied by the report planner. Export counters are
records written at a safe boundary; bootstrap counters are committed attempted
replicas, including invalid attempts. Neither advances the inference-query bar.
An unknown stream length is displayed as `?`, not as a made-up total.

The same compact metadata is stored as `report_progress` and exposed in status;
the current ETA also contains `current_report`. A report heartbeat or progress
tick does not scan result records. Presentation updates use the normal bounded
refresh/persistence intervals; stage boundaries and scientific commits force
their own durable updates. A resumed process can reuse saved compatible rates,
but its first counter tick establishes a new clock baseline so offline time is
never added to measured throughput.

The audited focused plan has 905,480 confirmable queries and the full plan has
14,179,485. These totals depend on supplied data/configuration and do not count
repeated epoch validation as independent committed queries. Training, validation,
correction, and reporting can consume substantial time while the query bar is static.
The panel normally has 300 episodes in focused; full epoch validation uses all
configured C3–C5 episodes. HPO ranking/final evaluation remains complete in both.

## Interpret estimates

A query fraction is a work-count fraction, not elapsed-time completion. For example,
4,400/905,480 is about 0.49%; multiplying elapsed time by 206 assumes equal costs
and a representative initial phase, neither of which is established. Preparation
and HPO concentrate cost early; later models or correction can have different costs.

Phase ETA uses compatible observed task rates. Unmeasured phases keep total ETA
unknown; partial ETA covers only calibrated work. Current-fit ranges describe
observed epoch rates under alternative early-stopping/epoch-limit outcomes, not a
calibrated confidence interval or a promised completion time. Focused's 24 hours and
full's 168 hours are cumulative operational ceilings, not ETAs.

Reporting measures wall time between published counters, including file/ledger
commit and coordinator overhead. Export streams and bootstrap timing families
have separate rates; compute-only worker service time is not used as report wall
time. At least two positive progress observations are required before a remaining
stage duration is shown. Unknown totals, new families, and unmeasured publication
work remain unestimated. A bootstrap-only remaining estimate requires its complete
remaining plan and calibration of every remaining bootstrap family. An estimate
for the whole report additionally requires a complete plan covering all remaining
exports, plots, and final publication stages, with compatible rates for every
family. Otherwise the display explicitly says the report total is unknown, even
when a current-stage estimate is available. The displayed observed min/max range
is a timing heuristic, not a confidence interval or completion guarantee.

Worker counts for energy correction do not parallelize statistics automatically.
Report CPU scheduling is a separate operational setting recorded with the run.
The GPU can be idle while SQLite aggregation, bootstrap, export, and plotting
are active. A completed inference-query bar therefore does not mean reporting
has finished; wait for the final operational status and report publication.

The reusable estimator supports rate histories and resource/dependency-aware
scheduling, but the main GPU executor is serial. Do not divide work by CPU core
count or infer distributed makespan. CPU correction service costs can overlap;
summing worker service seconds does not give wall time. VRAM allocation does not
measure GPU activity or pipeline throughput.

## Persistence and recovery details

Default terminal/progress/status/heartbeat intervals are 1/10/60/15 seconds.
Scientific commits and completed-epoch checkpoints retain their own safe boundaries;
final updates are forced. `tqdm`, `log`, and `none` change presentation only.
Temporary files or unfinished epochs do not advance durable scientific work.

States include pending, running, interrupted, succeeded, failed_retryable,
failed_terminal, unavailable, unsupported, and blocked. Terminal resolution is not
scientific success; `completed_with_failures` requires failure/coverage review.
Offline time is excluded from measured throughput. After a crash, conservative
budget reservations can remain charged as uncertain work and are not measured fit
time. Read [resume and recovery](resume_and_recovery.md).

Controlled tests cover first-render totals, resumed cursors, offline ETA behavior,
resource dependencies, progress-backend invariance, and cumulative accounting.
Server phase calibration and complete-run ETA accuracy remain TODO in
[development status](development_status.md).

Reporting tests also cover incremental stage visibility, distinct rate families,
bounded metadata work, preservation of inference-query counts, wall/commit timing,
unknown totals, and restart clocks without charging offline intervals.
