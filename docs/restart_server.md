# Start a new study or resume an existing one

Use the central entry point in the active `cuda_env`. A new run preserves previous
data and run directories and receives a new ID.

```bash
conda activate cuda_env
bash scripts/server.sh run --preset focused --threads 32 --budget-hours 24
```

For the expanded design, use `--preset full --budget-hours 168`. A larger budget
alone does not select full. Run setup for the selected profile when dependencies
or artifacts are missing. An already prepared installation can use selected native
validation. See the [server guide](../SERVER_GUIDE.md).

Request a safe stop and wait for exit before changing executable code/configuration.
To continue the same compatible scientific identity:

```bash
bash scripts/server.sh resume ID --threads 32
```

Resume uses the locked configuration and saved cumulative budget; no preset/config
change is accepted. `--budget-hours 48` extends the total ceiling, including already
accounted work. `scripts/restart_server.sh` is a compatibility wrapper for a **new**
run; it does not resume an existing ID or remove output directories.

Logs are under `artifacts/analyses/logs/ID.log`; durable state and reports are in
`runs/ID/`. Current-profile results remain work in progress until coverage, failures,
selection, and scientific assumptions have been reviewed. The
[profile comparison](experiment_profiles.md) and [TODO roadmap](development_status.md)
describe the intended scope and pending evidence.
