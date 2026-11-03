# Aggregate results from the focused execution

Run `conference-20261101T165101Z-1789282` is represented by six unchanged original
aggregate exports in `reports/`: `main_table.csv`, `core_aggregate.csv`,
`selected_baseline_validation.csv`, `performance.csv`, `contrasts.json` and
`contrasts_block_sensitivity.json`. Original unavailable forecast values are
preserved. The separately versioned forecast follow-up is under
[`analysis/forecast_v2`](../analysis/forecast_v2/README.md).

`protocol.public.json` is an explicitly identified extract, not the original
lock. It includes scientific settings, declared initial runtime configuration
and hashes of the restricted original protocol. Later scheduling changed from
8 workers/16 pending tasks to 16/32; original session history remains restricted.

Episode records, runtime history, full dataset metadata, calendars and the
original completion manifest are excluded. This directory cannot resume the
server run or independently reproduce episode-level bootstrap sampling.
See [data availability](../DATA_AVAILABILITY.md). The public artifact manifest
lists included files; historical analysis manifests may reference restricted inputs.
