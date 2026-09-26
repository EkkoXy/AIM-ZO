# Experiment scripts

The package installs four command-line entry points:

- `aimzo-prepare-data`
- `aimzo-train`
- `aimzo-eval`
- `aimzo-qa-eval`

`train.sh` and `evaluate.sh` are minimal shell wrappers. The Python entry
points write resolved configs, manifests, histories, checkpoint records, and
evaluation JSON files under the configured output directory.

Paper reproduction utilities:

- `generate_main_configs.py` validates and expands the main protocol registry.
- `run_main_experiment.py` resolves and runs one registry entry; `--dry-run`
  prints exact numerical settings without loading a model.
- `aggregate_official_results.py` aggregates official metrics with sample
  standard deviation and preserves paired F1/EM metrics.
