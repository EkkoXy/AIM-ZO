# Experiment scripts

The package installs four command-line entry points:

- `aimzo-prepare-data`
- `aimzo-train`
- `aimzo-eval`
- `aimzo-qa-eval`

`train.sh` and `evaluate.sh` are minimal shell wrappers. The Python entry
points write resolved configs, manifests, histories, checkpoint records, and
evaluation JSON files under the configured output directory.
