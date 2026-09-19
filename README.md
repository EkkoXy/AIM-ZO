# AIM-ZO

Activation-Informed Subspace Maintenance for Zeroth-Order LLM Fine-Tuning.

## Status

This is a submission-repository scaffold. Algorithm implementations, verified dependencies, frozen experiment configurations, and validated results have not yet been imported. It is not currently a runnable reproduction package.

## Layout

- `src/aimzo/`: method implementations, trainers, tasks, evaluation, and diagnostics.
- `configs/`: main experiments, baselines, ablations, and diagnostics.
- `scripts/`: future data preparation, training, evaluation, and table-generation entry points.
- `results/`: compact, evidence-linked official results; no model checkpoints.
- `figures/`: reproducible paper figures.
- `tests/`: implementation and artifact-validation tests.
- `docs/`: protocols, provenance, reproduction instructions, and implementation differences.
- `requirements/`: dependencies to be populated from the verified execution environment.

## Naming

The public method name is **AIM-ZO**; the Python package is `aimzo`.
Historical artifacts may use OSZO or MyZO. Keep original artifacts unchanged and record the mapping in the result manifest. Do not assume every historical configuration is part of the final method.

## Reproduction

No training command is advertised until the imported implementation has passed a clean-environment smoke test. See [the reproduction checklist](docs/reproduction.md).

## Results

Each published experiment is a self-contained unit:
`results/<category>/<model>/<task>/<method>/seed<seed>/`.
See [the artifact contract](docs/result-ledger.md). Templates are not experimental evidence.

## License and citation

A project license and citation metadata will be added after rights and anonymous-submission requirements are reviewed. Preserve all applicable third-party licenses and attribution. No project-wide license is granted by this scaffold.

