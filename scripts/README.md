# Planned entry points

No executable training scripts are supplied yet.

- prepare_data.sh: prepare public datasets using documented splits.
- run_main_results.sh: execute frozen main configurations.
- run_baselines.sh: execute selected baseline configurations.
- run_ablations.sh: execute ablation configurations.
- run_diagnostics.sh: execute offline diagnostic configurations.
- evaluate_official.sh: evaluate a dev-selected checkpoint on the official split.
- reproduce_tables.py: validate the registry and export CSV/LaTeX.

Implement these after code import, with explicit configuration arguments, nonzero failure exits, and no embedded credentials or machine-specific paths.

