# Official result registry and artifact contract

Status: empty. No verified results have been imported.

## Unit

`results/<main|baselines|ablations|diagnostics>/<model>/<task>/<method>/seed<seed>/`

Each unit contains:
- `config.yaml`: exact effective configuration, including overrides.
- `manifest.json`: identity, provenance, code revision, budget, hardware and evidence references.
- `metrics.json`: normalised metrics with explicit units and split.
- `official_eval.json`: compact official-evaluation evidence.
- `training_summary.json`: completed steps, best dev checkpoint, stopping status and elapsed time.

Templates are in `results/manifests/templates/`, outside result directories. Null means unknown, never zero. Do not use templates as table inputs.

## Provenance and anonymisation

Record both the internal source revision and the released implementation revision through an internal mapping. Public manifests should contain only an anonymity-reviewed revision/reference. Do not publish an internal repository URL or commit reference that reveals authorship.

Keep raw evidence immutable internally. Export redacted copies where needed and hash the actual distributed evidence. If both raw and redacted hashes are retained, label them separately. SHA-256 alone is not a substitute for accessible evidence.

## Aggregation

A future exporter must validate matching configurations and evaluation protocols, reject duplicates, and group by explicit run identity. Export CSV and LaTeX from the same registry.

Use sample standard deviation (ddof=1); n=1 has no estimated sample standard deviation. Keep percentage points distinct from relative percentages. Calculate standard deviation of a cross-task average from aligned per-seed aggregates, not by averaging task standard deviations.

Every table must identify its included run IDs and metric definitions. Unknown and unverified records must not silently enter official tables.

