# Result records

Experimental results have not yet been added. Format templates are provided under `results/manifests/templates/`.

## Directory layout

`results/<category>/<model>/<task>/<method>/seed<seed>/`

Categories are main experiments, baselines, ablations, and diagnostics.

## Record format

| File | Contents |
| --- | --- |
| `config.yaml` | Effective experiment configuration |
| `manifest.json` | Run identity, code revision, evaluation budget, hardware, and artifact references |
| `metrics.json` | Metrics, units, and evaluation split |
| `official_eval.json` | Official-evaluation output |
| `training_summary.json` | Completed steps, selected checkpoint, stopping status, and elapsed time |

A null value denotes unavailable information, not zero. Template files are not inputs to result tables.

## Reporting conventions

Results are grouped by configuration, evaluation protocol, and checkpoint-selection rule. Mean scores are accompanied by the number of seeds and, when at least two seeds are available, the sample standard deviation.

Percentage-point differences are distinguished from relative percentage changes. Uncertainty for a cross-task average requires aligned per-seed results and is not obtained by averaging task standard deviations.

Artifact references and checksums identify the evidence associated with each reported run.
