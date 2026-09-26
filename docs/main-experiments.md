# Main experiments

`configs/main/experiments.yaml` is the machine-readable paper protocol. It
records every released model/method/task group, seeds, precision, learning
rate, perturbation scale, checkpoint schedule, and objective-call budget.
`scripts/generate_main_configs.py` expands it into one training YAML per task
and seed.

Generate all resolved configurations, or validate without writing:

```bash
python scripts/generate_main_configs.py
python scripts/generate_main_configs.py --check
```

Filters are comma-separated and can be combined:

```bash
python scripts/generate_main_configs.py \
  --protocol opt13-agzo --task rte --seed 42 --check
```

Run one experiment after supplying local paths:

```bash
python scripts/run_main_experiment.py \
  --protocol opt13-agzo --task rte --seed 42 \
  --model-path /path/to/opt-13b --data-root /path/to/dataset
```

Add `--dry-run` to inspect the resolved precision, learning rate, epsilon,
forward-call budget, checkpoints, and output directory without loading a model.

## Budget matching

| Method | Updates | Calls per update | Total |
|---|---:|---:|---:|
| MeZO, CurvZO, LoZO | 20,000 | 2 | 40,000 |
| AIM-ZO | 2,500 | 16 | 40,000 |
| AGZO, HiZOO | 13,333 | 3 | 39,999 |
| ZO-Muon | 8,000 | 5 | 40,000 |

AIM-ZO uses fifteen one-sided probes and one center evaluation. Precision is
BF16 unless an executed protocol explicitly records FP16. Qwen3-0.6B HiZOO
and LoZO use FP16; final OPT-13B transfer protocols use BF16.

## Checkpoint and official evaluation

The official validation set is never used for checkpoint selection. The
trainer evaluates the train-derived dev split and saves `checkpoints/best`.
Classification maximizes accuracy; SQuAD, ReCoRD, and DROP minimize generation
CE. OPT protocols use five uniform checkpoints. Executed Qwen3-0.6B snapshots
evaluate every 500 steps; this historical difference remains in `eval_every`.

```bash
aimzo-eval --config CONFIG --checkpoint OUTPUT/checkpoints/best
aimzo-qa-eval --config CONFIG --checkpoint OUTPUT/checkpoints/best --scope official
python scripts/aggregate_official_results.py seed*/official_eval.json
```

MultiRC reports grouped F1a/EM over all answer rows. Generation tasks report
F1/EM. Aggregation uses the sample standard deviation. Deterministic zero-shot
evaluation is run once rather than once per training seed.

The released OPT-13B table uses one uniform full-parameter, generation-CE
protocol for SQuAD across methods. This differs from the original MeZO SQuAD
source protocol, which uses five prefix tokens and a generated-answer F1
objective; the result ledger labels this distinction explicitly. Large-model
evaluation-only settings are in
`configs/supplemental/large-model-evaluation.yaml`.
