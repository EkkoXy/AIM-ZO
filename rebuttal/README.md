# Supplementary experiments

This index collects experiment protocols, per-seed results, and supporting
diagnostics. Each study is stored with the corresponding experiment materials.

| Study | Results and protocol |
| --- | --- |
| Activation-gradient alignment | [Diagnostic](../docs/activation-gradient-alignment.md) |
| Active width at fixed maintained width | [Width experiments](../docs/width-experiments.md#active-width) |
| Shared-prefix and sampled-tail allocation | [Width experiments](../docs/width-experiments.md#shared-width) |
| AGZO, LoZO, and CurvZO consistency checks | [Consistency results](../docs/baseline-validation-and-tuning.md#implementation-consistency) |
| LoZO and CurvZO precision comparison | [Precision results](../docs/baseline-validation-and-tuning.md#precision-comparison) |
| Hyperparameter exploration | [Parameter records](../docs/baseline-validation-and-tuning.md#hyperparameter-exploration) |
| MeZO+LoRA and selected-layer AIM-ZO | [Independent appendix experiment](../experiments/appendix/lora-selected-layers/README.md) |

Scores are aggregated across the seeds listed in each protocol. Standard
deviations use the sample convention. Official evaluation scores are reported
at checkpoints selected on development data.

Reproduce the released aggregates and validate their underlying records:

```bash
python scripts/summarize_supplement.py --check
```

This command uses only the released CSV/JSON files and the Python standard
library. It performs no model calls. Source artifact identities are listed in
[the source manifest](../results/supplement-source-manifest.json).
