# MeZO+LoRA and selected-layer AIM-ZO

This independent appendix experiment compares Q/V LoRA adaptation with direct
Q/V weight updates on Qwen3-0.6B-Base across eleven tasks. Each method uses
seeds 42, 142, and 242, BF16, batch size 16, and 40000 training objective calls
per run.

## Configuration

| Setting | MeZO+LoRA | AIM-ZO |
| --- | --- | --- |
| Trainable parameters | Q/V LoRA factors, rank 8, alpha 16, dropout 0 | Direct Q/V projection weights |
| Steps | 20000 | 2500 |
| Learning rate / native lambda | `5e-5` | `0.006` |
| Epsilon | `0.01`, constant | `6e-5` to `1.5e-5`, cosine |
| Maintained / active width | - | `K=256`, `k=32` |
| Shared / tail columns | - | 24 / 8 |
| Center activation tokens | - | 512 |
| Development interval | 4000 steps | 500 steps |

Each run selects the lowest of five development CE measurements, with exact
ties resolved by the earlier step. Official evaluation uses frozen task-specific
examples, paired across methods and seeds. Test scores are used for reporting.

## Results and artifacts

- [Task results](RESULTS.md): mean and sample standard deviation for all eleven tasks.
- [Per-seed results](results/per-seed.csv): 66 runs, including selected steps,
  development CE, evaluation payload checksums, steps, and objective calls.
- [Development measurements](results/development.json): all five measurements per run.
- [Aggregate results](results/summary.csv).
- [Protocol](configs/protocol.json).

Scores in `per-seed.csv` use the unit interval; the summary and task table use
percentages. MultiRC reports F1a, ReCoRD reports EM, and SQuAD/DROP report F1;
the other tasks report accuracy. Sample standard deviations use three seeds.

Reproduce the aggregates from the repository root:

```bash
python scripts/summarize_supplement.py --check
```
