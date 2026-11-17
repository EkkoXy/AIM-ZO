# Width experiments

These OPT-2.7B/RTE experiments report official accuracy on 277 examples.
Each run has 2500 optimization steps and 40000 training objective calls.

## Active width

The maintained width is fixed at `K=128`. The active configurations are
`k=64` (48 shared and 16 sampled tail columns) and `k=128` (the full maintained
basis). Each configuration uses seeds 42, 142, and 242.

| Active width | Accuracy (%), mean +/- sample SD |
| --- | ---: |
| 64 | 65.46 +/- 3.62 |
| 128 | 63.66 +/- 1.78 |

The larger active width reduces mean accuracy by 1.81 percentage points in
this experiment. Increasing active width therefore does not provide a monotonic
gain at this fixed maintained width.

BF16, batch size 16, learning rate 0.0015, and a cosine epsilon schedule from
`8e-5` to `2e-5` are shared. Checkpoints are selected by maximum development
accuracy at 500-step intervals.

- [Protocol](../configs/ablations/active-width/protocol.json)
- [Per-seed results](../results/ablations/active-width/per-seed.csv)
- [Aggregate results](../results/ablations/active-width/summary.csv)

## Shared width

At maintained width `K=128` and active width `k=64`, we report the main-result
48+16 reference and the 64+0 shared-prefix experiment. Each series includes
seeds 42, 142, 242, 342, and 442.

| Shared + tail columns | Accuracy (%), mean +/- sample SD |
| --- | ---: |
| 48 + 16 | 67.51 +/- 3.18 |
| 64 + 0 | 67.00 +/- 2.16 |

- [Protocol and series labels](../configs/ablations/shared-width/protocol.json)
- [Per-seed results](../results/ablations/shared-width/per-seed.csv)
- [Aggregate results](../results/ablations/shared-width/summary.csv)
