# Activation-gradient alignment

We measure how much weight-gradient energy lies in a fresh activation subspace
at OPT-2.7B checkpoints from RTE and SST-2 training trajectories (seed 42).
Each task uses checkpoints trained on that task, at steps 500, 1500, and 2500.

For a module's mean gradient `G` and orthonormal activation basis `Q`, coverage
is `||G Q||_F^2 / ||G||_F^2`. We compute an uncentered rank-64 activation SVD
using 32 training examples and accumulate gradients in FP32 on 32 disjoint
training examples. Five random orthonormal subspaces provide controls. The
activation budget is 2048 valid tokens per module; sequence length is capped
at 512. Model parameters use BF16.

RTE covers all 32 layers and six projections per layer (`q`, `k`, `v`, `out`,
`fc1`, `fc2`), giving 576 module-checkpoint observations. SST-2 covers layers
0, 15, and 31 and their `q`, `v`, `fc1`, and `fc2` projections, giving 36
observations.

| Task | Checkpoint step | Mean activation coverage (%) | Mean random coverage (%) |
| --- | ---: | ---: | ---: |
| RTE | 500 | 83.50 | 2.09 |
| RTE | 1500 | 86.89 | 2.08 |
| RTE | 2500 | 86.79 | 2.07 |
| SST-2 | 500 | 85.75 | 1.97 |
| SST-2 | 1500 | 87.55 | 1.98 |
| SST-2 | 2500 | 87.25 | 1.97 |

Every measured module has positive excess coverage over its random controls.
This is a frozen-checkpoint subspace diagnostic; gradients are measured without
updating the model.

## Artifacts

- [Protocol](../configs/diagnostics/activation-gradient/protocol.json)
- [Per-module observations](../results/diagnostics/activation-gradient/per-module.csv)
- [Aggregate values](../results/diagnostics/activation-gradient/summary.csv)
- [Checkpoint reports and reference indices](../results/diagnostics/activation-gradient/)
- [Diagnostic script](../scripts/diagnose_activation_gradient.py)

The diagnostic script accepts a user-supplied checkpoint and prepared task data.
Run `python scripts/diagnose_activation_gradient.py --help` for its arguments.
Reference indices and data checksums are included in each checkpoint report.
