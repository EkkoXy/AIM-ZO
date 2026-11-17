# Baseline validation and tuning

## Implementation consistency

The checks compare perturbations and parameter updates against the author
kernels under paired randomness and inputs.

| Algorithm | Check | Result |
| --- | --- | --- |
| AGZO | 20 model steps per arm; nine compared trace fields; 60 objective calls per arm | All fields match; identical initial and final full-parameter hashes |
| LoZO | 52 model steps per arm; nine compared trace fields; 104 objective calls per arm | All fields match; identical initial and final full-parameter hashes |
| LoZO | Seven CPU kernel steps per precision, including a basis-refresh boundary | Exact perturbation and update equality in FP64, FP16, and BF16 |
| CurvZO | 60 CPU kernel steps per precision, including inclusion-probability correction and weight decay | Exact perturbation and update equality in FP64, FP16, and BF16 |

The CPU checks isolate the perturbation/update kernels using fixed scalar
gradients. Model-step checks use a shared objective and paired batches/seeds.

Machine-readable evidence: [AGZO](../results/audits/baseline-parity/agzo.json),
[LoZO](../results/audits/baseline-parity/lozo.json), and
[LoZO/CurvZO CPU checks](../results/audits/baseline-parity/lozo-curvzo-kernels.json).

## Precision comparison

OPT-2.7B/SQuAD uses the same configuration in BF16 and FP16: batch size 16,
learning rate `1e-7`, epsilon `1e-3`, and 20000 steps / 40000 training objective
calls. LoZO uses rank 2 and refresh interval 50; CurvZO uses sampling ratio
0.4, SGS alpha 0.1, and adaptive interval 20. Checkpoint selection minimizes
development CE at 1000-step intervals. Official evaluation uses the same
frozen 1000 examples across these runs.

| Method | Seeds | BF16 F1 (%) | FP16 F1 (%) |
| --- | --- | ---: | ---: |
| LoZO | 42, 142, 242 | 43.52 +/- 1.24 | 78.07 +/- 2.21 |
| CurvZO | 42 | 61.05 | 78.26 |

For seed 42, the fraction of zero projected gradients is 3.425% / 0.220%
for LoZO (BF16 / FP16), and 3.575% / 0.485% for CurvZO. The corresponding
minimum development CEs are 1.2465 / 0.5025 and 0.9053 / 0.5301.
These observations connect the precision comparison to the finite-difference
signal measured during training.

- [Protocol](../configs/diagnostics/precision/protocol.json)
- [Per-seed results](../results/diagnostics/precision/per-seed.csv)
- [Aggregate results](../results/diagnostics/precision/summary.csv)

## Hyperparameter exploration

The [parameter inventory](../results/tuning/candidate-parameters.csv) records
256 model/method/stage/learning-rate/epsilon candidate entries across OPT-2.7B,
OPT-13B, OPT-30B, and Qwen3-8B-Base. The `execution_supported` field distinguishes
entries with execution evidence from configuration-only entries. Method-family
and implementation-stage labels are retained so that distinct exploration
settings remain identifiable.

The [AGZO/SQuAD per-run table](../results/tuning/agzo-squad-per-run.csv)
contains 65 parameter groups and 185 run records. It includes epsilon,
learning rate, rank, perturbation form, normalization, other method parameters,
development CE, official metrics where available, and configuration checksums.
Completion status and logged steps distinguish complete runs from partial
trajectories. Configuration selection uses development measurements.

The [LoZO/CurvZO search records](../results/tuning/lozo-curvzo/runs.json)
include the recorded configurations and their development measurements. The
[search summary](../results/tuning/lozo-curvzo/search-summary.json) lists 12
completed short LoZO runs and six completed short CurvZO runs. LoZO explores
learning rates from `1e-7` to `3e-6`, ranks 1/2/4, and refresh intervals 50/100;
CurvZO explores learning rates from `2e-7` to `2e-6` in the completed short runs.
The records specify their data and selection settings, keeping short-search
development scores separate from full-run official results.

For LoZO with rank 1 and refresh interval 50, the short-run development CE is
0.7337 at learning rate `1e-6` and 7.1380 at `3e-6`. For CurvZO, the
corresponding short-run CE is 0.6396 at `1.2e-6` and 6.1730 at `2e-6`.
The larger learning rates degrade the development objective in these settings.
