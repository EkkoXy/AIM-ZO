# AIM-ZO

**Activation-Informed Subspace Maintenance for Zeroth-Order LLM Fine-Tuning**

AIM-ZO maintains a wide right subspace from forward activations with an online
Oja update. At each optimization step, it combines a shared high-score prefix
with sampled tail directions, draws rank-one Gaussian probes in the active
subspace, and aggregates a population of forward-only objective evaluations.

## Included code

- `src/aimzo/zo/methods/aimzo.py`: AIM-ZO update and subspace maintenance.
- `src/aimzo/zo/methods/aimzo_subspace.py`: Oja updates and active-basis selection.
- `src/aimzo/zo/methods/aimzo_perturbation.py`: structured probe construction.
- `src/aimzo/zo/methods/aimzo_population.py`: centered population estimator.
- `src/aimzo/trainers/hf_zoregular.py`: Hugging Face training and checkpointing.
- `configs/aimzo/`: runnable AIM-ZO configurations.
- `tests/`: deterministic unit tests for the core estimator.

Baseline implementations used by the same trainer are included under
`src/aimzo/zo/methods/` for protocol comparison.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Model weights and datasets are not included.

## Prepare a task

```bash
aimzo-prepare-data --task rte
```

Datasets are written under `data/dataset/<task>`. Supported paper tasks include
RTE, BoolQ, SST-2, WiC, WSC, MultiRC, COPA, ReCoRD, SQuAD, and DROP.

## Run AIM-ZO

Edit `backend.model_name` in the selected YAML so that it points to a local
Hugging Face model or a model identifier, then run:

```bash
CUDA_VISIBLE_DEVICES=0 aimzo-train configs/aimzo/opt-2.7b/rte.yaml
```

Evaluate a classification checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 aimzo-eval \
  --config configs/aimzo/opt-2.7b/rte.yaml \
  --checkpoint outputs/aimzo/opt-2.7b/rte/seed42/checkpoints/best
```

For SQuAD and DROP generated-answer F1/EM:

```bash
CUDA_VISIBLE_DEVICES=0 aimzo-qa-eval \
  --config configs/aimzo/opt-2.7b/squad.yaml \
  --checkpoint outputs/aimzo/opt-2.7b/squad/seed42/checkpoints/best
```

## Default estimator

The released default uses a wide maintained basis of width `K=128`, an active
basis with `h=48` shared columns and `k-h=16` sampled tail columns, `N=15`
rank-one probes, layer-Frobenius normalization, and a centered one-sided
population estimate. The YAML files record model-specific learning rates,
perturbation schedules, precision, step budgets, and checkpoint intervals.

See [implementation notes](docs/implementation-notes.md),
[protocols](docs/protocols.md), [reproduction](docs/reproduction.md), and the
[release smoke test](docs/smoke-test.md).

The complete paper matrix and deterministic config generator are described in
[main experiments](docs/main-experiments.md). Mechanism diagnostics and the
offline-artifact release policy are described in
[offline experiments](docs/offline-experiments.md).
