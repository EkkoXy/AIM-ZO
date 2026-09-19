# AIM-ZO

**Activation-Informed Subspace Maintenance for Zeroth-Order LLM Fine-Tuning**

AIM-ZO uses forward activation information to maintain an evolving candidate subspace. Each perturbation operates within a smaller active subspace formed from shared and sampled basis directions.

## Availability

Code and reproduction materials are being prepared. This repository currently contains the directory structure and result-format templates; a runnable implementation is not yet available.

## Repository structure

| Directory | Purpose |
| --- | --- |
| `src/aimzo/` | Method implementations and training utilities |
| `configs/` | Experiment configurations |
| `scripts/` | Training, evaluation, and analysis entry points |
| `results/` | Evaluation records and metadata |
| `figures/` | Paper figures |
| `tests/` | Tests |
| `docs/` | Reproduction and evaluation documentation |
| `requirements/` | Environment dependencies |

## Reproduction

Installation instructions and experiment commands will accompany the code release. See [reproduction](docs/reproduction.md) and [evaluation protocols](docs/protocols.md).

## Results

The planned result format is described in [result records](docs/result-ledger.md). Files under `results/manifests/templates/` are examples of the format, not experimental results.
