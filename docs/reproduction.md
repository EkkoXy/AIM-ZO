# Reproduction

1. Create a Python 3.12 environment (the validated stack used Python 3.12 and PyTorch 2.10.0+cu128) and install `pip install -e ".[dev]"`.
2. Prepare each dataset with `aimzo-prepare-data --task <task>`.
3. Set `backend.model_name` in the YAML to the required local model or model ID.
4. Run `aimzo-train <config>` on one GPU.
5. Use the five periodic development evaluations to select a checkpoint.
6. Run `aimzo-eval` for classification or `aimzo-qa-eval` for generated QA.
7. Repeat with seeds 42, 142, and 242 by changing both `data.seed` and
   `zo.seed`, and use separate output directories.

Example:

```bash
aimzo-prepare-data --task rte
CUDA_VISIBLE_DEVICES=0 aimzo-train configs/aimzo/opt-2.7b/rte.yaml
pytest -q
```

The repository excludes model weights, downloaded datasets, checkpoints, and
training outputs. Each run writes a resolved configuration, manifest, history,
checkpoint metadata, selection record, and evaluation record to its output
folder.
