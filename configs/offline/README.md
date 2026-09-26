# Offline diagnostics

The offline experiments test mechanism claims rather than downstream task
accuracy. They are published separately from `configs/main/` so an offline
capture statistic cannot be mistaken for an online training result.

The paper-facing bundle contains held-out gradient capture, activation-width
sweeps, persistent-Oja controls, estimator diagnostics, fixed-width space
comparisons, and cache equivalence checks.

Online ablations use only OPT-2.7B/RTE. Some frozen-checkpoint diagnostics in
the paper also inspect Qwen3-4B and OPT-30B; those are retained only when needed
to reproduce a paper table and are clearly marked as offline evidence.

Large gradient tensors, checkpoints, activation caches, and per-probe raw files
are omitted. The public manifest records lightweight artifacts and checksums of
omitted inputs.
