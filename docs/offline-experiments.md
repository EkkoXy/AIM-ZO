# Offline experiments

Offline diagnostics support mechanism claims that final task accuracy cannot
establish by itself. They remain separate from online ablations and main
results.

The public release includes lightweight summaries for paper tables and a
checksum manifest. It excludes weights, checkpoints, activation caches, full
gradients, and large per-probe records. Those inputs are reconstructed from
user-supplied checkpoints or verified against manifest checksums.

The diagnostics have four evidence levels:

1. **Held-out capture:** construct a space on calibration batches and measure
   gradient capture on disjoint evaluation batches.
2. **Estimator diagnostics:** compare one-sided RLOO and two-sided estimates
   against an FP32 autograd gradient at a frozen checkpoint.
3. **Space controls:** compare maintained Oja, current-batch, fixed activation,
   random, and dense spaces under the same estimator.
4. **Implementation checks:** compare cache modes with identical seeds and
   verify parameter checksums and peak memory.

Capture, cosine, and MSE are mechanism diagnostics, not online accuracy. The
top-prefix versus sampled-tail experiment is a linear-oracle diagnostic and
does not establish that one choice trains better.

All new online ablations use OPT-2.7B/RTE. Frozen Qwen3-4B and OPT-30B rows
remain only where they reproduce an existing paper table.
