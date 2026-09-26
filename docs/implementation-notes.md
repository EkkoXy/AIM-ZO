# AIM-ZO implementation

The public method key is `aimzo`; the Python implementation is
`HFAIMZOMethod`.

For each eligible matrix parameter, AIM-ZO maintains a wide orthonormal right
basis `Q` from forward activations. Oja updates refresh this historical basis
without storing gradients. An active basis is constructed from a shared prefix
and a sampled tail. Each population member draws low-rank factors `A` and
`B`, giving a structured direction `A B Q_active^T`.

The default estimator evaluates an unperturbed center and fifteen one-sided
probes. Probe rewards are centered by the population mean. The resulting
weighted low-rank factors are applied directly without materializing a dense
full-parameter perturbation. Configuration fields controlling maintained width,
shared width, sampled width, Oja step size, update interval, population size,
normalization, and cache behavior are under `zo.aimzo`.

The implementation also records forward-call counts, subspace timing,
perturbation checksums, update norms, CUDA peak memory, and checkpoint metadata.
