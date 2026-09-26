# Release smoke test

The release candidate was validated with:

- model: OPT-2.7B
- task: RTE
- precision: BF16
- parameter scope: full parameters
- training steps: 1
- AIM-ZO probes: 15 plus one unperturbed center evaluation
- maintained basis width: 128
- active basis: 48 shared plus 16 sampled tail columns
- Oja update interval: 1 for the smoke
- evaluation examples: 16

The run completed successfully, wrote all expected artifacts, recorded the public
method key `aimzo`, performed 16 objective calls, and updated Oja bases for 193
eligible layers. Peak CUDA memory was 6.69 GB allocated and 7.14 GB reserved.

The smoke is an implementation check rather than a reported experiment result.
