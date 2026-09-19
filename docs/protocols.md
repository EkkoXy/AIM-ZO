# Experiment protocols

Status: pending confirmation against the actual training code and official artifacts.

For every experiment group, freeze model revision, task and split, preprocessing, training and evaluation sample counts, seeds, optimizer settings, perturbation parameters, numerical precision, budget, and checkpoint-selection rule.

Count unperturbed centre evaluations separately from perturbed evaluations, and state whether reported totals include validation. Use **forward evaluations** for budgets and **directions/perturbations** for sampled objects.

Publish dev-selected official metrics separately from final-checkpoint metrics. Do not pool runs across different configurations, budgets, or selection rules. Different seed counts and task metrics may appear in one clearly labelled comparison table, but must not be represented as identical protocols.

Incomplete or early-stopped runs require explicit provenance and a documented inclusion decision. Do not silently replace a run by a more favourable configuration.

