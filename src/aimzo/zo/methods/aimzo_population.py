from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PopulationEstimate:
    loss_mean: float
    loss_std: float
    baseline: float
    mean_minus_center: float | None
    std_denominator: float
    clip_threshold: float
    clipped_count: int
    trim_extremes: int
    trimmed_seeds: frozenset[int]
    fitness_weights: tuple[tuple[int, float], ...]
    objective_deltas: tuple[float, ...]
    projected_grads: tuple[tuple[int, float], ...]


def population_sample_denominator(num_noise: int, baseline_mode: str) -> int:
    count = max(1, int(num_noise))
    return count if baseline_mode == "center" else max(1, count - 1)


def centered_population_estimate(
    probe_seeds: list[int],
    losses: list[float],
    *,
    center_loss: float | None,
    baseline_mode: str,
    eps: float,
    divide_by_eps: bool,
    normalize_std: bool,
    std_eps: float,
    clip_std: float,
    trim_extremes: int,
) -> PopulationEstimate:
    """Build centered one-sided population coefficients.

    The main AIMZO configuration uses the population mean, no epsilon division,
    no clipping, and no trimming. The extra arguments preserve checkpoint and
    configuration compatibility with earlier experiments.
    """
    if len(probe_seeds) != len(losses) or not losses:
        raise ValueError("population seeds and losses must have equal nonzero length")

    count = len(losses)
    loss_mean = sum(losses) / float(count)
    loss_variance = sum(
        (float(value) - float(loss_mean)) ** 2 for value in losses
    ) / float(max(1, count))
    loss_std = math.sqrt(max(float(loss_variance), 0.0))

    if baseline_mode == "center":
        if center_loss is None:
            raise ValueError("center population baseline requires center_loss")
        baseline = float(center_loss)
        sample_count = population_sample_denominator(count, baseline_mode)
    else:
        baseline = float(loss_mean)
        sample_count = population_sample_denominator(count, baseline_mode)

    mean_minus_center = (
        float(loss_mean) - float(center_loss) if center_loss is not None else None
    )
    std_denominator = loss_std + float(std_eps) if normalize_std else 1.0
    eps_denominator = float(eps) if divide_by_eps else 1.0
    denominator = eps_denominator * float(sample_count)

    clip_threshold = (
        float(clip_std) * float(loss_std)
        if float(clip_std) > 0.0 and loss_std > 0.0
        else 0.0
    )
    requested_trim = max(0, int(trim_extremes))
    trimmed_seeds: set[int] = set()
    if requested_trim > 0:
        max_each_side = max(0, (count - 1) // 2)
        trim_each_side = min(requested_trim, max_each_side)
        if trim_each_side > 0:
            ranked = sorted(
                zip(probe_seeds, losses, strict=False),
                key=lambda item: float(item[1]),
            )
            trimmed_seeds = {
                int(seed_value)
                for seed_value, _ in (
                    ranked[:trim_each_side] + ranked[-trim_each_side:]
                )
            }

    clipped_count = 0
    fitness_weights: list[tuple[int, float]] = []
    objective_deltas: list[float] = []
    projected_grads: list[tuple[int, float]] = []
    for probe_seed, value in zip(probe_seeds, losses, strict=False):
        raw_weight = float(value) - float(baseline)
        clipped_weight = 0.0 if int(probe_seed) in trimmed_seeds else raw_weight
        if clip_threshold > 0.0:
            clipped_weight = max(
                -float(clip_threshold),
                min(float(clip_threshold), clipped_weight),
            )
            if clipped_weight != raw_weight:
                clipped_count += 1
        fitness_weight = float(clipped_weight) / float(std_denominator)
        projected = float(fitness_weight) / float(denominator)
        fitness_weights.append((int(probe_seed), float(fitness_weight)))
        objective_deltas.append(float(raw_weight))
        projected_grads.append((int(probe_seed), projected))

    return PopulationEstimate(
        loss_mean=float(loss_mean),
        loss_std=float(loss_std),
        baseline=float(baseline),
        mean_minus_center=mean_minus_center,
        std_denominator=float(std_denominator),
        clip_threshold=float(clip_threshold),
        clipped_count=int(clipped_count),
        trim_extremes=int(requested_trim),
        trimmed_seeds=frozenset(trimmed_seeds),
        fitness_weights=tuple(fitness_weights),
        objective_deltas=tuple(objective_deltas),
        projected_grads=tuple(projected_grads),
    )
