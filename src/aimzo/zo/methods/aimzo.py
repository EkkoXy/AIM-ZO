"""AIM-ZO implementation.

The maintained presets exercise the narrow path documented in ``docs/implementation-notes.md``:
online Oja wide basis, top-plus-random-tail active basis, rank-one Gaussian ABQ
probes, layer-Frobenius normalization, and centered one-sided population
aggregation. ``agzo.py`` subclasses this class only to load historical
``zo.method=agzo`` experiments.

Compatibility-only diagnostic keys retain the name ``agzo`` so baseline
checkpoints and traces remain loadable. AIM-ZO uses its own public method key,
configuration section, class, and registry entry.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from typing import Any, Callable

import torch

from aimzo.config import AGZOConfig, AIMZOConfig, ZOConfig
from aimzo.source_reproduction.targets import (
    AGZO_PAPER_GAP_STATUS,
    AGZO_PAPER_TARGET_ACCURACY,
    AGZO_SOURCE_TARGET_ACCURACY,
)
from aimzo.zo.params import ParameterList, select_trainable_parameters

from .base import (
    CandidateEvaluation,
    HFZOMethodStepResult,
    METHOD_SOURCE_COMMITS,
    ObjectiveFn,
    UpdateTraceAccumulator,
    bounded_tensor_fingerprint,
    evaluate_objective,
    method_config_diagnostics,
)
from .lowdim_muon import _zeropower_via_newton_schulz
from .aimzo_perturbation import (
    perturbation_normalization_scale,
    sample_probe_a,
    sample_probe_b,
)
from .aimzo_population import centered_population_estimate
from .aimzo_subspace import (
    initialize_oja_basis,
    oja_covariance_action,
    orthonormalize_columns,
    select_top_tail_basis,
)
from .aimzo_utils import stable_int_seed as _stable_int_seed
from .aimzo_utils import stable_json_checksum as _stable_json_checksum


def _two_side_residual_projected_grads(
    plus_minus_values: list[tuple[int, float, float]],
    *,
    residual_lambda: float,
    eps: float,
    divide_by_eps: bool,
    center_coefficients: bool = True,
) -> list[tuple[int, float]]:
    """Return coefficients for centered odd + lambda * even fitness."""
    count = len(plus_minus_values)
    if count < 2:
        raise ValueError("two-side residual interpolation requires at least 2 probes")
    odd = [(plus - minus) / 2.0 for _, plus, minus in plus_minus_values]
    even = [(plus + minus) / 2.0 for _, plus, minus in plus_minus_values]
    odd_mean = sum(odd) / float(count) if center_coefficients else 0.0
    even_mean = sum(even) / float(count) if center_coefficients else 0.0
    # The update path applies uniform 1/count probe weights. This factor makes
    # the aggregate use the same 1/(count-1) sample normalization as RLOO.
    scale = float(count) / float(count - 1)
    if divide_by_eps:
        scale /= float(eps)
    return [
        (
            int(seed),
            scale
            * (
                (odd_value - odd_mean)
                + float(residual_lambda) * (even_value - even_mean)
            ),
        )
        for (seed, _, _), odd_value, even_value in zip(
            plus_minus_values, odd, even, strict=True
        )
    ]


class HFAIMZOMethod:
    name = "aimzo"

    def __init__(self, config: ZOConfig) -> None:
        self.config = config
        method_config = config.aimzo if self.name == "aimzo" else config.agzo
        self.method_config = self._normalize_method_config(method_config)
        self._cov_momentum: dict[str, torch.Tensor] = {}
        self._oja_right_bases: dict[str, torch.Tensor] = {}
        self._cov_momentum_updates: Counter[str] = Counter()
        self._oja_basis_updates: Counter[str] = Counter()
        self._oja_q_loss: dict[str, float] = {}
        self._oja_q_orth_error: dict[str, float] = {}
        self._oja_q_delta_norm: dict[str, float] = {}
        self._oja_q_overlap: dict[str, float] = {}
        self._oja_spectral_strengths: dict[str, torch.Tensor] = {}
        self._oja_sampling_probability_min: dict[str, float] = {}
        self._oja_sampling_probability_max: dict[str, float] = {}
        self._oja_sampling_selected_index_mean: dict[str, float] = {}
        self._oja_q_ema_y: dict[str, torch.Tensor] = {}
        self._oja_q_adam_m: dict[str, torch.Tensor] = {}
        self._oja_q_adam_v: dict[str, torch.Tensor] = {}
        self._oja_q_adam_steps: Counter[str] = Counter()
        self._abh_a_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._abh_b_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._abh_noise_pre_norm_ratio: dict[str, float] = {}
        self._abh_noise_post_norm_ratio: dict[str, float] = {}
        self._abh_noise_scale_factor: dict[str, float] = {}
        self._abh_muon_small_b_pre_norm: dict[str, float] = {}
        self._abh_muon_small_b_post_norm: dict[str, float] = {}
        self._abh_muon_small_b_norm_ratio: dict[str, float] = {}
        self._abh_muon_small_b_momentum: dict[str, torch.Tensor] = {}
        self._abh_ab_momentum: dict[str, torch.Tensor] = {}
        self._abh_ab_adam_v: dict[str, torch.Tensor] = {}
        self._abh_ab_adam_steps: Counter[str] = Counter()
        self._abh_ab_momentum_norm: dict[str, float] = {}
        self._abh_adamu_b_momentum: dict[str, torch.Tensor] = {}
        self._abh_adamu_b_step_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._abh_adamu_b_cache_step: int | None = None
        self._abh_adamu_b_probe_norm: dict[str, float] = {}
        self._abh_adamu_b_update_norm: dict[str, float] = {}
        self._abh_adamu_b_update_ratio: dict[str, float] = {}
        self._abh_loren_q_a: dict[str, torch.Tensor] = {}
        self._abh_loren_q_a_norm: dict[str, float] = {}
        self._abh_loren_q_a_grad_norm: dict[str, float] = {}
        self._abh_loren_q_a_delta_norm: dict[str, float] = {}
        self._abh_loren_q_a_delta_ratio: dict[str, float] = {}
        self._abh_loren_q_scale_mean: dict[str, float] = {}
        self._abh_q_dropout_effective_keep: dict[str, float] = {}
        self._abh_meazo_scale_v = 0.0
        self._abh_meazo_scale_step = 0
        self._abh_meazo_scale_factor = 1.0
        self._abh_meazo_scale_v_hat = 0.0
        self._abh_seed_pool: dict[int, dict[str, float | int]] = {}
        self._abh_seed_pool_total_evals = 0
        self._abh_seed_pool_selected_roles: dict[int, str] = {}
        self._active_step_number = 0
        self._active_noise_seed: int | None = None
        self._active_population_dense_seeds: set[int] = set()
        self._active_population_seed_indices: dict[int, int] = {}

    def step(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        seed: int,
        step: int,
    ) -> HFZOMethodStepResult:
        self._active_step_number = int(step)
        self._active_population_seed_indices = {}
        self._abh_q_dropout_effective_keep.clear()
        abh_active = self._uses_abh_perturbation() and (
            int(step) > int(self.method_config.abh_warmup_steps)
        )
        eps = float(self.config.eps)
        learning_rate = float(self.config.learning_rate)
        parameter_scope = self.config.parameter_scope
        parameters = select_trainable_parameters(model, parameter_scope)
        if not parameters:
            raise ValueError(
                f"No trainable parameters found for scope {parameter_scope!r}"
            )
        estimator_mode = str(self.method_config.estimator_mode)
        profile = self._new_profile()
        if (
            self._uses_lagged_oja_basis()
            and estimator_mode == "two_side"
            and abh_active
            and not self._uses_source_profile()
            and self._lagged_oja_has_initialized_bases(parameters)
        ):
            return self._lagged_oja_two_side_step(
                model,
                objective_fn,
                parameters=parameters,
                seed=seed,
                eps=eps,
                learning_rate=learning_rate,
                parameter_scope=parameter_scope,
            )

        subspace_started_at = time.perf_counter()
        source_parameter_names = (
            {name for name, _ in parameters} if self._uses_source_profile() else None
        )
        basis_seed = (
            None if str(self.method_config.basis_seed_mode) == "ambient" else seed
        )
        random_probe_oja_basis = (
            self._uses_oja_abh_basis()
            and str(getattr(self.method_config, "abh_oja_basis_source", "center"))
            == "random_probe"
            and estimator_mode == "loren_population"
            and not self._uses_seed_pool_dense_noise()
        )
        has_lagged_oja_basis = (
            random_probe_oja_basis
            and self._lagged_oja_has_initialized_bases(parameters)
        )
        center_capture_started_at = time.perf_counter()
        center, capture = self._capture_center_objective(
            model,
            objective_fn,
            source_parameter_names=source_parameter_names,
            profile=profile,
            basis_seed=basis_seed,
            stream_oja_basis=(
                self._uses_oja_abh_basis()
                and not self._uses_seed_pool_dense_noise()
                and not has_lagged_oja_basis
            ),
            use_abh=abh_active and not self._uses_seed_pool_dense_noise(),
        )
        self._profile_add(
            profile,
            "center_capture_total_time",
            time.perf_counter() - center_capture_started_at,
        )
        basis_started_at = time.perf_counter()
        if has_lagged_oja_basis:
            bases = self._lagged_oja_bases(parameters, seed=seed)
        elif capture.bases:
            bases = dict(capture.bases)
        else:
            bases = self._build_bases(
                capture.activations,
                seed=basis_seed,
                parameters=dict(parameters),
                use_abh=abh_active,
                profile=profile,
            )
        self._profile_add(
            profile,
            "basis_total_time",
            time.perf_counter() - basis_started_at,
        )
        subspace_build_time = time.perf_counter() - subspace_started_at

        noise_plan = self._noise_plan(parameters, bases)
        activation_basis_checksum = self._activation_basis_checksum(bases)
        records_source_trace = self._records_source_trace()
        basis_trace = self._source_basis_trace(bases) if records_source_trace else []
        perturbation_trace: list[dict[str, Any]] = []
        source_update_trace: list[dict[str, Any]] = []
        noise_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        num_abh_noise = self._effective_abh_num_noise(estimator_mode)
        multi_probe_projected_grads_raw: list[tuple[int, float]] = []
        multi_probe_projected_grads: list[tuple[int, float]] = []
        multi_probe_plus_minus_values: list[tuple[int, float, float]] = []
        multi_probe_objective_deltas: list[float] = []
        multi_probe_candidates: list[CandidateEvaluation] = []
        deferred_population_minus_candidates: list[CandidateEvaluation] = []
        multi_probe_selected_seeds: list[int] = []
        multi_probe_seed_score_rows: list[dict[str, float | int]] = []
        multi_probe_update_weights: dict[int, float] = {}
        loren_population_clip_std = 0.0
        loren_population_clip_threshold = 0.0
        loren_population_clipped_count = 0
        loren_population_trim_extremes = 0
        loren_population_trimmed_count = 0
        loren_population_baseline_value: float | None = None
        loren_population_mean_minus_center: float | None = None
        deferred_population_residual_lambda = float(
            getattr(
                self.method_config,
                "abh_loren_population_deferred_residual_lambda",
                -1.0,
            )
        )
        seed_pool_selection_mode: str | None = None
        single_probe_seed = int(seed)
        if (
            estimator_mode == "two_side"
            and num_abh_noise <= 1
            and self._uses_abh_seed_pool(num_abh_noise, estimator_mode)
        ):
            (
                multi_probe_selected_seeds,
                seed_pool_selection_mode,
            ) = self._select_abh_seed_pool_seeds(seed, 1)
            if multi_probe_selected_seeds:
                single_probe_seed = int(multi_probe_selected_seeds[0])
        if estimator_mode == "loren_population":
            if num_abh_noise < 2:
                raise ValueError(
                    "loren_population requires zo.oja_abq.abh_num_noise >= 2"
                )
            if self._uses_abh_seed_pool(num_abh_noise, estimator_mode):
                (
                    multi_probe_selected_seeds,
                    seed_pool_selection_mode,
                ) = self._select_abh_seed_pool_seeds(seed, num_abh_noise)
            else:
                multi_probe_selected_seeds = [
                    int(seed) + 1_000_003 * int(probe_idx)
                    for probe_idx in range(num_abh_noise)
                ]
                seed_pool_selection_mode = "loren_population_sequential"
            dense_count = min(
                int(num_abh_noise),
                max(
                    0,
                    int(
                        getattr(
                            self.method_config,
                            "abh_population_dense_noise_count",
                            0,
                        )
                    ),
                ),
            )
            self._active_population_dense_seeds = (
                set(multi_probe_selected_seeds[-dense_count:])
                if dense_count > 0
                else set()
            )
            self._active_population_seed_indices = {
                int(probe_seed): int(probe_idx)
                for probe_idx, probe_seed in enumerate(multi_probe_selected_seeds)
            }
            random_probe_oja_seed: int | None = None
            if random_probe_oja_basis and multi_probe_selected_seeds:
                stable_basis_seed = _stable_int_seed(
                    {
                        "method": "agzo_oja_random_probe_basis_source",
                        "seed": int(seed),
                        "step": int(self._active_step_number),
                        "selected_seeds": [
                            int(value) for value in multi_probe_selected_seeds
                        ],
                    }
                )
                random_probe_oja_seed = int(
                    multi_probe_selected_seeds[
                        stable_basis_seed % len(multi_probe_selected_seeds)
                    ]
                )
                self._profile_add(
                    profile,
                    "abh_oja_random_probe_basis_seed",
                    random_probe_oja_seed,
                )
            loren_candidates: list[CandidateEvaluation] = []
            loren_values: list[float] = []
            try:
                for probe_idx, probe_seed in enumerate(multi_probe_selected_seeds):
                    probe_noise_cache: dict[tuple[Any, ...], torch.Tensor] = {}
                    self._apply_agzo_perturbation(
                        parameters,
                        bases,
                        seed=probe_seed,
                        eps=eps,
                        scaling=1.0,
                        source_trace=(
                            perturbation_trace if records_source_trace else None
                        ),
                        profile=profile,
                        profile_prefix=f"loren_{probe_idx}_apply_plus",
                        noise_cache=probe_noise_cache,
                    )
                    try:
                        if random_probe_oja_seed is not None and int(probe_seed) == int(
                            random_probe_oja_seed
                        ):
                            probe_plus, _probe_capture = self._capture_objective(
                                model,
                                objective_fn,
                                candidate_id=f"loren_plus_{probe_idx}",
                                source_parameter_names=source_parameter_names,
                                profile=profile,
                                basis_seed=basis_seed,
                                stream_oja_basis=True,
                                use_abh=abh_active,
                            )
                        else:
                            probe_plus = evaluate_objective(
                                f"loren_plus_{probe_idx}", objective_fn
                            )
                    finally:
                        self._apply_agzo_perturbation(
                            parameters,
                            bases,
                            seed=probe_seed,
                            eps=eps,
                            scaling=-1.0,
                            source_trace=(
                                perturbation_trace if records_source_trace else None
                            ),
                            profile=profile,
                            profile_prefix=f"loren_{probe_idx}_restore_center",
                            noise_cache=probe_noise_cache,
                        )
                    loren_candidates.append(probe_plus)
                    if probe_plus.ok and probe_plus.objective_value is not None:
                        loren_values.append(float(probe_plus.objective_value))
            finally:
                if len(loren_values) != int(num_abh_noise):
                    self._active_population_dense_seeds = set()
            multi_probe_candidates.extend(loren_candidates)
            if len(loren_values) == int(num_abh_noise):
                population_baseline = str(
                    getattr(
                        self.method_config,
                        "abh_loren_population_baseline",
                        "population_mean",
                    )
                )
                fweight_clip_std = float(
                    getattr(
                        self.method_config,
                        "abh_loren_population_clip_std",
                        0.0,
                    )
                )
                fweight_trim_extremes = max(
                    0,
                    int(
                        getattr(
                            self.method_config,
                            "abh_loren_population_trim_extremes",
                            0,
                        )
                    ),
                )
                population = centered_population_estimate(
                    multi_probe_selected_seeds,
                    loren_values,
                    center_loss=(
                        float(center.objective_value)
                        if center.objective_value is not None
                        else None
                    ),
                    baseline_mode=population_baseline,
                    eps=eps,
                    divide_by_eps=bool(
                        self.method_config.abh_loren_population_divide_by_eps
                    ),
                    normalize_std=bool(
                        getattr(
                            self.method_config,
                            "abh_loren_population_normalize_std",
                            False,
                        )
                    ),
                    std_eps=float(
                        getattr(
                            self.method_config,
                            "abh_loren_population_std_eps",
                            1e-8,
                        )
                    ),
                    clip_std=fweight_clip_std,
                    trim_extremes=fweight_trim_extremes,
                )
                f_mean = population.loss_mean
                f_std = population.loss_std
                f_std_denom = population.std_denominator
                fweight_clip_threshold = population.clip_threshold
                fweight_clipped_count = population.clipped_count
                trimmed_seed_set = set(population.trimmed_seeds)
                loren_fweights = list(population.fitness_weights)
                loren_population_baseline_value = population.baseline
                loren_population_mean_minus_center = population.mean_minus_center
                multi_probe_objective_deltas.extend(population.objective_deltas)
                multi_probe_projected_grads_raw.extend(population.projected_grads)
                multi_probe_projected_grads.extend(population.projected_grads)
                if deferred_population_residual_lambda >= 0.0:
                    deferred_plus_minus_values: list[tuple[int, float, float]] = []
                    for probe_idx, (probe_seed, plus_value) in enumerate(
                        zip(
                            multi_probe_selected_seeds,
                            loren_values,
                            strict=True,
                        )
                    ):
                        probe_noise_cache: dict[tuple[Any, ...], torch.Tensor] = {}
                        self._apply_agzo_perturbation(
                            parameters,
                            bases,
                            seed=probe_seed,
                            eps=eps,
                            scaling=-1.0,
                            source_trace=(
                                perturbation_trace if records_source_trace else None
                            ),
                            profile=profile,
                            profile_prefix=f"loren_{probe_idx}_apply_minus_deferred",
                            noise_cache=probe_noise_cache,
                        )
                        try:
                            probe_minus = evaluate_objective(
                                f"loren_minus_{probe_idx}", objective_fn
                            )
                        finally:
                            self._apply_agzo_perturbation(
                                parameters,
                                bases,
                                seed=probe_seed,
                                eps=eps,
                                scaling=1.0,
                                source_trace=(
                                    perturbation_trace if records_source_trace else None
                                ),
                                profile=profile,
                                profile_prefix=(
                                    f"loren_{probe_idx}_restore_center_deferred"
                                ),
                                noise_cache=probe_noise_cache,
                            )
                        deferred_population_minus_candidates.append(probe_minus)
                        if probe_minus.ok and probe_minus.objective_value is not None:
                            deferred_plus_minus_values.append(
                                (
                                    int(probe_seed),
                                    float(plus_value),
                                    float(probe_minus.objective_value),
                                )
                            )
                    if len(deferred_plus_minus_values) == int(num_abh_noise):
                        deferred_projected_grads = _two_side_residual_projected_grads(
                            deferred_plus_minus_values,
                            residual_lambda=deferred_population_residual_lambda,
                            eps=eps,
                            divide_by_eps=bool(
                                self.method_config.abh_loren_population_divide_by_eps
                            ),
                            center_coefficients=bool(
                                self.method_config.abh_two_side_residual_center
                            ),
                        )
                        # Population coefficients already carry the sample
                        # denominator and use unit per-probe update weights.
                        # The shared helper targets the two-side path's 1/K
                        # update weights, so remove its compensating K factor.
                        multi_probe_projected_grads = [
                            (probe_seed, value / float(num_abh_noise))
                            for probe_seed, value in deferred_projected_grads
                        ]
                        multi_probe_projected_grads_raw = list(
                            multi_probe_projected_grads
                        )
                        multi_probe_objective_deltas = [
                            float(plus_value) - float(minus_value)
                            for _, plus_value, minus_value in deferred_plus_minus_values
                        ]
                if fweight_clip_std > 0.0 and profile is not None:
                    profile["abh_loren_population_clip_std"] = float(fweight_clip_std)
                    profile["abh_loren_population_clip_threshold"] = float(
                        fweight_clip_threshold
                    )
                    profile["abh_loren_population_clipped_count"] = int(
                        fweight_clipped_count
                    )
                loren_population_clip_std = float(fweight_clip_std)
                loren_population_clip_threshold = float(fweight_clip_threshold)
                loren_population_clipped_count = int(fweight_clipped_count)
                loren_population_trim_extremes = int(fweight_trim_extremes)
                loren_population_trimmed_count = int(len(trimmed_seed_set))
                if fweight_trim_extremes > 0 and profile is not None:
                    profile["abh_loren_population_trim_extremes"] = int(
                        fweight_trim_extremes
                    )
                    profile["abh_loren_population_trimmed_count"] = int(
                        len(trimmed_seed_set)
                    )
                self._update_abh_loren_q_population_state(
                    parameters,
                    bases,
                    probe_fweights=loren_fweights,
                    eps=eps,
                )
                if self._uses_abh_seed_pool(num_abh_noise, estimator_mode):
                    multi_probe_seed_score_rows = (
                        self._update_abh_seed_pool_population_scores(
                            probe_fweights=loren_fweights,
                            probe_losses=[
                                (int(probe_seed), float(value))
                                for probe_seed, value in zip(
                                    multi_probe_selected_seeds,
                                    loren_values,
                                    strict=False,
                                )
                            ],
                            loss_mean=float(f_mean),
                            loss_std=float(f_std),
                            fweight_normalizer=float(f_std_denom),
                        )
                    )
                multi_probe_update_weights = {
                    int(probe_seed): 1.0 for probe_seed in multi_probe_selected_seeds
                }
                self._profile_add(
                    profile,
                    "abh_population_dense.member_count",
                    len(self._active_population_dense_seeds),
                )
            plus = multi_probe_candidates[0] if multi_probe_candidates else center
            minus = None
        elif estimator_mode == "two_side" and num_abh_noise > 1:
            if self._uses_abh_seed_pool(num_abh_noise, estimator_mode):
                (
                    multi_probe_selected_seeds,
                    seed_pool_selection_mode,
                ) = self._select_abh_seed_pool_seeds(seed, num_abh_noise)
            else:
                multi_probe_selected_seeds = [
                    int(seed) + 1_000_003 * int(probe_idx)
                    for probe_idx in range(num_abh_noise)
                ]
                seed_pool_selection_mode = "sequential"
            for probe_idx, probe_seed in enumerate(multi_probe_selected_seeds):
                probe_noise_cache: dict[tuple[Any, ...], torch.Tensor] = {}
                self._apply_agzo_perturbation(
                    parameters,
                    bases,
                    seed=probe_seed,
                    eps=eps,
                    scaling=1.0,
                    source_trace=perturbation_trace if records_source_trace else None,
                    profile=profile,
                    profile_prefix=f"multi_{probe_idx}_apply_plus",
                    noise_cache=probe_noise_cache,
                )
                try:
                    probe_plus = evaluate_objective(f"plus_{probe_idx}", objective_fn)
                    self._apply_agzo_perturbation(
                        parameters,
                        bases,
                        seed=probe_seed,
                        eps=eps,
                        scaling=-2.0,
                        source_trace=perturbation_trace
                        if records_source_trace
                        else None,
                        profile=profile,
                        profile_prefix=f"multi_{probe_idx}_apply_plus_to_minus",
                        noise_cache=probe_noise_cache,
                    )
                    probe_minus = evaluate_objective(f"minus_{probe_idx}", objective_fn)
                finally:
                    self._apply_agzo_perturbation(
                        parameters,
                        bases,
                        seed=probe_seed,
                        eps=eps,
                        scaling=1.0,
                        source_trace=perturbation_trace
                        if records_source_trace
                        else None,
                        profile=profile,
                        profile_prefix=f"multi_{probe_idx}_restore_center",
                        noise_cache=probe_noise_cache,
                    )
                multi_probe_candidates.extend([probe_plus, probe_minus])
                if (
                    center.ok
                    and probe_plus.ok
                    and probe_minus.ok
                    and probe_plus.objective_value is not None
                    and probe_minus.objective_value is not None
                ):
                    probe_delta = float(probe_plus.objective_value) - float(
                        probe_minus.objective_value
                    )
                    probe_projected_grad = (
                        _source_projected_grad(
                            probe_plus.objective_value,
                            probe_minus.objective_value,
                            eps,
                            dtype=getattr(torch, self.method_config.source_scalar_dtype),
                        )
                        if self._uses_source_profile()
                        else probe_delta
                        / (
                            2.0 * float(eps)
                            if bool(self.method_config.abh_projected_grad_divide_by_eps)
                            else 2.0
                        )
                    )
                    multi_probe_objective_deltas.append(probe_delta)
                    multi_probe_projected_grads_raw.append(
                        (int(probe_seed), float(probe_projected_grad))
                    )
                    multi_probe_projected_grads.append(
                        (int(probe_seed), float(probe_projected_grad))
                    )
                    multi_probe_plus_minus_values.append(
                        (
                            int(probe_seed),
                            float(probe_plus.objective_value),
                            float(probe_minus.objective_value),
                        )
                    )
                    if self._uses_abh_seed_pool(num_abh_noise, estimator_mode):
                        multi_probe_seed_score_rows.append(
                            {
                                "seed": int(probe_seed),
                                "s_raw": float(probe_projected_grad),
                                "f_plus": float(probe_plus.objective_value),
                                "f_minus": float(probe_minus.objective_value),
                            }
                        )
            if multi_probe_seed_score_rows and center.objective_value is not None:
                multi_probe_seed_score_rows = self._update_abh_seed_pool_scores(
                    multi_probe_seed_score_rows,
                    loss0=float(center.objective_value),
                    eps=eps,
                )
            residual_lambda = float(
                getattr(self.method_config, "abh_two_side_residual_lambda", -1.0)
            )
            if residual_lambda >= 0.0 and len(multi_probe_plus_minus_values) >= 2:
                multi_probe_projected_grads = _two_side_residual_projected_grads(
                    multi_probe_plus_minus_values,
                    residual_lambda=residual_lambda,
                    eps=eps,
                    divide_by_eps=bool(
                        self.method_config.abh_projected_grad_divide_by_eps
                    ),
                    center_coefficients=bool(
                        self.method_config.abh_two_side_residual_center
                    ),
                )
                multi_probe_projected_grads_raw = list(multi_probe_projected_grads)
            multi_probe_update_weights = self._multi_probe_update_weights(
                multi_probe_projected_grads,
                multi_probe_seed_score_rows,
            )
            if multi_probe_seed_score_rows and center.objective_value is not None:
                multi_probe_projected_grads = (
                    self._apply_seed_pool_rma_damping_to_projected_grads(
                        multi_probe_projected_grads,
                        multi_probe_seed_score_rows,
                    )
                )
            multi_probe_projected_grads = [
                (int(seed_value), self._clip_projected_grad(value))
                for seed_value, value in multi_probe_projected_grads
            ]
            plus = multi_probe_candidates[0] if multi_probe_candidates else center
            minus = (
                multi_probe_candidates[1] if len(multi_probe_candidates) > 1 else None
            )
        elif estimator_mode == "two_side":
            self._apply_agzo_perturbation(
                parameters,
                bases,
                seed=single_probe_seed,
                eps=eps,
                scaling=1.0,
                source_trace=perturbation_trace if records_source_trace else None,
                profile=profile,
                profile_prefix="apply_plus",
                noise_cache=noise_cache,
            )
            try:
                plus = evaluate_objective("plus", objective_fn)
                self._apply_agzo_perturbation(
                    parameters,
                    bases,
                    seed=single_probe_seed,
                    eps=eps,
                    scaling=-2.0,
                    source_trace=perturbation_trace if records_source_trace else None,
                    profile=profile,
                    profile_prefix="apply_plus_to_minus",
                    noise_cache=noise_cache,
                )
                minus = evaluate_objective("minus", objective_fn)
            finally:
                self._apply_agzo_perturbation(
                    parameters,
                    bases,
                    seed=single_probe_seed,
                    eps=eps,
                    scaling=1.0,
                    source_trace=perturbation_trace if records_source_trace else None,
                    profile=profile,
                    profile_prefix="restore_center",
                    noise_cache=noise_cache,
                )
        else:
            plus = self._evaluate_candidate(
                parameters,
                bases,
                objective_fn,
                candidate_id="plus",
                seed=seed,
                eps=eps,
                scaling=1.0,
                source_trace=perturbation_trace if records_source_trace else None,
                profile=profile,
                profile_prefix="evaluate_plus",
                noise_cache=noise_cache,
            )
            minus = None

        if multi_probe_candidates and estimator_mode == "loren_population":
            population_values = [
                candidate.objective_value
                for candidate in multi_probe_candidates
                if candidate.objective_value is not None
            ]
            f_plus = (
                sum(float(value) for value in population_values)
                / float(len(population_values))
                if population_values
                else None
            )
            f_minus = None
        elif multi_probe_candidates:
            plus_values = [
                candidate.objective_value
                for index, candidate in enumerate(multi_probe_candidates)
                if index % 2 == 0 and candidate.objective_value is not None
            ]
            minus_values = [
                candidate.objective_value
                for index, candidate in enumerate(multi_probe_candidates)
                if index % 2 == 1 and candidate.objective_value is not None
            ]
            f_plus = (
                sum(float(value) for value in plus_values) / float(len(plus_values))
                if plus_values
                else None
            )
            f_minus = (
                sum(float(value) for value in minus_values) / float(len(minus_values))
                if minus_values
                else None
            )
        else:
            f_plus = plus.objective_value
            f_minus = minus.objective_value if minus is not None else None
        update_skipped = not (center.ok and plus.ok and f_plus is not None)
        if estimator_mode == "two_side":
            update_skipped = update_skipped or not (
                minus is not None and minus.ok and f_minus is not None
            )
        if multi_probe_candidates:
            update_skipped = update_skipped or len(multi_probe_projected_grads) != int(
                num_abh_noise
            )
        if (
            estimator_mode == "loren_population"
            and deferred_population_residual_lambda >= 0.0
        ):
            update_skipped = (
                update_skipped
                or len(deferred_population_minus_candidates) != int(num_abh_noise)
                or any(
                    not candidate.ok or candidate.objective_value is None
                    for candidate in deferred_population_minus_candidates
                )
            )
        objective_delta = 0.0
        projected_grad_raw = 0.0
        projected_grad = 0.0
        curvature_diagnostics = self._curvature_damping_diagnostics()
        update_trace = UpdateTraceAccumulator(
            defer_samples=bool(
                getattr(
                    self.method_config,
                    "abh_population_deferred_update_trace",
                    False,
                )
            )
        )
        multi_probe_applied_weighted_update = False
        seed_pool_screening_active = self._uses_abh_seed_pool_screening(
            num_abh_noise,
            estimator_mode,
        )
        guided_delta_checksum = self._guided_delta_checksum(
            parameters,
            bases,
            seed=seed,
            projected_grad=0.0,
            update_scale=0.0,
        )
        if not update_skipped:
            if multi_probe_projected_grads:
                objective_delta = self._weighted_probe_value_mean(
                    [
                        (
                            seed_value,
                            delta_value,
                        )
                        for (seed_value, _), delta_value in zip(
                            multi_probe_projected_grads_raw,
                            multi_probe_objective_deltas,
                            strict=False,
                        )
                    ],
                    multi_probe_update_weights,
                )
                projected_grad = self._weighted_probe_value_mean(
                    multi_probe_projected_grads,
                    multi_probe_update_weights,
                )
                projected_grad_raw = self._weighted_probe_value_mean(
                    multi_probe_projected_grads_raw,
                    multi_probe_update_weights,
                )
                guided_delta_checksum = _stable_json_checksum(
                    {
                        "seed": int(seed),
                        "projected_grad": _optional_finite_float(projected_grad),
                        "update_scale": -float(learning_rate),
                        "abh_num_noise": int(num_abh_noise),
                        "abh_multi_update_weighting": str(
                            self.method_config.abh_multi_update_weighting
                        ),
                        "abh_multi_update_curvature_lambda": float(
                            self.method_config.abh_multi_update_curvature_lambda
                        ),
                        "probe_update_weights": {
                            str(probe_seed): _optional_finite_float(weight)
                            for probe_seed, weight in sorted(
                                multi_probe_update_weights.items()
                            )
                        },
                        "probe_projected_grads": [
                            _optional_finite_float(value)
                            for _, value in multi_probe_projected_grads
                        ],
                    }
                )
                if seed_pool_screening_active:
                    guided_delta_checksum = _stable_json_checksum(
                        {
                            "seed": int(seed),
                            "update_skipped_for_seed_pool_screening": True,
                            "abh_num_noise": int(num_abh_noise),
                            "probe_update_weights": {
                                str(probe_seed): _optional_finite_float(weight)
                                for probe_seed, weight in sorted(
                                    multi_probe_update_weights.items()
                                )
                            },
                        }
                    )
                    multi_probe_applied_weighted_update = True
                elif self._uses_combined_multi_structured_update():
                    combined_update = (
                        self._apply_agzo_combined_varied_q_update
                        if self._uses_fused_varied_q_population_update()
                        else self._apply_agzo_combined_multi_structured_update
                    )
                    guided_delta_checksum = combined_update(
                        parameters,
                        bases,
                        probe_projected_grads=multi_probe_projected_grads,
                        probe_update_weights=multi_probe_update_weights,
                        learning_rate=learning_rate,
                        weight_decay=float(self.config.weight_decay)
                        / float(len(multi_probe_projected_grads)),
                        update_trace=update_trace,
                        source_trace=source_update_trace
                        if records_source_trace
                        else None,
                        profile=profile,
                    )
                    multi_probe_applied_weighted_update = True
                else:
                    summary_last_only = bool(
                        getattr(
                            self.method_config,
                            "abh_population_summary_last_only",
                            False,
                        )
                    )
                    for probe_index, (
                        probe_seed,
                        probe_projected_grad,
                    ) in enumerate(multi_probe_projected_grads):
                        probe_weight = float(
                            multi_probe_update_weights.get(
                                int(probe_seed),
                                1.0 / float(len(multi_probe_projected_grads)),
                            )
                        )
                        guided_delta_checksum = self._apply_agzo_update(
                            parameters,
                            bases,
                            seed=probe_seed,
                            projected_grad=probe_projected_grad,
                            learning_rate=learning_rate * probe_weight,
                            weight_decay=(
                                float(self.config.weight_decay) * probe_weight
                            ),
                            update_trace=update_trace,
                            source_trace=(
                                source_update_trace if records_source_trace else None
                            ),
                            profile=profile,
                            noise_cache={},
                            collect_parameter_summaries=(
                                not summary_last_only
                                or probe_index == len(multi_probe_projected_grads) - 1
                            ),
                        )
                    multi_probe_applied_weighted_update = True
            elif estimator_mode == "one_side":
                objective_delta = float(f_plus) - float(center.objective_value)
                projected_grad_raw = (
                    _source_projected_grad(
                        f_plus, center.objective_value, eps, two_side=False,
                        dtype=getattr(torch, self.method_config.source_scalar_dtype),
                    )
                    if self._uses_source_profile()
                    else objective_delta
                    / (
                        float(eps)
                        if bool(self.method_config.abh_projected_grad_divide_by_eps)
                        else 1.0
                    )
                )
                projected_grad, curvature_diagnostics = self._apply_curvature_damping(
                    projected_grad_raw,
                    f_plus=f_plus,
                    f_minus=None,
                    loss0=center.objective_value,
                    eps=eps,
                )
                projected_grad = self._apply_meazo_scale(projected_grad)
                projected_grad = self._clip_projected_grad(projected_grad)
            else:
                objective_delta = float(f_plus) - float(f_minus)
                projected_grad_raw = (
                    _source_projected_grad(
                        f_plus, f_minus, eps,
                        dtype=getattr(torch, self.method_config.source_scalar_dtype),
                    )
                    if self._uses_source_profile()
                    else objective_delta
                    / (
                        2.0 * float(eps)
                        if bool(self.method_config.abh_projected_grad_divide_by_eps)
                        else 2.0
                    )
                )
                projected_grad, curvature_diagnostics = self._apply_curvature_damping(
                    projected_grad_raw,
                    f_plus=f_plus,
                    f_minus=f_minus,
                    loss0=center.objective_value,
                    eps=eps,
                )
                projected_grad = self._apply_meazo_scale(projected_grad)
                projected_grad = self._clip_projected_grad(projected_grad)
            if not multi_probe_applied_weighted_update:
                guided_delta_checksum = self._apply_agzo_update(
                    parameters,
                    bases,
                    seed=single_probe_seed if estimator_mode == "two_side" else seed,
                    projected_grad=projected_grad,
                    learning_rate=learning_rate,
                    weight_decay=float(self.config.weight_decay),
                    update_trace=update_trace,
                    source_trace=source_update_trace if records_source_trace else None,
                    profile=profile,
                    noise_cache=noise_cache,
                )
        update_norm, parameter_delta_checksum = update_trace.finish()
        self._active_population_dense_seeds = set()

        candidate_results = [center.to_dict(seed=seed, eps=eps, scaling=0.0)]
        if multi_probe_candidates and estimator_mode == "loren_population":
            for probe_idx, candidate in enumerate(multi_probe_candidates):
                probe_seed = (
                    int(multi_probe_selected_seeds[probe_idx])
                    if probe_idx < len(multi_probe_selected_seeds)
                    else int(seed) + 1_000_003 * int(probe_idx)
                )
                candidate_results.append(
                    candidate.to_dict(seed=probe_seed, eps=eps, scaling=1.0)
                )
            for probe_idx, candidate in enumerate(deferred_population_minus_candidates):
                probe_seed = int(multi_probe_selected_seeds[probe_idx])
                candidate_results.append(
                    candidate.to_dict(seed=probe_seed, eps=eps, scaling=-1.0)
                )
        elif multi_probe_candidates:
            for probe_idx, candidate in enumerate(multi_probe_candidates):
                if multi_probe_selected_seeds:
                    probe_seed = int(multi_probe_selected_seeds[int(probe_idx // 2)])
                else:
                    probe_seed = int(seed) + 1_000_003 * int(probe_idx // 2)
                scaling = 1.0 if probe_idx % 2 == 0 else -1.0
                candidate_results.append(
                    candidate.to_dict(seed=probe_seed, eps=eps, scaling=scaling)
                )
        else:
            candidate_results.append(
                plus.to_dict(seed=single_probe_seed, eps=eps, scaling=1.0)
            )
            if minus is not None:
                candidate_results.append(
                    minus.to_dict(seed=single_probe_seed, eps=eps, scaling=-1.0)
                )
        objective_calls = len(candidate_results)
        method_diagnostics = {
            "authority_source": "paper+author_source",
            "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
            "estimator_mode": estimator_mode,
            "method_config": method_config_diagnostics(
                eps=eps,
                learning_rate=learning_rate,
                parameter_scope=parameter_scope,
                weight_decay=float(self.config.weight_decay),
                agzo=self.method_config,
            ),
            "objective_calls": objective_calls,
            "num_objective_calls": objective_calls,
            "loss0": center.objective_value,
            "subspace_build_time": float(subspace_build_time),
            "matched_linear_layers": int(capture.matched_linear_layers),
            "activation_layers_seen": len(capture.activations) + len(capture.bases),
            "activation_layers_used": len(bases),
            "agzo_rank": int(self.method_config.rank),
            "rank": int(self.method_config.rank),
            "max_activation_tokens": int(self.method_config.max_activation_tokens),
            "subspace_backend": str(self.method_config.subspace_backend),
            "power_iterations": int(self.method_config.power_iterations),
            "basis_seed_mode": str(self.method_config.basis_seed_mode),
            "perturbation_form": str(self.method_config.perturbation_form),
            "abh_normalization": str(self.method_config.abh_normalization),
            "abh_layer_fro_scale": float(self.method_config.abh_layer_fro_scale),
            "abh_active": bool(abh_active),
            "abh_warmup_steps": int(self.method_config.abh_warmup_steps),
            "abh_warmup_dense_noise": bool(self.method_config.abh_warmup_dense_noise),
            "abh_warmup_dense_active": bool(self._uses_warmup_dense_noise()),
            "abh_momentum_beta": float(self.method_config.abh_momentum_beta),
            "abh_a_refresh_interval": int(self.method_config.abh_a_refresh_interval),
            "abh_right_rank": int(self.method_config.abh_right_rank),
            "abh_oja_wide_right_rank": int(
                getattr(self.method_config, "abh_oja_wide_right_rank", 0)
            ),
            "abh_oja_active_top_count": int(
                getattr(self.method_config, "abh_oja_active_top_count", 0)
            ),
            "abh_oja_active_tail_count": int(
                getattr(self.method_config, "abh_oja_active_tail_count", 0)
            ),
            "abh_oja_active_resample_per_probe": bool(
                getattr(
                    self.method_config,
                    "abh_oja_active_resample_per_probe",
                    False,
                )
            ),
            "abh_right_rank_policy": str(self.method_config.abh_right_rank_policy),
            "abh_right_rank_ratio": float(self.method_config.abh_right_rank_ratio),
            "abh_right_rank_min": int(self.method_config.abh_right_rank_min),
            "abh_activation_token_policy": str(
                self.method_config.abh_activation_token_policy
            ),
            "abh_activation_token_multiplier": float(
                self.method_config.abh_activation_token_multiplier
            ),
            "abh_oja_eta": float(self.method_config.abh_oja_eta),
            "abh_oja_eta_effective": float(self._effective_oja_eta()),
            "abh_oja_eta_decay_interval": int(
                self.method_config.abh_oja_eta_decay_interval
            ),
            "abh_oja_eta_decay_factor": float(
                self.method_config.abh_oja_eta_decay_factor
            ),
            "abh_oja_eta_schedule": self._oja_eta_schedule_kind(),
            "abh_oja_q_update_rule": str(self.method_config.abh_oja_q_update_rule),
            "abh_oja_q_ema_beta": float(self.method_config.abh_oja_q_ema_beta),
            "abh_oja_q_optimizer": str(self.method_config.abh_oja_q_optimizer),
            "abh_oja_q_lr": float(self._effective_osd_q_lr()),
            "abh_oja_q_orth_lambda": float(self.method_config.abh_oja_q_orth_lambda),
            "abh_oja_q_loss_mean": _dict_mean(self._oja_q_loss),
            "abh_oja_q_orth_error_mean": _dict_mean(self._oja_q_orth_error),
            "abh_oja_q_delta_norm_mean": _dict_mean(self._oja_q_delta_norm),
            "abh_oja_q_delta_norm_max": _dict_max(self._oja_q_delta_norm),
            "abh_oja_q_overlap_mean": _dict_mean(self._oja_q_overlap),
            "abh_oja_q_overlap_min": _dict_min(self._oja_q_overlap),
            "abh_oja_sampling_probability_min": _dict_min(
                self._oja_sampling_probability_min
            ),
            "abh_oja_sampling_probability_max": _dict_max(
                self._oja_sampling_probability_max
            ),
            "abh_oja_sampling_selected_index_mean": _dict_mean(
                self._oja_sampling_selected_index_mean
            ),
            "abh_q_dropout_keep_fraction": float(
                self.method_config.abh_q_dropout_keep_fraction
            ),
            "abh_q_dropout_effective_keep_mean": _dict_mean(
                self._abh_q_dropout_effective_keep
            ),
            "abh_oja_update_interval_effective": int(
                self._effective_oja_update_interval()
            ),
            "abh_oja_update_schedule": self._oja_update_schedule_kind(),
            "abh_noise_cache": str(self.method_config.abh_noise_cache),
            "abh_left_factor": str(self.method_config.abh_left_factor),
            "abh_a_seed_mode": str(self.method_config.abh_a_seed_mode),
            "abh_a_refresh_block": int(self._abh_a_refresh_block()),
            "abh_num_noise": int(num_abh_noise),
            "projected_grad_raw": _optional_finite_float(projected_grad_raw),
            "abh_projected_grad_clip": float(self.method_config.abh_projected_grad_clip),
            "abh_projected_grad_divide_by_eps": bool(
                self.method_config.abh_projected_grad_divide_by_eps
            ),
            "abh_projected_grad_was_clipped": bool(
                projected_grad
                != (
                    curvature_diagnostics["abh_projected_grad_after_curvature_damping"]
                    if curvature_diagnostics[
                        "abh_projected_grad_after_curvature_damping"
                    ]
                    is not None
                    else projected_grad
                )
            ),
            "abh_meazo_scale_beta": float(self.method_config.abh_meazo_scale_beta),
            "abh_meazo_scale_eps": float(self.method_config.abh_meazo_scale_eps),
            "abh_meazo_scale_step": int(self._abh_meazo_scale_step),
            "abh_meazo_scale_v": float(self._abh_meazo_scale_v),
            "abh_meazo_scale_v_hat": float(self._abh_meazo_scale_v_hat),
            "abh_meazo_scale_factor": float(self._abh_meazo_scale_factor),
            **curvature_diagnostics,
            "abh_multi_projected_grad_mean": (
                float(projected_grad) if multi_probe_projected_grads else None
            ),
            "abh_multi_projected_grad_raw_mean": (
                sum(value for _, value in multi_probe_projected_grads_raw)
                / float(len(multi_probe_projected_grads_raw))
                if multi_probe_projected_grads_raw
                else None
            ),
            "abh_multi_projected_grad_abs_mean": (
                sum(abs(value) for _, value in multi_probe_projected_grads)
                / float(len(multi_probe_projected_grads))
                if multi_probe_projected_grads
                else None
            ),
            "abh_multi_projected_grad_raw_abs_mean": (
                sum(abs(value) for _, value in multi_probe_projected_grads_raw)
                / float(len(multi_probe_projected_grads_raw))
                if multi_probe_projected_grads_raw
                else None
            ),
            "abh_multi_projected_grad_count": len(multi_probe_projected_grads),
            "abh_multi_update_weighting": str(
                self.method_config.abh_multi_update_weighting
            ),
            "abh_multi_update_curvature_lambda": float(
                self.method_config.abh_multi_update_curvature_lambda
            ),
            "abh_multi_applied_weighted_update": bool(
                multi_probe_applied_weighted_update
            ),
            "abh_multi_update_softmax_tau": float(
                self.method_config.abh_multi_update_softmax_tau
            ),
            "abh_multi_update_weights": {
                str(probe_seed): _optional_finite_float(weight)
                for probe_seed, weight in sorted(multi_probe_update_weights.items())
            },
            "abh_seed_pool_dense_noise": bool(
                self.method_config.abh_seed_pool_dense_noise
            ),
            "abh_seed_pool_screen_until_step": int(
                self.method_config.abh_seed_pool_screen_until_step
            ),
            "abh_seed_pool_update_c_over_eps_threshold": float(
                getattr(
                    self.method_config,
                    "abh_seed_pool_update_c_over_eps_threshold",
                    0.0,
                )
            ),
            "abh_seed_pool_update_curvature_signal_ratio_threshold": float(
                getattr(
                    self.method_config,
                    "abh_seed_pool_update_curvature_signal_ratio_threshold",
                    0.0,
                )
            ),
            "abh_seed_pool_update_filter_roles": str(
                getattr(self.method_config, "abh_seed_pool_update_filter_roles", "")
            ),
            "abh_seed_pool_screening_active": bool(seed_pool_screening_active),
            "abh_seed_pool_enabled": bool(
                self._uses_abh_seed_pool(num_abh_noise, estimator_mode)
            ),
            "abh_seed_pool_selection_mode": seed_pool_selection_mode,
            "abh_seed_pool_size": int(
                getattr(self.method_config, "abh_seed_pool_size", 128)
            ),
            "abh_seed_pool_current_size": int(len(self._abh_seed_pool)),
            "abh_seed_pool_total_evals": int(self._abh_seed_pool_total_evals),
            "abh_seed_pool_warmup_steps": int(
                getattr(self.method_config, "abh_seed_pool_warmup_steps", 50)
            ),
            "abh_seed_pool_score_mode": str(
                getattr(
                    self.method_config,
                    "abh_seed_pool_score_mode",
                    "abs_s_minus_curvature",
                )
            ),
            "abh_seed_pool_selected_seeds": [
                int(value) for value in multi_probe_selected_seeds
            ],
            "abh_oja_active_sampling_policy": str(
                getattr(
                    self.method_config,
                    "abh_oja_active_sampling_policy",
                    "top_tail",
                )
            ),
            "abh_oja_population_candidate_roles": [
                self._oja_population_candidate_role(int(value))
                for value in multi_probe_selected_seeds
            ],
            "abh_seed_pool_selected_roles": {
                str(seed): str(self._abh_seed_pool_selected_roles.get(int(seed), ""))
                for seed in multi_probe_selected_seeds
            },
            "abh_seed_pool_score_rows": multi_probe_seed_score_rows,
            "abh_seed_pool_snapshot": self._abh_seed_pool_snapshot(limit=8),
            "abh_seed_pool_score_mean": _mean_optional(
                [
                    float(row["score"])
                    for row in multi_probe_seed_score_rows
                    if "score" in row
                ]
            ),
            "abh_seed_pool_abs_s_mean": _mean_optional(
                [
                    float(row["abs_s"])
                    for row in multi_probe_seed_score_rows
                    if "abs_s" in row
                ]
            ),
            "abh_seed_pool_c_over_eps_mean": _mean_optional(
                [
                    float(row["c_over_eps"])
                    for row in multi_probe_seed_score_rows
                    if "c_over_eps" in row
                ]
            ),
            "abh_seed_pool_m_abs_s_mean": _mean_optional(
                [
                    float(row["m_abs_s"])
                    for row in multi_probe_seed_score_rows
                    if "m_abs_s" in row
                ]
            ),
            "abh_seed_pool_m_h_mean": _mean_optional(
                [
                    float(row["m_h"])
                    for row in multi_probe_seed_score_rows
                    if "m_h" in row
                ]
            ),
            "abh_seed_pool_rma_damping_factor_mean": _mean_optional(
                [
                    float(row["rma_damping_factor"])
                    for row in multi_probe_seed_score_rows
                    if "rma_damping_factor" in row
                ]
            ),
            "abh_multi_update_combined_before_muon": bool(
                multi_probe_projected_grads
                and self._uses_combined_multi_structured_update()
                and str(self.method_config.abh_update_transform)
                in {
                    "muon_ns_r",
                    "muon_ns_r_momentum",
                    "muon_ns_small_b",
                    "muon_ns_small_b_momentum",
                }
            ),
            "abh_multi_update_combined_before_transform": bool(
                multi_probe_projected_grads
                and self._uses_combined_multi_structured_update()
            ),
            "abh_population_summary_last_only": bool(
                getattr(
                    self.method_config,
                    "abh_population_summary_last_only",
                    False,
                )
            ),
            "abh_population_deferred_update_trace": bool(
                getattr(
                    self.method_config,
                    "abh_population_deferred_update_trace",
                    False,
                )
            ),
            "abh_population_cache_restore_factors": bool(
                getattr(
                    self.method_config,
                    "abh_population_cache_restore_factors",
                    False,
                )
            ),
            "abh_population_fused_shared_q_update": bool(
                self._uses_fused_shared_q_population_update()
            ),
            "abh_population_fused_varied_q_update": bool(
                getattr(
                    self.method_config,
                    "abh_population_fused_varied_q_update",
                    False,
                )
            ),
            "abh_mezo_mix_interval": int(self.method_config.abh_mezo_mix_interval),
            "abh_mezo_mix_dense_steps": int(self.method_config.abh_mezo_mix_dense_steps),
            "abh_mezo_mix_dense_active": bool(self._uses_mezo_mix_dense_noise()),
            "abh_dense_residual_ratio": float(
                self.method_config.abh_dense_residual_ratio
            ),
            "abh_population_dense_noise_count": int(
                getattr(self.method_config, "abh_population_dense_noise_count", 0)
            ),
            "abh_population_dense_active_count": len(
                self._active_population_dense_seeds
            ),
            "abh_update_transform": str(self.method_config.abh_update_transform),
            "abh_probe_transform": str(self.method_config.abh_probe_transform),
            "abh_adamu_alpha": float(self.method_config.abh_adamu_alpha),
            "abh_adamu_beta1": float(self.method_config.abh_adamu_beta1),
            "abh_adamu_beta2": float(self.method_config.abh_adamu_beta2),
            "abh_adamu_eps": float(self.method_config.abh_adamu_eps),
            "abh_loren_q_damping": float(self.method_config.abh_loren_q_damping),
            "abh_loren_q_a_init_std": float(self.method_config.abh_loren_q_a_init_std),
            "abh_loren_q_lr_cov": float(self.method_config.abh_loren_q_lr_cov),
            "abh_loren_q_a_eps_power": float(self.method_config.abh_loren_q_a_eps_power),
            "abh_loren_population_divide_by_eps": bool(
                self.method_config.abh_loren_population_divide_by_eps
            ),
            "abh_two_side_residual_lambda": float(
                getattr(self.method_config, "abh_two_side_residual_lambda", -1.0)
            ),
            "abh_two_side_residual_center": bool(
                getattr(self.method_config, "abh_two_side_residual_center", True)
            ),
            "abh_loren_population_deferred_residual_lambda": float(
                getattr(
                    self.method_config,
                    "abh_loren_population_deferred_residual_lambda",
                    -1.0,
                )
            ),
            "abh_loren_population_baseline": str(
                getattr(
                    self.method_config,
                    "abh_loren_population_baseline",
                    "population_mean",
                )
            ),
            "abh_loren_population_baseline_value": (
                None
                if loren_population_baseline_value is None
                else float(loren_population_baseline_value)
            ),
            "abh_loren_population_mean_minus_center": (
                None
                if loren_population_mean_minus_center is None
                else float(loren_population_mean_minus_center)
            ),
            "abh_loren_population_normalize_std": bool(
                getattr(
                    self.method_config,
                    "abh_loren_population_normalize_std",
                    False,
                )
            ),
            "abh_loren_population_std_eps": float(
                getattr(self.method_config, "abh_loren_population_std_eps", 1e-8)
            ),
            "abh_loren_population_clip_std": float(
                getattr(self.method_config, "abh_loren_population_clip_std", 0.0)
            ),
            "abh_loren_population_clip_std_effective": float(loren_population_clip_std),
            "abh_loren_population_clip_threshold": float(
                loren_population_clip_threshold
            ),
            "abh_loren_population_clipped_count": int(loren_population_clipped_count),
            "abh_loren_population_trim_extremes": int(loren_population_trim_extremes),
            "abh_loren_population_trimmed_count": int(loren_population_trimmed_count),
            "abh_loren_q_a_parameters": len(self._abh_loren_q_a),
            "abh_loren_q_a_norm_mean": _dict_mean(self._abh_loren_q_a_norm),
            "abh_loren_q_a_grad_norm_mean": _dict_mean(self._abh_loren_q_a_grad_norm),
            "abh_loren_q_a_delta_norm_mean": _dict_mean(self._abh_loren_q_a_delta_norm),
            "abh_loren_q_a_delta_ratio_mean": _dict_mean(
                self._abh_loren_q_a_delta_ratio
            ),
            "abh_loren_q_scale_mean": _dict_mean(self._abh_loren_q_scale_mean),
            "abh_adamu_b_momentum_parameters": len(self._abh_adamu_b_momentum),
            "abh_adamu_b_probe_norm_mean": _dict_mean(self._abh_adamu_b_probe_norm),
            "abh_adamu_b_update_norm_mean": _dict_mean(self._abh_adamu_b_update_norm),
            "abh_adamu_b_update_ratio_mean": _dict_mean(self._abh_adamu_b_update_ratio),
            "cov_momentum_layers": len(self._cov_momentum),
            "oja_basis_layers": len(self._oja_right_bases),
            "activation_basis_checksum": activation_basis_checksum,
            "guided_delta_checksum": guided_delta_checksum,
            "abh_noise_pre_norm_ratio_mean": _dict_mean(self._abh_noise_pre_norm_ratio),
            "abh_noise_pre_norm_ratio_max": _dict_max(self._abh_noise_pre_norm_ratio),
            "abh_noise_post_norm_ratio_mean": _dict_mean(
                self._abh_noise_post_norm_ratio
            ),
            "abh_noise_scale_factor_mean": _dict_mean(self._abh_noise_scale_factor),
            "abh_noise_scale_factor_max": _dict_max(self._abh_noise_scale_factor),
            "abh_muon_small_b_pre_norm_mean": _dict_mean(
                self._abh_muon_small_b_pre_norm
            ),
            "abh_muon_small_b_post_norm_mean": _dict_mean(
                self._abh_muon_small_b_post_norm
            ),
            "abh_muon_small_b_norm_ratio_mean": _dict_mean(
                self._abh_muon_small_b_norm_ratio
            ),
            "abh_muon_small_b_momentum_parameters": len(
                self._abh_muon_small_b_momentum
            ),
            "abh_ab_momentum_parameters": len(self._abh_ab_momentum),
            "abh_ab_adam_v_parameters": len(self._abh_ab_adam_v),
            "abh_ab_momentum_norm_mean": _dict_mean(self._abh_ab_momentum_norm),
            "abh_ab_momentum_norm_max": _dict_max(self._abh_ab_momentum_norm),
            "agzo_parameter_count": int(noise_plan["agzo_parameter_count"]),
            "fallback_parameter_count": int(noise_plan["fallback_parameter_count"]),
            "fallback_reason_counts": dict(noise_plan["fallback_reason_counts"]),
            "all_linear_hooks_failed": (
                capture.matched_linear_layers > 0
                and not capture.activations
                and not capture.bases
            ),
            "agzo_source_target_accuracy": AGZO_SOURCE_TARGET_ACCURACY,
            "agzo_paper_target_accuracy": AGZO_PAPER_TARGET_ACCURACY,
            "agzo_paper_gap_status": AGZO_PAPER_GAP_STATUS,
        }
        self._attach_profile(
            method_diagnostics,
            profile,
            center=center,
            plus=plus,
            minus=minus,
        )
        if records_source_trace:
            method_diagnostics.update(
                {
                    "perturbation_trace_policy": "bounded_source_debug",
                    "basis_trace": basis_trace,
                    "perturbation_trace": perturbation_trace,
                    "perturbation_trace_truncated": len(parameters) > 16,
                    "update_trace": source_update_trace,
                    "update_trace_truncated": len(parameters) > 16,
                    "source_update_semantics": "direct_param_add",
                }
            )
        return HFZOMethodStepResult(
            zo_method=self.name,
            seed=int(seed),
            eps=eps,
            learning_rate=learning_rate,
            f_plus=f_plus,
            f_minus=f_minus,
            objective_delta=objective_delta,
            projected_grad=projected_grad,
            candidate_results=candidate_results,
            update_norm=update_norm,
            parameter_delta_checksum=parameter_delta_checksum,
            method_diagnostics=method_diagnostics,
        )

    def _lagged_oja_two_side_step(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        parameters: ParameterList,
        seed: int,
        eps: float,
        learning_rate: float,
        parameter_scope: str,
    ) -> HFZOMethodStepResult:
        subspace_started_at = time.perf_counter()
        parameter_map = dict(parameters)
        bases = self._lagged_oja_bases(parameters, seed=seed)
        subspace_build_time = time.perf_counter() - subspace_started_at

        noise_plan = self._noise_plan(parameters, bases)
        activation_basis_checksum = self._activation_basis_checksum(bases)
        records_source_trace = self._records_source_trace()
        perturbation_trace: list[dict[str, Any]] = []
        source_update_trace: list[dict[str, Any]] = []
        update_source = self._lagged_oja_update_source(seed=seed)
        plus_capture: _ActivationCapture | None = None
        minus_capture: _ActivationCapture | None = None

        self._apply_agzo_perturbation(
            parameters,
            bases,
            seed=seed,
            eps=eps,
            scaling=1.0,
            source_trace=perturbation_trace if records_source_trace else None,
        )
        try:
            if update_source == "plus":
                plus, plus_capture = self._capture_objective(
                    model,
                    objective_fn,
                    candidate_id="plus",
                )
            else:
                plus = evaluate_objective("plus", objective_fn)
            self._apply_agzo_perturbation(
                parameters,
                bases,
                seed=seed,
                eps=eps,
                scaling=-2.0,
                source_trace=perturbation_trace if records_source_trace else None,
            )
            if update_source == "minus":
                minus, minus_capture = self._capture_objective(
                    model,
                    objective_fn,
                    candidate_id="minus",
                )
            else:
                minus = evaluate_objective("minus", objective_fn)
        finally:
            self._apply_agzo_perturbation(
                parameters,
                bases,
                seed=seed,
                eps=eps,
                scaling=1.0,
                source_trace=perturbation_trace if records_source_trace else None,
            )

        basis_update_capture = (
            plus_capture if update_source == "plus" else minus_capture
        )
        basis_update_layers_seen = 0
        basis_update_layers_used = 0
        if basis_update_capture is not None:
            updated_bases = self._build_bases(
                basis_update_capture.activations,
                seed=None
                if str(self.method_config.basis_seed_mode) == "ambient"
                else seed,
                parameters=parameter_map,
                use_abh=True,
            )
            basis_update_layers_seen = len(basis_update_capture.activations)
            basis_update_layers_used = len(updated_bases)

        f_plus = plus.objective_value
        f_minus = minus.objective_value
        update_skipped = not (
            plus.ok and minus.ok and f_plus is not None and f_minus is not None
        )
        objective_delta = 0.0
        projected_grad_raw = 0.0
        projected_grad = 0.0
        update_trace = UpdateTraceAccumulator(
            defer_samples=bool(
                getattr(
                    self.method_config,
                    "abh_population_deferred_update_trace",
                    False,
                )
            )
        )
        guided_delta_checksum = self._guided_delta_checksum(
            parameters,
            bases,
            seed=seed,
            projected_grad=0.0,
            update_scale=0.0,
        )
        if not update_skipped:
            objective_delta = float(f_plus) - float(f_minus)
            projected_grad_raw = objective_delta / (2.0 * eps)
            projected_grad = self._clip_projected_grad(projected_grad_raw)
            guided_delta_checksum = self._apply_agzo_update(
                parameters,
                bases,
                seed=seed,
                projected_grad=projected_grad,
                learning_rate=learning_rate,
                weight_decay=float(self.config.weight_decay),
                update_trace=update_trace,
                source_trace=source_update_trace if records_source_trace else None,
            )
        update_norm, parameter_delta_checksum = update_trace.finish()

        candidate_results = [
            plus.to_dict(seed=seed, eps=eps, scaling=1.0),
            minus.to_dict(seed=seed, eps=eps, scaling=-1.0),
        ]
        objective_calls = len(candidate_results)
        method_diagnostics = {
            "authority_source": "paper+author_source",
            "method_source_commit": METHOD_SOURCE_COMMITS[self.name],
            "estimator_mode": "two_side_lagged_oja",
            "method_config": method_config_diagnostics(
                eps=eps,
                learning_rate=learning_rate,
                parameter_scope=parameter_scope,
                weight_decay=float(self.config.weight_decay),
                agzo=self.method_config,
            ),
            "objective_calls": objective_calls,
            "num_objective_calls": objective_calls,
            "loss0": None,
            "subspace_build_time": float(subspace_build_time),
            "matched_linear_layers": int(basis_update_capture.matched_linear_layers)
            if basis_update_capture is not None
            else 0,
            "activation_layers_seen": int(basis_update_layers_seen),
            "activation_layers_used": int(basis_update_layers_used),
            "agzo_rank": int(self.method_config.rank),
            "rank": int(self.method_config.rank),
            "max_activation_tokens": int(self.method_config.max_activation_tokens),
            "subspace_backend": str(self.method_config.subspace_backend),
            "power_iterations": int(self.method_config.power_iterations),
            "basis_seed_mode": str(self.method_config.basis_seed_mode),
            "perturbation_form": str(self.method_config.perturbation_form),
            "abh_normalization": str(self.method_config.abh_normalization),
            "abh_layer_fro_scale": float(self.method_config.abh_layer_fro_scale),
            "abh_active": True,
            "abh_warmup_steps": int(self.method_config.abh_warmup_steps),
            "abh_warmup_dense_noise": bool(self.method_config.abh_warmup_dense_noise),
            "abh_warmup_dense_active": bool(self._uses_warmup_dense_noise()),
            "abh_momentum_beta": float(self.method_config.abh_momentum_beta),
            "abh_a_refresh_interval": int(self.method_config.abh_a_refresh_interval),
            "abh_right_rank": int(self.method_config.abh_right_rank),
            "abh_oja_wide_right_rank": int(
                getattr(self.method_config, "abh_oja_wide_right_rank", 0)
            ),
            "abh_oja_active_top_count": int(
                getattr(self.method_config, "abh_oja_active_top_count", 0)
            ),
            "abh_oja_active_tail_count": int(
                getattr(self.method_config, "abh_oja_active_tail_count", 0)
            ),
            "abh_oja_active_resample_per_probe": bool(
                getattr(
                    self.method_config,
                    "abh_oja_active_resample_per_probe",
                    False,
                )
            ),
            "abh_right_rank_policy": str(self.method_config.abh_right_rank_policy),
            "abh_right_rank_ratio": float(self.method_config.abh_right_rank_ratio),
            "abh_right_rank_min": int(self.method_config.abh_right_rank_min),
            "abh_activation_token_policy": str(
                self.method_config.abh_activation_token_policy
            ),
            "abh_activation_token_multiplier": float(
                self.method_config.abh_activation_token_multiplier
            ),
            "abh_oja_eta": float(self.method_config.abh_oja_eta),
            "abh_oja_eta_effective": float(self._effective_oja_eta()),
            "abh_oja_eta_decay_interval": int(
                self.method_config.abh_oja_eta_decay_interval
            ),
            "abh_oja_eta_decay_factor": float(
                self.method_config.abh_oja_eta_decay_factor
            ),
            "abh_oja_eta_schedule": self._oja_eta_schedule_kind(),
            "abh_oja_q_update_rule": str(self.method_config.abh_oja_q_update_rule),
            "abh_oja_q_ema_beta": float(self.method_config.abh_oja_q_ema_beta),
            "abh_oja_q_optimizer": str(self.method_config.abh_oja_q_optimizer),
            "abh_oja_q_lr": float(self._effective_osd_q_lr()),
            "abh_oja_q_orth_lambda": float(self.method_config.abh_oja_q_orth_lambda),
            "abh_oja_q_loss_mean": _dict_mean(self._oja_q_loss),
            "abh_oja_q_orth_error_mean": _dict_mean(self._oja_q_orth_error),
            "abh_oja_q_delta_norm_mean": _dict_mean(self._oja_q_delta_norm),
            "abh_oja_q_delta_norm_max": _dict_max(self._oja_q_delta_norm),
            "abh_oja_q_overlap_mean": _dict_mean(self._oja_q_overlap),
            "abh_oja_q_overlap_min": _dict_min(self._oja_q_overlap),
            "abh_oja_sampling_probability_min": _dict_min(
                self._oja_sampling_probability_min
            ),
            "abh_oja_sampling_probability_max": _dict_max(
                self._oja_sampling_probability_max
            ),
            "abh_oja_sampling_selected_index_mean": _dict_mean(
                self._oja_sampling_selected_index_mean
            ),
            "abh_q_dropout_keep_fraction": float(
                self.method_config.abh_q_dropout_keep_fraction
            ),
            "abh_q_dropout_effective_keep_mean": _dict_mean(
                self._abh_q_dropout_effective_keep
            ),
            "abh_oja_update_interval_effective": int(
                self._effective_oja_update_interval()
            ),
            "abh_oja_update_schedule": self._oja_update_schedule_kind(),
            "abh_noise_cache": str(self.method_config.abh_noise_cache),
            "abh_left_factor": str(self.method_config.abh_left_factor),
            "projected_grad_raw": _optional_finite_float(projected_grad_raw),
            "abh_projected_grad_clip": float(self.method_config.abh_projected_grad_clip),
            "abh_projected_grad_was_clipped": bool(
                projected_grad != projected_grad_raw
            ),
            "abh_meazo_scale_beta": float(self.method_config.abh_meazo_scale_beta),
            "abh_meazo_scale_eps": float(self.method_config.abh_meazo_scale_eps),
            "abh_meazo_scale_step": int(self._abh_meazo_scale_step),
            "abh_meazo_scale_v": float(self._abh_meazo_scale_v),
            "abh_meazo_scale_v_hat": float(self._abh_meazo_scale_v_hat),
            "abh_meazo_scale_factor": float(self._abh_meazo_scale_factor),
            "abh_mezo_mix_interval": int(self.method_config.abh_mezo_mix_interval),
            "abh_mezo_mix_dense_steps": int(self.method_config.abh_mezo_mix_dense_steps),
            "abh_mezo_mix_dense_active": bool(self._uses_mezo_mix_dense_noise()),
            "abh_dense_residual_ratio": float(
                self.method_config.abh_dense_residual_ratio
            ),
            "abh_update_transform": str(self.method_config.abh_update_transform),
            "abh_probe_transform": str(self.method_config.abh_probe_transform),
            "abh_adamu_alpha": float(self.method_config.abh_adamu_alpha),
            "abh_adamu_beta1": float(self.method_config.abh_adamu_beta1),
            "abh_adamu_beta2": float(self.method_config.abh_adamu_beta2),
            "abh_adamu_eps": float(self.method_config.abh_adamu_eps),
            "abh_adamu_b_momentum_parameters": len(self._abh_adamu_b_momentum),
            "abh_adamu_b_probe_norm_mean": _dict_mean(self._abh_adamu_b_probe_norm),
            "abh_adamu_b_update_norm_mean": _dict_mean(self._abh_adamu_b_update_norm),
            "abh_adamu_b_update_ratio_mean": _dict_mean(self._abh_adamu_b_update_ratio),
            "abh_oja_lagged_basis": True,
            "abh_oja_lagged_update_source": str(update_source),
            "cov_momentum_layers": len(self._cov_momentum),
            "oja_basis_layers": len(self._oja_right_bases),
            "activation_basis_checksum": activation_basis_checksum,
            "guided_delta_checksum": guided_delta_checksum,
            "abh_noise_pre_norm_ratio_mean": _dict_mean(self._abh_noise_pre_norm_ratio),
            "abh_noise_pre_norm_ratio_max": _dict_max(self._abh_noise_pre_norm_ratio),
            "abh_noise_post_norm_ratio_mean": _dict_mean(
                self._abh_noise_post_norm_ratio
            ),
            "abh_noise_scale_factor_mean": _dict_mean(self._abh_noise_scale_factor),
            "abh_noise_scale_factor_max": _dict_max(self._abh_noise_scale_factor),
            "abh_muon_small_b_pre_norm_mean": _dict_mean(
                self._abh_muon_small_b_pre_norm
            ),
            "abh_muon_small_b_post_norm_mean": _dict_mean(
                self._abh_muon_small_b_post_norm
            ),
            "abh_muon_small_b_norm_ratio_mean": _dict_mean(
                self._abh_muon_small_b_norm_ratio
            ),
            "abh_ab_momentum_parameters": len(self._abh_ab_momentum),
            "abh_ab_adam_v_parameters": len(self._abh_ab_adam_v),
            "abh_ab_momentum_norm_mean": _dict_mean(self._abh_ab_momentum_norm),
            "abh_ab_momentum_norm_max": _dict_max(self._abh_ab_momentum_norm),
            "agzo_parameter_count": int(noise_plan["agzo_parameter_count"]),
            "fallback_parameter_count": int(noise_plan["fallback_parameter_count"]),
            "fallback_reason_counts": dict(noise_plan["fallback_reason_counts"]),
            "all_linear_hooks_failed": (
                basis_update_capture is not None
                and basis_update_capture.matched_linear_layers > 0
                and not basis_update_capture.activations
            ),
            "agzo_source_target_accuracy": AGZO_SOURCE_TARGET_ACCURACY,
            "agzo_paper_target_accuracy": AGZO_PAPER_TARGET_ACCURACY,
            "agzo_paper_gap_status": AGZO_PAPER_GAP_STATUS,
        }
        if records_source_trace:
            method_diagnostics.update(
                {
                    "perturbation_trace_policy": "bounded_source_debug",
                    "basis_trace": self._source_basis_trace(bases),
                    "perturbation_trace": perturbation_trace,
                    "perturbation_trace_truncated": len(parameters) > 16,
                    "update_trace": source_update_trace,
                    "update_trace_truncated": len(parameters) > 16,
                    "source_update_semantics": "direct_param_add",
                }
            )
        return HFZOMethodStepResult(
            zo_method=self.name,
            seed=int(seed),
            eps=eps,
            learning_rate=learning_rate,
            f_plus=f_plus,
            f_minus=f_minus,
            objective_delta=objective_delta,
            projected_grad=projected_grad,
            candidate_results=candidate_results,
            update_norm=update_norm,
            parameter_delta_checksum=parameter_delta_checksum,
            method_diagnostics=method_diagnostics,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "cov_momentum": {
                name: tensor.detach().cpu()
                for name, tensor in self._cov_momentum.items()
            },
            "oja_right_bases": {
                name: tensor.detach().cpu()
                for name, tensor in self._oja_right_bases.items()
            },
            "cov_momentum_updates": dict(self._cov_momentum_updates),
            "oja_basis_updates": dict(self._oja_basis_updates),
            "oja_q_loss": dict(self._oja_q_loss),
            "oja_q_orth_error": dict(self._oja_q_orth_error),
            "oja_spectral_strengths": {
                name: tensor.detach().cpu()
                for name, tensor in self._oja_spectral_strengths.items()
            },
            "oja_q_ema_y": {
                name: tensor.detach().cpu()
                for name, tensor in self._oja_q_ema_y.items()
            },
            "oja_q_adam_m": {
                name: tensor.detach().cpu()
                for name, tensor in self._oja_q_adam_m.items()
            },
            "oja_q_adam_v": {
                name: tensor.detach().cpu()
                for name, tensor in self._oja_q_adam_v.items()
            },
            "oja_q_adam_steps": dict(self._oja_q_adam_steps),
            "abh_muon_small_b_momentum": {
                name: tensor.detach().cpu()
                for name, tensor in self._abh_muon_small_b_momentum.items()
            },
            "abh_ab_momentum": {
                name: tensor.detach().cpu()
                for name, tensor in self._abh_ab_momentum.items()
            },
            "abh_ab_adam_v": {
                name: tensor.detach().cpu()
                for name, tensor in self._abh_ab_adam_v.items()
            },
            "abh_ab_adam_steps": dict(self._abh_ab_adam_steps),
            "abh_adamu_b_momentum": {
                name: tensor.detach().cpu()
                for name, tensor in self._abh_adamu_b_momentum.items()
            },
            "abh_loren_q_a": {
                name: tensor.detach().cpu()
                for name, tensor in self._abh_loren_q_a.items()
            },
            "abh_meazo_scale_v": float(self._abh_meazo_scale_v),
            "abh_meazo_scale_step": int(self._abh_meazo_scale_step),
            "abh_seed_pool": {
                str(seed): dict(row) for seed, row in self._abh_seed_pool.items()
            },
            "abh_seed_pool_total_evals": int(self._abh_seed_pool_total_evals),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        momentum = state_dict.get("cov_momentum", {})
        if isinstance(momentum, dict):
            self._cov_momentum = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in momentum.items()
                if isinstance(tensor, torch.Tensor)
            }
        oja_bases = state_dict.get("oja_right_bases", {})
        if isinstance(oja_bases, dict):
            self._oja_right_bases = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in oja_bases.items()
                if isinstance(tensor, torch.Tensor)
            }
        updates = state_dict.get("cov_momentum_updates", {})
        if isinstance(updates, dict):
            self._cov_momentum_updates = Counter(
                {str(name): int(value) for name, value in updates.items()}
            )
        oja_updates = state_dict.get("oja_basis_updates", {})
        if isinstance(oja_updates, dict):
            self._oja_basis_updates = Counter(
                {str(name): int(value) for name, value in oja_updates.items()}
            )
        q_loss = state_dict.get("oja_q_loss", {})
        if isinstance(q_loss, dict):
            self._oja_q_loss = {
                str(name): float(value) for name, value in q_loss.items()
            }
        q_orth_error = state_dict.get("oja_q_orth_error", {})
        if isinstance(q_orth_error, dict):
            self._oja_q_orth_error = {
                str(name): float(value) for name, value in q_orth_error.items()
            }
        spectral_strengths = state_dict.get("oja_spectral_strengths", {})
        if isinstance(spectral_strengths, dict):
            self._oja_spectral_strengths = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in spectral_strengths.items()
                if isinstance(tensor, torch.Tensor)
            }
        q_ema_y = state_dict.get("oja_q_ema_y", {})
        if isinstance(q_ema_y, dict):
            self._oja_q_ema_y = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in q_ema_y.items()
                if isinstance(tensor, torch.Tensor)
            }
        adam_m = state_dict.get("oja_q_adam_m", {})
        if isinstance(adam_m, dict):
            self._oja_q_adam_m = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in adam_m.items()
                if isinstance(tensor, torch.Tensor)
            }
        adam_v = state_dict.get("oja_q_adam_v", {})
        if isinstance(adam_v, dict):
            self._oja_q_adam_v = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in adam_v.items()
                if isinstance(tensor, torch.Tensor)
            }
        adam_steps = state_dict.get("oja_q_adam_steps", {})
        if isinstance(adam_steps, dict):
            self._oja_q_adam_steps = Counter(
                {str(name): int(value) for name, value in adam_steps.items()}
            )
        small_b_momentum = state_dict.get("abh_muon_small_b_momentum", {})
        if isinstance(small_b_momentum, dict):
            self._abh_muon_small_b_momentum = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in small_b_momentum.items()
                if isinstance(tensor, torch.Tensor)
            }
        ab_momentum = state_dict.get("abh_ab_momentum", {})
        if isinstance(ab_momentum, dict):
            self._abh_ab_momentum = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in ab_momentum.items()
                if isinstance(tensor, torch.Tensor)
            }
        ab_adam_v = state_dict.get("abh_ab_adam_v", {})
        if isinstance(ab_adam_v, dict):
            self._abh_ab_adam_v = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in ab_adam_v.items()
                if isinstance(tensor, torch.Tensor)
            }
        ab_adam_steps = state_dict.get("abh_ab_adam_steps", {})
        if isinstance(ab_adam_steps, dict):
            self._abh_ab_adam_steps = Counter(
                {str(name): int(value) for name, value in ab_adam_steps.items()}
            )
        adamu_b_momentum = state_dict.get("abh_adamu_b_momentum", {})
        if isinstance(adamu_b_momentum, dict):
            self._abh_adamu_b_momentum = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in adamu_b_momentum.items()
                if isinstance(tensor, torch.Tensor)
            }
        loren_q_a = state_dict.get("abh_loren_q_a", {})
        if isinstance(loren_q_a, dict):
            self._abh_loren_q_a = {
                str(name): tensor.detach().float().contiguous()
                for name, tensor in loren_q_a.items()
                if isinstance(tensor, torch.Tensor)
            }
        self._abh_meazo_scale_v = float(state_dict.get("abh_meazo_scale_v", 0.0))
        self._abh_meazo_scale_step = int(state_dict.get("abh_meazo_scale_step", 0))
        seed_pool = state_dict.get("abh_seed_pool", {})
        if isinstance(seed_pool, dict):
            self._abh_seed_pool = {
                int(seed): {
                    "score": float(row.get("score", 0.0)),
                    "count": int(row.get("count", 0)),
                    "created_step": int(row.get("created_step", 0)),
                }
                for seed, row in seed_pool.items()
                if isinstance(row, dict)
            }
        self._abh_seed_pool_total_evals = int(
            state_dict.get("abh_seed_pool_total_evals", 0)
        )
        self._abh_a_cache = {}
        self._abh_adamu_b_step_cache = {}
        self._abh_adamu_b_cache_step = None

    def _records_source_trace(self) -> bool:
        return os.getenv("AIMZO_AGZO_SOURCE_TRACE") == "1"

    def _profiles_runtime(self) -> bool:
        return os.getenv("AIMZO_AGZO_PROFILE") == "1"

    def _new_profile(self) -> dict[str, Any] | None:
        if not self._profiles_runtime():
            return None
        return {
            "enabled": True,
            "timings": {},
        }

    def _profile_add(
        self,
        profile: dict[str, Any] | None,
        key: str,
        value: float | int,
    ) -> None:
        if profile is None:
            return
        timings = profile.setdefault("timings", {})
        timings[str(key)] = float(timings.get(str(key), 0.0)) + float(value)

    def _attach_profile(
        self,
        diagnostics: dict[str, Any],
        profile: dict[str, Any] | None,
        *,
        center: CandidateEvaluation | None = None,
        plus: CandidateEvaluation | None = None,
        minus: CandidateEvaluation | None = None,
    ) -> None:
        if profile is None:
            return
        timings = dict(profile.get("timings", {}))
        if center is not None:
            timings["objective.center_time"] = float(center.elapsed_time)
        if plus is not None:
            timings["objective.plus_time"] = float(plus.elapsed_time)
        if minus is not None:
            timings["objective.minus_time"] = float(minus.elapsed_time)
        diagnostics["profile"] = {
            "enabled": True,
            "timings": {key: float(timings[key]) for key in sorted(timings)},
        }

    def _uses_source_profile(self) -> bool:
        return self.config.source_compatibility_profile == "agzo_source_and_paper_gap"

    def _bounded_tensor_trace(self, tensor: torch.Tensor) -> dict[str, Any]:
        detached = tensor.detach().cpu().contiguous()
        flat = detached.reshape(-1)
        if detached.is_floating_point():
            total = float(detached.float().sum().item())
        else:
            total = int(detached.long().sum().item())
        return {
            "shape": [int(dim) for dim in detached.shape],
            "dtype": str(tensor.dtype),
            "sum": total,
            "first_values": flat[:16].tolist(),
        }

    def _source_basis_trace(
        self,
        bases: dict[str, torch.Tensor],
    ) -> list[dict[str, Any]]:
        return [
            {
                "name": str(name),
                "basis": self._bounded_tensor_trace(bases[name]),
            }
            for name in sorted(bases)[:16]
        ]

    def _capture_center_objective(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        source_parameter_names: set[str] | None = None,
        profile: dict[str, Any] | None = None,
        basis_seed: int | None = None,
        stream_oja_basis: bool = False,
        use_abh: bool = False,
    ) -> tuple[CandidateEvaluation, "_ActivationCapture"]:
        return self._capture_objective(
            model,
            objective_fn,
            candidate_id="center",
            source_parameter_names=source_parameter_names,
            profile=profile,
            basis_seed=basis_seed,
            stream_oja_basis=stream_oja_basis,
            use_abh=use_abh,
        )

    def _capture_objective(
        self,
        model: torch.nn.Module,
        objective_fn: ObjectiveFn,
        *,
        candidate_id: str,
        source_parameter_names: set[str] | None = None,
        profile: dict[str, Any] | None = None,
        basis_seed: int | None = None,
        stream_oja_basis: bool = False,
        use_abh: bool = False,
    ) -> tuple[CandidateEvaluation, "_ActivationCapture"]:
        if self._uses_seed_pool_dense_noise() and not stream_oja_basis:
            capture = _ActivationCapture(
                max_tokens=int(self.method_config.max_activation_tokens),
                basis_builder=None,
            )
            candidate = evaluate_objective(candidate_id, objective_fn)
            return candidate, capture

        basis_builder = None
        if stream_oja_basis and self._uses_oja_abh_basis():

            def basis_builder(
                module_name: str,
                activation: torch.Tensor,
            ) -> torch.Tensor | None:
                return self._basis_from_activation(
                    activation,
                    seed=basis_seed,
                    module_name=module_name,
                    use_abh=use_abh,
                    profile=profile,
                )

        if self.method_config.stream_activation_basis and not self._uses_abh_perturbation():
            selected_parameters = dict(model.named_parameters())

            def basis_builder(module_name: str, activation: torch.Tensor) -> torch.Tensor | None:
                parameter_name = "weight" if module_name == "" else f"{module_name}.weight"
                return self._build_bases(
                    {module_name: activation}, seed=basis_seed,
                    parameters=selected_parameters, use_abh=use_abh, profile=profile,
                ).get(parameter_name)

        capture = _ActivationCapture(
            max_tokens=int(self.method_config.max_activation_tokens),
            basis_builder=basis_builder,
        )
        pattern = re.compile(str(self.method_config.target_module_regex))
        handles: list[torch.utils.hooks.RemovableHandle] = []

        register_started_at = time.perf_counter()
        for module_name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not pattern.fullmatch(module_name):
                continue
            parameter_name = "weight" if module_name == "" else f"{module_name}.weight"
            if (
                source_parameter_names is not None
                and parameter_name not in source_parameter_names
            ):
                continue
            capture.matched_linear_layers += 1
            handles.append(module.register_forward_hook(capture.hook(module_name)))
        self._profile_add(
            profile,
            f"{candidate_id}_hook_register_time",
            time.perf_counter() - register_started_at,
        )

        try:
            candidate = evaluate_objective(candidate_id, objective_fn)
        finally:
            remove_started_at = time.perf_counter()
            for handle in handles:
                handle.remove()
            self._profile_add(
                profile,
                f"{candidate_id}_hook_remove_time",
                time.perf_counter() - remove_started_at,
            )
        self._profile_add(profile, f"{candidate_id}_hook_call_time", capture.hook_time)
        self._profile_add(profile, f"{candidate_id}_hook_calls", capture.hook_calls)
        return candidate, capture

    def _effective_abh_right_rank(self, feature_dim: int) -> int:
        feature_dim = int(feature_dim)
        if feature_dim <= 0:
            return 0
        max_rank = min(int(self.method_config.abh_right_rank), feature_dim)
        if str(self.method_config.abh_right_rank_policy) == "proportional":
            ratio_rank = int(
                math.ceil(float(self.method_config.abh_right_rank_ratio) * feature_dim)
            )
            min_rank = min(int(self.method_config.abh_right_rank_min), feature_dim)
            return max(1, min(max_rank, max(min_rank, ratio_rank)))
        return max_rank

    def _effective_oja_wide_right_rank(self, feature_dim: int) -> int:
        active_rank = self._effective_abh_right_rank(feature_dim)
        if active_rank <= 0:
            return 0
        wide_rank = int(getattr(self.method_config, "abh_oja_wide_right_rank", 0))
        if wide_rank <= 0:
            return active_rank
        return min(int(feature_dim), max(active_rank, wide_rank))

    def _effective_activation_max_tokens(self, feature_dim: int) -> int:
        fixed_max = int(self.method_config.max_activation_tokens)
        if str(self.method_config.abh_activation_token_policy) != "rank_multiple":
            return fixed_max
        right_rank = (
            self._effective_oja_wide_right_rank(feature_dim)
            if self._uses_oja_abh_basis()
            else self._effective_abh_right_rank(feature_dim)
        )
        if right_rank <= 0:
            return fixed_max
        rank_multiple = int(
            math.ceil(
                float(self.method_config.abh_activation_token_multiplier) * right_rank
            )
        )
        rank_multiple = max(1, rank_multiple)
        if fixed_max > 0:
            return min(fixed_max, rank_multiple)
        return rank_multiple

    def _lagged_oja_bases(
        self,
        parameters: ParameterList,
        *,
        seed: int,
    ) -> dict[str, torch.Tensor]:
        pattern = re.compile(str(self.method_config.target_module_regex))
        bases: dict[str, torch.Tensor] = {}
        for name, param in parameters:
            if param.data.ndim != 2:
                continue
            module_name = _module_name_from_parameter_name(name)
            if not pattern.fullmatch(module_name):
                continue
            feature_dim = int(param.data.shape[1])
            wide_rank = self._effective_oja_wide_right_rank(feature_dim)
            if wide_rank <= 0:
                continue
            previous = self._oja_right_bases.get(module_name)
            if previous is None or tuple(previous.shape) != (feature_dim, wide_rank):
                continue
            active = self._oja_active_basis(module_name, previous)
            bases[name] = active.to(device=param.device).detach().contiguous()
        return bases

    def _lagged_oja_has_initialized_bases(self, parameters: ParameterList) -> bool:
        pattern = re.compile(str(self.method_config.target_module_regex))
        for name, param in parameters:
            if param.data.ndim != 2:
                continue
            module_name = _module_name_from_parameter_name(name)
            if not pattern.fullmatch(module_name):
                continue
            feature_dim = int(param.data.shape[1])
            wide_rank = self._effective_oja_wide_right_rank(feature_dim)
            if wide_rank <= 0:
                continue
            previous = self._oja_right_bases.get(module_name)
            if previous is not None and tuple(previous.shape) == (
                feature_dim,
                wide_rank,
            ):
                return True
        return False

    def _oja_active_basis(
        self,
        module_name: str,
        q: torch.Tensor,
    ) -> torch.Tensor:
        active_rank = min(
            self._effective_abh_right_rank(int(q.shape[0])), int(q.shape[1])
        )
        if active_rank <= 0:
            return q[:, :0].t().detach().contiguous()
        sampling_policy = str(
            getattr(self.method_config, "abh_oja_active_sampling_policy", "top_tail")
        )
        if sampling_policy != "top_tail":
            active = self._oja_weighted_active_basis(
                module_name,
                q,
                active_rank,
                sampling_policy,
            )
            if active is not None:
                return active.t().detach().contiguous()
        top_count = int(getattr(self.method_config, "abh_oja_active_top_count", 0))
        tail_count = int(getattr(self.method_config, "abh_oja_active_tail_count", 0))
        return select_top_tail_basis(
            q,
            module_name=module_name,
            step=self._active_step_number,
            noise_seed=self._active_noise_seed,
            active_rank=active_rank,
            top_count=top_count,
            tail_count=tail_count,
        )

    def _oja_weighted_active_basis(
        self,
        module_name: str,
        q: torch.Tensor,
        active_rank: int,
        sampling_policy: str,
    ) -> torch.Tensor | None:
        rank = int(q.shape[1])
        if rank < active_rank:
            return q[:, :active_rank]
        stable_seed = _stable_int_seed(
            {
                "method": "agzo_oja_active_weighted",
                "module": str(module_name),
                "step": int(self._active_step_number),
                "policy": str(sampling_policy),
                "active_rank": int(active_rank),
                "q_shape": [int(dim) for dim in q.shape],
                "noise_seed": self._active_noise_seed,
            }
        )
        generator = torch.Generator(device=q.device)
        generator.manual_seed(int(stable_seed))
        if sampling_policy == "top48_fixed_last16":
            if active_rank != 64 or rank < 128:
                return q[:, :active_rank]
            return torch.cat([q[:, :48], q[:, 112:128]], dim=1)
        if sampling_policy == "top64_tail64_energy002":
            if rank < 128:
                return q[:, :active_rank]
            return torch.cat(
                [
                    q[:, :64].mul(math.sqrt(0.98)),
                    q[:, 64:128].mul(math.sqrt(0.02)),
                ],
                dim=1,
            )
        if sampling_policy == "population_3top64_1top64tail16":
            if rank < 64:
                return q[:, :active_rank]
            candidate_index = self._active_population_seed_indices.get(
                int(self._active_noise_seed)
                if self._active_noise_seed is not None
                else -1
            )
            if candidate_index is None or candidate_index % 4 != 3:
                return q[:, :64]
            tail_available = max(0, min(rank, 128) - 64)
            if tail_available < 16:
                return q[:, :64]
            tail_indices = (
                torch.randperm(
                    tail_available,
                    device=q.device,
                    generator=generator,
                )[:16]
                + 64
            )
            return torch.cat([q[:, :64], q[:, tail_indices]], dim=1)
        if sampling_policy == "tail64":
            tail_start = int(active_rank)
            tail_end = min(rank, tail_start + active_rank)
            if tail_end - tail_start < active_rank:
                return q[:, :active_rank]
            return q[:, tail_start:tail_end]
        if sampling_policy == "top48_random_orth16":
            return self._oja_top_random_orth_basis(
                q,
                active_rank,
                top_count=48,
                random_count=16,
                generator=generator,
            )
        if sampling_policy == "plumage_all64":
            return self._oja_plumage_active_basis(
                module_name,
                q,
                active_rank,
                fixed_head_count=0,
                generator=generator,
            )
        if sampling_policy == "plumage_top48_tail16":
            return self._oja_plumage_active_basis(
                module_name,
                q,
                active_rank,
                fixed_head_count=48,
                generator=generator,
            )
        if sampling_policy == "fixed32_weighted32_floor025_tau64_a05":
            return self._oja_fixed_head_weighted_tail_basis(
                q,
                active_rank,
                head_count=32,
                floor=0.25,
                tau=64.0,
                alpha=0.5,
                generator=generator,
            )
        if sampling_policy == "fixed16_weighted48_floor020_tau64_a07":
            return self._oja_fixed_head_weighted_tail_basis(
                q,
                active_rank,
                head_count=16,
                floor=0.20,
                tau=64.0,
                alpha=0.7,
                generator=generator,
            )
        if sampling_policy == "pure_hi16_w5_mid12_tail10":
            return self._oja_block_weighted_active_basis(
                q,
                active_rank,
                first16_weight=5.0,
                mid_weight=1.2,
                tail_weight=1.0,
                generator=generator,
            )
        if sampling_policy == "pure_hi16_w6_mid12_tail10":
            return self._oja_block_weighted_active_basis(
                q,
                active_rank,
                first16_weight=6.0,
                mid_weight=1.2,
                tail_weight=1.0,
                generator=generator,
            )
        return None

    def _oja_population_candidate_role(self, probe_seed: int) -> str:
        policy = str(
            getattr(self.method_config, "abh_oja_active_sampling_policy", "top_tail")
        )
        if policy == "population_3top64_1top64tail16":
            candidate_index = self._active_population_seed_indices.get(int(probe_seed))
            if candidate_index is not None and candidate_index % 4 == 3:
                return "top64_tail16"
            return "top64"
        if policy == "top64_tail64_energy002":
            return "top64_tail64_energy002"
        return policy

    def _candidate_oja_basis(
        self,
        name: str,
        q_t: torch.Tensor,
    ) -> torch.Tensor:
        policy = str(
            getattr(self.method_config, "abh_oja_active_sampling_policy", "top_tail")
        )
        resample_per_probe = bool(
            getattr(
                self.method_config,
                "abh_oja_active_resample_per_probe",
                False,
            )
        )
        if not resample_per_probe and policy not in {
            "top64_tail64_energy002",
            "population_3top64_1top64tail16",
        }:
            return q_t
        module_name = _module_name_from_parameter_name(name)
        wide_q = self._oja_right_bases.get(module_name)
        if (
            wide_q is None
            or int(wide_q.shape[0]) != int(q_t.shape[1])
            or int(wide_q.shape[1]) < 64
        ):
            return q_t
        return self._oja_active_basis(
            module_name,
            wide_q.to(device=q_t.device, dtype=q_t.dtype),
        )

    def _oja_fixed_head_weighted_tail_basis(
        self,
        q: torch.Tensor,
        active_rank: int,
        *,
        head_count: int,
        floor: float,
        tau: float,
        alpha: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        rank = int(q.shape[1])
        head_count = max(0, min(int(head_count), int(active_rank), rank))
        remaining = int(active_rank) - head_count
        if remaining <= 0 or rank <= head_count:
            return q[:, :active_rank]
        index = torch.arange(rank - head_count, device=q.device, dtype=torch.float32)
        weights = float(floor) + (1.0 - float(floor)) * torch.pow(
            1.0 + index / float(tau),
            -float(alpha),
        )
        tail_indices = (
            torch.multinomial(
                weights,
                remaining,
                replacement=False,
                generator=generator,
            )
            + head_count
        )
        selected = torch.cat(
            [
                torch.arange(head_count, device=q.device, dtype=torch.long),
                tail_indices.sort().values,
            ]
        )
        return q[:, selected]

    def _oja_top_random_orth_basis(
        self,
        q: torch.Tensor,
        active_rank: int,
        *,
        top_count: int,
        random_count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if int(top_count) + int(random_count) != int(active_rank):
            return q[:, :active_rank]
        if int(q.shape[0]) < int(active_rank):
            return q[:, :active_rank]
        top = q[:, : int(top_count)].float()
        random = torch.randn(
            int(q.shape[0]),
            int(random_count),
            device=q.device,
            dtype=torch.float32,
            generator=generator,
        )
        random = random - top.matmul(top.t().matmul(random))
        try:
            random, _ = torch.linalg.qr(random, mode="reduced")
        except RuntimeError:
            return q[:, :active_rank]
        active = torch.cat([top, random[:, : int(random_count)]], dim=1)
        return active.to(device=q.device, dtype=q.dtype)

    def _oja_plumage_active_basis(
        self,
        module_name: str,
        q: torch.Tensor,
        active_rank: int,
        *,
        fixed_head_count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        rank = int(q.shape[1])
        fixed_head_count = max(
            0,
            min(int(fixed_head_count), int(active_rank), rank),
        )
        sample_count = int(active_rank) - fixed_head_count
        if sample_count <= 0:
            return q[:, :active_rank]
        candidate_start = fixed_head_count
        candidate_count = rank - candidate_start
        if candidate_count < sample_count:
            return q[:, :active_rank]

        strengths = self._oja_spectral_strengths.get(module_name)
        if strengths is None or int(strengths.numel()) != rank:
            strengths = torch.ones(rank, device=q.device, dtype=torch.float64)
        else:
            strengths = strengths.to(device=q.device, dtype=torch.float64)
        probabilities = self._plumage_inclusion_probabilities(
            strengths[candidate_start:],
            sample_count,
        )
        sampled = self._systematic_fixed_size_sample(
            probabilities,
            sample_count,
            generator=generator,
        )
        sampled = sampled + candidate_start
        if fixed_head_count > 0:
            selected = torch.cat(
                [
                    torch.arange(
                        fixed_head_count,
                        device=q.device,
                        dtype=torch.long,
                    ),
                    sampled,
                ]
            )
        else:
            selected = sampled
        selected = selected.sort().values
        self._oja_sampling_probability_min[module_name] = float(
            probabilities.min().detach().cpu().item()
        )
        self._oja_sampling_probability_max[module_name] = float(
            probabilities.max().detach().cpu().item()
        )
        self._oja_sampling_selected_index_mean[module_name] = float(
            selected.float().mean().detach().cpu().item()
        )
        return q[:, selected]

    @staticmethod
    def _plumage_inclusion_probabilities(
        strengths: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        strengths = torch.nan_to_num(
            strengths.detach().to(dtype=torch.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        count = int(strengths.numel())
        sample_count = max(0, min(int(sample_count), count))
        if sample_count <= 0:
            return torch.zeros_like(strengths)
        if sample_count >= count:
            return torch.ones_like(strengths)
        if float(strengths.sum().item()) <= 0.0:
            return torch.full_like(strengths, float(sample_count) / float(count))

        sorted_strengths, _ = torch.sort(strengths, descending=True)
        suffix_sum = torch.flip(
            torch.cumsum(torch.flip(sorted_strengths, dims=[0]), dim=0),
            dims=[0],
        )
        saturated = torch.arange(
            sample_count,
            device=strengths.device,
            dtype=torch.long,
        )
        tau = suffix_sum[:sample_count] / (
            float(sample_count) - saturated.to(dtype=torch.float64)
        )
        left_ok = torch.ones_like(tau, dtype=torch.bool)
        if sample_count > 1:
            left_ok[1:] = sorted_strengths[: sample_count - 1] >= tau[1:]
        right_ok = sorted_strengths[:sample_count] <= tau
        valid = left_ok & right_ok
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        saturated_count = (
            int(valid_indices[0].item()) if valid_indices.numel() > 0 else 0
        )
        threshold = tau[saturated_count].clamp_min(torch.finfo(torch.float64).tiny)
        probabilities = torch.clamp(strengths / threshold, max=1.0)
        return probabilities

    @staticmethod
    def _systematic_fixed_size_sample(
        probabilities: torch.Tensor,
        sample_count: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        sample_count = int(sample_count)
        if sample_count <= 0:
            return torch.empty(
                0,
                device=probabilities.device,
                dtype=torch.long,
            )
        cumulative = probabilities.to(dtype=torch.float64).cumsum(dim=0)
        offset = torch.rand(
            (),
            device=probabilities.device,
            dtype=torch.float64,
            generator=generator,
        )
        thresholds = offset + torch.arange(
            sample_count,
            device=probabilities.device,
            dtype=torch.float64,
        )
        thresholds = thresholds.clamp_max(
            cumulative[-1] - torch.finfo(torch.float64).eps
        )
        return torch.searchsorted(cumulative, thresholds, right=False)

    def _oja_block_weighted_active_basis(
        self,
        q: torch.Tensor,
        active_rank: int,
        *,
        first16_weight: float,
        mid_weight: float,
        tail_weight: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        rank = int(q.shape[1])
        if rank <= active_rank:
            return q[:, :active_rank]
        weights = torch.full(
            (rank,),
            float(tail_weight),
            device=q.device,
            dtype=torch.float32,
        )
        weights[: min(64, rank)] = float(mid_weight)
        weights[: min(16, rank)] = float(first16_weight)
        selected = torch.multinomial(
            weights,
            int(active_rank),
            replacement=False,
            generator=generator,
        )
        return q[:, selected.sort().values]

    def _lagged_oja_update_source(self, *, seed: int) -> str:
        source = str(self.method_config.abh_oja_lagged_update_source)
        if source in {"plus", "minus"}:
            return source
        stable = _stable_int_seed(
            {
                "method": "agzo_abh_oja_lagged_update_source",
                "seed": int(seed),
                "step": int(self._active_step_number),
            }
        )
        return "plus" if stable % 2 == 0 else "minus"

    def _build_bases(
        self,
        activations: dict[str, torch.Tensor],
        *,
        seed: int | None = None,
        parameters: dict[str, torch.nn.Parameter] | None = None,
        use_abh: bool = False,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        bases: dict[str, torch.Tensor] = {}
        for module_name, activation in activations.items():
            basis = self._basis_from_activation(
                activation,
                seed=seed,
                module_name=module_name,
                use_abh=use_abh,
                profile=profile,
            )
            if basis is None:
                continue
            parameter_name = "weight" if module_name == "" else f"{module_name}.weight"
            param = parameters.get(parameter_name) if parameters is not None else None
            if (
                self._uses_source_profile()
                and not self._uses_abh_perturbation()
                and param is not None
            ):
                basis = basis / (basis.norm(p=2, dim=1, keepdim=True) + 1e-12)
                basis = basis.to(device=param.device, dtype=param.dtype)
            bases[parameter_name] = basis
        return bases

    def _basis_from_activation(
        self,
        activation: torch.Tensor,
        *,
        seed: int | None = None,
        module_name: str | None = None,
        use_abh: bool = False,
        profile: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        started_at = time.perf_counter()
        if activation.ndim < 2 or not torch.is_floating_point(activation):
            return None
        tokens = activation.reshape(-1, activation.shape[-1])
        if tokens.numel() == 0:
            return None
        max_tokens = self._effective_activation_max_tokens(int(tokens.shape[1]))
        if max_tokens > 0 and tokens.shape[0] > max_tokens:
            tokens = tokens[:max_tokens]
        tokens = tokens.float()
        token_norm = torch.linalg.vector_norm(tokens)
        if not torch.isfinite(token_norm) or float(token_norm.item()) <= 0.0:
            return None
        if self._uses_oja_abh_basis() and module_name is not None:
            result = self._update_oja_right_basis(
                module_name,
                tokens,
                seed=seed,
                profile=profile,
            )
            self._profile_add(
                profile,
                "basis_from_activation_total_time",
                time.perf_counter() - started_at,
            )
            return result
        if self._uses_covariance_momentum_basis() and module_name is not None:
            cov = self._update_cov_momentum(module_name, tokens)
            if self._uses_momentum_basis_only() or use_abh:
                result = self._basis_from_covariance_rank(cov)
                self._profile_add(
                    profile,
                    "basis_from_activation_total_time",
                    time.perf_counter() - started_at,
                )
                return result
        backend = str(self.method_config.subspace_backend)
        if backend == "power_iteration":
            result = self._basis_from_activation_power_iteration(tokens, seed=seed)
            self._profile_add(
                profile,
                "basis_from_activation_total_time",
                time.perf_counter() - started_at,
            )
            return result
        if backend != "svd":
            return None
        result = self._basis_from_activation_svd(tokens)
        self._profile_add(
            profile,
            "basis_from_activation_total_time",
            time.perf_counter() - started_at,
        )
        return result

    def _basis_from_activation_svd(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        try:
            _, _, vh = torch.linalg.svd(tokens, full_matrices=False)
        except RuntimeError:
            return None
        basis_rank = min(
            int(self.method_config.rank),
            int(tokens.shape[1]),
            int(vh.shape[0]),
        )
        if basis_rank <= 0:
            return None
        return vh[:basis_rank].detach().contiguous()

    def _basis_from_activation_power_iteration(
        self,
        tokens: torch.Tensor,
        *,
        seed: int | None = None,
    ) -> torch.Tensor | None:
        feature_dim = int(tokens.shape[1])
        basis_rank = min(int(self.method_config.rank), feature_dim)
        if basis_rank <= 0:
            return None
        generator = None
        if seed is not None:
            generator = torch.Generator(device=tokens.device)
            generator.manual_seed(int(seed))
        try:
            q = torch.randn(
                (feature_dim, basis_rank),
                device=tokens.device,
                dtype=tokens.dtype,
                generator=generator,
            )
            q, _ = torch.linalg.qr(q, mode="reduced")
            for _ in range(int(self.method_config.power_iterations)):
                q = tokens.t().matmul(tokens.matmul(q))
                q, _ = torch.linalg.qr(q, mode="reduced")
        except RuntimeError:
            return None
        return q.t().detach().contiguous()

    def _basis_from_covariance_rank(
        self,
        covariance: torch.Tensor,
    ) -> torch.Tensor | None:
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            return None
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
        except RuntimeError:
            return None
        if eigenvalues.numel() == 0 or not torch.isfinite(eigenvalues).all():
            return None
        positive = torch.clamp(eigenvalues, min=0.0)
        max_eigenvalue = float(positive.max().item())
        if max_eigenvalue <= 0.0:
            return None
        basis_rank = min(
            int(self.method_config.rank),
            int(covariance.shape[0]),
            int(eigenvectors.shape[1]),
        )
        if basis_rank <= 0:
            return None
        order = torch.argsort(eigenvalues, descending=True)
        basis = eigenvectors[:, order[:basis_rank]]
        return basis.t().detach().contiguous()

    def _update_cov_momentum(
        self,
        module_name: str,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        covariance = self._activation_covariance(tokens)
        beta = float(self.method_config.abh_momentum_beta)
        previous = self._cov_momentum.get(module_name)
        if previous is not None and tuple(previous.shape) == tuple(covariance.shape):
            previous = previous.to(device=covariance.device, dtype=covariance.dtype)
            updated = previous.mul(beta).add(covariance, alpha=1.0 - beta)
        else:
            updated = covariance.detach().clone()
        updated = updated.detach().contiguous()
        self._cov_momentum[module_name] = updated
        self._cov_momentum_updates[module_name] += 1
        return updated

    def _activation_covariance(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.detach().float()
        denom = max(1, int(tokens.shape[0]))
        covariance = tokens.t().matmul(tokens).div(float(denom))
        return covariance.contiguous()

    def _update_oja_right_basis(
        self,
        module_name: str,
        tokens: torch.Tensor,
        *,
        seed: int | None = None,
        profile: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        total_started_at = time.perf_counter()
        feature_dim = int(tokens.shape[1])
        wide_rank = self._effective_oja_wide_right_rank(feature_dim)
        if wide_rank <= 0:
            return None
        previous = self._oja_right_bases.get(module_name)
        if previous is not None and tuple(previous.shape) == (feature_dim, wide_rank):
            q = previous.to(device=tokens.device, dtype=torch.float32)
        else:
            init_started_at = time.perf_counter()
            q = self._initialize_oja_right_basis(
                module_name,
                feature_dim=feature_dim,
                right_rank=wide_rank,
                device=tokens.device,
                seed=seed,
            )
            self._profile_add(
                profile,
                "oja_init_time",
                time.perf_counter() - init_started_at,
            )
        if previous is not None and not self._should_update_oja_basis(module_name):
            self._profile_add(profile, "oja_update_skipped", 1)
            self._profile_add(
                profile,
                "oja_total_time",
                time.perf_counter() - total_started_at,
            )
            return self._oja_active_basis(module_name, q)
        q_before_update = q.detach().float().clone()
        tokens = tokens.detach().float()
        update_rule = str(self.method_config.abh_oja_q_update_rule)
        if update_rule == "osd":
            q = self._update_oja_right_basis_osd(
                module_name,
                tokens,
                q,
                profile=profile,
            )
            if q is None:
                return None
        else:
            eta = float(self._effective_oja_eta())
            if eta > 0.0:
                gram_started_at = time.perf_counter()
                y = oja_covariance_action(tokens, q)
                self._profile_add(
                    profile,
                    "oja_hthq_time",
                    time.perf_counter() - gram_started_at,
                )
                if update_rule in {"ema_oja", "tangent_ema_oja"}:
                    if update_rule == "tangent_ema_oja":
                        tangent_started_at = time.perf_counter()
                        y = y - q.matmul(q.t().matmul(y))
                        self._profile_add(
                            profile,
                            "oja_tangent_project_time",
                            time.perf_counter() - tangent_started_at,
                        )
                    beta = float(self.method_config.abh_oja_q_ema_beta)
                    previous_y = self._oja_q_ema_y.get(module_name)
                    if previous_y is None or tuple(previous_y.shape) != tuple(y.shape):
                        previous_y = torch.zeros_like(y)
                    else:
                        previous_y = previous_y.to(device=y.device, dtype=y.dtype)
                    y = previous_y.mul(beta).add(y, alpha=1.0 - beta)
                    self._oja_q_ema_y[module_name] = y.detach().contiguous()
                q = q.add(y, alpha=eta)
            try:
                qr_started_at = time.perf_counter()
                q = orthonormalize_columns(q)
                self._profile_add(
                    profile,
                    "oja_qr_time",
                    time.perf_counter() - qr_started_at,
                )
            except RuntimeError:
                return None
            self._record_oja_q_metrics(module_name, q)
        q = q[:, :wide_rank].detach().contiguous()
        projected = tokens.matmul(q)
        strengths = projected.pow(2).mean(dim=0).clamp_min(0.0).sqrt()
        self._oja_spectral_strengths[module_name] = strengths.detach().contiguous()
        self._record_oja_q_change_metrics(module_name, q_before_update, q)
        self._oja_right_bases[module_name] = q
        self._oja_basis_updates[module_name] += 1
        self._profile_add(
            profile,
            "oja_total_time",
            time.perf_counter() - total_started_at,
        )
        return self._oja_active_basis(module_name, q)

    def _update_oja_right_basis_osd(
        self,
        module_name: str,
        tokens: torch.Tensor,
        q: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        started_at = time.perf_counter()
        q_lr = float(self._effective_osd_q_lr())
        if q_lr <= 0.0:
            q_lr = float(self._effective_oja_eta())
        q_param = q.detach().float().clone().requires_grad_(True)
        lambda_q = float(self.method_config.abh_oja_q_orth_lambda)
        try:
            with torch.enable_grad():
                h_norm = torch.linalg.vector_norm(tokens)
                if not torch.isfinite(h_norm) or float(h_norm.item()) <= 0.0:
                    return q.detach().float()
                h_normalized = tokens.div(h_norm + 1e-12)
                hq_started_at = time.perf_counter()
                hq = h_normalized.matmul(q_param)
                self._profile_add(
                    profile,
                    "oja_osd_hq_time",
                    time.perf_counter() - hq_started_at,
                )
                recon_started_at = time.perf_counter()
                reconstruction = hq.matmul(q_param.t())
                reconstruction_loss = (reconstruction - h_normalized).pow(2).sum()
                self._profile_add(
                    profile,
                    "oja_osd_recon_time",
                    time.perf_counter() - recon_started_at,
                )
                gram = q_param.t().matmul(q_param)
                eye = torch.eye(
                    int(gram.shape[0]),
                    device=gram.device,
                    dtype=gram.dtype,
                )
                orth_error_tensor = (gram - eye).pow(2).sum()
                loss = reconstruction_loss
                if lambda_q > 0.0:
                    loss = loss + lambda_q * orth_error_tensor
                backward_started_at = time.perf_counter()
                loss.backward()
                self._profile_add(
                    profile,
                    "oja_osd_backward_time",
                    time.perf_counter() - backward_started_at,
                )
                step_started_at = time.perf_counter()
                updated = self._apply_osd_q_optimizer_step(
                    module_name,
                    q_param,
                    q_lr=q_lr,
                )
                self._profile_add(
                    profile,
                    "oja_osd_optimizer_time",
                    time.perf_counter() - step_started_at,
                )
        except RuntimeError:
            return None
        self._oja_q_loss[module_name] = float(loss.detach().float().item())
        self._oja_q_orth_error[module_name] = float(
            orth_error_tensor.detach().float().sqrt().item()
        )
        self._profile_add(
            profile,
            "oja_osd_total_time",
            time.perf_counter() - started_at,
        )
        return updated

    def _apply_osd_q_optimizer_step(
        self,
        module_name: str,
        q_param: torch.Tensor,
        *,
        q_lr: float,
    ) -> torch.Tensor:
        grad = q_param.grad
        if grad is None:
            return q_param.detach().float().contiguous()
        grad = grad.detach().float()
        q_data = q_param.detach().float()
        if str(self.method_config.abh_oja_q_optimizer) != "adamw":
            return q_data.add(grad, alpha=-float(q_lr)).contiguous()

        m = self._oja_q_adam_m.get(module_name)
        v = self._oja_q_adam_v.get(module_name)
        if m is None or tuple(m.shape) != tuple(q_data.shape):
            m = torch.zeros_like(q_data)
            v = torch.zeros_like(q_data)
            self._oja_q_adam_steps[module_name] = 0
        else:
            m = m.to(device=q_data.device, dtype=q_data.dtype)
            v = v.to(device=q_data.device, dtype=q_data.dtype)
        self._oja_q_adam_steps[module_name] += 1
        step = int(self._oja_q_adam_steps[module_name])
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        weight_decay = 0.01
        m = m.mul(beta1).add(grad, alpha=1.0 - beta1)
        v = v.mul(beta2).addcmul(grad, grad, value=1.0 - beta2)
        m_hat = m.div(1.0 - beta1**step)
        v_hat = v.div(1.0 - beta2**step)
        update = m_hat.div(v_hat.sqrt().add(eps))
        if weight_decay > 0.0:
            update = update.add(q_data, alpha=weight_decay)
        updated = q_data.add(update, alpha=-float(q_lr)).contiguous()
        self._oja_q_adam_m[module_name] = m.detach().contiguous()
        self._oja_q_adam_v[module_name] = v.detach().contiguous()
        return updated

    def _record_oja_q_metrics(self, module_name: str, q: torch.Tensor) -> None:
        try:
            gram = q.float().t().matmul(q.float())
            eye = torch.eye(int(gram.shape[0]), device=gram.device, dtype=gram.dtype)
            self._oja_q_orth_error[module_name] = float(
                torch.linalg.vector_norm(gram - eye).detach().cpu().item()
            )
        except RuntimeError:
            return

    def _record_oja_q_change_metrics(
        self,
        module_name: str,
        previous_q: torch.Tensor,
        updated_q: torch.Tensor,
    ) -> None:
        try:
            previous = previous_q.float()
            updated = updated_q.float()
            if tuple(previous.shape) != tuple(updated.shape):
                return
            delta = torch.linalg.vector_norm(updated - previous)
            self._oja_q_delta_norm[module_name] = float(delta.detach().cpu().item())
            gram = previous.t().matmul(updated)
            rank = max(1, int(min(previous.shape[1], updated.shape[1])))
            overlap = gram.pow(2).sum().div(float(rank))
            self._oja_q_overlap[module_name] = float(overlap.detach().cpu().item())
        except RuntimeError:
            return

    def _effective_osd_q_lr(self) -> float:
        configured = float(self.method_config.abh_oja_q_lr)
        if configured > 0.0:
            return configured
        interval = max(1, int(self._effective_oja_update_interval()))
        return 1.0 / float(interval)

    def _should_update_oja_basis(self, module_name: str) -> bool:
        interval = self._effective_oja_update_interval()
        if interval <= 1:
            return True
        if module_name not in self._oja_right_bases:
            return True
        active_index = max(
            0,
            int(self._active_step_number) - int(self.method_config.abh_warmup_steps),
        )
        return active_index % interval == 0

    def _effective_oja_update_interval(self) -> int:
        interval = max(1, int(self.method_config.abh_oja_update_interval))
        late_start = int(self.method_config.abh_oja_late_update_start_step)
        if late_start > 0 and int(self._active_step_number) >= late_start:
            return max(1, int(self.method_config.abh_oja_late_update_interval))
        return interval

    def _oja_update_schedule_kind(self) -> str:
        base = max(1, int(self.method_config.abh_oja_update_interval))
        late_start = int(self.method_config.abh_oja_late_update_start_step)
        late = max(1, int(self.method_config.abh_oja_late_update_interval))
        if late_start > 0 and late != base:
            return "late_step_interval"
        if base > 1:
            return "fixed_interval"
        return "every_step"

    def _effective_oja_eta(self) -> float:
        eta = float(self.method_config.abh_oja_eta)
        interval = int(self.method_config.abh_oja_eta_decay_interval)
        if interval <= 0:
            return eta
        factor = float(self.method_config.abh_oja_eta_decay_factor)
        step_number = max(1, int(self._active_step_number))
        decay_count = max(0, (step_number - 1) // interval)
        return eta * (factor**decay_count)

    def _oja_eta_schedule_kind(self) -> str:
        interval = int(self.method_config.abh_oja_eta_decay_interval)
        factor = float(self.method_config.abh_oja_eta_decay_factor)
        if interval <= 0 or factor == 1.0:
            return "constant"
        return "step_decay"

    def _initialize_oja_right_basis(
        self,
        module_name: str,
        *,
        feature_dim: int,
        right_rank: int,
        device: torch.device,
        seed: int | None = None,
    ) -> torch.Tensor:
        return initialize_oja_basis(
            module_name=module_name,
            feature_dim=feature_dim,
            right_rank=right_rank,
            device=device,
            seed=seed,
        )

    def _evaluate_candidate(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
        objective_fn: ObjectiveFn,
        *,
        candidate_id: str,
        seed: int,
        eps: float,
        scaling: float,
        source_trace: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
        profile_prefix: str = "evaluate_candidate",
        noise_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
    ) -> CandidateEvaluation:
        self._apply_agzo_perturbation(
            parameters,
            bases,
            seed=seed,
            eps=eps,
            scaling=scaling,
            source_trace=source_trace,
            profile=profile,
            profile_prefix=f"{profile_prefix}_apply",
            noise_cache=noise_cache,
        )
        try:
            return evaluate_objective(candidate_id, objective_fn)
        finally:
            self._apply_agzo_perturbation(
                parameters,
                bases,
                seed=seed,
                eps=eps,
                scaling=-float(scaling),
                source_trace=source_trace,
                profile=profile,
                profile_prefix=f"{profile_prefix}_restore",
                noise_cache=noise_cache,
            )

    @torch.no_grad()
    def _apply_agzo_perturbation(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
        *,
        seed: int,
        eps: float,
        scaling: float,
        source_trace: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
        profile_prefix: str = "apply_perturbation",
        noise_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
    ) -> None:
        total_started_at = time.perf_counter()
        previous_noise_seed = self._active_noise_seed
        self._active_noise_seed = int(seed)
        torch.manual_seed(int(seed))
        try:
            for index, (name, param) in enumerate(parameters):
                noise_started_at = time.perf_counter()
                basis = bases.get(name)
                can_use_lowrank_components = (
                    source_trace is None
                    and not self._uses_source_profile()
                    and not self._uses_population_dense_noise()
                    and basis is not None
                    and param.data.ndim == 2
                    and int(basis.shape[1]) == int(param.data.shape[1])
                )
                cache_restore_factors = (
                    bool(
                        getattr(
                            self.method_config,
                            "abh_population_cache_restore_factors",
                            False,
                        )
                    )
                    and self._can_cache_population_restore_factors()
                )
                lowrank_cache_key = (
                    "agzo_population_lowrank_components",
                    int(self._active_step_number),
                    int(seed),
                    str(name),
                    id(basis),
                    str(param.device),
                    str(param.dtype),
                )
                cached_lowrank_components = (
                    noise_cache.get(lowrank_cache_key)
                    if cache_restore_factors and noise_cache is not None
                    else None
                )
                if cached_lowrank_components is not None:
                    lowrank_components = cached_lowrank_components
                    _, cached_small_right, _ = lowrank_components
                    # Preserve the legacy seed-replay RNG position for any
                    # dense fallback tensors that follow this parameter.
                    torch.randn(
                        tuple(int(dim) for dim in cached_small_right.shape),
                        device=param.device,
                        dtype=param.dtype,
                    )
                    self._profile_add(profile, "population_factor_cache.hit", 1)
                else:
                    lowrank_components = (
                        self._abh_small_b_components_for_param(
                            name,
                            param,
                            basis,
                            projected_grad=1.0,
                            profile=profile,
                        )
                        if can_use_lowrank_components
                        else None
                    )
                    if (
                        lowrank_components is not None
                        and cache_restore_factors
                        and noise_cache is not None
                    ):
                        noise_cache[lowrank_cache_key] = lowrank_components
                        self._profile_add(profile, "population_factor_cache.store", 1)
                if lowrank_components is not None:
                    a_matrix, small_right, right_basis = lowrank_components
                    right_factor = small_right.to(
                        device=param.device,
                        dtype=param.dtype,
                    ).matmul(right_basis)
                    dense_residual_ratio = float(
                        self.method_config.abh_dense_residual_ratio
                    )
                    keep_scale = math.sqrt(max(0.0, 1.0 - dense_residual_ratio))
                    self._profile_add(
                        profile,
                        f"{profile_prefix}.noise_time",
                        time.perf_counter() - noise_started_at,
                    )
                    add_started_at = time.perf_counter()
                    param.addmm_(
                        a_matrix.to(device=param.device, dtype=param.dtype),
                        right_factor,
                        alpha=float(eps) * float(scaling) * keep_scale,
                    )
                    if dense_residual_ratio > 0.0:
                        residual_started_at = time.perf_counter()
                        residual = torch.randn_like(param.data)
                        residual_norm = _safe_tensor_norm(residual)
                        lowrank_norm = _safe_tensor_norm(small_right)
                        if (
                            residual_norm is not None
                            and lowrank_norm is not None
                            and float(residual_norm) > 0.0
                            and float(lowrank_norm) > 0.0
                        ):
                            residual = residual.mul(
                                float(lowrank_norm) / (float(residual_norm) + 1e-12)
                            )
                            param.add_(
                                residual,
                                alpha=(
                                    float(eps)
                                    * float(scaling)
                                    * math.sqrt(dense_residual_ratio)
                                ),
                            )
                        self._profile_add(
                            profile,
                            f"{profile_prefix}.dense_residual_time",
                            time.perf_counter() - residual_started_at,
                        )
                    self._profile_add(
                        profile,
                        f"{profile_prefix}.param_add_time",
                        time.perf_counter() - add_started_at,
                    )
                    continue
                z = self._noise_for_param(
                    name,
                    param,
                    bases,
                    profile=profile,
                    noise_cache=noise_cache,
                )
                self._profile_add(
                    profile,
                    f"{profile_prefix}.noise_time",
                    time.perf_counter() - noise_started_at,
                )
                param_before = (
                    self._bounded_tensor_trace(param)
                    if source_trace is not None and index < 16
                    else None
                )
                add_started_at = time.perf_counter()
                if self._uses_source_profile():
                    perturb = (float(scaling) * z * float(eps)).to(
                        dtype=param.dtype,
                        device=param.device,
                    )
                    param.add_(perturb)
                else:
                    perturb = z.mul(float(eps) * float(scaling))
                    param.add_(z, alpha=float(eps) * float(scaling))
                self._profile_add(
                    profile,
                    f"{profile_prefix}.param_add_time",
                    time.perf_counter() - add_started_at,
                )
                if source_trace is not None and index < 16:
                    source_trace.append(
                        {
                            "name": str(name),
                            "scaling_factor": float(scaling),
                            "z": self._bounded_tensor_trace(z),
                            "perturb": self._bounded_tensor_trace(perturb),
                            "param_before": param_before,
                            "param_after": self._bounded_tensor_trace(param),
                        }
                    )
        finally:
            self._active_noise_seed = previous_noise_seed
        self._profile_add(
            profile,
            f"{profile_prefix}.total_time",
            time.perf_counter() - total_started_at,
        )

    def _can_cache_population_restore_factors(self) -> bool:
        return (
            str(self.method_config.estimator_mode) == "loren_population"
            and str(self.method_config.abh_update_transform) == "none"
            and str(self.method_config.abh_probe_transform) == "none"
            and str(self.method_config.abh_left_factor)
            in {"orthogonal_ab_random", "gaussian_ab_random"}
            and float(self.method_config.abh_dense_residual_ratio) == 0.0
        )

    @torch.no_grad()
    def _apply_agzo_update(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
        *,
        seed: int,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
        source_trace: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
        noise_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
        collect_parameter_summaries: bool = True,
    ) -> str:
        total_started_at = time.perf_counter()
        previous_noise_seed = self._active_noise_seed
        self._active_noise_seed = int(seed)
        torch.manual_seed(int(seed))
        parameter_summaries = []
        try:
            for index, (name, param) in enumerate(parameters):
                noise_started_at = time.perf_counter()
                z, update = self._noise_and_update_for_param(
                    name,
                    param,
                    bases,
                    projected_grad=float(projected_grad),
                    profile=profile,
                    noise_cache=noise_cache,
                )
                self._profile_add(
                    profile,
                    "update.noise_time",
                    time.perf_counter() - noise_started_at,
                )
                tensor_started_at = time.perf_counter()
                if weight_decay:
                    update = update.add(param.data, alpha=float(weight_decay))
                scale = -float(learning_rate)
                self._profile_add(
                    profile,
                    "update.update_tensor_time",
                    time.perf_counter() - tensor_started_at,
                )
                trace_started_at = time.perf_counter()
                update_trace.add_delta(name, update, scale=scale)
                self._profile_add(
                    profile,
                    "update.trace_time",
                    time.perf_counter() - trace_started_at,
                )
                param_before = (
                    self._bounded_tensor_trace(param)
                    if source_trace is not None and index < 16
                    else None
                )
                if collect_parameter_summaries:
                    parameter_summaries.append(
                        self._guided_delta_parameter_summary(
                            name,
                            param,
                            bases,
                            update,
                        )
                    )
                add_started_at = time.perf_counter()
                param.add_(update, alpha=scale)
                self._profile_add(
                    profile,
                    "update.param_add_time",
                    time.perf_counter() - add_started_at,
                )
                if source_trace is not None and index < 16:
                    source_trace.append(
                        {
                            "name": str(name),
                            "z": self._bounded_tensor_trace(z),
                            "grad": self._bounded_tensor_trace(update),
                            "param_before": param_before,
                            "param_after": self._bounded_tensor_trace(param),
                        }
                    )
        finally:
            self._active_noise_seed = previous_noise_seed
        self._profile_add(
            profile,
            "update.total_time",
            time.perf_counter() - total_started_at,
        )
        return self._guided_delta_payload_checksum(
            seed=seed,
            projected_grad=projected_grad,
            update_scale=-float(learning_rate),
            parameter_summaries=parameter_summaries,
        )

    def _noise_and_update_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        bases: dict[str, torch.Tensor],
        *,
        projected_grad: float,
        profile: dict[str, Any] | None = None,
        noise_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transform = str(self.method_config.abh_update_transform)
        if transform not in {
            "ab_adam",
            "ab_momentum",
            "muon_ns_small_b",
            "muon_ns_small_b_momentum",
        }:
            if str(self.method_config.abh_probe_transform) == "adamu_b":
                pair = self._abh_adamu_b_noise_and_update_for_param(
                    name,
                    param,
                    bases.get(name),
                    projected_grad=float(projected_grad),
                    profile=profile,
                )
                if pair is not None:
                    return pair
            z = self._noise_for_param(
                name,
                param,
                bases,
                profile=profile,
                noise_cache=noise_cache,
            )
            return z, z.mul(float(projected_grad))
        if (
            not self._uses_active_abh_noise()
            or self._uses_population_dense_noise()
            or self._uses_seed_pool_dense_noise()
            or self._uses_mezo_mix_dense_noise()
            or self._uses_warmup_dense_noise()
        ):
            z = self._noise_for_param(
                name,
                param,
                bases,
                profile=profile,
                noise_cache=noise_cache,
            )
            return z, z.mul(float(projected_grad))
        basis = bases.get(name)
        if (
            basis is None
            or param.data.ndim != 2
            or int(basis.shape[1]) != int(param.data.shape[1])
        ):
            z = self._noise_for_param(
                name,
                param,
                bases,
                profile=profile,
                noise_cache=noise_cache,
            )
            return z, z.mul(float(projected_grad))
        if transform in {"ab_adam", "ab_momentum"}:
            pair = self._abh_ab_momentum_noise_and_update_for_param(
                name,
                param,
                basis,
                projected_grad=float(projected_grad),
                use_adam=(transform == "ab_adam"),
                profile=profile,
            )
            if pair is not None:
                return pair
        pair = self._abh_muon_small_b_noise_and_update_for_param(
            name,
            param,
            basis,
            projected_grad=float(projected_grad),
            use_momentum=(transform == "muon_ns_small_b_momentum"),
            profile=profile,
        )
        if pair is not None:
            return pair
        z = self._noise_for_param(
            name,
            param,
            bases,
            profile=profile,
            noise_cache=noise_cache,
        )
        return z, z.mul(float(projected_grad))

    def _noise_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        bases: dict[str, torch.Tensor],
        *,
        profile: dict[str, Any] | None = None,
        noise_cache: dict[tuple[Any, ...], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cache_key = self._noise_cache_key(name, param, bases)
        if noise_cache is not None and cache_key is not None:
            cached = noise_cache.get(cache_key)
            if cached is not None:
                self._profile_add(profile, "noise_cache.hit", 1)
                return cached
        if (
            self._uses_population_dense_noise()
            or self._uses_seed_pool_dense_noise()
            or self._uses_mezo_mix_dense_noise()
            or self._uses_warmup_dense_noise()
        ):
            noise = torch.randn_like(param.data)
            if noise_cache is not None and cache_key is not None:
                noise_cache[cache_key] = noise
                self._profile_add(profile, "noise_cache.store", 1)
            if self._uses_population_dense_noise():
                self._profile_add(profile, "abh_population_dense.dense_parameters", 1)
            elif self._uses_seed_pool_dense_noise():
                self._profile_add(profile, "abh_seed_pool_dense.dense_parameters", 1)
            elif self._uses_warmup_dense_noise():
                self._profile_add(profile, "abh_warmup_dense.dense_parameters", 1)
            else:
                self._profile_add(profile, "abh_mezo_mix.dense_parameters", 1)
            return noise
        basis = bases.get(name)
        if (
            basis is not None
            and param.data.ndim == 2
            and int(basis.shape[1]) == int(param.data.shape[1])
        ):
            if self._uses_active_abh_noise():
                noise = self._abh_noise_for_param(name, param, basis, profile=profile)
                if noise_cache is not None and cache_key is not None:
                    noise_cache[cache_key] = noise
                    self._profile_add(profile, "noise_cache.store", 1)
                return noise
            rank = self._effective_basis_rank(param, basis)
            if rank <= 0:
                noise = torch.randn_like(param.data)
                if noise_cache is not None and cache_key is not None:
                    noise_cache[cache_key] = noise
                    self._profile_add(profile, "noise_cache.store", 1)
                return noise
            effective_basis = basis[:rank]
            coeffs = torch.randn(
                (int(param.data.shape[0]), rank),
                device=param.device,
                dtype=param.dtype,
            )
            noise = coeffs.matmul(
                effective_basis.to(device=param.device, dtype=param.dtype)
            )
            if rank > 1:
                noise = noise.div(rank**0.5)
            if noise_cache is not None and cache_key is not None:
                noise_cache[cache_key] = noise
                self._profile_add(profile, "noise_cache.store", 1)
            return noise
        noise = torch.randn_like(param.data)
        if noise_cache is not None and cache_key is not None:
            noise_cache[cache_key] = noise
            self._profile_add(profile, "noise_cache.store", 1)
        return noise

    def _uses_mezo_mix_dense_noise(self) -> bool:
        if not self._uses_active_abh_noise():
            return False
        interval = int(self.method_config.abh_mezo_mix_interval)
        dense_steps = int(self.method_config.abh_mezo_mix_dense_steps)
        if interval <= 0 or dense_steps <= 0:
            return False
        active_index = max(
            0,
            int(self._active_step_number) - int(self.method_config.abh_warmup_steps) - 1,
        )
        return active_index % interval >= interval - dense_steps

    def _uses_population_dense_noise(self) -> bool:
        return (
            self._active_noise_seed is not None
            and int(self._active_noise_seed) in self._active_population_dense_seeds
        )

    def _uses_seed_pool_dense_noise(self) -> bool:
        return (
            bool(self.method_config.abh_seed_pool_dense_noise)
            and self._uses_active_abh_noise()
            and int(getattr(self.method_config, "abh_num_noise", 1)) > 1
        )

    def _uses_abh_seed_pool_screening(
        self,
        num_abh_noise: int,
        estimator_mode: str,
    ) -> bool:
        screen_until_step = int(
            getattr(self.method_config, "abh_seed_pool_screen_until_step", 0)
        )
        return (
            screen_until_step > 0
            and int(self._active_step_number) <= screen_until_step
            and self._uses_abh_seed_pool(num_abh_noise, estimator_mode)
        )

    def _uses_warmup_dense_noise(self) -> bool:
        return (
            bool(self.method_config.abh_warmup_dense_noise)
            and self._uses_abh_perturbation()
            and int(self.method_config.abh_warmup_steps) > 0
            and int(self._active_step_number) <= int(self.method_config.abh_warmup_steps)
        )

    def _noise_cache_key(
        self,
        name: str,
        param: torch.nn.Parameter,
        bases: dict[str, torch.Tensor],
    ) -> tuple[Any, ...] | None:
        if not self._uses_active_abh_noise():
            return None
        if str(self.method_config.abh_noise_cache) != "step_full":
            return None
        basis = bases.get(name)
        basis_id = id(basis) if basis is not None else None
        return (
            int(self._active_step_number),
            str(name),
            tuple(int(dim) for dim in param.data.shape),
            str(param.device),
            str(param.dtype),
            basis_id,
        )

    def _abh_noise_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        q_t: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        q_t = self._candidate_oja_basis(name, q_t)
        rank = self._effective_basis_rank(param, q_t)
        if rank <= 0:
            return torch.randn_like(param.data)
        q_t = self._abh_q_dropout_basis(name, q_t)
        rank = min(rank, int(q_t.shape[0]))
        span_rank = int(q_t.shape[0])
        matmul_started_at = time.perf_counter()
        basis = q_t.to(device=param.device, dtype=param.dtype)
        left_factor_mode = str(self.method_config.abh_left_factor)
        if left_factor_mode == "dense_left":
            if str(self.method_config.abh_probe_transform) == "loren_q":
                raw_left = self._abh_loren_q_raw_b_matrix(
                    name,
                    param,
                    int(param.data.shape[0]),
                    span_rank,
                )
                left_factor = self._abh_loren_q_transform_b(
                    name,
                    param,
                    raw_left,
                    span_rank,
                )
            else:
                left_factor = torch.randn(
                    (int(param.data.shape[0]), span_rank),
                    device=param.device,
                    dtype=param.dtype,
                )
            noise = left_factor.matmul(basis)
        elif left_factor_mode in {"orthogonal_ab_random", "gaussian_ab_random"}:
            a_started_at = time.perf_counter()
            a_matrix = self._abh_probe_a_matrix(
                name,
                param,
                rank,
                orthogonal=(left_factor_mode == "orthogonal_ab_random"),
            )
            self._profile_add(
                profile,
                "abh_noise.a_matrix_time",
                time.perf_counter() - a_started_at,
            )
            b_started_at = time.perf_counter()
            b_matrix, _ = self._abh_probe_and_update_b_matrices(
                name,
                param,
                rank,
                span_rank,
            )
            self._profile_add(
                profile,
                "abh_noise.b_matrix_time",
                time.perf_counter() - b_started_at,
            )
            if str(self.method_config.abh_probe_transform) == "loren_r_flat":
                raw_r = a_matrix.float().matmul(b_matrix.float())
                r_matrix = self._abh_loren_r_flat_transform(
                    name,
                    param,
                    raw_r,
                    span_rank,
                ).to(device=param.device, dtype=param.dtype)
                noise = r_matrix.matmul(basis)
            else:
                right_factor = b_matrix.matmul(basis)
                noise = a_matrix.matmul(right_factor)
        elif left_factor_mode == "orthogonal_b_random_a":
            a_started_at = time.perf_counter()
            a_matrix = self._abh_probe_a_matrix(name, param, rank)
            self._profile_add(
                profile,
                "abh_noise.a_matrix_time",
                time.perf_counter() - a_started_at,
            )
            b_started_at = time.perf_counter()
            b_matrix = self._abh_b_matrix(name, param, rank, span_rank)
            self._profile_add(
                profile,
                "abh_noise.b_matrix_time",
                time.perf_counter() - b_started_at,
            )
            right_factor = b_matrix.matmul(basis)
            noise = a_matrix.matmul(right_factor)
        else:
            a_started_at = time.perf_counter()
            a_matrix = self._abh_a_matrix(name, param, rank)
            self._profile_add(
                profile,
                "abh_noise.a_matrix_time",
                time.perf_counter() - a_started_at,
            )
            b_started_at = time.perf_counter()
            b_matrix, _ = self._abh_probe_and_update_b_matrices(
                name,
                param,
                rank,
                span_rank,
            )
            self._profile_add(
                profile,
                "abh_noise.b_matrix_time",
                time.perf_counter() - b_started_at,
            )
            right_factor = b_matrix.matmul(basis)
            noise = a_matrix.matmul(right_factor)
        self._profile_add(
            profile,
            "abh_noise.matmul_time",
            time.perf_counter() - matmul_started_at,
        )
        dense_residual_ratio = float(self.method_config.abh_dense_residual_ratio)
        if dense_residual_ratio > 0.0:
            dense_started_at = time.perf_counter()
            residual = torch.randn_like(param.data)
            noise_norm = _safe_tensor_norm(noise)
            residual_norm = _safe_tensor_norm(residual)
            if (
                noise_norm is not None
                and residual_norm is not None
                and float(noise_norm) > 0.0
                and float(residual_norm) > 0.0
            ):
                residual = residual.mul(
                    float(noise_norm) / (float(residual_norm) + 1e-12)
                )
                keep_scale = math.sqrt(max(0.0, 1.0 - dense_residual_ratio))
                residual_scale = math.sqrt(dense_residual_ratio)
                noise = noise.mul(keep_scale).add(residual, alpha=residual_scale)
            self._profile_add(
                profile,
                "abh_noise.dense_residual_time",
                time.perf_counter() - dense_started_at,
            )
        normalization = str(self.method_config.abh_normalization)
        layer_fro_scale = float(self.method_config.abh_layer_fro_scale)
        target_norm = (float(param.data.numel()) ** 0.5) * layer_fro_scale
        pre_norm_value = _safe_tensor_norm(noise)
        if target_norm > 0.0 and pre_norm_value is not None:
            self._abh_noise_pre_norm_ratio[str(name)] = float(pre_norm_value) / float(
                target_norm
            )
        if normalization == "none":
            if target_norm > 0.0 and pre_norm_value is not None:
                self._abh_noise_post_norm_ratio[str(name)] = float(
                    pre_norm_value
                ) / float(target_norm)
                self._abh_noise_scale_factor[str(name)] = 1.0
            return noise
        if normalization == "layer_fro":
            norm_started_at = time.perf_counter()
            if pre_norm_value is not None and float(pre_norm_value) > 0.0:
                scale_factor = target_norm / (float(pre_norm_value) + 1e-12)
                noise = noise.mul(scale_factor)
                self._abh_noise_scale_factor[str(name)] = float(scale_factor)
                post_norm_value = _safe_tensor_norm(noise)
                if target_norm > 0.0 and post_norm_value is not None:
                    self._abh_noise_post_norm_ratio[str(name)] = float(
                        post_norm_value
                    ) / float(target_norm)
            self._profile_add(
                profile,
                "abh_noise.normalization_time",
                time.perf_counter() - norm_started_at,
            )
            return noise
        norm_started_at = time.perf_counter()
        normalizer = float(rank * span_rank) ** 0.5
        if normalizer > 0.0:
            noise = noise.div(normalizer)
            self._abh_noise_scale_factor[str(name)] = 1.0 / float(normalizer)
            post_norm_value = _safe_tensor_norm(noise)
            if target_norm > 0.0 and post_norm_value is not None:
                self._abh_noise_post_norm_ratio[str(name)] = float(
                    post_norm_value
                ) / float(target_norm)
        self._profile_add(
            profile,
            "abh_noise.normalization_time",
            time.perf_counter() - norm_started_at,
        )
        return noise

    def _abh_muon_small_b_noise_and_update_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        q_t: torch.Tensor,
        *,
        projected_grad: float,
        use_momentum: bool = False,
        profile: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        rank = self._effective_basis_rank(param, q_t)
        if rank <= 0:
            return None
        q_t = self._abh_q_dropout_basis(name, q_t)
        rank = min(rank, int(q_t.shape[0]))
        span_rank = int(q_t.shape[0])
        a_matrix = self._abh_a_matrix(name, param, rank)
        b_matrix = torch.randn(
            (rank, span_rank),
            device=param.device,
            dtype=param.dtype,
        )
        basis = q_t.to(device=param.device, dtype=param.dtype)
        right_factor = b_matrix.matmul(basis)
        noise = a_matrix.matmul(right_factor)

        normalization_scale = 1.0
        normalization = str(self.method_config.abh_normalization)
        layer_fro_scale = float(self.method_config.abh_layer_fro_scale)
        target_norm = (float(param.data.numel()) ** 0.5) * layer_fro_scale
        pre_norm_value = _safe_tensor_norm(noise)
        if target_norm > 0.0 and pre_norm_value is not None:
            self._abh_noise_pre_norm_ratio[str(name)] = float(pre_norm_value) / float(
                target_norm
            )
        if normalization == "layer_fro":
            if pre_norm_value is not None and float(pre_norm_value) > 0.0:
                normalization_scale = target_norm / (float(pre_norm_value) + 1e-12)
                noise = noise.mul(normalization_scale)
        elif normalization != "none":
            normalizer = float(rank * span_rank) ** 0.5
            if normalizer > 0.0:
                normalization_scale = 1.0 / float(normalizer)
                noise = noise.mul(normalization_scale)
        self._abh_noise_scale_factor[str(name)] = float(normalization_scale)
        post_norm_value = _safe_tensor_norm(noise)
        if target_norm > 0.0 and post_norm_value is not None:
            self._abh_noise_post_norm_ratio[str(name)] = float(post_norm_value) / float(
                target_norm
            )

        small_update = b_matrix.float().mul(
            float(projected_grad) * float(normalization_scale)
        )
        transform_source = small_update
        if use_momentum:
            beta = float(self.method_config.abh_momentum_beta)
            momentum_key = str(name)
            previous = self._abh_muon_small_b_momentum.get(momentum_key)
            if previous is None or tuple(previous.shape) != tuple(small_update.shape):
                transform_source = small_update.detach().clone()
            else:
                transform_source = previous.to(
                    device=small_update.device,
                    dtype=small_update.dtype,
                )
                transform_source.mul_(beta).add_(small_update, alpha=1.0 - beta)
            self._abh_muon_small_b_momentum[momentum_key] = (
                transform_source.detach().float().contiguous()
            )
        pre_small_norm = _safe_tensor_norm(small_update)
        pre_transform_norm = _safe_tensor_norm(transform_source)
        if pre_transform_norm is not None:
            self._abh_muon_small_b_pre_norm[str(name)] = float(pre_transform_norm)
        elif pre_small_norm is not None:
            self._abh_muon_small_b_pre_norm[str(name)] = float(pre_small_norm)
        transformed_small = _zeropower_via_newton_schulz(transform_source).to(
            device=param.device,
            dtype=param.dtype,
        )
        post_small_norm = _safe_tensor_norm(transformed_small)
        if post_small_norm is not None:
            self._abh_muon_small_b_post_norm[str(name)] = float(post_small_norm)
        if (
            pre_transform_norm is not None
            and float(pre_transform_norm) > 0.0
            and post_small_norm is not None
        ):
            self._abh_muon_small_b_norm_ratio[str(name)] = float(
                post_small_norm
            ) / float(pre_transform_norm)
        update = a_matrix.matmul(transformed_small.matmul(basis))
        self._profile_add(profile, "abh_muon_small_b.parameters", 1)
        return noise, update

    def _abh_probe_and_update_b_matrices(
        self,
        name: str,
        param: torch.nn.Parameter,
        rank: int,
        span_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if str(self.method_config.abh_probe_transform) == "loren_q":
            b_matrix = self._abh_loren_q_b_matrix(name, param, rank, span_rank)
            return b_matrix, b_matrix
        if str(self.method_config.abh_probe_transform) == "loren_r_flat":
            b_matrix = self._abh_loren_q_raw_b_matrix(name, param, rank, span_rank)
            return b_matrix.to(device=param.device, dtype=param.dtype), b_matrix.to(
                device=param.device,
                dtype=param.dtype,
            )
        if str(self.method_config.abh_probe_transform) != "adamu_b":
            b_matrix = torch.randn(
                (int(rank), int(span_rank)),
                device=param.device,
                dtype=param.dtype,
            )
            return b_matrix, b_matrix

        active_step = int(self._active_step_number)
        if self._abh_adamu_b_cache_step != active_step:
            self._abh_adamu_b_step_cache.clear()
            self._abh_adamu_b_cache_step = active_step

        active_seed = int(self._active_noise_seed or 0)
        cache_key = (
            active_step,
            active_seed,
            str(name),
            int(rank),
            int(span_rank),
            str(param.device),
            str(param.dtype),
        )
        cached = self._abh_adamu_b_step_cache.get(cache_key)
        if cached is not None:
            return cached

        alpha = float(self.method_config.abh_adamu_alpha)
        beta1 = float(self.method_config.abh_adamu_beta1)
        beta2 = float(self.method_config.abh_adamu_beta2)
        eps = float(self.method_config.abh_adamu_eps)
        state_key = f"{name}:{rank}:{span_rank}:{param.device}:{param.dtype}"

        dot = torch.randn(
            (int(rank), int(span_rank)),
            device=param.device,
            dtype=torch.float32,
        ).mul(math.sqrt(max(alpha, 0.0)))
        previous = self._abh_adamu_b_momentum.get(state_key)
        if previous is None or tuple(previous.shape) != tuple(dot.shape):
            previous_center = torch.zeros_like(dot)
        else:
            previous_center = previous.to(device=param.device, dtype=torch.float32)
        ddot = previous_center.add(
            torch.randn_like(dot),
            alpha=math.sqrt(max(1.0 - alpha, 0.0)),
        )
        probe = dot.mul(beta1).add(ddot, alpha=1.0 - beta1)
        v = dot.square().mul(beta2).add(ddot.square(), alpha=1.0 - beta2)
        update = probe.div(v.sqrt().add(eps))

        self._abh_adamu_b_momentum[state_key] = probe.detach().float().contiguous()
        probe_norm = _safe_tensor_norm(probe)
        update_norm = _safe_tensor_norm(update)
        if probe_norm is not None:
            self._abh_adamu_b_probe_norm[str(name)] = float(probe_norm)
        if update_norm is not None:
            self._abh_adamu_b_update_norm[str(name)] = float(update_norm)
        if (
            probe_norm is not None
            and float(probe_norm) > 0.0
            and update_norm is not None
        ):
            self._abh_adamu_b_update_ratio[str(name)] = float(update_norm) / float(
                probe_norm
            )

        result = (
            probe.to(device=param.device, dtype=param.dtype).detach().contiguous(),
            update.to(device=param.device, dtype=param.dtype).detach().contiguous(),
        )
        self._abh_adamu_b_step_cache[cache_key] = result
        return result

    def _abh_loren_q_state_key(
        self,
        name: str,
        param: torch.nn.Parameter,
        span_rank: int,
    ) -> str:
        return ":".join(
            [
                str(name),
                str(int(self._abh_a_refresh_block())),
                str(int(span_rank)),
                str(param.device),
            ]
        )

    def _abh_loren_q_a_vector(
        self,
        name: str,
        param: torch.nn.Parameter,
        span_rank: int,
    ) -> torch.Tensor:
        key = self._abh_loren_q_state_key(name, param, span_rank)
        cached = self._abh_loren_q_a.get(key)
        if cached is not None and tuple(cached.shape) == (int(span_rank),):
            return cached.to(device=param.device, dtype=torch.float32)
        std = float(self.method_config.abh_loren_q_a_init_std)
        seed = _stable_int_seed(
            {
                "method": "agzo_abh_loren_q_a",
                "name": str(name),
                "block": int(self._abh_a_refresh_block()),
                "span_rank": int(span_rank),
                "device": str(param.device),
            }
        )
        generator = torch.Generator(device=param.device)
        generator.manual_seed(seed)
        vector = torch.randn(
            (int(span_rank),),
            device=param.device,
            dtype=torch.float32,
            generator=generator,
        ).mul(float(std))
        self._abh_loren_q_a[key] = vector.detach().float().contiguous()
        norm = _safe_tensor_norm(vector)
        if norm is not None:
            self._abh_loren_q_a_norm[str(name)] = float(norm)
        return vector

    def _abh_loren_q_raw_b_matrix(
        self,
        name: str,
        param: torch.nn.Parameter,
        rank: int,
        span_rank: int,
    ) -> torch.Tensor:
        active_seed = int(self._active_noise_seed or 0)
        seed = _stable_int_seed(
            {
                "method": "agzo_abh_loren_q_b",
                "name": str(name),
                "probe_seed": int(active_seed),
                "rank": int(rank),
                "span_rank": int(span_rank),
                "block": int(self._abh_a_refresh_block()),
            }
        )
        generator = torch.Generator(device=param.device)
        generator.manual_seed(seed)
        return torch.randn(
            (int(rank), int(span_rank)),
            device=param.device,
            dtype=torch.float32,
            generator=generator,
        )

    def _abh_loren_q_transform_b(
        self,
        name: str,
        param: torch.nn.Parameter,
        raw_b: torch.Tensor,
        span_rank: int,
    ) -> torch.Tensor:
        a_vec = self._abh_loren_q_a_vector(name, param, span_rank)
        damping = float(self.method_config.abh_loren_q_damping)
        sq_norm_a = float(torch.dot(a_vec, a_vec).item())
        if sq_norm_a <= 1e-12:
            transformed = raw_b
        else:
            damping_sqrt = math.sqrt(max(damping, 1e-12))
            damping_sq_norm_a = math.sqrt(max(damping + sq_norm_a, 1e-12))
            dot_au = raw_b.matmul(a_vec)
            alpha = (
                (damping_sqrt + damping_sq_norm_a)
                * dot_au
                / (sq_norm_a * damping_sq_norm_a)
            )
            transformed = raw_b - alpha.unsqueeze(1).mul(a_vec.unsqueeze(0))
        raw_norm = _safe_tensor_norm(raw_b)
        transformed_norm = _safe_tensor_norm(transformed)
        if (
            raw_norm is not None
            and float(raw_norm) > 0.0
            and transformed_norm is not None
        ):
            self._abh_loren_q_scale_mean[str(name)] = float(transformed_norm) / float(
                raw_norm
            )
        norm = _safe_tensor_norm(a_vec)
        if norm is not None:
            self._abh_loren_q_a_norm[str(name)] = float(norm)
        return (
            transformed.to(device=param.device, dtype=param.dtype).detach().contiguous()
        )

    def _abh_loren_q_b_matrix(
        self,
        name: str,
        param: torch.nn.Parameter,
        rank: int,
        span_rank: int,
    ) -> torch.Tensor:
        raw_b = self._abh_loren_q_raw_b_matrix(name, param, rank, span_rank)
        return self._abh_loren_q_transform_b(name, param, raw_b, span_rank)

    def _abh_loren_r_flat_transform(
        self,
        name: str,
        param: torch.nn.Parameter,
        raw_r: torch.Tensor,
        span_rank: int,
    ) -> torch.Tensor:
        a_vec = self._abh_loren_q_a_vector(name, param, span_rank)
        damping = float(self.method_config.abh_loren_q_damping)
        rows = raw_r.reshape(-1, int(span_rank)).to(dtype=torch.float32)
        row_count = max(1, int(rows.shape[0]))
        sq_norm_a = float(torch.dot(a_vec, a_vec).item())
        if sq_norm_a <= 1e-12:
            transformed = rows
        else:
            flat_sq_norm_a = float(row_count) * float(sq_norm_a)
            dot_au = rows.matmul(a_vec).sum()
            damping_sqrt = math.sqrt(max(damping, 1e-12))
            damping_sq_norm_a = math.sqrt(max(damping + flat_sq_norm_a, 1e-12))
            alpha = (
                (damping_sqrt + damping_sq_norm_a)
                * dot_au
                / (flat_sq_norm_a * damping_sq_norm_a)
            )
            transformed = rows - alpha.mul(a_vec).unsqueeze(0)
        raw_norm = _safe_tensor_norm(rows)
        transformed_norm = _safe_tensor_norm(transformed)
        if (
            raw_norm is not None
            and float(raw_norm) > 0.0
            and transformed_norm is not None
        ):
            self._abh_loren_q_scale_mean[str(name)] = float(transformed_norm) / float(
                raw_norm
            )
        norm = _safe_tensor_norm(a_vec)
        if norm is not None:
            self._abh_loren_q_a_norm[str(name)] = float(norm)
        return transformed.reshape_as(raw_r).detach().float().contiguous()

    def _abh_loren_r_flat_update_components_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        q_t: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if str(self.method_config.abh_probe_transform) != "loren_r_flat":
            return None
        if str(self.method_config.abh_left_factor) not in {
            "orthogonal_ab_random",
            "gaussian_ab_random",
        }:
            return None
        if param.data.ndim != 2 or int(q_t.shape[1]) != int(param.data.shape[1]):
            return None
        rank = self._effective_basis_rank(param, q_t)
        if rank <= 0:
            return None
        q_t = self._abh_q_dropout_basis(name, q_t)
        rank = min(rank, int(q_t.shape[0]))
        span_rank = int(q_t.shape[0])
        basis = q_t.to(device=param.device, dtype=param.dtype)
        left_factor_mode = str(self.method_config.abh_left_factor)
        a_matrix = self._abh_probe_a_matrix(
            name,
            param,
            rank,
            orthogonal=(left_factor_mode == "orthogonal_ab_random"),
        )
        raw_b = self._abh_loren_q_raw_b_matrix(name, param, rank, span_rank)
        raw_r = a_matrix.float().matmul(raw_b.float())
        r_matrix = self._abh_loren_r_flat_transform(
            name,
            param,
            raw_r,
            span_rank,
        )

        normalization = str(self.method_config.abh_normalization)
        layer_fro_scale = float(self.method_config.abh_layer_fro_scale)
        target_norm = (float(param.data.numel()) ** 0.5) * layer_fro_scale
        pre_norm_value = _safe_tensor_norm(r_matrix)
        if target_norm > 0.0 and pre_norm_value is not None:
            self._abh_noise_pre_norm_ratio[str(name)] = float(pre_norm_value) / float(
                target_norm
            )
        if normalization == "layer_fro":
            if pre_norm_value is not None and float(pre_norm_value) > 0.0:
                scale_factor = target_norm / (float(pre_norm_value) + 1e-12)
                r_matrix = r_matrix.mul(scale_factor)
                self._abh_noise_scale_factor[str(name)] = float(scale_factor)
                post_norm_value = _safe_tensor_norm(r_matrix)
                if target_norm > 0.0 and post_norm_value is not None:
                    self._abh_noise_post_norm_ratio[str(name)] = float(
                        post_norm_value
                    ) / float(target_norm)
        elif normalization == "none":
            if target_norm > 0.0 and pre_norm_value is not None:
                self._abh_noise_post_norm_ratio[str(name)] = float(
                    pre_norm_value
                ) / float(target_norm)
                self._abh_noise_scale_factor[str(name)] = 1.0
        else:
            normalizer = float(rank * span_rank) ** 0.5
            if normalizer > 0.0:
                r_matrix = r_matrix.div(normalizer)
                self._abh_noise_scale_factor[str(name)] = 1.0 / float(normalizer)
                post_norm_value = _safe_tensor_norm(r_matrix)
                if target_norm > 0.0 and post_norm_value is not None:
                    self._abh_noise_post_norm_ratio[str(name)] = float(
                        post_norm_value
                    ) / float(target_norm)
        self._profile_add(profile, "abh_loren_r_flat_update_components.parameters", 1)
        return (
            r_matrix.to(device=param.device, dtype=torch.float32).detach().contiguous(),
            basis.detach().contiguous(),
        )

    def _update_abh_loren_q_population_state(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
        *,
        probe_fweights: list[tuple[int, float]],
        eps: float,
    ) -> None:
        if str(self.method_config.abh_probe_transform) != "loren_q":
            if str(self.method_config.abh_probe_transform) != "loren_r_flat":
                return
        lr_cov = float(self.method_config.abh_loren_q_lr_cov)
        if lr_cov <= 0.0 or not probe_fweights:
            return
        denom = max(1, len(probe_fweights) - 1)
        previous_seed = self._active_noise_seed
        try:
            for name, param in parameters:
                basis = bases.get(name)
                if (
                    basis is None
                    or param.data.ndim != 2
                    or int(basis.shape[1]) != int(param.data.shape[1])
                ):
                    continue
                rank = self._effective_basis_rank(param, basis)
                if rank <= 0:
                    continue
                q_t = self._abh_q_dropout_basis(name, basis)
                rank = min(rank, int(q_t.shape[0]))
                span_rank = int(q_t.shape[0])
                coeff_rows = (
                    int(param.data.shape[0])
                    if str(self.method_config.abh_left_factor) == "dense_left"
                    else int(rank)
                )
                a_vec = self._abh_loren_q_a_vector(name, param, span_rank)
                damping = float(self.method_config.abh_loren_q_damping)
                sq_norm_a = float(torch.dot(a_vec, a_vec).item())
                if sq_norm_a <= 1e-12:
                    continue
                damping_sqrt = math.sqrt(max(damping, 1e-12))
                damping_sq_norm_a = math.sqrt(max(damping + sq_norm_a, 1e-12))
                grad = torch.zeros_like(a_vec)
                for probe_seed, fweight in probe_fweights:
                    self._active_noise_seed = int(probe_seed)
                    raw_b = self._abh_loren_q_raw_b_matrix(
                        name,
                        param,
                        coeff_rows,
                        span_rank,
                    )
                    if str(self.method_config.abh_probe_transform) == "loren_r_flat":
                        if str(self.method_config.abh_left_factor) in {
                            "orthogonal_ab_random",
                            "gaussian_ab_random",
                        }:
                            a_matrix = self._abh_probe_a_matrix(
                                name,
                                param,
                                rank,
                                orthogonal=(
                                    str(self.method_config.abh_left_factor)
                                    == "orthogonal_ab_random"
                                ),
                            )
                            raw_rows = a_matrix.float().matmul(raw_b.float())
                        else:
                            raw_rows = raw_b.float()
                        row_count = max(
                            1, int(raw_rows.reshape(-1, span_rank).shape[0])
                        )
                        rows = raw_rows.reshape(-1, span_rank).to(
                            device=a_vec.device,
                            dtype=a_vec.dtype,
                        )
                        flat_sq_norm_a = float(row_count) * float(sq_norm_a)
                        flat_damping_sq_norm_a = math.sqrt(
                            max(damping + flat_sq_norm_a, 1e-12)
                        )
                        dot_au = rows.matmul(a_vec).sum()
                        c1_sum = dot_au.mul(rows.sum(dim=0)).sub(
                            a_vec.mul(float(row_count))
                        )
                        c2_scale = (
                            (damping_sqrt + flat_damping_sq_norm_a)
                            * (dot_au.square() - flat_sq_norm_a)
                            / (flat_sq_norm_a * flat_damping_sq_norm_a)
                        )
                        c2_sum = a_vec.mul(float(row_count)).mul(c2_scale)
                        grad = grad + float(fweight) * (c1_sum - c2_sum) / float(
                            flat_damping_sq_norm_a
                        )
                    else:
                        dot_au = raw_b.matmul(a_vec)
                        c1 = dot_au.unsqueeze(1).mul(raw_b) - a_vec.unsqueeze(0)
                        c2_scale = (
                            (damping_sqrt + damping_sq_norm_a)
                            * (dot_au.square() - sq_norm_a)
                            / (sq_norm_a * damping_sq_norm_a)
                        )
                        c2 = c2_scale.unsqueeze(1).mul(a_vec.unsqueeze(0))
                        grad = grad + float(fweight) * (c1 - c2).sum(dim=0) / float(
                            damping_sq_norm_a
                        )
                eps_scale = float(eps) ** float(
                    self.method_config.abh_loren_q_a_eps_power
                )
                grad = grad.mul(eps_scale / float(denom))
                old_norm = _safe_tensor_norm(a_vec)
                grad_norm = _safe_tensor_norm(grad)
                delta = grad.mul(-lr_cov)
                delta_norm = _safe_tensor_norm(delta)
                if grad_norm is not None:
                    self._abh_loren_q_a_grad_norm[str(name)] = float(grad_norm)
                if delta_norm is not None:
                    self._abh_loren_q_a_delta_norm[str(name)] = float(delta_norm)
                    self._abh_loren_q_a_delta_ratio[str(name)] = float(
                        delta_norm
                    ) / max(float(old_norm or 0.0), 1e-12)
                a_vec = a_vec.add(delta).detach().float().contiguous()
                key = self._abh_loren_q_state_key(name, param, span_rank)
                self._abh_loren_q_a[key] = a_vec
                norm = _safe_tensor_norm(a_vec)
                if norm is not None:
                    self._abh_loren_q_a_norm[str(name)] = float(norm)
        finally:
            self._active_noise_seed = previous_seed

    def _abh_adamu_b_noise_and_update_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        q_t: torch.Tensor | None,
        *,
        projected_grad: float,
        profile: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        left_factor_mode = str(self.method_config.abh_left_factor)
        if left_factor_mode not in {
            "orthogonal_a",
            "orthogonal_ab_random",
            "gaussian_ab_random",
        }:
            return None
        if q_t is None:
            return None
        if param.data.ndim != 2 or int(q_t.shape[1]) != int(param.data.shape[1]):
            return None
        rank = self._effective_basis_rank(param, q_t)
        if rank <= 0:
            return None
        q_t = self._abh_q_dropout_basis(name, q_t)
        rank = min(rank, int(q_t.shape[0]))
        span_rank = int(q_t.shape[0])
        if left_factor_mode in {"orthogonal_ab_random", "gaussian_ab_random"}:
            a_matrix = self._abh_probe_a_matrix(
                name,
                param,
                rank,
                orthogonal=(left_factor_mode == "orthogonal_ab_random"),
            )
        else:
            a_matrix = self._abh_a_matrix(name, param, rank)
        b_probe, b_update = self._abh_probe_and_update_b_matrices(
            name,
            param,
            rank,
            span_rank,
        )
        basis = q_t.to(device=param.device, dtype=param.dtype)
        probe_right = b_probe.to(device=param.device, dtype=param.dtype).matmul(basis)
        noise = a_matrix.matmul(probe_right)

        normalization_scale = 1.0
        normalization = str(self.method_config.abh_normalization)
        layer_fro_scale = float(self.method_config.abh_layer_fro_scale)
        target_norm = (float(param.data.numel()) ** 0.5) * layer_fro_scale
        pre_norm_value = _safe_tensor_norm(noise)
        if target_norm > 0.0 and pre_norm_value is not None:
            self._abh_noise_pre_norm_ratio[str(name)] = float(pre_norm_value) / float(
                target_norm
            )
        if normalization == "layer_fro":
            if pre_norm_value is not None and float(pre_norm_value) > 0.0:
                normalization_scale = target_norm / (float(pre_norm_value) + 1e-12)
                noise = noise.mul(normalization_scale)
        elif normalization != "none":
            normalizer = float(rank * span_rank) ** 0.5
            if normalizer > 0.0:
                normalization_scale = 1.0 / float(normalizer)
                noise = noise.mul(normalization_scale)
        self._abh_noise_scale_factor[str(name)] = float(normalization_scale)
        post_norm_value = _safe_tensor_norm(noise)
        if target_norm > 0.0 and post_norm_value is not None:
            self._abh_noise_post_norm_ratio[str(name)] = float(post_norm_value) / float(
                target_norm
            )

        small_update = b_update.float().mul(
            float(projected_grad) * float(normalization_scale)
        )
        update = (
            a_matrix.float()
            .matmul(small_update)
            .to(
                device=param.device,
                dtype=param.dtype,
            )
            .matmul(basis)
        )
        self._profile_add(profile, "abh_adamu_b.parameters", 1)
        return noise, update

    def _abh_ab_momentum_noise_and_update_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        q_t: torch.Tensor,
        *,
        projected_grad: float,
        use_adam: bool = False,
        profile: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if str(self.method_config.abh_left_factor) != "orthogonal_a":
            return None
        rank = self._effective_basis_rank(param, q_t)
        if rank <= 0:
            return None
        q_t = self._abh_q_dropout_basis(name, q_t)
        rank = min(rank, int(q_t.shape[0]))
        span_rank = int(q_t.shape[0])
        a_matrix = self._abh_a_matrix(name, param, rank)
        b_matrix = torch.randn(
            (rank, span_rank),
            device=param.device,
            dtype=param.dtype,
        )
        basis = q_t.to(device=param.device, dtype=param.dtype)
        right_factor = b_matrix.matmul(basis)
        noise = a_matrix.matmul(right_factor)

        normalization_scale = 1.0
        normalization = str(self.method_config.abh_normalization)
        layer_fro_scale = float(self.method_config.abh_layer_fro_scale)
        target_norm = (float(param.data.numel()) ** 0.5) * layer_fro_scale
        pre_norm_value = _safe_tensor_norm(noise)
        if target_norm > 0.0 and pre_norm_value is not None:
            self._abh_noise_pre_norm_ratio[str(name)] = float(pre_norm_value) / float(
                target_norm
            )
        if normalization == "layer_fro":
            if pre_norm_value is not None and float(pre_norm_value) > 0.0:
                normalization_scale = target_norm / (float(pre_norm_value) + 1e-12)
                noise = noise.mul(normalization_scale)
        elif normalization != "none":
            normalizer = float(rank * span_rank) ** 0.5
            if normalizer > 0.0:
                normalization_scale = 1.0 / float(normalizer)
                noise = noise.mul(normalization_scale)
        self._abh_noise_scale_factor[str(name)] = float(normalization_scale)
        post_norm_value = _safe_tensor_norm(noise)
        if target_norm > 0.0 and post_norm_value is not None:
            self._abh_noise_post_norm_ratio[str(name)] = float(post_norm_value) / float(
                target_norm
            )

        ab_update = a_matrix.float().matmul(
            b_matrix.float().mul(float(projected_grad) * float(normalization_scale))
        )
        key = str(name)
        beta = float(self.method_config.abh_momentum_beta)
        previous = self._abh_ab_momentum.get(key)
        if previous is None or tuple(previous.shape) != tuple(ab_update.shape):
            first_moment = torch.zeros_like(ab_update)
        else:
            first_moment = previous.to(device=ab_update.device, dtype=ab_update.dtype)
        first_moment.mul_(beta).add_(ab_update, alpha=1.0 - beta)
        self._abh_ab_momentum[key] = first_moment.detach().float().contiguous()

        if use_adam:
            beta2 = float(getattr(self.method_config, "abh_adam_beta2", 0.999))
            adam_eps = float(getattr(self.method_config, "abh_adam_eps", 1e-8))
            previous_second = self._abh_ab_adam_v.get(key)
            if previous_second is None or tuple(previous_second.shape) != tuple(
                ab_update.shape
            ):
                second_moment = torch.zeros_like(ab_update)
                self._abh_ab_adam_steps[key] = 0
            else:
                second_moment = previous_second.to(
                    device=ab_update.device,
                    dtype=ab_update.dtype,
                )
            second_moment.mul_(beta2).addcmul_(
                ab_update,
                ab_update,
                value=1.0 - beta2,
            )
            self._abh_ab_adam_steps[key] += 1
            step = max(1, int(self._abh_ab_adam_steps[key]))
            first_hat = (
                first_moment / (1.0 - beta**step) if beta < 1.0 else first_moment
            )
            second_hat = (
                second_moment / (1.0 - beta2**step) if beta2 < 1.0 else second_moment
            )
            transform = first_hat.div(second_hat.sqrt().add(adam_eps))
            self._abh_ab_adam_v[key] = second_moment.detach().float().contiguous()
            self._profile_add(profile, "abh_ab_adam.parameters", 1)
        else:
            transform = first_moment
            self._profile_add(profile, "abh_ab_momentum.parameters", 1)

        transform_norm = _safe_tensor_norm(transform)
        if transform_norm is not None:
            self._abh_ab_momentum_norm[key] = float(transform_norm)

        update = transform.to(device=param.device, dtype=param.dtype).matmul(basis)
        return noise, update

    def _abh_ab_momentum_or_adam_transform(
        self,
        key: str,
        left_update: torch.Tensor,
        *,
        use_adam: bool,
        profile: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        beta = float(self.method_config.abh_momentum_beta)
        previous = self._abh_ab_momentum.get(key)
        if previous is None or tuple(previous.shape) != tuple(left_update.shape):
            first_moment = torch.zeros_like(left_update)
        else:
            first_moment = previous.to(
                device=left_update.device,
                dtype=left_update.dtype,
            )
        first_moment.mul_(beta).add_(left_update, alpha=1.0 - beta)
        self._abh_ab_momentum[key] = first_moment.detach().float().contiguous()

        if use_adam:
            beta2 = float(getattr(self.method_config, "abh_adam_beta2", 0.999))
            adam_eps = float(getattr(self.method_config, "abh_adam_eps", 1e-8))
            previous_second = self._abh_ab_adam_v.get(key)
            if previous_second is None or tuple(previous_second.shape) != tuple(
                left_update.shape
            ):
                second_moment = torch.zeros_like(left_update)
                self._abh_ab_adam_steps[key] = 0
            else:
                second_moment = previous_second.to(
                    device=left_update.device,
                    dtype=left_update.dtype,
                )
            second_moment.mul_(beta2).addcmul_(
                left_update,
                left_update,
                value=1.0 - beta2,
            )
            self._abh_ab_adam_steps[key] += 1
            step = max(1, int(self._abh_ab_adam_steps[key]))
            first_hat = (
                first_moment / (1.0 - beta**step) if beta < 1.0 else first_moment
            )
            second_hat = (
                second_moment / (1.0 - beta2**step) if beta2 < 1.0 else second_moment
            )
            transformed = first_hat.div(second_hat.sqrt().add(adam_eps))
            self._abh_ab_adam_v[key] = second_moment.detach().float().contiguous()
            self._profile_add(profile, "abh_ab_adam.parameters", 1)
        else:
            transformed = first_moment
            self._profile_add(profile, "abh_ab_momentum.parameters", 1)

        transform_norm = _safe_tensor_norm(transformed)
        if transform_norm is not None:
            self._abh_ab_momentum_norm[key] = float(transform_norm)
        return transformed

    def _abh_r_muon_transform(
        self,
        key: str,
        left_update: torch.Tensor,
        *,
        use_momentum: bool,
        profile: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if use_momentum:
            beta = float(self.method_config.abh_momentum_beta)
            previous = self._abh_ab_momentum.get(key)
            if previous is not None and tuple(previous.shape) == tuple(
                left_update.shape
            ):
                transform_source = previous.to(
                    device=left_update.device,
                    dtype=left_update.dtype,
                )
                transform_source.mul_(beta).add_(left_update, alpha=1.0 - beta)
            else:
                transform_source = left_update.detach().clone()
            self._abh_ab_momentum[key] = transform_source.detach().float().contiguous()
        else:
            transform_source = left_update

        pre_transform_norm = _safe_tensor_norm(transform_source)
        if pre_transform_norm is not None:
            self._abh_ab_momentum_norm[key] = float(pre_transform_norm)
            self._abh_muon_small_b_pre_norm[key] = float(pre_transform_norm)
        transformed = _zeropower_via_newton_schulz(transform_source).to(
            device=left_update.device,
            dtype=left_update.dtype,
        )
        post_transform_norm = _safe_tensor_norm(transformed)
        if post_transform_norm is not None:
            self._abh_muon_small_b_post_norm[key] = float(post_transform_norm)
        if (
            pre_transform_norm is not None
            and float(pre_transform_norm) > 0.0
            and post_transform_norm is not None
        ):
            self._abh_muon_small_b_norm_ratio[key] = float(post_transform_norm) / float(
                pre_transform_norm
            )
        self._profile_add(profile, "abh_muon_r_momentum.parameters", 1)
        return transformed

    @torch.no_grad()
    def _apply_agzo_combined_varied_q_update(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
        *,
        probe_projected_grads: list[tuple[int, float]],
        probe_update_weights: dict[int, float] | None = None,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
        source_trace: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> str:
        """Apply independently sampled probe directions with one write per matrix."""
        del source_trace  # This optimized path is intentionally not a trace path.
        total_started_at = time.perf_counter()
        probe_count = max(1, len(probe_projected_grads))
        weights = probe_update_weights or {
            int(seed): 1.0 / float(probe_count) for seed, _ in probe_projected_grads
        }
        accumulators: dict[
            str,
            tuple[
                torch.nn.Parameter,
                list[torch.Tensor],
                list[torch.Tensor],
            ],
        ] = {}

        for probe_seed, projected_grad in probe_projected_grads:
            probe_weight = float(weights.get(int(probe_seed), 1.0 / probe_count))
            previous_noise_seed = self._active_noise_seed
            self._active_noise_seed = int(probe_seed)
            torch.manual_seed(int(probe_seed))
            try:
                for name, param in parameters:
                    basis = bases.get(name)
                    components = (
                        self._abh_small_b_components_for_param(
                            name,
                            param,
                            basis,
                            projected_grad=float(projected_grad) * probe_weight,
                            use_update_preconditioner=True,
                            profile=profile,
                        )
                        if basis is not None
                        and param.data.ndim == 2
                        and int(basis.shape[1]) == int(param.data.shape[1])
                        else None
                    )
                    if components is None:
                        z = self._noise_for_param(name, param, bases, profile=profile)
                        update = z.mul(float(projected_grad) * probe_weight)
                        update_trace.add_delta(
                            name, update, scale=-float(learning_rate)
                        )
                        param.add_(update, alpha=-float(learning_rate))
                        continue
                    a_matrix, small_update, right_basis = components
                    left = a_matrix.to(device=param.device, dtype=param.dtype)
                    right = small_update.to(
                        device=param.device, dtype=param.dtype
                    ).matmul(right_basis.to(device=param.device, dtype=param.dtype))
                    key = str(name)
                    if key not in accumulators:
                        accumulators[key] = (param, [], [])
                    accumulators[key][1].append(left)
                    accumulators[key][2].append(right)
            finally:
                self._active_noise_seed = previous_noise_seed

        if weight_decay:
            raise ValueError("varied-Q fused update requires zero weight decay")
        for name, (param, left_parts, right_parts) in accumulators.items():
            left = torch.cat(left_parts, dim=1)
            right = torch.cat(right_parts, dim=0)
            update_trace.add_factorized_delta(
                name, left, right, scale=-float(learning_rate)
            )
            param.addmm_(left, right, alpha=-float(learning_rate))

        self._profile_add(
            profile,
            "update.combined_varied_q_total_time",
            time.perf_counter() - total_started_at,
        )
        return _stable_json_checksum(
            {
                "kind": "combined_varied_q",
                "probe_projected_grads": probe_projected_grads,
                "probe_update_weights": sorted(weights.items()),
                "learning_rate": float(learning_rate),
            }
        )

    @torch.no_grad()
    def _apply_agzo_combined_multi_structured_update(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
        *,
        probe_projected_grads: list[tuple[int, float]],
        probe_update_weights: dict[int, float] | None = None,
        learning_rate: float,
        weight_decay: float,
        update_trace: UpdateTraceAccumulator,
        source_trace: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> str:
        total_started_at = time.perf_counter()
        probe_count = max(1, len(probe_projected_grads))
        if probe_update_weights is None:
            probe_update_weights = {
                int(seed): 1.0 / float(probe_count) for seed, _ in probe_projected_grads
            }
        small_accumulators: dict[
            str,
            tuple[
                torch.nn.Parameter,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        left_accumulators: dict[
            str,
            tuple[
                torch.nn.Parameter,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        parameter_lookup = {str(name): (name, param) for name, param in parameters}
        left_factor_mode = str(self.method_config.abh_left_factor)
        direct_probe_seed_a = left_factor_mode in {
            "orthogonal_ab_random",
            "gaussian_ab_random",
        } or (
            str(getattr(self.method_config, "abh_a_seed_mode", "block")) == "probe"
            and left_factor_mode == "orthogonal_a"
        )
        parameter_summaries: list[dict[str, Any]] = []

        for probe_seed, projected_grad in probe_projected_grads:
            probe_weight = float(
                probe_update_weights.get(int(probe_seed), 1.0 / float(probe_count))
            )
            previous_noise_seed = self._active_noise_seed
            self._active_noise_seed = int(probe_seed)
            torch.manual_seed(int(probe_seed))
            try:
                for name, param in parameters:
                    basis = bases.get(name)
                    if direct_probe_seed_a:
                        if (
                            basis is None
                            or param.data.ndim != 2
                            or int(basis.shape[1]) != int(param.data.shape[1])
                        ):
                            z = self._noise_for_param(
                                name, param, bases, profile=profile
                            )
                            update = z.mul(float(projected_grad) * probe_weight)
                            update_trace.add_delta(
                                name,
                                update,
                                scale=-float(learning_rate),
                            )
                            param.add_(update, alpha=-float(learning_rate))
                            continue
                        r_flat_components = (
                            self._abh_loren_r_flat_update_components_for_param(
                                name,
                                param,
                                basis,
                                profile=profile,
                            )
                            if str(self.method_config.abh_probe_transform)
                            == "loren_r_flat"
                            else None
                        )
                        if r_flat_components is not None:
                            r_matrix, right_basis = r_flat_components
                            r_update = r_matrix.mul(
                                float(projected_grad) * probe_weight
                            )
                            key = str(name)
                            previous = left_accumulators.get(key)
                            if previous is None:
                                left_accumulators[key] = (
                                    param,
                                    r_update.detach().float().clone(),
                                    right_basis,
                                )
                            else:
                                previous_param, accumulator, previous_basis = previous
                                accumulator.add_(r_update.detach().float())
                                left_accumulators[key] = (
                                    previous_param,
                                    accumulator,
                                    previous_basis,
                                )
                            continue
                        components = self._abh_small_b_components_for_param(
                            name,
                            param,
                            basis,
                            projected_grad=float(projected_grad) * probe_weight,
                            use_update_preconditioner=True,
                            profile=profile,
                        )
                        if components is None:
                            z = self._noise_for_param(
                                name, param, bases, profile=profile
                            )
                            update = z.mul(float(projected_grad) * probe_weight)
                            update_trace.add_delta(
                                name,
                                update,
                                scale=-float(learning_rate),
                            )
                            param.add_(update, alpha=-float(learning_rate))
                            continue
                        a_matrix, small_update, right_basis = components
                        left_update = a_matrix.float().matmul(small_update)
                        key = str(name)
                        previous = left_accumulators.get(key)
                        if previous is None:
                            left_accumulators[key] = (
                                param,
                                left_update.detach().float().clone(),
                                right_basis,
                            )
                        else:
                            previous_param, accumulator, previous_basis = previous
                            accumulator.add_(left_update.detach().float())
                            left_accumulators[key] = (
                                previous_param,
                                accumulator,
                                previous_basis,
                            )
                        continue
                    if (
                        basis is None
                        or param.data.ndim != 2
                        or int(basis.shape[1]) != int(param.data.shape[1])
                    ):
                        z = self._noise_for_param(name, param, bases)
                        update = z.mul(float(projected_grad) * probe_weight)
                        update_trace.add_delta(
                            name,
                            update,
                            scale=-float(learning_rate),
                        )
                        param.add_(update, alpha=-float(learning_rate))
                        continue
                    components = self._abh_small_b_components_for_param(
                        name,
                        param,
                        basis,
                        projected_grad=float(projected_grad) * probe_weight,
                        use_update_preconditioner=True,
                        profile=profile,
                    )
                    if components is None:
                        z = self._noise_for_param(name, param, bases)
                        update = z.mul(float(projected_grad) * probe_weight)
                        update_trace.add_delta(
                            name,
                            update,
                            scale=-float(learning_rate),
                        )
                        param.add_(update, alpha=-float(learning_rate))
                        continue
                    a_matrix, small_update, right_basis = components
                    key = str(name)
                    previous = small_accumulators.get(key)
                    if previous is None:
                        small_accumulators[key] = (
                            param,
                            a_matrix,
                            small_update.detach().float().clone(),
                            right_basis,
                        )
                    else:
                        previous_param, previous_a, accumulator, previous_basis = (
                            previous
                        )
                        accumulator.add_(small_update.detach().float())
                        small_accumulators[key] = (
                            previous_param,
                            previous_a,
                            accumulator,
                            previous_basis,
                        )
            finally:
                self._active_noise_seed = previous_noise_seed

        if direct_probe_seed_a:
            total_weight_decay = float(weight_decay) * float(probe_count)
            for key, (param, left_update, right_basis) in left_accumulators.items():
                name, _ = parameter_lookup[key]
                transform = str(self.method_config.abh_update_transform)
                if transform in {"ab_momentum", "ab_adam"}:
                    left_update = self._abh_ab_momentum_or_adam_transform(
                        key,
                        left_update,
                        use_adam=(transform == "ab_adam"),
                        profile=profile,
                    )
                elif transform in {"muon_ns_r", "muon_ns_r_momentum"}:
                    left_update = self._abh_r_muon_transform(
                        key,
                        left_update,
                        use_momentum=(transform == "muon_ns_r_momentum"),
                        profile=profile,
                    )
                if total_weight_decay:
                    param.mul_(1.0 - float(learning_rate) * total_weight_decay)
                update_trace.add_delta(
                    f"{name}.left_lowrank",
                    left_update,
                    scale=-float(learning_rate),
                )
                parameter_summaries.append(
                    self._guided_delta_parameter_summary(
                        name,
                        param,
                        bases,
                        left_update,
                    )
                )
                right_basis = right_basis.to(device=param.device, dtype=param.dtype)
                param.addmm_(
                    left_update.to(device=param.device, dtype=param.dtype),
                    right_basis,
                    alpha=-float(learning_rate),
                )
            self._profile_add(
                profile,
                "update.combined_multi_probe_seed_a_lowrank_total_time",
                time.perf_counter() - total_started_at,
            )
            return self._guided_delta_payload_checksum(
                seed=int(probe_projected_grads[0][0]) if probe_projected_grads else 0,
                projected_grad=sum(value for _, value in probe_projected_grads)
                / float(probe_count),
                update_scale=-float(learning_rate),
                parameter_summaries=parameter_summaries,
            )

        for key, (
            param,
            a_matrix,
            small_update,
            right_basis,
        ) in small_accumulators.items():
            name, _ = parameter_lookup[key]
            transform = str(self.method_config.abh_update_transform)
            if transform in {"ab_momentum", "ab_adam"}:
                left_update = a_matrix.float().matmul(small_update)
                transformed_left = self._abh_ab_momentum_or_adam_transform(
                    key,
                    left_update,
                    use_adam=(transform == "ab_adam"),
                    profile=profile,
                )
                update = transformed_left.to(
                    device=param.device, dtype=param.dtype
                ).matmul(right_basis.to(device=param.device, dtype=param.dtype))
                if weight_decay:
                    update = update.add(param.data, alpha=float(weight_decay))
                update_trace.add_delta(name, update, scale=-float(learning_rate))
                parameter_summaries.append(
                    self._guided_delta_parameter_summary(name, param, bases, update)
                )
                param.add_(update, alpha=-float(learning_rate))
                continue
            if transform in {"muon_ns_r", "muon_ns_r_momentum"}:
                left_update = a_matrix.float().matmul(small_update)
                transformed_left = self._abh_r_muon_transform(
                    key,
                    left_update,
                    use_momentum=(transform == "muon_ns_r_momentum"),
                    profile=profile,
                )
                update = transformed_left.to(
                    device=param.device, dtype=param.dtype
                ).matmul(right_basis.to(device=param.device, dtype=param.dtype))
                if weight_decay:
                    update = update.add(param.data, alpha=float(weight_decay))
                update_trace.add_delta(name, update, scale=-float(learning_rate))
                parameter_summaries.append(
                    self._guided_delta_parameter_summary(name, param, bases, update)
                )
                param_before = (
                    self._bounded_tensor_trace(param)
                    if source_trace is not None and len(source_trace) < 16
                    else None
                )
                param.add_(update, alpha=-float(learning_rate))
                if source_trace is not None and len(source_trace) < 16:
                    source_trace.append(
                        {
                            "name": str(name),
                            "grad": self._bounded_tensor_trace(update),
                            "param_before": param_before,
                            "param_after": self._bounded_tensor_trace(param),
                        }
                    )
                continue
            transform_source = small_update
            if str(self.method_config.abh_update_transform) == "muon_ns_small_b_momentum":
                beta = float(self.method_config.abh_momentum_beta)
                previous = self._abh_muon_small_b_momentum.get(key)
                if previous is not None and tuple(previous.shape) == tuple(
                    small_update.shape
                ):
                    transform_source = previous.to(
                        device=small_update.device,
                        dtype=small_update.dtype,
                    )
                    transform_source.mul_(beta).add_(small_update, alpha=1.0 - beta)
                else:
                    transform_source = small_update.detach().clone()
                self._abh_muon_small_b_momentum[key] = (
                    transform_source.detach().float().contiguous()
                )
            pre_transform_norm = _safe_tensor_norm(transform_source)
            if pre_transform_norm is not None:
                self._abh_muon_small_b_pre_norm[key] = float(pre_transform_norm)
            transformed_small = _zeropower_via_newton_schulz(transform_source).to(
                device=param.device,
                dtype=param.dtype,
            )
            post_small_norm = _safe_tensor_norm(transformed_small)
            if post_small_norm is not None:
                self._abh_muon_small_b_post_norm[key] = float(post_small_norm)
            if (
                pre_transform_norm is not None
                and float(pre_transform_norm) > 0.0
                and post_small_norm is not None
            ):
                self._abh_muon_small_b_norm_ratio[key] = float(post_small_norm) / float(
                    pre_transform_norm
                )
            update = a_matrix.matmul(transformed_small.matmul(right_basis))
            if weight_decay:
                update = update.add(param.data, alpha=float(weight_decay))
            update_trace.add_delta(name, update, scale=-float(learning_rate))
            parameter_summaries.append(
                self._guided_delta_parameter_summary(name, param, bases, update)
            )
            param_before = (
                self._bounded_tensor_trace(param)
                if source_trace is not None and len(source_trace) < 16
                else None
            )
            param.add_(update, alpha=-float(learning_rate))
            if source_trace is not None and len(source_trace) < 16:
                source_trace.append(
                    {
                        "name": str(name),
                        "grad": self._bounded_tensor_trace(update),
                        "param_before": param_before,
                        "param_after": self._bounded_tensor_trace(param),
                    }
                )

        self._profile_add(
            profile,
            "update.combined_multi_structured_total_time",
            time.perf_counter() - total_started_at,
        )
        return self._guided_delta_payload_checksum(
            seed=int(probe_projected_grads[0][0]) if probe_projected_grads else 0,
            projected_grad=sum(value for _, value in probe_projected_grads)
            / float(probe_count),
            update_scale=-float(learning_rate),
            parameter_summaries=parameter_summaries,
        )

    def _abh_small_b_components_for_param(
        self,
        name: str,
        param: torch.nn.Parameter,
        q_t: torch.Tensor,
        *,
        projected_grad: float,
        use_update_preconditioner: bool = False,
        profile: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        left_factor_mode = str(self.method_config.abh_left_factor)
        if str(self.method_config.abh_probe_transform) == "loren_r_flat":
            return None
        if left_factor_mode not in {
            "orthogonal_a",
            "orthogonal_ab_random",
            "gaussian_ab_random",
        }:
            return None
        # The population fast path calls this helper directly instead of going
        # through ``_abh_noise_for_param``.  Apply the same probe-seeded active
        # Oja-basis selection here so top-48 stays shared while the configured
        # tail-16 is independently resampled for every probe.  Restore and
        # update replay the same seed and therefore recover the same basis.
        q_t = self._candidate_oja_basis(name, q_t)
        rank = self._effective_basis_rank(param, q_t)
        if rank <= 0:
            return None
        q_t = self._abh_q_dropout_basis(name, q_t)
        rank = min(rank, int(q_t.shape[0]))
        span_rank = int(q_t.shape[0])
        if left_factor_mode in {"orthogonal_ab_random", "gaussian_ab_random"}:
            a_matrix = self._abh_probe_a_matrix(
                name,
                param,
                rank,
                orthogonal=(left_factor_mode == "orthogonal_ab_random"),
            )
        else:
            a_matrix = self._abh_a_matrix(name, param, rank)
        b_probe, b_update = self._abh_probe_and_update_b_matrices(
            name,
            param,
            rank,
            span_rank,
        )
        b_matrix = b_update if use_update_preconditioner else b_probe
        basis = q_t.to(device=param.device, dtype=param.dtype)
        normalization = str(self.method_config.abh_normalization)
        normalization_scale, pre_norm_value, target_norm = (
            perturbation_normalization_scale(
                normalization=normalization,
                a_matrix=a_matrix,
                b_matrix=b_matrix,
                q_active=q_t,
                parameter_numel=param.data.numel(),
                layer_fro_scale=float(self.method_config.abh_layer_fro_scale),
            )
        )
        if normalization != "none" and target_norm > 0.0 and pre_norm_value is not None:
            self._abh_noise_pre_norm_ratio[str(name)] = float(pre_norm_value) / float(
                target_norm
            )
        self._abh_noise_scale_factor[str(name)] = float(normalization_scale)
        if normalization == "layer_fro" and target_norm > 0.0:
            self._abh_noise_post_norm_ratio[str(name)] = (
                float(pre_norm_value or 0.0)
                * float(normalization_scale)
                / float(target_norm)
            )
        small_update = b_matrix.float().mul(
            float(projected_grad) * float(normalization_scale)
        )
        self._profile_add(profile, "abh_multi_muon_small_b.parameters", 1)
        return a_matrix, small_update, basis

    def _abh_q_dropout_basis(
        self,
        name: str,
        q_t: torch.Tensor,
    ) -> torch.Tensor:
        keep_fraction = float(
            getattr(self.method_config, "abh_q_dropout_keep_fraction", 1.0)
        )
        span_rank = int(q_t.shape[0])
        if keep_fraction >= 1.0 or span_rank <= 1:
            self._abh_q_dropout_effective_keep[str(name)] = 1.0
            return q_t
        keep_rank = max(1, min(span_rank, int(math.ceil(span_rank * keep_fraction))))
        indices = torch.randperm(span_rank, device=q_t.device)[:keep_rank]
        self._abh_q_dropout_effective_keep[str(name)] = float(keep_rank) / float(
            span_rank
        )
        return q_t.index_select(0, indices).contiguous()

    def _abh_a_matrix(
        self,
        name: str,
        param: torch.nn.Parameter,
        rank: int,
    ) -> torch.Tensor:
        block = self._abh_a_refresh_block()
        seed_mode = str(getattr(self.method_config, "abh_a_seed_mode", "block"))
        probe_seed = (
            int(self._active_noise_seed)
            if seed_mode == "probe" and self._active_noise_seed is not None
            else None
        )
        # Probe-seeded A changes every probe and every step. Caching those matrices
        # in the long-lived block cache leaks GPU memory during multi-probe runs.
        cache_enabled = probe_seed is None
        cache_key = (
            str(name),
            int(block),
            probe_seed,
            int(rank),
            int(param.data.shape[0]),
            str(param.device),
            str(param.dtype),
        )
        if cache_enabled:
            cached = self._abh_a_cache.get(cache_key)
            if cached is not None:
                return cached
            self._abh_a_cache = {
                key: value
                for key, value in self._abh_a_cache.items()
                if not (
                    len(key) == 7
                    and key[0] == str(name)
                    and key[2] is None
                    and key[3] == int(rank)
                    and key[4] == int(param.data.shape[0])
                    and key[5] == str(param.device)
                    and key[6] == str(param.dtype)
                )
            }
        seed = _stable_int_seed(
            {
                "method": "agzo_abh_a",
                "name": str(name),
                "block": int(block),
                "probe_seed": probe_seed,
                "rank": int(rank),
                "out_dim": int(param.data.shape[0]),
            }
        )
        generator = torch.Generator(device=param.device)
        generator.manual_seed(seed)
        matrix_dtype = (
            torch.float32
            if param.data.dtype in {torch.float16, torch.bfloat16}
            else param.data.dtype
        )
        raw = torch.randn(
            (int(param.data.shape[0]), int(rank)),
            device=param.device,
            dtype=matrix_dtype,
            generator=generator,
        )
        if str(self.method_config.abh_left_factor) == "gaussian_a_random_b":
            result = raw.to(dtype=param.dtype).detach().contiguous()
        else:
            q, _ = torch.linalg.qr(raw, mode="reduced")
            result = q.to(dtype=param.dtype).detach().contiguous()
        if cache_enabled:
            self._abh_a_cache[cache_key] = result
        return result

    def _abh_probe_a_matrix(
        self,
        name: str,
        param: torch.nn.Parameter,
        rank: int,
        *,
        orthogonal: bool = True,
    ) -> torch.Tensor:
        probe_seed = (
            int(self._active_noise_seed) if self._active_noise_seed is not None else 0
        )
        return sample_probe_a(
            name=name,
            probe_seed=probe_seed,
            out_dim=int(param.data.shape[0]),
            rank=rank,
            device=param.device,
            dtype=param.dtype,
            orthogonal=orthogonal,
        )

    def _abh_b_matrix(
        self,
        name: str,
        param: torch.nn.Parameter,
        rank: int,
        span_rank: int,
    ) -> torch.Tensor:
        block = self._abh_a_refresh_block()
        cache_key = (
            str(name),
            int(block),
            int(rank),
            int(span_rank),
            str(param.device),
            str(param.dtype),
        )
        cached = self._abh_b_cache.get(cache_key)
        if cached is not None:
            return cached
        self._abh_b_cache = {
            key: value
            for key, value in self._abh_b_cache.items()
            if not (
                len(key) == 6
                and key[0] == str(name)
                and key[2] == int(rank)
                and key[3] == int(span_rank)
                and key[4] == str(param.device)
                and key[5] == str(param.dtype)
            )
        }
        seed = _stable_int_seed(
            {
                "method": "agzo_abh_b",
                "name": str(name),
                "block": int(block),
                "rank": int(rank),
                "span_rank": int(span_rank),
            }
        )
        generator = torch.Generator(device=param.device)
        generator.manual_seed(seed)
        result = (
            torch.randn(
                (int(rank), int(span_rank)),
                device=param.device,
                dtype=param.dtype,
                generator=generator,
            )
            .detach()
            .contiguous()
        )
        self._abh_b_cache[cache_key] = result
        return result

    def _abh_probe_b_matrix(
        self,
        name: str,
        param: torch.nn.Parameter,
        rank: int,
        span_rank: int,
    ) -> torch.Tensor:
        probe_seed = (
            int(self._active_noise_seed) if self._active_noise_seed is not None else 0
        )
        return sample_probe_b(
            name=name,
            probe_seed=probe_seed,
            block=self._abh_a_refresh_block(),
            rank=rank,
            span_rank=span_rank,
            device=param.device,
            dtype=param.dtype,
        )

    def _abh_a_refresh_block(self) -> int:
        interval = max(1, int(self.method_config.abh_a_refresh_interval))
        active_index = max(
            0,
            int(self._active_step_number) - int(self.method_config.abh_warmup_steps) - 1,
        )
        return active_index // interval

    def _noise_plan(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
    ) -> dict[str, int | Counter[str]]:
        agzo_parameter_count = 0
        fallback_parameter_count = 0
        fallback_reason_counts: Counter[str] = Counter()
        for name, param in parameters:
            reason = self._fallback_reason(name, param, bases)
            if reason is None:
                agzo_parameter_count += int(param.numel())
                continue
            fallback_parameter_count += int(param.numel())
            fallback_reason_counts[reason] += int(param.numel())
        return {
            "agzo_parameter_count": agzo_parameter_count,
            "fallback_parameter_count": fallback_parameter_count,
            "fallback_reason_counts": fallback_reason_counts,
        }

    def _fallback_reason(
        self,
        name: str,
        param: torch.nn.Parameter,
        bases: dict[str, torch.Tensor],
    ) -> str | None:
        basis = bases.get(name)
        if basis is None:
            return "non_linear_weight"
        if param.data.ndim != 2:
            return "non_2d_parameter"
        if int(basis.shape[1]) != int(param.data.shape[1]):
            return "basis_shape_mismatch"
        if self._effective_basis_rank(param, basis) <= 0:
            return "basis_shape_mismatch"
        return None

    def _activation_basis_checksum(self, bases: dict[str, torch.Tensor]) -> str:
        payload = [
            {
                "name": str(name),
                "summary": _tensor_scalar_summary(name, bases[name]),
            }
            for name in sorted(bases)
        ]
        return _stable_json_checksum(payload)

    def _guided_delta_checksum(
        self,
        parameters: ParameterList,
        bases: dict[str, torch.Tensor],
        *,
        seed: int,
        projected_grad: float | None = None,
        update_scale: float | None = None,
    ) -> str:
        parameter_summaries = []
        with torch.random.fork_rng(devices=_cuda_device_indices(parameters, bases)):
            torch.manual_seed(int(seed))
            for name, param in sorted(parameters, key=lambda item: item[0]):
                z = self._noise_for_param(name, param, bases)
                update = z.mul(float(projected_grad or 0.0))
                parameter_summaries.append(
                    self._guided_delta_parameter_summary(
                        name,
                        param,
                        bases,
                        update,
                    )
                )
        return self._guided_delta_payload_checksum(
            seed=seed,
            projected_grad=projected_grad,
            update_scale=update_scale,
            parameter_summaries=parameter_summaries,
        )

    def _guided_delta_parameter_summary(
        self,
        name: str,
        param: torch.nn.Parameter,
        bases: dict[str, torch.Tensor],
        update: torch.Tensor,
    ) -> dict[str, Any]:
        basis = bases.get(name)
        fallback_reason = self._fallback_reason(name, param, bases)
        update_fingerprint, _ = bounded_tensor_fingerprint(
            name,
            update,
            error_prefix="agzo guided delta",
        )
        return {
            "name": str(name),
            "shape": [int(dim) for dim in param.shape],
            "dtype": str(param.dtype),
            "device_type": str(param.device.type),
            "numel": int(param.numel()),
            "agzo_basis_applies": fallback_reason is None,
            "effective_rank": (
                self._effective_basis_rank(param, basis) if basis is not None else 0
            ),
            "fallback_reason": fallback_reason,
            "basis_summary": (
                _tensor_scalar_summary(name, basis) if basis is not None else None
            ),
            "guided_update_fingerprint": update_fingerprint,
        }

    def _guided_delta_payload_checksum(
        self,
        *,
        seed: int,
        projected_grad: float | None,
        update_scale: float | None,
        parameter_summaries: list[dict[str, Any]],
    ) -> str:
        return _stable_json_checksum(
            {
                "seed": int(seed),
                "projected_grad": _optional_finite_float(projected_grad),
                "update_scale": _optional_finite_float(update_scale),
                "parameters": parameter_summaries,
            }
        )

    def _effective_basis_rank(
        self,
        param: torch.nn.Parameter,
        basis: torch.Tensor,
    ) -> int:
        if param.data.ndim != 2:
            return 0
        return min(
            int(self.method_config.rank),
            int(basis.shape[0]),
            int(basis.shape[1]),
            int(param.data.shape[0]),
            int(param.data.shape[1]),
        )

    def _uses_abh_perturbation(self) -> bool:
        return str(self.method_config.perturbation_form) in {"abh", "abh_oja"}

    def _uses_oja_abh_basis(self) -> bool:
        return str(self.method_config.perturbation_form) == "abh_oja"

    def _uses_lagged_oja_basis(self) -> bool:
        return self._uses_oja_abh_basis() and bool(
            self.method_config.abh_oja_lagged_basis
        )

    def _uses_momentum_basis_only(self) -> bool:
        return str(self.method_config.perturbation_form) == "basis_momentum"

    def _uses_covariance_momentum_basis(self) -> bool:
        return (
            str(self.method_config.perturbation_form) == "abh"
            or self._uses_momentum_basis_only()
        )

    def _uses_active_abh_noise(self) -> bool:
        return self._uses_abh_perturbation() and (
            int(self._active_step_number) > int(self.method_config.abh_warmup_steps)
        )

    def _effective_abh_num_noise(self, estimator_mode: str) -> int:
        if estimator_mode not in {"two_side", "loren_population"}:
            return 1
        if not self._uses_active_abh_noise():
            return 1
        return max(1, int(getattr(self.method_config, "abh_num_noise", 1)))

    def _uses_combined_multi_structured_update(self) -> bool:
        return (
            self._uses_active_abh_noise()
            and int(getattr(self.method_config, "abh_num_noise", 1)) > 1
            and (
                str(self.method_config.abh_update_transform)
                in {
                    "ab_adam",
                    "ab_momentum",
                    "muon_ns_r",
                    "muon_ns_r_momentum",
                    "muon_ns_small_b",
                    "muon_ns_small_b_momentum",
                }
                or self._uses_fused_shared_q_population_update()
                or self._uses_fused_varied_q_population_update()
            )
        )

    def _uses_fused_varied_q_population_update(self) -> bool:
        return (
            bool(
                getattr(
                    self.method_config,
                    "abh_population_fused_varied_q_update",
                    False,
                )
            )
            and str(self.method_config.estimator_mode) == "loren_population"
            and bool(
                getattr(
                    self.method_config,
                    "abh_oja_active_resample_per_probe",
                    False,
                )
            )
            and str(self.method_config.abh_update_transform) == "none"
            and str(self.method_config.abh_probe_transform) == "none"
            and str(self.method_config.abh_left_factor)
            in {"orthogonal_ab_random", "gaussian_ab_random"}
            and float(self.method_config.abh_dense_residual_ratio) == 0.0
            and float(self.config.weight_decay) == 0.0
        )

    def _uses_fused_shared_q_population_update(self) -> bool:
        return (
            bool(
                getattr(
                    self.method_config,
                    "abh_population_fused_shared_q_update",
                    False,
                )
            )
            and str(self.method_config.estimator_mode) == "loren_population"
            and not bool(
                getattr(
                    self.method_config,
                    "abh_oja_active_resample_per_probe",
                    False,
                )
            )
            and str(self.method_config.abh_update_transform) == "none"
            and str(self.method_config.abh_probe_transform) == "none"
            and str(self.method_config.abh_left_factor)
            in {"orthogonal_ab_random", "gaussian_ab_random"}
            and float(self.method_config.abh_dense_residual_ratio) == 0.0
            and float(self.config.weight_decay) == 0.0
        )

    def _uses_abh_seed_pool(self, num_abh_noise: int, estimator_mode: str) -> bool:
        return (
            bool(getattr(self.method_config, "abh_seed_pool_enabled", False))
            and estimator_mode in {"two_side", "loren_population"}
            and self._uses_active_abh_noise()
            and (int(num_abh_noise) > 1 or bool(self._fixed_abh_seed_pool_seeds()))
        )

    def _select_abh_seed_pool_seeds(
        self,
        base_seed: int,
        probe_count: int,
    ) -> tuple[list[int], str]:
        probe_count = max(1, int(probe_count))
        fixed_seeds = self._fixed_abh_seed_pool_seeds()
        if fixed_seeds:
            selected: list[int] = []
            roles: dict[int, str] = {}
            offset_seed = _stable_int_seed(
                {
                    "method": "agzo_abh_fixed_seed_pool_offset",
                    "base_seed": int(base_seed),
                    "active_step": int(self._active_step_number),
                    "pool_size": int(len(fixed_seeds)),
                }
            )
            offset = int(offset_seed % len(fixed_seeds))
            index = 0
            while len(selected) < probe_count:
                seed = int(fixed_seeds[(offset + index) % len(fixed_seeds)])
                if seed not in selected:
                    selected.append(seed)
                    roles[seed] = "offline_fixed"
                    self._ensure_abh_seed_pool_seed(seed)
                index += 1
                if (
                    index > len(fixed_seeds) + probe_count
                    and len(selected) < probe_count
                ):
                    break
            while len(selected) < probe_count:
                seed = self._new_abh_pool_seed(
                    base_seed, role="fixed_fill", index=index
                )
                selected.append(seed)
                roles[int(seed)] = "fixed_fill"
                self._ensure_abh_seed_pool_seed(seed)
                index += 1
            self._abh_seed_pool_selected_roles = roles
            self._evict_abh_seed_pool(exclude=set(selected))
            return selected, "offline_fixed_seed_pool"

        self._decay_abh_seed_pool_scores()
        warmup_steps = int(getattr(self.method_config, "abh_seed_pool_warmup_steps", 50))
        if int(self._active_step_number) <= warmup_steps:
            self._abh_seed_pool_selected_roles = {}
            seeds = [
                self._new_abh_pool_seed(base_seed, role="warmup", index=index)
                for index in range(probe_count)
            ]
            for seed in seeds:
                self._ensure_abh_seed_pool_seed(seed)
                self._abh_seed_pool_selected_roles[int(seed)] = "warmup"
            self._evict_abh_seed_pool(exclude=set(seeds))
            return seeds, "warmup_random"

        if (
            str(getattr(self.method_config, "abh_seed_pool_score_mode", ""))
            == "population_rma"
        ):
            return self._select_abh_population_rma_pool_seeds(
                base_seed,
                probe_count,
            )

        selected: list[int] = []
        roles: dict[int, str] = {}
        if self._abh_seed_pool:
            seed = self._select_top_elite_seed(base_seed, exclude=set())
            if seed is not None:
                selected.append(int(seed))
                roles[int(seed)] = "top"
        soft = self._select_soft_elite_seed(base_seed, exclude=set(selected))
        if soft is not None:
            selected.append(int(soft))
            roles[int(soft)] = "soft"
        ucb = self._select_ucb_seed(exclude=set(selected))
        if ucb is not None:
            selected.append(int(ucb))
            roles[int(ucb)] = "ucb"
        fresh = self._new_abh_pool_seed(base_seed, role="fresh", index=0)
        selected.append(fresh)
        roles[int(fresh)] = "fresh"

        index = 0
        while len(set(selected)) < probe_count:
            fill = self._new_abh_pool_seed(base_seed, role="fill", index=index)
            selected.append(fill)
            roles[int(fill)] = "fill"
            index += 1
        deduped: list[int] = []
        deduped_roles: dict[int, str] = {}
        for seed in selected:
            if seed not in deduped:
                seed = int(seed)
                deduped.append(seed)
                deduped_roles[seed] = roles.get(seed, "unknown")
            if len(deduped) == probe_count:
                break
        for seed in deduped:
            self._ensure_abh_seed_pool_seed(seed)
        self._abh_seed_pool_selected_roles = deduped_roles
        self._evict_abh_seed_pool(exclude=set(deduped))
        return deduped, "seed_pool"

    def _select_abh_population_rma_pool_seeds(
        self,
        base_seed: int,
        probe_count: int,
    ) -> tuple[list[int], str]:
        selected: list[int] = []
        roles: dict[int, str] = {}
        top_count = int(getattr(self.method_config, "abh_seed_pool_top_select_count", 1))
        ucb_count = int(getattr(self.method_config, "abh_seed_pool_ucb_select_count", 1))
        fresh_count = int(
            getattr(self.method_config, "abh_seed_pool_fresh_select_count", 1)
        )
        if top_count == 1 and ucb_count == 1 and fresh_count == 1 and probe_count >= 6:
            top_count, ucb_count, fresh_count = 3, 2, 1

        for index in range(max(0, top_count)):
            if len(selected) >= probe_count:
                break
            seed = self._select_soft_elite_seed(
                base_seed + 7919 * index,
                exclude=set(selected),
            )
            if seed is not None:
                selected.append(int(seed))
                roles[int(seed)] = f"top20_{index}"

        for index in range(max(0, ucb_count)):
            if len(selected) >= probe_count:
                break
            seed = self._select_ucb_seed(exclude=set(selected))
            if seed is not None:
                selected.append(int(seed))
                roles[int(seed)] = f"ucb_{index}"

        for index in range(max(0, fresh_count)):
            if len(selected) >= probe_count:
                break
            seed = self._new_abh_pool_seed(base_seed, role="fresh", index=index)
            selected.append(int(seed))
            roles[int(seed)] = f"fresh_{index}"

        fill_index = 0
        while len(set(selected)) < probe_count:
            seed = self._new_abh_pool_seed(
                base_seed,
                role="population_fill",
                index=fill_index,
            )
            selected.append(int(seed))
            roles[int(seed)] = f"fill_{fill_index}"
            fill_index += 1
        deduped: list[int] = []
        deduped_roles: dict[int, str] = {}
        for seed in selected:
            seed = int(seed)
            if seed in deduped:
                continue
            deduped.append(seed)
            deduped_roles[seed] = roles.get(seed, "unknown")
            if len(deduped) >= probe_count:
                break
        for seed in deduped:
            self._ensure_abh_seed_pool_seed(seed)
        self._abh_seed_pool_selected_roles = deduped_roles
        self._evict_abh_seed_pool(exclude=set(deduped))
        return deduped, "population_rma_seed_pool"

    def _fixed_abh_seed_pool_seeds(self) -> list[int]:
        raw = str(getattr(self.method_config, "abh_seed_pool_fixed_seeds", "") or "")
        if not raw.strip():
            return []
        seeds: list[int] = []
        for part in raw.replace("\n", ",").split(","):
            value = part.strip()
            if not value:
                continue
            seeds.append(int(value))
        return seeds

    def _decay_abh_seed_pool_scores(self) -> None:
        gamma = float(getattr(self.method_config, "abh_seed_pool_gamma", 0.999))
        if gamma == 1.0:
            return
        for row in self._abh_seed_pool.values():
            row["score"] = float(row.get("score", 0.0)) * gamma

    def _seed_pool_score(self, seed: int) -> float:
        return float(self._abh_seed_pool.get(int(seed), {}).get("score", 0.0))

    def _abh_seed_pool_snapshot(
        self, *, limit: int = 8
    ) -> list[dict[str, float | int]]:
        if not self._abh_seed_pool:
            return []
        rows: list[dict[str, float | int]] = []
        for seed, state in sorted(
            self._abh_seed_pool.items(),
            key=lambda item: float(item[1].get("score", 0.0)),
            reverse=True,
        )[: max(0, int(limit))]:
            rows.append(
                {
                    "seed": int(seed),
                    "score": float(state.get("score", 0.0)),
                    "count": int(state.get("count", 0)),
                    "m_s": float(state.get("m_s", 0.0)),
                    "m_abs_s": float(state.get("m_abs_s", 0.0)),
                    "m_h": float(state.get("m_h", 0.0)),
                    "m_s_std": float(state.get("m_s_std", 0.0)),
                    "m_snr": float(state.get("m_snr", 0.0)),
                    "m_sign_consistency": float(state.get("m_sign_consistency", 0.0)),
                    "m_noise_ratio": float(state.get("m_noise_ratio", 0.0)),
                }
            )
        return rows

    def _new_abh_pool_seed(self, base_seed: int, *, role: str, index: int) -> int:
        return _stable_int_seed(
            {
                "method": "agzo_abh_seed_pool",
                "base_seed": int(base_seed),
                "active_step": int(self._active_step_number),
                "role": str(role),
                "index": int(index),
                "pool_generation": int(self._abh_seed_pool_total_evals),
            }
        )

    def _ensure_abh_seed_pool_seed(self, seed: int) -> None:
        seed = int(seed)
        if seed not in self._abh_seed_pool:
            self._abh_seed_pool[seed] = {
                "score": 0.0,
                "count": 0,
                "m_s": 0.0,
                "m_abs_s": 0.0,
                "m_h": 0.0,
                "m_s2": 0.0,
                "created_step": int(self._active_step_number),
            }

    def _evict_abh_seed_pool(self, *, exclude: set[int] | None = None) -> None:
        max_size = int(getattr(self.method_config, "abh_seed_pool_size", 128))
        exclude = set() if exclude is None else {int(value) for value in exclude}
        while len(self._abh_seed_pool) > max_size:
            evictable = [
                seed for seed in self._abh_seed_pool.keys() if int(seed) not in exclude
            ]
            if not evictable:
                return
            victim = self._select_abh_seed_pool_eviction_victim(evictable)
            self._abh_seed_pool.pop(int(victim), None)

    def _select_abh_seed_pool_eviction_victim(self, seeds: list[int]) -> int:
        min_survival_count = int(
            getattr(self.method_config, "abh_seed_pool_min_survival_count", 0)
        )
        mature = [
            int(seed)
            for seed in seeds
            if int(self._abh_seed_pool[int(seed)].get("count", 0)) >= min_survival_count
        ]
        candidates = mature if mature else [int(seed) for seed in seeds]

        bad_count_threshold = int(
            getattr(self.method_config, "abh_seed_pool_bad_count_threshold", 0)
        )
        bad_score_quantile = float(
            getattr(self.method_config, "abh_seed_pool_bad_score_quantile", 0.0)
        )
        if bad_count_threshold > 0 and bad_score_quantile > 0.0 and candidates:
            scored = sorted(self._seed_pool_score(seed) for seed in candidates)
            cutoff_index = max(
                0,
                min(
                    len(scored) - 1,
                    int(math.ceil(len(scored) * bad_score_quantile)) - 1,
                ),
            )
            bad_score_cutoff = float(scored[cutoff_index])
            confirmed_bad = [
                int(seed)
                for seed in candidates
                if int(self._abh_seed_pool[int(seed)].get("count", 0))
                >= bad_count_threshold
                and self._seed_pool_score(int(seed)) <= bad_score_cutoff
            ]
            if confirmed_bad:
                return int(min(confirmed_bad, key=self._seed_pool_score))

        return int(min(candidates, key=self._seed_pool_score))

    def _select_soft_elite_seed(
        self,
        base_seed: int,
        *,
        exclude: set[int],
    ) -> int | None:
        if not self._abh_seed_pool:
            return None
        top_fraction = float(
            getattr(self.method_config, "abh_seed_pool_top_fraction", 0.2)
        )
        pool_items = sorted(
            self._abh_seed_pool.keys(), key=self._seed_pool_score, reverse=True
        )
        top_count = max(1, int(math.ceil(len(pool_items) * top_fraction)))
        candidates = [seed for seed in pool_items[:top_count] if seed not in exclude]
        if not candidates:
            return None
        temperature = float(getattr(self.method_config, "abh_seed_pool_temperature", 1.0))
        scores = [self._seed_pool_score(seed) / temperature for seed in candidates]
        max_score = max(scores)
        weights = [
            math.exp(max(-60.0, min(60.0, score - max_score))) for score in scores
        ]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return int(candidates[0])
        draw_seed = _stable_int_seed(
            {
                "method": "agzo_abh_seed_pool_soft_elite",
                "base_seed": int(base_seed),
                "active_step": int(self._active_step_number),
                "pool_total_evals": int(self._abh_seed_pool_total_evals),
            }
        )
        threshold = (draw_seed / float(2**63 - 1)) * total
        cumulative = 0.0
        for seed, weight in zip(candidates, weights):
            cumulative += weight
            if cumulative >= threshold:
                return int(seed)
        return int(candidates[-1])

    def _select_top_elite_seed(
        self,
        base_seed: int,
        *,
        exclude: set[int],
    ) -> int | None:
        if not self._abh_seed_pool:
            return None
        pool_items = sorted(
            self._abh_seed_pool.keys(), key=self._seed_pool_score, reverse=True
        )
        top_k = int(getattr(self.method_config, "abh_seed_pool_top_k", 1))
        top_k = max(1, min(top_k, len(pool_items)))
        candidates = [seed for seed in pool_items[:top_k] if seed not in exclude]
        if not candidates:
            return None
        if len(candidates) == 1:
            return int(candidates[0])
        temperature = float(getattr(self.method_config, "abh_seed_pool_temperature", 1.0))
        scores = [self._seed_pool_score(seed) / temperature for seed in candidates]
        max_score = max(scores)
        weights = [
            math.exp(max(-60.0, min(60.0, score - max_score))) for score in scores
        ]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            return int(candidates[0])
        draw_seed = _stable_int_seed(
            {
                "method": "agzo_abh_seed_pool_top_elite",
                "base_seed": int(base_seed),
                "active_step": int(self._active_step_number),
                "pool_total_evals": int(self._abh_seed_pool_total_evals),
                "top_k": int(top_k),
            }
        )
        threshold = (draw_seed / float(2**63 - 1)) * total
        cumulative = 0.0
        for seed, weight in zip(candidates, weights):
            cumulative += weight
            if cumulative >= threshold:
                return int(seed)
        return int(candidates[-1])

    def _select_ucb_seed(self, *, exclude: set[int]) -> int | None:
        candidates = [
            seed for seed in self._abh_seed_pool.keys() if seed not in exclude
        ]
        if not candidates:
            return None
        lambda_u = float(getattr(self.method_config, "abh_seed_pool_ucb_lambda", 0.5))
        log_total = math.log(float(self._abh_seed_pool_total_evals) + 1.0)

        def ucb(seed: int) -> float:
            row = self._abh_seed_pool[int(seed)]
            count = float(row.get("count", 0))
            bonus = lambda_u * math.sqrt(log_total / (count + 1.0))
            return float(row.get("score", 0.0)) + bonus

        return int(max(candidates, key=ucb))

    def _update_abh_seed_pool_scores(
        self,
        rows: list[dict[str, float | int]],
        *,
        loss0: float,
        eps: float,
    ) -> list[dict[str, float | int]]:
        rho = float(getattr(self.method_config, "abh_seed_pool_rho", 0.2))
        lambda_c = float(
            getattr(self.method_config, "abh_seed_pool_curvature_lambda", 1.0)
        )
        score_mode = str(
            getattr(
                self.method_config,
                "abh_seed_pool_score_mode",
                "abs_s_minus_curvature",
            )
        )
        updated_rows: list[dict[str, float | int]] = []
        for row in rows:
            seed = int(row["seed"])
            self._ensure_abh_seed_pool_seed(seed)
            f_plus = float(row["f_plus"])
            f_minus = float(row["f_minus"])
            s_raw = float(row["s_raw"])
            c_proxy = max(f_plus + f_minus - 2.0 * float(loss0), 0.0)
            c_over_eps = c_proxy / max(float(eps), 1e-30)
            abs_s = abs(s_raw)
            curvature_signal_ratio = (
                c_over_eps / max(abs_s, 1e-12) if c_over_eps > 0.0 else 0.0
            )
            learning_rate = float(getattr(self.config, "learning_rate", 0.0))
            taylor_risk = (
                learning_rate * c_proxy / max(2.0 * float(eps) ** 2, 1e-30)
                if c_proxy > 0.0
                else 0.0
            )
            taylor_expected_gain = learning_rate * (abs_s**2) * (1.0 - taylor_risk)
            if score_mode == "negative_curvature":
                score = -lambda_c * c_over_eps
            elif score_mode == "signal_ratio_damped":
                score = abs_s / (1.0 + lambda_c * curvature_signal_ratio)
            elif score_mode == "taylor_expected_gain":
                score = taylor_expected_gain
            elif score_mode in {"rma_signal_ratio", "rma_zscore"}:
                score = 0.0
            else:
                score = abs_s - lambda_c * c_over_eps
            state = self._abh_seed_pool[seed]
            previous_count = int(state.get("count", 0))
            if score_mode in {"rma_signal_ratio", "rma_zscore"}:
                if previous_count <= 0:
                    state["m_s"] = float(s_raw)
                    state["m_abs_s"] = float(abs_s)
                    state["m_h"] = float(c_over_eps)
                    state["m_s2"] = float(s_raw) * float(s_raw)
                else:
                    state["m_s"] = (1.0 - rho) * float(
                        state.get("m_s", 0.0)
                    ) + rho * float(s_raw)
                    state["m_abs_s"] = (1.0 - rho) * float(
                        state.get("m_abs_s", 0.0)
                    ) + rho * float(abs_s)
                    state["m_h"] = (1.0 - rho) * float(
                        state.get("m_h", 0.0)
                    ) + rho * float(c_over_eps)
                    state["m_s2"] = (1.0 - rho) * float(
                        state.get("m_s2", 0.0)
                    ) + rho * float(s_raw) * float(s_raw)
                state["count"] = previous_count + 1
            else:
                state["score"] = (1.0 - rho) * float(
                    state.get("score", 0.0)
                ) + rho * score
                state["count"] = previous_count + 1
            m_s_state = float(state.get("m_s", 0.0))
            m_abs_s_state = float(state.get("m_abs_s", 0.0))
            m_s2_state = float(state.get("m_s2", 0.0))
            m_s_var = max(m_s2_state - m_s_state * m_s_state, 0.0)
            m_s_std = math.sqrt(m_s_var)
            m_snr = abs(m_s_state) / (m_s_std + 1e-12)
            m_sign_consistency = abs(m_s_state) / (m_abs_s_state + 1e-12)
            m_sign_consistency = max(0.0, min(1.0, m_sign_consistency))
            m_noise_ratio = 1.0 - m_sign_consistency
            state["m_s_std"] = float(m_s_std)
            state["m_snr"] = float(m_snr)
            state["m_sign_consistency"] = float(m_sign_consistency)
            state["m_noise_ratio"] = float(m_noise_ratio)
            self._abh_seed_pool_total_evals += 1
            scored = dict(row)
            scored.update(
                {
                    "abs_s": float(abs_s),
                    "c_proxy": float(c_proxy),
                    "c_over_eps": float(c_over_eps),
                    "curvature_signal_ratio": float(curvature_signal_ratio),
                    "taylor_risk": float(taylor_risk),
                    "taylor_expected_gain": float(taylor_expected_gain),
                    "score": float(score),
                    "score_mode": score_mode,
                    "selection_role": str(
                        self._abh_seed_pool_selected_roles.get(seed, "unknown")
                    ),
                    "ema_score": float(state["score"]),
                    "m_s": float(state.get("m_s", 0.0)),
                    "m_abs_s": float(state.get("m_abs_s", 0.0)),
                    "m_h": float(state.get("m_h", 0.0)),
                    "m_s2": float(state.get("m_s2", 0.0)),
                    "m_s_std": float(state.get("m_s_std", 0.0)),
                    "m_snr": float(state.get("m_snr", 0.0)),
                    "m_sign_consistency": float(state.get("m_sign_consistency", 0.0)),
                    "m_noise_ratio": float(state.get("m_noise_ratio", 0.0)),
                    "count": int(state["count"]),
                }
            )
            updated_rows.append(scored)
        if score_mode in {"rma_signal_ratio", "rma_zscore"}:
            self._refresh_abh_seed_pool_rma_scores()
            for row in updated_rows:
                seed = int(row["seed"])
                state = self._abh_seed_pool.get(seed, {})
                row["score"] = float(state.get("score", 0.0))
                row["ema_score"] = float(state.get("score", 0.0))
                row["m_s_std"] = float(state.get("m_s_std", 0.0))
                row["m_snr"] = float(state.get("m_snr", 0.0))
                row["m_sign_consistency"] = float(state.get("m_sign_consistency", 0.0))
                row["m_noise_ratio"] = float(state.get("m_noise_ratio", 0.0))
        self._evict_abh_seed_pool(exclude={int(row["seed"]) for row in rows})
        return updated_rows

    def _update_abh_seed_pool_population_scores(
        self,
        *,
        probe_fweights: list[tuple[int, float]],
        probe_losses: list[tuple[int, float]],
        loss_mean: float,
        loss_std: float = 0.0,
        fweight_normalizer: float = 1.0,
    ) -> list[dict[str, float | int]]:
        rho = float(getattr(self.method_config, "abh_seed_pool_rho", 0.2))
        loss_by_seed = {int(seed): float(loss) for seed, loss in probe_losses}
        sorted_losses = sorted(
            loss_by_seed.items(),
            key=lambda item: float(item[1]),
        )
        rank_by_seed = {
            int(seed): int(rank)
            for rank, (seed, _) in enumerate(sorted_losses, start=1)
        }
        rows: list[dict[str, float | int]] = []
        for seed_value, fweight_value in probe_fweights:
            seed = int(seed_value)
            fweight = float(fweight_value)
            reward = -fweight
            self._ensure_abh_seed_pool_seed(seed)
            state = self._abh_seed_pool[seed]
            previous_count = int(state.get("count", 0))
            if previous_count <= 0:
                state["m_population_reward"] = float(reward)
                state["m_population_abs_reward"] = abs(float(reward))
                state["m_population_reward2"] = float(reward) * float(reward)
                state["m_population_fweight"] = float(fweight)
                state["m_population_rank"] = float(rank_by_seed.get(seed, 0))
            else:
                state["m_population_reward"] = (1.0 - rho) * float(
                    state.get("m_population_reward", 0.0)
                ) + rho * float(reward)
                state["m_population_abs_reward"] = (1.0 - rho) * float(
                    state.get("m_population_abs_reward", 0.0)
                ) + rho * abs(float(reward))
                state["m_population_reward2"] = (1.0 - rho) * float(
                    state.get("m_population_reward2", 0.0)
                ) + rho * float(reward) * float(reward)
                state["m_population_fweight"] = (1.0 - rho) * float(
                    state.get("m_population_fweight", 0.0)
                ) + rho * float(fweight)
                state["m_population_rank"] = (1.0 - rho) * float(
                    state.get("m_population_rank", 0.0)
                ) + rho * float(rank_by_seed.get(seed, 0))
            state["count"] = previous_count + 1
            reward_mean = float(state.get("m_population_reward", 0.0))
            reward2_mean = float(state.get("m_population_reward2", 0.0))
            reward_var = max(reward2_mean - reward_mean * reward_mean, 0.0)
            reward_std = math.sqrt(reward_var)
            reward_snr = reward_mean / (reward_std + 1e-12)
            state["m_population_reward_std"] = float(reward_std)
            state["m_population_reward_snr"] = float(reward_snr)
            state["score"] = float(reward_mean)
            self._abh_seed_pool_total_evals += 1
            row = {
                "seed": int(seed),
                "fweight": float(fweight),
                "population_reward": float(reward),
                "population_loss": float(loss_by_seed.get(seed, float("nan"))),
                "population_loss_mean": float(loss_mean),
                "population_loss_std": float(loss_std),
                "population_fweight_normalizer": float(fweight_normalizer),
                "population_rank": int(rank_by_seed.get(seed, 0)),
                "score": float(state.get("score", 0.0)),
                "ema_score": float(state.get("score", 0.0)),
                "score_mode": "population_rma",
                "selection_role": str(
                    self._abh_seed_pool_selected_roles.get(seed, "unknown")
                ),
                "m_population_reward": float(state.get("m_population_reward", 0.0)),
                "m_population_abs_reward": float(
                    state.get("m_population_abs_reward", 0.0)
                ),
                "m_population_fweight": float(state.get("m_population_fweight", 0.0)),
                "m_population_rank": float(state.get("m_population_rank", 0.0)),
                "m_population_reward_std": float(
                    state.get("m_population_reward_std", 0.0)
                ),
                "m_population_reward_snr": float(
                    state.get("m_population_reward_snr", 0.0)
                ),
                "count": int(state["count"]),
            }
            rows.append(row)
        self._evict_abh_seed_pool(exclude={int(row["seed"]) for row in rows})
        return rows

    def _refresh_abh_seed_pool_rma_scores(self) -> None:
        active = {
            int(seed): row
            for seed, row in self._abh_seed_pool.items()
            if int(row.get("count", 0)) > 0
        }
        if not active:
            return

        def z_values(values: dict[int, float]) -> dict[int, float]:
            if not values:
                return {}
            vals = list(values.values())
            mean_value = sum(vals) / float(len(vals))
            variance = sum((value - mean_value) ** 2 for value in vals) / float(
                max(1, len(vals) - 1)
            )
            std_value = math.sqrt(max(variance, 0.0)) + 1e-12
            return {
                seed: (value - mean_value) / std_value for seed, value in values.items()
            }

        m_h_values = {
            seed: max(float(row.get("m_h", 0.0)), 0.0) for seed, row in active.items()
        }
        mean_positive_h = (
            sum(m_h_values.values()) / float(len(m_h_values)) if m_h_values else 0.0
        )
        z_abs_s = z_values(
            {seed: float(row.get("m_abs_s", 0.0)) for seed, row in active.items()}
        )
        z_signed_s = z_values(
            {seed: abs(float(row.get("m_s", 0.0))) for seed, row in active.items()}
        )
        z_h = z_values(m_h_values)
        z_noise = z_values(
            {seed: float(row.get("m_noise_ratio", 0.0)) for seed, row in active.items()}
        )
        z_count = z_values(
            {
                seed: 1.0 - 1.0 / math.sqrt(float(row.get("count", 0)) + 1.0)
                for seed, row in active.items()
            }
        )
        abs_s_lambda = float(
            getattr(self.method_config, "abh_seed_pool_abs_s_lambda", 1.0)
        )
        signed_s_lambda = float(
            getattr(self.method_config, "abh_seed_pool_signed_s_lambda", 0.0)
        )
        h_lambda = float(
            getattr(self.method_config, "abh_seed_pool_curvature_lambda", 1.0)
        )
        noise_lambda = float(
            getattr(self.method_config, "abh_seed_pool_noise_lambda", 0.0)
        )
        count_lambda = float(
            getattr(self.method_config, "abh_seed_pool_count_lambda", 0.0)
        )
        score_mode = str(
            getattr(self.method_config, "abh_seed_pool_score_mode", "rma_zscore")
        )
        if score_mode == "rma_signal_ratio":
            ratio_values: dict[int, float] = {}
            h_scale = max(mean_positive_h, 1e-12)
            for seed, row in active.items():
                normalized_h = max(float(row.get("m_h", 0.0)), 0.0) / h_scale
                ratio_values[seed] = float(row.get("m_abs_s", 0.0)) / (
                    1.0 + h_lambda * normalized_h
                )
            z_ratio = z_values(ratio_values)
            for seed, row in active.items():
                row["score"] = (
                    abs_s_lambda * z_ratio.get(seed, 0.0)
                    + signed_s_lambda * z_signed_s.get(seed, 0.0)
                    - noise_lambda * z_noise.get(seed, 0.0)
                    + count_lambda * z_count.get(seed, 0.0)
                )
            return
        for seed, row in active.items():
            row["score"] = (
                abs_s_lambda * z_abs_s.get(seed, 0.0)
                + signed_s_lambda * z_signed_s.get(seed, 0.0)
                - h_lambda * z_h.get(seed, 0.0)
                - noise_lambda * z_noise.get(seed, 0.0)
                + count_lambda * z_count.get(seed, 0.0)
            )

    def _multi_probe_update_weights(
        self,
        probe_values: list[tuple[int, float]],
        score_rows: list[dict[str, float | int]],
    ) -> dict[int, float]:
        if not probe_values:
            return {}
        uniform = {
            int(seed): 1.0 / float(len(probe_values)) for seed, _ in probe_values
        }
        if str(self.method_config.abh_multi_update_weighting) == "uniform":
            return self._filter_multi_probe_update_weights(uniform, score_rows)
        if str(self.method_config.abh_multi_update_weighting) in {
            "active_best_current_score",
            "active_max_abs_s",
            "active_min_abs_s",
            "active_signal_ratio",
        }:
            if not score_rows:
                return self._filter_multi_probe_update_weights(uniform, score_rows)
            row_by_seed = {int(row["seed"]): row for row in score_rows}
            active_roles = {
                "offline_fixed",
                "top",
                "soft",
                "soft_elite",
                "top_elite",
                "warmup",
            }
            candidates: list[int] = []
            for seed, _ in probe_values:
                row = row_by_seed.get(int(seed))
                if row is None:
                    continue
                role = str(row.get("selection_role", "unknown"))
                if role in active_roles:
                    candidates.append(int(seed))
            if not candidates:
                return {int(seed): 0.0 for seed, _ in probe_values}
            lambda_c = float(
                getattr(self.method_config, "abh_multi_update_curvature_lambda", 1.0)
            )

            def current_score(seed: int) -> float:
                row = row_by_seed[int(seed)]
                abs_s = abs(float(row.get("abs_s", row.get("s_raw", 0.0))))
                c_over_eps = max(0.0, float(row.get("c_over_eps", 0.0)))
                if (
                    str(self.method_config.abh_multi_update_weighting)
                    == "active_max_abs_s"
                ):
                    return abs_s
                if (
                    str(self.method_config.abh_multi_update_weighting)
                    == "active_min_abs_s"
                ):
                    return -abs_s
                if (
                    str(self.method_config.abh_multi_update_weighting)
                    == "active_signal_ratio"
                ):
                    return abs_s / (1.0 + lambda_c * c_over_eps)
                return abs_s - lambda_c * c_over_eps

            best_seed = max(candidates, key=current_score)
            weights = {
                int(seed): 1.0 if int(seed) == int(best_seed) else 0.0
                for seed, _ in probe_values
            }
            return self._filter_multi_probe_update_weights(weights, score_rows)
        if str(self.method_config.abh_multi_update_weighting) == "active_min_curvature":
            if not score_rows:
                return self._filter_multi_probe_update_weights(uniform, score_rows)
            row_by_seed = {int(row["seed"]): row for row in score_rows}
            active_roles = {
                "offline_fixed",
                "top",
                "soft",
                "soft_elite",
                "top_elite",
                "warmup",
            }
            candidates: list[int] = []
            for seed, _ in probe_values:
                row = row_by_seed.get(int(seed))
                if row is None:
                    continue
                role = str(row.get("selection_role", "unknown"))
                c_over_eps = float(row.get("c_over_eps", 0.0))
                if role in active_roles and math.isfinite(c_over_eps):
                    candidates.append(int(seed))
            if not candidates:
                return {int(seed): 0.0 for seed, _ in probe_values}
            best_seed = min(
                candidates,
                key=lambda seed: float(row_by_seed[int(seed)].get("c_over_eps", 0.0)),
            )
            weights = {
                int(seed): 1.0 if int(seed) == int(best_seed) else 0.0
                for seed, _ in probe_values
            }
            return self._filter_multi_probe_update_weights(weights, score_rows)
        if str(self.method_config.abh_multi_update_weighting) == "best_score":
            if not score_rows:
                return self._filter_multi_probe_update_weights(uniform, score_rows)
            min_count = int(
                getattr(self.method_config, "abh_seed_pool_update_min_count", 0)
            )
            score_by_seed: dict[int, float] = {}
            count_by_seed: dict[int, int] = {}
            for row in score_rows:
                seed = int(row["seed"])
                score = float(row.get("score", row.get("ema_score", 0.0)))
                if math.isfinite(score):
                    score_by_seed[seed] = score
                    count_by_seed[seed] = int(row.get("count", 0))
            candidates = [
                int(seed)
                for seed, _ in probe_values
                if int(seed) in score_by_seed
                and int(count_by_seed.get(int(seed), 0)) >= min_count
            ]
            if not candidates:
                return {int(seed): 0.0 for seed, _ in probe_values}
            best_seed = max(candidates, key=lambda seed: score_by_seed[int(seed)])
            weights = {
                int(seed): 1.0 if int(seed) == int(best_seed) else 0.0
                for seed, _ in probe_values
            }
            return self._filter_multi_probe_update_weights(weights, score_rows)
        score_by_seed: dict[int, float] = {}
        for row in score_rows:
            seed = int(row["seed"])
            score = float(row.get("score", 0.0))
            if math.isfinite(score):
                score_by_seed[seed] = score
        seeds: list[int] = []
        scores: list[float] = []
        for seed, _ in probe_values:
            seed = int(seed)
            if seed not in score_by_seed:
                return self._filter_multi_probe_update_weights(uniform, score_rows)
            seeds.append(seed)
            scores.append(float(score_by_seed[seed]))
        if len(scores) <= 1:
            return self._filter_multi_probe_update_weights(uniform, score_rows)
        mean_score = sum(scores) / float(len(scores))
        variance = sum((score - mean_score) ** 2 for score in scores) / float(
            len(scores)
        )
        std_score = math.sqrt(max(variance, 0.0))
        if not math.isfinite(std_score) or std_score <= 1e-12:
            return self._filter_multi_probe_update_weights(uniform, score_rows)
        tau = float(self.method_config.abh_multi_update_softmax_tau)
        logits = [(score - mean_score) / (std_score + 1e-12) / tau for score in scores]
        max_logit = max(logits)
        exp_values = [
            math.exp(max(-60.0, min(60.0, logit - max_logit))) for logit in logits
        ]
        total = sum(exp_values)
        if not math.isfinite(total) or total <= 0.0:
            return self._filter_multi_probe_update_weights(uniform, score_rows)
        weights = {
            int(seed): float(value) / float(total)
            for seed, value in zip(seeds, exp_values, strict=False)
        }
        return self._filter_multi_probe_update_weights(weights, score_rows)

    def _apply_seed_pool_rma_damping_to_projected_grads(
        self,
        probe_values: list[tuple[int, float]],
        score_rows: list[dict[str, float | int]],
    ) -> list[tuple[int, float]]:
        damping_lambda = float(
            getattr(self.method_config, "abh_curvature_damping_lambda", 0.0)
        )
        if (
            damping_lambda <= 0.0
            or str(getattr(self.method_config, "abh_seed_pool_score_mode", ""))
            != "rma_zscore"
            or not probe_values
        ):
            return probe_values
        active_h = [
            max(float(row.get("m_h", 0.0)), 0.0)
            for row in self._abh_seed_pool.values()
            if int(row.get("count", 0)) > 0
        ]
        mean_h = sum(active_h) / float(len(active_h)) if active_h else 0.0
        if mean_h <= 0.0 or not math.isfinite(mean_h):
            return probe_values
        row_by_seed = {int(row["seed"]): row for row in score_rows}
        damped: list[tuple[int, float]] = []
        for seed, value in probe_values:
            seed = int(seed)
            state = self._abh_seed_pool.get(seed, {})
            h_value = max(float(state.get("m_h", 0.0)), 0.0)
            normalized_h = h_value / (mean_h + 1e-12)
            factor = 1.0 / (1.0 + damping_lambda * normalized_h)
            if not math.isfinite(factor):
                factor = 1.0
            factor = max(0.0, min(1.0, factor))
            damped_value = float(value) * factor
            row = row_by_seed.get(seed)
            if row is not None:
                row["rma_damping_m_h"] = float(h_value)
                row["rma_damping_mean_h"] = float(mean_h)
                row["rma_damping_normalized_h"] = float(normalized_h)
                row["rma_damping_factor"] = float(factor)
                row["s_after_rma_damping"] = float(damped_value)
            damped.append((seed, damped_value))
        return damped

    def _filter_multi_probe_update_weights(
        self,
        weights: dict[int, float],
        score_rows: list[dict[str, float | int]],
    ) -> dict[int, float]:
        c_threshold = float(
            getattr(
                self.method_config,
                "abh_seed_pool_update_c_over_eps_threshold",
                0.0,
            )
        )
        ratio_threshold = float(
            getattr(
                self.method_config,
                "abh_seed_pool_update_curvature_signal_ratio_threshold",
                0.0,
            )
        )
        roles_raw = str(
            getattr(self.method_config, "abh_seed_pool_update_filter_roles", "")
        )
        min_count = int(getattr(self.method_config, "abh_seed_pool_update_min_count", 0))
        roles = {role.strip() for role in roles_raw.split(",") if role.strip()}
        if (
            (c_threshold <= 0.0 and ratio_threshold <= 0.0 and min_count <= 0)
            or not weights
            or not score_rows
        ):
            return weights

        row_by_seed = {int(row["seed"]): row for row in score_rows}
        filtered: dict[int, float] = {}
        for seed, weight in weights.items():
            row = row_by_seed.get(int(seed))
            if row is None:
                filtered[int(seed)] = float(weight)
                continue
            role = str(row.get("selection_role", "unknown"))
            count = int(row.get("count", 0))
            c_over_eps = float(row.get("c_over_eps", 0.0))
            abs_s = abs(float(row.get("abs_s", row.get("s_raw", 0.0))))
            curvature_signal_ratio = (
                c_over_eps / max(abs_s, 1e-12) if c_over_eps > 0.0 else 0.0
            )
            c_filtered = c_threshold > 0.0 and c_over_eps > c_threshold
            ratio_filtered = (
                ratio_threshold > 0.0 and curvature_signal_ratio > ratio_threshold
            )
            if roles and role in roles and (c_filtered or ratio_filtered):
                filtered[int(seed)] = 0.0
            elif min_count > 0 and count < min_count:
                filtered[int(seed)] = 0.0
            else:
                filtered[int(seed)] = float(weight)

        total = sum(max(0.0, float(value)) for value in filtered.values())
        if total <= 0.0 or not math.isfinite(total):
            return {int(seed): 0.0 for seed in weights}
        return {
            int(seed): max(0.0, float(value)) / total
            for seed, value in filtered.items()
        }

    def _weighted_probe_value_mean(
        self,
        probe_values: list[tuple[int, float]],
        probe_weights: dict[int, float],
    ) -> float:
        if not probe_values:
            return 0.0
        weighted_total = 0.0
        weight_total = 0.0
        for seed, value in probe_values:
            weight = float(probe_weights.get(int(seed), 0.0))
            if not math.isfinite(weight) or weight <= 0.0:
                continue
            weighted_total += weight * float(value)
            weight_total += weight
        if weight_total > 0.0:
            return weighted_total / weight_total
        return sum(float(value) for _, value in probe_values) / float(len(probe_values))

    def _clip_projected_grad(self, value: float) -> float:
        clip = float(getattr(self.method_config, "abh_projected_grad_clip", 0.0))
        projected_grad = float(value)
        if clip <= 0.0:
            return projected_grad
        return max(-clip, min(clip, projected_grad))

    def _apply_meazo_scale(self, value: float) -> float:
        beta = float(getattr(self.method_config, "abh_meazo_scale_beta", 0.0))
        value = float(value)
        if beta <= 0.0 or not math.isfinite(value):
            self._abh_meazo_scale_factor = 1.0
            self._abh_meazo_scale_v_hat = float(self._abh_meazo_scale_v)
            return value
        self._abh_meazo_scale_step += 1
        self._abh_meazo_scale_v = (
            beta * float(self._abh_meazo_scale_v) + (1.0 - beta) * value * value
        )
        v_hat = float(self._abh_meazo_scale_v)
        if bool(getattr(self.method_config, "abh_meazo_scale_bias_correction", True)):
            correction = 1.0 - beta ** max(1, int(self._abh_meazo_scale_step))
            if correction > 0.0:
                v_hat = v_hat / correction
        eps = float(getattr(self.method_config, "abh_meazo_scale_eps", 1e-8))
        denominator = math.sqrt(max(v_hat, 0.0)) + eps
        factor = 1.0 / denominator if denominator > 0.0 else 1.0
        self._abh_meazo_scale_v_hat = float(v_hat)
        self._abh_meazo_scale_factor = float(factor)
        return value * factor

    def _curvature_damping_diagnostics(self) -> dict[str, float | bool | None]:
        return {
            "abh_curvature_damping_lambda": float(
                getattr(self.method_config, "abh_curvature_damping_lambda", 0.0)
            ),
            "abh_curvature_damping_active": False,
            "abh_directional_curvature": None,
            "abh_positive_directional_curvature": None,
            "abh_curvature_damping_factor": 1.0,
            "abh_projected_grad_after_curvature_damping": None,
        }

    def _apply_curvature_damping(
        self,
        projected_grad: float,
        *,
        f_plus: float | None,
        f_minus: float | None,
        loss0: float | None,
        eps: float,
    ) -> tuple[float, dict[str, float | bool | None]]:
        diagnostics = self._curvature_damping_diagnostics()
        value = float(projected_grad)
        diagnostics["abh_projected_grad_after_curvature_damping"] = value
        damping_lambda = float(
            getattr(self.method_config, "abh_curvature_damping_lambda", 0.0)
        )
        if damping_lambda <= 0.0 or f_plus is None or f_minus is None or loss0 is None:
            return value, diagnostics

        curvature = (float(f_plus) + float(f_minus) - 2.0 * float(loss0)) / (
            float(eps) ** 2
        )
        positive_curvature = max(0.0, float(curvature))
        factor = 1.0 / (1.0 + damping_lambda * positive_curvature)
        damped = value * factor
        diagnostics.update(
            {
                "abh_curvature_damping_active": True,
                "abh_directional_curvature": float(curvature),
                "abh_positive_directional_curvature": float(positive_curvature),
                "abh_curvature_damping_factor": float(factor),
                "abh_projected_grad_after_curvature_damping": float(damped),
            }
        )
        return float(damped), diagnostics

    def _normalize_method_config(
        self, value: AGZOConfig | AIMZOConfig | dict[str, Any]
    ) -> AGZOConfig:
        if isinstance(value, AGZOConfig):
            return value
        return AGZOConfig(**dict(value))


def _cuda_device_indices(
    parameters: ParameterList,
    bases: dict[str, torch.Tensor],
) -> list[int]:
    devices: set[int] = set()
    for _, param in parameters:
        if param.device.type == "cuda" and param.device.index is not None:
            devices.add(int(param.device.index))
    for basis in bases.values():
        if basis.device.type == "cuda" and basis.device.index is not None:
            devices.add(int(basis.device.index))
    return sorted(devices)


def _dict_mean(values: dict[str, float]) -> float | None:
    if not values:
        return None
    finite = [float(value) for value in values.values() if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _mean_optional(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _dict_min(values: dict[str, float]) -> float | None:
    if not values:
        return None
    finite = [float(value) for value in values.values() if math.isfinite(float(value))]
    if not finite:
        return None
    return float(min(finite))


def _dict_max(values: dict[str, float]) -> float | None:
    if not values:
        return None
    finite = [float(value) for value in values.values() if math.isfinite(float(value))]
    if not finite:
        return None
    return float(max(finite))


def _safe_tensor_norm(tensor: torch.Tensor) -> float | None:
    try:
        value = torch.linalg.vector_norm(tensor.float())
        if not torch.isfinite(value):
            return None
        return float(value.detach().cpu().item())
    except RuntimeError:
        return None


def _module_name_from_parameter_name(name: str) -> str:
    if name == "weight":
        return ""
    suffix = ".weight"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _source_projected_grad(
    f_plus: float,
    f_minus: float,
    eps: float,
    *,
    two_side: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> float:
    denominator = (2 * float(eps)) if two_side else float(eps)
    return float(
        (
            (
                torch.tensor(float(f_plus), dtype=dtype)
                - torch.tensor(float(f_minus), dtype=dtype)
            )
            / denominator
        ).item()
    )


class _ActivationCapture:
    def __init__(
        self,
        *,
        max_tokens: int = 0,
        basis_builder: Callable[[str, torch.Tensor], torch.Tensor | None] | None = None,
    ) -> None:
        self.max_tokens = int(max_tokens)
        self.basis_builder = basis_builder
        self.matched_linear_layers = 0
        self.activations: dict[str, torch.Tensor] = {}
        self.bases: dict[str, torch.Tensor] = {}
        self.hook_calls = 0
        self.hook_time = 0.0

    def hook(self, module_name: str):
        def _hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            _output: Any,
        ) -> None:
            started_at = time.perf_counter()
            self.hook_calls += 1
            try:
                parameter_name = (
                    "weight" if module_name == "" else f"{module_name}.weight"
                )
                if module_name in self.activations or parameter_name in self.bases:
                    return
                tensor = _first_tensor(inputs)
                if tensor is None:
                    return
                if (
                    self.basis_builder is None
                    and self.max_tokens > 0
                    and tensor.ndim >= 2
                    and torch.is_floating_point(tensor)
                ):
                    tokens = tensor.reshape(-1, tensor.shape[-1])
                    if tokens.shape[0] > self.max_tokens:
                        tensor = tokens[: self.max_tokens]
                tensor = tensor.detach()
                if self.basis_builder is not None:
                    basis = self.basis_builder(module_name, tensor)
                    if basis is not None:
                        self.bases[parameter_name] = basis
                    return
                self.activations[module_name] = tensor
            finally:
                self.hook_time += time.perf_counter() - started_at

        return _hook


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _optional_finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _tensor_scalar_summary(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    fingerprint, _ = bounded_tensor_fingerprint(
        name,
        tensor,
        error_prefix="agzo tensor summary",
    )
    summary = dict(fingerprint["summary"])
    base = {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "device_type": str(tensor.device.type),
        "numel": int(tensor.numel()),
        "summary_kind": "bounded_sample",
        "sample_count": summary["sample_count"],
    }
    if int(tensor.numel()) == 0:
        return {
            **base,
            "min": None,
            "max": None,
            "mean": None,
            "l2_norm": None,
            "sample": [],
        }
    return {
        **base,
        "min": summary["min"],
        "max": summary["max"],
        "mean": summary["mean"],
        "l2_norm": summary["l2"],
        "sample": list(fingerprint["sample"]),
    }
