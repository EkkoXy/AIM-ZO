from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from aimzo.protocols import (
    FIXED_ROLLOUT_OBJECTIVES,
    FIXED_ROLLOUT_PROTOCOL,
    FRESH_ROLLOUT_PROTOCOL,
    SUPPORTED_LOGPROB_GRANULARITIES as ALLOWED_LOGPROB_GRANULARITIES,
    SUPPORTED_OBJECTIVES as ALLOWED_OBJECTIVES,
    SUPPORTED_ROLLOUT_PROTOCOLS as ALLOWED_ROLLOUT_PROTOCOLS,
    ZOREGULAR_SUPERVISED_OBJECTIVES,
    build_objective_protocol,
)
from aimzo.tasks.zoregular import (
    ZO_REGULAR_CLASSIFICATION_TASKS,
    ZO_REGULAR_GENERATION_TASKS,
    ZO_REGULAR_TASKS,
    is_zoregular_task,
)


__all__ = [
    "AGZOConfig",
    "ALLOWED_ZO_METHODS",
    "ALLOWED_LOSS_AGGREGATIONS",
    "ALLOWED_LOGPROB_GRANULARITIES",
    "ALLOWED_OBJECTIVES",
    "ALLOWED_PARAMETER_SCOPES",
    "ALLOWED_PROMPT_MODES",
    "ALLOWED_REFERENCE_POLICIES",
    "ALLOWED_ROLLOUT_PROTOCOLS",
    "ALLOWED_TASKS",
    "BackendConfig",
    "CurvZOConfig",
    "DataConfig",
    "FIXED_ROLLOUT_OBJECTIVES",
    "FIXED_ROLLOUT_PROTOCOL",
    "FRESH_ROLLOUT_PROTOCOL",
    "GenerationConfig",
    "HiZOOConfig",
    "HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES",
    "LOZOConfig",
    "LowDimMuonConfig",
    "PGAPConfig",
    "SVD0Config",
    "LoggingConfig",
    "AIMZOConfig",
    "ObjectiveConfig",
    "OjaABQConfig",
    "TrainerConfig",
    "ExperimentConfig",
    "ZOConfig",
    "build_zo_step_seed_sequence",
    "load_config",
    "load_experiment_config",
    "normalize_hizoo_config",
    "validate_config",
    "validate_real_vllm_train_config",
]


ALLOWED_PROMPT_MODES = frozenset({"base", "chat", "chat_user"})
ALLOWED_LOSS_AGGREGATIONS = frozenset(
    {
        "sequence_sum_then_sample_mean",
        "sequence_mean",
        "group_mean",
        "token_mean_then_sample_mean",
    }
)
ALLOWED_REFERENCE_POLICIES = frozenset({"initial_adapter", "center_at_step_start"})
ALLOWED_ZO_SEED_MODES = frozenset({"source_numpy_randint", "aimzo_step"})
ALLOWED_ZOREGULAR_BATCH_SAMPLER_POLICIES = frozenset(
    {
        "torch_random",
        "torch_random_hf_dataloader",
        "hf_trainer_persistent_generator",
        "accelerate_seedable",
    }
)
ALLOWED_SOURCE_CHECKPOINT_POLICIES = frozenset({"default", "final", "source_style"})
REASONING_TASKS = frozenset({"gsm8k", "math", "deepscaler"})
ALLOWED_TASKS = frozenset({*REASONING_TASKS, *ZO_REGULAR_TASKS})
ALLOWED_PARAMETER_SCOPES = frozenset({"lora", "full_parameters"})
ALLOWED_ZO_METHODS = frozenset(
    {
        "mezo",
        "hizoo",
        "agzo",
        "aimzo",
        "zomuon",
        "zomopi",
        "curvzo",
        "lozo",
        "oja_abq",
        "svd0",
        "pgap",
    }
)
SOURCE_COMPAT_OBJECTIVES = frozenset({"source_sst2_candidate_scoring"})
HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES = {
    "constant0": 0.0,
    "constant1e-6": 1e-6,
    "constant1e-8": 1e-8,
    "constant1e-10": 1e-10,
    "constant1e-12": 1e-12,
    "constant1e-2": 1e-2,
    "constant1e-4": 1e-4,
}
AGZO_BASIS_SEED_MODES = frozenset({"perturbation_seed", "ambient"})
_HIZOO_UNSET = object()


@dataclass(frozen=True)
class DataConfig:
    task: str = "gsm8k"
    data_root: str | None = None
    train_file: str | None = None
    eval_file: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    zoregular_train_dev_samples: int | None = None
    seed: int = 42
    zoregular_sampling_policy: str = "split"
    zoregular_batch_sampler_policy: str = "torch_random"


@dataclass(frozen=True)
class BackendConfig:
    name: str = "hf"
    model_name: str = "sshleifer/tiny-gpt2"
    dtype: str = "float32"
    trust_remote_code: bool = False
    lora_rank: int = 2
    lora_alpha: int = 4
    lora_target_modules: tuple[str, ...] = ("c_attn",)
    vllm_worker_mode: str = "fake"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.5
    max_model_len: int | None = None
    worker_extension_cls: str = "aimzo.backends.vllm.worker_extension.WorkerExtension"
    lora_adapter_path: str | None = None
    real_worker_lora_strategy: str = "in_memory"
    candidate_parallelism: int = 2


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    num_samples: int = 4
    prompt_mode: str | None = None


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str = "policy_loss"
    format_reward: float = 0.1
    rollout_protocol: str = "fixed_rollout_per_step"
    loss_aggregation: str = "sequence_sum_then_sample_mean"
    kl_beta: float = 0.0
    reference_policy: str = "initial_adapter"
    grpo_clip_epsilon: float = 0.2
    logprob_granularity: str = "sequence"


@dataclass(frozen=True, init=False)
class HiZOOConfig:
    hessian_smooth: float = 0.0
    hessian_init: float = 1.0
    hessian_smooth_type: str = "constant0"
    hessian_state_dtype: str = "parameter"

    def __init__(
        self,
        hessian_smooth: float | object = _HIZOO_UNSET,
        hessian_init: float = 1.0,
        hessian_smooth_type: str | object = _HIZOO_UNSET,
        hessian_state_dtype: str = "parameter",
    ) -> None:
        smooth_provided = hessian_smooth is not _HIZOO_UNSET
        smooth_type_provided = hessian_smooth_type is not _HIZOO_UNSET
        smooth_type = (
            "constant0" if not smooth_type_provided else str(hessian_smooth_type)
        )
        if smooth_provided:
            smooth = float(hessian_smooth)
        elif smooth_type in HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES:
            smooth = float(HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES[smooth_type])
        else:
            smooth = 0.0

        object.__setattr__(self, "hessian_smooth", smooth)
        object.__setattr__(self, "hessian_init", float(hessian_init))
        object.__setattr__(self, "hessian_smooth_type", smooth_type)
        object.__setattr__(self, "hessian_state_dtype", str(hessian_state_dtype))
        object.__setattr__(self, "_hessian_smooth_explicit", smooth_provided)
        object.__setattr__(self, "_hessian_smooth_type_explicit", smooth_type_provided)


@dataclass(frozen=True)
class AGZOConfig:
    rank: int = 1
    target_module_regex: str = ".*"
    fallback: str = "gaussian"
    max_activation_tokens: int = 2048
    source_scalar_dtype: str = "bfloat16"
    stream_activation_basis: bool = False
    estimator_mode: str = "two_side"
    subspace_backend: str = "power_iteration"
    power_iterations: int = 3
    basis_seed_mode: str = "perturbation_seed"
    perturbation_form: str = "basis"
    abh_normalization: str = "none"
    abh_layer_fro_scale: float = 1.0
    abh_warmup_steps: int = 0
    abh_warmup_dense_noise: bool = False
    abh_momentum_beta: float = 0.0
    abh_a_refresh_interval: int = 1
    abh_right_rank: int = 1
    abh_oja_wide_right_rank: int = 0
    abh_oja_active_top_count: int = 0
    abh_oja_active_tail_count: int = 0
    abh_oja_active_sampling_policy: str = "top_tail"
    abh_oja_active_resample_per_probe: bool = False
    abh_population_summary_last_only: bool = False
    abh_population_deferred_update_trace: bool = False
    abh_population_cache_restore_factors: bool = False
    abh_population_fused_shared_q_update: bool = False
    # Combine the final weighted population update even when each probe uses
    # its own active Oja tail. Probe directions remain independently sampled.
    abh_population_fused_varied_q_update: bool = False
    abh_oja_basis_source: str = "center"
    abh_right_rank_policy: str = "fixed"
    abh_right_rank_ratio: float = 0.0
    abh_right_rank_min: int = 1
    abh_activation_token_policy: str = "fixed"
    abh_activation_token_multiplier: float = 2.0
    abh_oja_eta: float = 1.0
    abh_oja_eta_decay_interval: int = 0
    abh_oja_eta_decay_factor: float = 1.0
    abh_oja_update_interval: int = 1
    abh_oja_late_update_start_step: int = 0
    abh_oja_late_update_interval: int = 1
    abh_oja_lagged_basis: bool = False
    abh_oja_lagged_update_source: str = "random"
    abh_oja_q_update_rule: str = "qr_oja"
    abh_oja_q_ema_beta: float = 0.9
    abh_oja_q_optimizer: str = "sgd"
    abh_oja_q_lr: float = 0.0
    abh_oja_q_orth_lambda: float = 0.0
    abh_q_dropout_keep_fraction: float = 1.0
    abh_noise_cache: str = "step_full"
    abh_num_noise: int = 1
    abh_population_dense_noise_count: int = 0
    abh_left_factor: str = "orthogonal_a"
    abh_a_seed_mode: str = "block"
    abh_mezo_mix_interval: int = 0
    abh_mezo_mix_dense_steps: int = 0
    abh_dense_residual_ratio: float = 0.0
    abh_update_transform: str = "none"
    abh_probe_transform: str = "none"
    abh_adamu_alpha: float = 0.5
    abh_adamu_beta1: float = 0.9
    abh_adamu_beta2: float = 0.01
    abh_adamu_eps: float = 1e-8
    abh_loren_q_damping: float = 0.1
    abh_loren_q_a_init_std: float = 1.0
    abh_loren_q_lr_cov: float = 1e-3
    abh_loren_q_a_eps_power: float = 2.0
    abh_loren_population_divide_by_eps: bool = True
    abh_loren_population_baseline: str = "population_mean"
    abh_loren_population_normalize_std: bool = False
    abh_loren_population_std_eps: float = 1e-8
    abh_loren_population_clip_std: float = 0.0
    abh_loren_population_trim_extremes: int = 0
    # Negative keeps the one-sided population estimator. Values in [0, 1]
    # evaluate all plus candidates before matched minus candidates.
    abh_loren_population_deferred_residual_lambda: float = -1.0
    # Negative keeps standard two-sided ZO. Values in [0, 1] interpolate
    # centered odd fitness with the finite-radius even residual.
    abh_two_side_residual_lambda: float = -1.0
    # Disable only for the exact centering ablation. The default preserves the
    # RLOO/sample-covariance estimator used by deferred two-sided runs.
    abh_two_side_residual_center: bool = True
    abh_adam_beta2: float = 0.999
    abh_adam_eps: float = 1e-8
    abh_projected_grad_clip: float = 0.0
    abh_projected_grad_divide_by_eps: bool = True
    abh_curvature_damping_lambda: float = 0.0
    abh_meazo_scale_beta: float = 0.0
    abh_meazo_scale_eps: float = 1e-8
    abh_meazo_scale_bias_correction: bool = True
    abh_seed_pool_enabled: bool = False
    abh_seed_pool_size: int = 128
    abh_seed_pool_warmup_steps: int = 50
    abh_seed_pool_gamma: float = 0.999
    abh_seed_pool_rho: float = 0.2
    abh_seed_pool_score_mode: str = "abs_s_minus_curvature"
    abh_seed_pool_curvature_lambda: float = 1.0
    abh_seed_pool_abs_s_lambda: float = 1.0
    abh_seed_pool_signed_s_lambda: float = 0.0
    abh_seed_pool_noise_lambda: float = 0.0
    abh_seed_pool_count_lambda: float = 0.0
    abh_seed_pool_ucb_lambda: float = 0.5
    abh_seed_pool_top_k: int = 1
    abh_seed_pool_top_select_count: int = 1
    abh_seed_pool_ucb_select_count: int = 1
    abh_seed_pool_fresh_select_count: int = 1
    abh_seed_pool_top_fraction: float = 0.2
    abh_seed_pool_temperature: float = 1.0
    abh_seed_pool_dense_noise: bool = False
    abh_seed_pool_screen_until_step: int = 0
    abh_seed_pool_update_c_over_eps_threshold: float = 0.0
    abh_seed_pool_update_curvature_signal_ratio_threshold: float = 0.0
    abh_seed_pool_update_min_count: int = 0
    abh_seed_pool_update_filter_roles: str = ""
    abh_seed_pool_min_survival_count: int = 0
    abh_seed_pool_bad_count_threshold: int = 0
    abh_seed_pool_bad_score_quantile: float = 0.0
    abh_seed_pool_fixed_seeds: str = ""
    abh_multi_update_weighting: str = "uniform"
    abh_multi_update_curvature_lambda: float = 1.0
    abh_multi_update_softmax_tau: float = 1.0


@dataclass(frozen=True)
class AIMZOConfig(AGZOConfig):
    """Configuration for AIM-ZO structured population optimization."""


@dataclass(frozen=True)
class OjaABQConfig(AGZOConfig):
    perturbation_form: str = "abh_oja"


@dataclass(frozen=True)
class LOZOConfig:
    rank: int = 2
    step_interval: int = 50
    normalization: str = "source"
    fallback: str = "gaussian"


@dataclass(frozen=True)
class SVD0Config:
    rank: int = 24
    update_interval: int = 1000
    probe_count: int = 1
    fallback: str = "gaussian"
    target_ndim: int = 2


@dataclass(frozen=True)
class PGAPConfig:
    rank: int = 128
    update_interval: int = 100
    probe_count: int = 10
    delta_init: float = 2.0
    delta_final: float = 0.0
    delta_decay_steps: int = 0
    fallback: str = "gaussian"
    target_ndim: int = 2


@dataclass(frozen=True)
class LowDimMuonConfig:
    rank: int = 64
    step_interval: int = 500
    k_start: int = 32
    reset_v_on_p_refresh: bool = True
    phase2_steps: int = 20000
    beta: float = 0.0
    num_samples: int = 8
    multiple_sample: bool = True
    perturbation_mode: str = "two_side"
    phase2_optimizer: str = "muon"
    pion_steps: int = 5
    pion_promotion_steps: int = 2
    one_d_lr: float = 1e-7
    refresh_indexing: str = "aimzo"
    source_debug_trace: bool = False


@dataclass(frozen=True)
class CurvZOConfig:
    sample_ratio: float = 0.4
    sgs_alpha: float = 0.1
    sensitivity_init: float = 0.01
    sensitivity_beta: float = 0.1
    sensitivity_global_beta: float = 0.1
    adaptive_every: int = 20
    sampling_mode: str = "poisson_pps"
    normalize_energy_by_numel: bool = False


@dataclass(frozen=True)
class ZOConfig:
    method: str = "mezo"
    estimator: str = "mezo_two_point"
    eps: float = 1e-3
    eps_schedule: str = "constant"
    eps_schedule_min_ratio: float = 0.0
    eps_schedule_total_steps: int | None = None
    eps_schedule_step_interval: int = 1
    eps_schedule_step_factor: float = 1.0
    learning_rate: float = 1e-3
    lr_schedule: str = "constant"
    lr_warmup_steps: int = 0
    lr_schedule_min_ratio: float = 0.0
    lr_schedule_step_interval: int = 1
    lr_schedule_step_factor: float = 1.0
    lr_schedule_milestones: str = ""
    parameter_scope: str = "full_parameters"
    seed: int = 42
    seed_mode: str = "aimzo_step"
    fixed_seed_pool: str = ""
    source_compatibility_profile: str | None = None
    source_update_order: bool = False
    weight_decay: float = 0.0
    noise_backend: str = "gpu_seeded"
    restore_strategy: str = "seed_replay"
    integrity_every: int = 0
    num_noise: int = 1
    mezo_two_point_divide_by_eps: bool = True
    mezo_population_divide_by_eps: bool = True
    mezo_loren_enabled: bool = False
    mezo_loren_damping: float = 0.1
    mezo_loren_lr_cov: float = 1e-3
    mezo_loren_beta1: float = 0.9
    mezo_loren_a_init_std: float = 1.0
    mezo_loren_a_eps_power: float = 2.0
    hizoo: HiZOOConfig | dict[str, Any] = field(default_factory=HiZOOConfig)
    agzo: AGZOConfig | dict[str, Any] = field(default_factory=AGZOConfig)
    aimzo: AIMZOConfig | dict[str, Any] = field(default_factory=AIMZOConfig)
    oja_abq: OjaABQConfig | dict[str, Any] = field(default_factory=OjaABQConfig)
    lozo: LOZOConfig | dict[str, Any] = field(default_factory=LOZOConfig)
    svd0: SVD0Config | dict[str, Any] = field(default_factory=SVD0Config)
    pgap: PGAPConfig | dict[str, Any] = field(default_factory=PGAPConfig)
    lowdim_muon: LowDimMuonConfig | dict[str, Any] = field(
        default_factory=LowDimMuonConfig
    )
    curvzo: CurvZOConfig | dict[str, Any] = field(default_factory=CurvZOConfig)


@dataclass(frozen=True)
class TrainerConfig:
    output_dir: str = "outputs/hf_gsm8k_smoke"
    train_batch_size: int = 1
    eval_batch_size: int = 4
    max_steps: int = 1
    eval_every: int = 1
    checkpoint_milestones: tuple[int, ...] = ()
    save_every: int = 1
    max_periodic_checkpoints: int = 5
    save_checkpoints: bool = True
    resume_from_checkpoint: str | None = None
    source_checkpoint_policy: str = "default"
    best_checkpoint_metric: str = "loss"
    best_checkpoint_mode: str = "min"
    trace_required: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    tensorboard: bool = False
    tensorboard_dir: str | None = None
    tensorboard_samples: int = 8
    tensorboard_log_text: bool = True
    history_snapshot_interval: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    zo: ZOConfig = field(default_factory=ZOConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.trainer.output_dir).expanduser()


def validate_config(config: ExperimentConfig) -> None:
    _validate_string("data.task", config.data.task)
    if config.data.task not in ALLOWED_TASKS:
        allowed = ", ".join(sorted(ALLOWED_TASKS))
        raise ValueError(f"data.task must be one of: {allowed}")
    _validate_string(
        "data.zoregular_sampling_policy", config.data.zoregular_sampling_policy
    )
    if config.data.zoregular_sampling_policy not in {
        "split",
        "mezo_train_dev",
        "multirc_grouped_shuffled",
    }:
        raise ValueError(
            "data.zoregular_sampling_policy must be one of: "
            "split, mezo_train_dev, multirc_grouped_shuffled"
        )
    _validate_string(
        "data.zoregular_batch_sampler_policy",
        config.data.zoregular_batch_sampler_policy,
    )
    if (
        config.data.zoregular_batch_sampler_policy
        not in ALLOWED_ZOREGULAR_BATCH_SAMPLER_POLICIES
    ):
        raise ValueError(
            "data.zoregular_batch_sampler_policy must be one of: "
            "torch_random, torch_random_hf_dataloader, "
            "hf_trainer_persistent_generator, accelerate_seedable"
        )
    if config.data.zoregular_sampling_policy == "mezo_train_dev":
        if config.data.max_train_samples is None:
            raise ValueError(
                "data.zoregular_sampling_policy='mezo_train_dev' requires "
                "data.max_train_samples"
            )
    if (
        config.data.zoregular_sampling_policy == "multirc_grouped_shuffled"
        and config.data.task != "multirc"
    ):
        raise ValueError(
            "data.zoregular_sampling_policy='multirc_grouped_shuffled' is only "
            "valid for data.task='multirc'"
        )
    _validate_optional_string("data.data_root", config.data.data_root)
    _validate_optional_string("data.train_file", config.data.train_file)
    _validate_optional_string("data.eval_file", config.data.eval_file)
    _validate_string("backend.name", config.backend.name)
    _validate_string("backend.model_name", config.backend.model_name)
    _validate_string("backend.dtype", config.backend.dtype)
    _validate_bool("backend.trust_remote_code", config.backend.trust_remote_code)
    _validate_vllm_worker_mode(config.backend.vllm_worker_mode)
    _validate_string(
        "backend.worker_extension_cls", config.backend.worker_extension_cls
    )
    _validate_optional_string(
        "backend.lora_adapter_path", config.backend.lora_adapter_path
    )
    _validate_string(
        "backend.real_worker_lora_strategy", config.backend.real_worker_lora_strategy
    )
    _validate_optional_string(
        "trainer.resume_from_checkpoint", config.trainer.resume_from_checkpoint
    )
    _validate_bool("trainer.save_checkpoints", config.trainer.save_checkpoints)
    _validate_string(
        "trainer.source_checkpoint_policy", config.trainer.source_checkpoint_policy
    )
    if (
        config.trainer.source_checkpoint_policy
        not in ALLOWED_SOURCE_CHECKPOINT_POLICIES
    ):
        allowed = ", ".join(sorted(ALLOWED_SOURCE_CHECKPOINT_POLICIES))
        raise ValueError(f"trainer.source_checkpoint_policy must be one of: {allowed}")
    _validate_bool("trainer.trace_required", config.trainer.trace_required)
    _validate_string("objective.name", config.objective.name)
    _validate_zoregular_objective_task_pair(config)
    _validate_string("objective.rollout_protocol", config.objective.rollout_protocol)
    _validate_string("objective.loss_aggregation", config.objective.loss_aggregation)
    _validate_loss_aggregation(config.objective.loss_aggregation)
    _validate_string("objective.reference_policy", config.objective.reference_policy)
    _validate_reference_policy(config.objective.reference_policy)
    _validate_string(
        "objective.logprob_granularity", config.objective.logprob_granularity
    )
    _validate_logprob_granularity(config.objective.logprob_granularity)
    _validate_optional_string("generation.prompt_mode", config.generation.prompt_mode)
    _validate_string("zo.estimator", config.zo.estimator)
    _validate_string("zo.method", config.zo.method)
    if config.zo.method not in ALLOWED_ZO_METHODS:
        allowed = ", ".join(sorted(ALLOWED_ZO_METHODS))
        raise ValueError(f"zo.method must be one of: {allowed}")
    _validate_zo_method_configs(config.zo)
    _validate_string("zo.seed_mode", config.zo.seed_mode)
    if config.zo.seed_mode not in ALLOWED_ZO_SEED_MODES:
        allowed = ", ".join(sorted(ALLOWED_ZO_SEED_MODES))
        raise ValueError(f"zo.seed_mode must be one of: {allowed}")
    _validate_string("zo.fixed_seed_pool", config.zo.fixed_seed_pool)
    _validate_optional_string(
        "zo.source_compatibility_profile", config.zo.source_compatibility_profile
    )
    _validate_bool("zo.source_update_order", config.zo.source_update_order)
    _validate_string("zo.parameter_scope", config.zo.parameter_scope)
    if config.zo.parameter_scope not in ALLOWED_PARAMETER_SCOPES:
        allowed = ", ".join(sorted(ALLOWED_PARAMETER_SCOPES))
        raise ValueError(f"zo.parameter_scope must be one of: {allowed}")
    _validate_string("zo.noise_backend", config.zo.noise_backend)
    if config.zo.noise_backend != "gpu_seeded":
        raise ValueError("zo.noise_backend must be exactly 'gpu_seeded'")
    _validate_string("zo.restore_strategy", config.zo.restore_strategy)
    if config.zo.restore_strategy != "seed_replay":
        raise ValueError("zo.restore_strategy must be exactly 'seed_replay'")
    _validate_string("trainer.output_dir", config.trainer.output_dir)
    _validate_bool("logging.tensorboard", config.logging.tensorboard)
    _validate_optional_string("logging.tensorboard_dir", config.logging.tensorboard_dir)
    _validate_at_least(
        "logging.tensorboard_samples", config.logging.tensorboard_samples, 0
    )
    _validate_bool("logging.tensorboard_log_text", config.logging.tensorboard_log_text)
    _validate_optional_at_least(
        "logging.history_snapshot_interval",
        config.logging.history_snapshot_interval,
        1,
    )
    _validate_objective_rollout_matrix(
        objective_name=config.objective.name,
        rollout_protocol=config.objective.rollout_protocol,
        loss_aggregation=config.objective.loss_aggregation,
        reference_policy=config.objective.reference_policy,
        logprob_granularity=config.objective.logprob_granularity,
    )
    _validate_optional_at_least(
        "data.max_train_samples", config.data.max_train_samples, 1
    )
    _validate_optional_at_least(
        "data.max_eval_samples", config.data.max_eval_samples, 1
    )
    _validate_optional_at_least(
        "data.zoregular_train_dev_samples",
        config.data.zoregular_train_dev_samples,
        0,
    )
    _validate_integer("data.seed", config.data.seed)
    _validate_at_least("backend.lora_rank", config.backend.lora_rank, 1)
    _validate_at_least("backend.lora_alpha", config.backend.lora_alpha, 1)
    _validate_at_least(
        "backend.tensor_parallel_size", config.backend.tensor_parallel_size, 1
    )
    _validate_at_least(
        "backend.candidate_parallelism", config.backend.candidate_parallelism, 1
    )
    _validate_at_least("trainer.max_steps", config.trainer.max_steps, 1)
    milestones = tuple(config.trainer.checkpoint_milestones)
    if any(type(step) is not int or not 1 <= step <= config.trainer.max_steps for step in milestones):
        raise ValueError("trainer.checkpoint_milestones must be integers within max_steps")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("trainer.checkpoint_milestones must be sorted and unique")
    _validate_at_least("trainer.train_batch_size", config.trainer.train_batch_size, 1)
    _validate_at_least("trainer.eval_batch_size", config.trainer.eval_batch_size, 1)
    _validate_at_least("trainer.eval_every", config.trainer.eval_every, 0)
    _validate_at_least("trainer.save_every", config.trainer.save_every, 0)
    _validate_at_least(
        "trainer.max_periodic_checkpoints",
        config.trainer.max_periodic_checkpoints,
        0,
    )
    _validate_string(
        "trainer.best_checkpoint_metric",
        config.trainer.best_checkpoint_metric,
    )
    _validate_string(
        "trainer.best_checkpoint_mode",
        config.trainer.best_checkpoint_mode,
    )
    if config.trainer.best_checkpoint_mode not in {"min", "max"}:
        raise ValueError("trainer.best_checkpoint_mode must be one of: min, max")
    _validate_at_least("generation.num_samples", config.generation.num_samples, 1)
    _validate_at_least("generation.max_new_tokens", config.generation.max_new_tokens, 1)
    _validate_finite("generation.temperature", config.generation.temperature)
    if float(config.generation.temperature) < 0.0:
        raise ValueError("generation.temperature must be >= 0")
    _validate_finite("generation.top_p", config.generation.top_p)
    top_p = float(config.generation.top_p)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("generation.top_p must be in (0, 1]")
    _validate_finite(
        "backend.gpu_memory_utilization", config.backend.gpu_memory_utilization
    )
    gpu_memory_utilization = float(config.backend.gpu_memory_utilization)
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("backend.gpu_memory_utilization must be in (0, 1]")
    _validate_optional_at_least(
        "backend.max_model_len", config.backend.max_model_len, 1
    )
    _validate_prompt_mode(config.generation.prompt_mode)
    _validate_finite("zo.eps", config.zo.eps)
    if float(config.zo.eps) <= 0.0:
        raise ValueError("zo.eps must be finite and > 0")
    _validate_string("zo.eps_schedule", config.zo.eps_schedule)
    if str(config.zo.eps_schedule) not in {
        "constant",
        "cosine_decay",
        "linear_decay",
        "step_decay",
    }:
        raise ValueError(
            "zo.eps_schedule must be one of: constant, cosine_decay, linear_decay, step_decay"
        )
    _validate_finite("zo.eps_schedule_min_ratio", config.zo.eps_schedule_min_ratio)
    if not 0.0 <= float(config.zo.eps_schedule_min_ratio) <= 1.0:
        raise ValueError("zo.eps_schedule_min_ratio must be finite and in [0, 1]")
    _validate_optional_at_least(
        "zo.eps_schedule_total_steps",
        config.zo.eps_schedule_total_steps,
        1,
    )
    _validate_at_least(
        "zo.eps_schedule_step_interval", config.zo.eps_schedule_step_interval, 1
    )
    _validate_finite("zo.eps_schedule_step_factor", config.zo.eps_schedule_step_factor)
    if not 0.0 < float(config.zo.eps_schedule_step_factor) <= 1.0:
        raise ValueError("zo.eps_schedule_step_factor must be finite and in (0, 1]")
    _validate_finite("zo.learning_rate", config.zo.learning_rate)
    _validate_string("zo.lr_schedule", config.zo.lr_schedule)
    if str(config.zo.lr_schedule) not in {
        "constant",
        "cosine_decay",
        "linear_decay",
        "step_decay",
        "milestone_step_decay",
    }:
        raise ValueError(
            "zo.lr_schedule must be one of: constant, cosine_decay, linear_decay, "
            "milestone_step_decay, step_decay"
        )
    _validate_at_least("zo.lr_warmup_steps", config.zo.lr_warmup_steps, 0)
    _validate_finite("zo.lr_schedule_min_ratio", config.zo.lr_schedule_min_ratio)
    if not 0.0 <= float(config.zo.lr_schedule_min_ratio) <= 1.0:
        raise ValueError("zo.lr_schedule_min_ratio must be finite and in [0, 1]")
    _validate_at_least(
        "zo.lr_schedule_step_interval", config.zo.lr_schedule_step_interval, 1
    )
    _validate_finite("zo.lr_schedule_step_factor", config.zo.lr_schedule_step_factor)
    if not 0.0 < float(config.zo.lr_schedule_step_factor) <= 1.0:
        raise ValueError("zo.lr_schedule_step_factor must be finite and in (0, 1]")
    _validate_string("zo.lr_schedule_milestones", config.zo.lr_schedule_milestones)
    _validate_finite("zo.weight_decay", config.zo.weight_decay)
    _validate_at_least("zo.integrity_every", config.zo.integrity_every, 0)
    _validate_at_least("zo.num_noise", config.zo.num_noise, 1)
    if config.zo.num_noise != 1 and str(config.zo.estimator) != "mezo_population":
        raise ValueError(
            "zo.num_noise must be exactly 1 unless zo.estimator='mezo_population'"
        )
    _validate_finite("zo.mezo_loren_damping", config.zo.mezo_loren_damping)
    if float(config.zo.mezo_loren_damping) <= 0.0:
        raise ValueError("zo.mezo_loren_damping must be finite and > 0")
    _validate_finite("zo.mezo_loren_lr_cov", config.zo.mezo_loren_lr_cov)
    if float(config.zo.mezo_loren_lr_cov) < 0.0:
        raise ValueError("zo.mezo_loren_lr_cov must be finite and >= 0")
    _validate_finite("zo.mezo_loren_beta1", config.zo.mezo_loren_beta1)
    if not 0.0 <= float(config.zo.mezo_loren_beta1) <= 1.0:
        raise ValueError("zo.mezo_loren_beta1 must be finite and in [0, 1]")
    _validate_finite("zo.mezo_loren_a_init_std", config.zo.mezo_loren_a_init_std)
    if float(config.zo.mezo_loren_a_init_std) < 0.0:
        raise ValueError("zo.mezo_loren_a_init_std must be finite and >= 0")
    _validate_finite("zo.mezo_loren_a_eps_power", config.zo.mezo_loren_a_eps_power)
    if float(config.zo.mezo_loren_a_eps_power) < 0.0:
        raise ValueError("zo.mezo_loren_a_eps_power must be finite and >= 0")
    _validate_finite("objective.format_reward", config.objective.format_reward)
    _validate_finite("objective.kl_beta", config.objective.kl_beta)
    if float(config.objective.kl_beta) < 0.0:
        raise ValueError("objective.kl_beta must be finite and >= 0")
    _validate_finite("objective.grpo_clip_epsilon", config.objective.grpo_clip_epsilon)
    grpo_clip_epsilon = float(config.objective.grpo_clip_epsilon)
    if not 0.0 <= grpo_clip_epsilon <= 1.0:
        raise ValueError("objective.grpo_clip_epsilon must be in [0, 1]")
    _validate_surrogate_objective_config(config.objective)
    _validate_lora_target_modules(config.backend.lora_target_modules)


def build_zo_step_seed_sequence(
    seed: int,
    seed_mode: str,
    steps: int,
    max_randint: int = 1_000_000_000,
) -> list[int]:
    if int(steps) < 0:
        raise ValueError("steps must be >= 0")
    if seed_mode == "aimzo_step":
        return [int(seed) + step for step in range(int(steps))]
    if seed_mode == "source_numpy_randint":
        import numpy as np

        rng = np.random.RandomState(int(seed))
        return [int(value) for value in rng.randint(int(max_randint), size=int(steps))]
    allowed = ", ".join(sorted(ALLOWED_ZO_SEED_MODES))
    raise ValueError(f"zo.seed_mode must be one of: {allowed}")


def validate_real_vllm_train_config(
    config: ExperimentConfig,
    *,
    logprob_backend: Any | None = None,
) -> None:
    validate_config(config)
    if config.backend.name != "vllm":
        raise ValueError("real vLLM training requires backend.name='vllm'")
    if config.backend.vllm_worker_mode != "real":
        raise ValueError("real vLLM training requires backend.vllm_worker_mode='real'")
    _validate_vllm_zo_method(config)
    if is_zoregular_task(config.data.task):
        _validate_real_vllm_zoregular_config(config)
        return
    if config.data.task not in REASONING_TASKS:
        allowed = ", ".join(sorted(REASONING_TASKS))
        raise ValueError(
            "real vLLM training supports reasoning or ZORegular tasks; "
            f"reasoning data.task must be one of: {allowed}"
        )
    if config.zo.estimator != "mezo_two_point":
        raise ValueError(
            "real vLLM training supports only zo.estimator='mezo_two_point'"
        )
    if config.zo.parameter_scope not in {"lora", "full_parameters"}:
        raise ValueError(
            "real vLLM training supports zo.parameter_scope='lora' or 'full_parameters'"
        )
    if config.backend.candidate_parallelism not in {1, 2}:
        raise ValueError(
            "backend.candidate_parallelism must be 1 or 2 for real vLLM training"
        )
    if config.objective.name not in {
        "reward",
        "policy_loss",
        "grpo_like_without_kl",
        "grpo_with_kl",
        "grpo_surrogate_without_kl",
        "grpo_surrogate_with_kl",
    }:
        raise ValueError(
            "real vLLM training objective must be one of: "
            "grpo_like_without_kl, grpo_surrogate_with_kl, "
            "grpo_surrogate_without_kl, grpo_with_kl, policy_loss, reward"
        )
    if config.zo.parameter_scope == "full_parameters" and config.objective.name in {
        "grpo_with_kl",
        "grpo_surrogate_with_kl",
    }:
        raise ValueError(
            "full_parameters real vLLM training does not support "
            f"{config.objective.name} because reference_policy requires "
            "a stable reference state/logprob path"
        )
    _validate_objective_rollout_matrix(
        objective_name=config.objective.name,
        rollout_protocol=config.objective.rollout_protocol,
        loss_aggregation=config.objective.loss_aggregation,
        reference_policy=config.objective.reference_policy,
        logprob_granularity=config.objective.logprob_granularity,
    )
    if (
        config.objective.name in FIXED_ROLLOUT_OBJECTIVES
        and logprob_backend is not None
        and not _has_checkable_logprob_support(logprob_backend)
    ):
        raise ValueError(f"real {config.objective.name} requires vLLM logprob support")


def _validate_vllm_zo_method(config: ExperimentConfig) -> None:
    if config.zo.method != "mezo":
        raise ValueError(
            "real vLLM training supports only zo.method='mezo' in this version"
        )


def _validate_real_vllm_zoregular_config(config: ExperimentConfig) -> None:
    if config.data.data_root is None:
        raise ValueError("real vLLM ZORegular training requires data.data_root")
    if config.zo.estimator != "mezo_two_point":
        raise ValueError(
            "real vLLM training supports only zo.estimator='mezo_two_point'"
        )
    if config.zo.parameter_scope not in {"lora", "full_parameters"}:
        raise ValueError(
            "real vLLM training supports zo.parameter_scope='lora' or 'full_parameters'"
        )
    expected_candidate_parallelism = (
        1 if config.zo.parameter_scope == "full_parameters" else 2
    )
    if config.backend.candidate_parallelism != expected_candidate_parallelism:
        raise ValueError(
            "backend.candidate_parallelism must be exactly "
            f"{expected_candidate_parallelism} for "
            f"{config.zo.parameter_scope} real vLLM ZORegular training"
        )
    if config.objective.name not in {"policy_loss", *ZOREGULAR_SUPERVISED_OBJECTIVES}:
        supported = ", ".join(sorted({"policy_loss", *ZOREGULAR_SUPERVISED_OBJECTIVES}))
        raise ValueError(f"real vLLM ZORegular objective must be one of: {supported}")


def _validate_zoregular_objective_task_pair(config: ExperimentConfig) -> None:
    objective_name = config.objective.name
    source_objective = objective_name in SOURCE_COMPAT_OBJECTIVES
    if objective_name not in ZOREGULAR_SUPERVISED_OBJECTIVES and not source_objective:
        return
    task_name = config.data.task
    if objective_name == "source_sst2_candidate_scoring" and task_name != "sst2":
        raise ValueError(
            "objective.name='source_sst2_candidate_scoring' is supported only "
            "for SST-2 data.task='sst2'"
        )
    if not is_zoregular_task(task_name):
        raise ValueError(
            f"ZORegular CE objective {objective_name!r} requires a ZORegular task; "
            f"got data.task={task_name!r}"
        )
    if (
        objective_name
        in {"zoregular_classification_ce", "source_sst2_candidate_scoring"}
        and task_name not in ZO_REGULAR_CLASSIFICATION_TASKS
    ):
        raise ValueError(
            f"data.task={task_name!r} does not support "
            f"objective.name={objective_name!r}"
        )
    if (
        objective_name == "zoregular_generation_ce"
        and task_name not in ZO_REGULAR_GENERATION_TASKS
        and task_name not in {"copa", "record"}
    ):
        raise ValueError(
            f"data.task={task_name!r} does not support "
            "objective.name='zoregular_generation_ce'"
        )


def _validate_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _validate_string(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _validate_optional_string(name: str, value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")


def _validate_at_least(name: str, value: Any, minimum: int) -> None:
    numeric = _validate_integer(name, value)
    if numeric < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _validate_optional_at_least(name: str, value: Any, minimum: int) -> None:
    if value is None:
        return
    _validate_at_least(name, value, minimum)


def _validate_finite(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")


def _validate_lora_target_modules(modules: Any) -> None:
    if not isinstance(modules, list | tuple):
        raise ValueError("backend.lora_target_modules must be a list of strings")
    if not all(isinstance(module, str) for module in modules):
        raise ValueError("backend.lora_target_modules must contain strings")


def _validate_prompt_mode(prompt_mode: str | None) -> None:
    if prompt_mode is None:
        return
    if str(prompt_mode).strip().lower() not in ALLOWED_PROMPT_MODES:
        allowed = ", ".join(sorted(ALLOWED_PROMPT_MODES))
        raise ValueError(f"generation.prompt_mode must be one of: {allowed}")


def _validate_loss_aggregation(value: str) -> None:
    if value not in ALLOWED_LOSS_AGGREGATIONS:
        allowed = ", ".join(sorted(ALLOWED_LOSS_AGGREGATIONS))
        raise ValueError(f"objective.loss_aggregation must be one of: {allowed}")


def _validate_reference_policy(value: str) -> None:
    if value not in ALLOWED_REFERENCE_POLICIES:
        allowed = ", ".join(sorted(ALLOWED_REFERENCE_POLICIES))
        raise ValueError(f"objective.reference_policy must be one of: {allowed}")


def _validate_logprob_granularity(value: str) -> None:
    if value not in ALLOWED_LOGPROB_GRANULARITIES:
        allowed = ", ".join(sorted(ALLOWED_LOGPROB_GRANULARITIES))
        raise ValueError(f"objective.logprob_granularity must be one of: {allowed}")


def _validate_surrogate_objective_config(objective: ObjectiveConfig) -> None:
    if not objective.name.startswith("grpo_surrogate_"):
        return
    if objective.logprob_granularity != "token":
        raise ValueError(
            f"{objective.name} requires objective.logprob_granularity='token'"
        )
    if objective.loss_aggregation != "token_mean_then_sample_mean":
        raise ValueError(
            f"{objective.name} requires "
            "objective.loss_aggregation='token_mean_then_sample_mean'"
        )


def _validate_zo_method_configs(zo: ZOConfig) -> None:
    hizoo = normalize_hizoo_config(zo.hizoo)
    agzo = _normalize_nested_config("agzo", AGZOConfig, zo.agzo)
    aimzo = _normalize_nested_config("aimzo", AIMZOConfig, zo.aimzo)
    oja_abq = _normalize_nested_config("oja_abq", OjaABQConfig, zo.oja_abq)
    lozo = _normalize_nested_config("lozo", LOZOConfig, zo.lozo)
    svd0 = _normalize_nested_config("svd0", SVD0Config, zo.svd0)
    pgap = _normalize_nested_config("pgap", PGAPConfig, zo.pgap)
    lowdim_muon = _normalize_nested_config(
        "lowdim_muon", LowDimMuonConfig, zo.lowdim_muon
    )
    curvzo = _normalize_nested_config("curvzo", CurvZOConfig, zo.curvzo)

    _validate_string("zo.hizoo.hessian_smooth_type", hizoo.hessian_smooth_type)
    expected_hessian_smooth = HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES.get(
        hizoo.hessian_smooth_type
    )
    if expected_hessian_smooth is None and hizoo.hessian_smooth_type != "custom":
        allowed = ", ".join(sorted(HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES))
        raise ValueError(
            "zo.hizoo.hessian_smooth_type must be one of source constant "
            f"scheduler keys or 'custom': {allowed}"
        )
    _validate_finite("zo.hizoo.hessian_smooth", hizoo.hessian_smooth)
    if float(hizoo.hessian_smooth) < 0.0:
        raise ValueError("zo.hizoo.hessian_smooth must be finite and >= 0")
    if expected_hessian_smooth is not None and not math.isclose(
        float(hizoo.hessian_smooth),
        expected_hessian_smooth,
        rel_tol=0.0,
        abs_tol=max(1e-15, abs(expected_hessian_smooth) * 1e-12),
    ):
        raise ValueError(
            "zo.hizoo.hessian_smooth must match "
            f"zo.hizoo.hessian_smooth_type={hizoo.hessian_smooth_type!r} "
            f"({expected_hessian_smooth:g})"
        )
    _validate_finite("zo.hizoo.hessian_init", hizoo.hessian_init)
    if float(hizoo.hessian_init) <= 0.0:
        raise ValueError("zo.hizoo.hessian_init must be finite and > 0")
    if hizoo.hessian_state_dtype not in {"parameter", "float32"}:
        raise ValueError(
            "zo.hizoo.hessian_state_dtype must be 'parameter' or 'float32'"
        )

    _validate_agzo_like_config("zo.agzo", agzo)
    _validate_agzo_like_config("zo.aimzo", aimzo)
    _validate_agzo_like_config("zo.oja_abq", oja_abq, allowed_forms={"abh_oja"})

    _validate_at_least("zo.lozo.rank", lozo.rank, 1)
    _validate_at_least("zo.lozo.step_interval", lozo.step_interval, 1)
    _validate_string("zo.lozo.normalization", lozo.normalization)
    if str(lozo.normalization) not in {"source", "rank"}:
        raise ValueError("zo.lozo.normalization must be one of: rank, source")
    _validate_string("zo.lozo.fallback", lozo.fallback)
    if str(lozo.fallback) != "gaussian":
        raise ValueError("zo.lozo.fallback must be exactly 'gaussian'")

    _validate_at_least("zo.svd0.rank", svd0.rank, 1)
    _validate_at_least("zo.svd0.update_interval", svd0.update_interval, 1)
    _validate_at_least("zo.svd0.probe_count", svd0.probe_count, 1)
    _validate_at_least("zo.svd0.target_ndim", svd0.target_ndim, 1)
    _validate_string("zo.svd0.fallback", svd0.fallback)
    if str(svd0.fallback) != "gaussian":
        raise ValueError("zo.svd0.fallback must be exactly 'gaussian'")

    _validate_at_least("zo.pgap.rank", pgap.rank, 1)
    _validate_at_least("zo.pgap.update_interval", pgap.update_interval, 1)
    _validate_at_least("zo.pgap.probe_count", pgap.probe_count, 1)
    _validate_at_least("zo.pgap.target_ndim", pgap.target_ndim, 1)
    _validate_finite("zo.pgap.delta_init", pgap.delta_init)
    _validate_finite("zo.pgap.delta_final", pgap.delta_final)
    if float(pgap.delta_init) < 0.0 or float(pgap.delta_final) < 0.0:
        raise ValueError("zo.pgap delta values must be finite and >= 0")
    _validate_at_least("zo.pgap.delta_decay_steps", pgap.delta_decay_steps, 0)
    _validate_string("zo.pgap.fallback", pgap.fallback)
    if str(pgap.fallback) != "gaussian":
        raise ValueError("zo.pgap.fallback must be exactly 'gaussian'")

    _validate_at_least("zo.lowdim_muon.rank", lowdim_muon.rank, 1)
    _validate_at_least("zo.lowdim_muon.step_interval", lowdim_muon.step_interval, 1)
    _validate_at_least("zo.lowdim_muon.k_start", lowdim_muon.k_start, 1)
    _validate_at_least("zo.lowdim_muon.phase2_steps", lowdim_muon.phase2_steps, 0)
    _validate_at_least("zo.lowdim_muon.pion_steps", lowdim_muon.pion_steps, 1)
    _validate_at_least(
        "zo.lowdim_muon.pion_promotion_steps",
        lowdim_muon.pion_promotion_steps,
        0,
    )
    _validate_at_least("zo.lowdim_muon.num_samples", lowdim_muon.num_samples, 1)
    _validate_finite("zo.lowdim_muon.beta", lowdim_muon.beta)
    _validate_finite("zo.lowdim_muon.one_d_lr", lowdim_muon.one_d_lr)
    _validate_bool("zo.lowdim_muon.multiple_sample", lowdim_muon.multiple_sample)
    _validate_bool(
        "zo.lowdim_muon.reset_v_on_p_refresh", lowdim_muon.reset_v_on_p_refresh
    )
    _validate_bool("zo.lowdim_muon.source_debug_trace", lowdim_muon.source_debug_trace)
    if str(lowdim_muon.phase2_optimizer) not in {
        "sgd",
        "muon",
        "muon_svd",
        "muon_ns",
        "ns",
        "pion",
        "pion_ns",
    }:
        raise ValueError(
            "zo.lowdim_muon.phase2_optimizer must be one of: "
            "sgd, muon, muon_svd, muon_ns, ns, pion, pion_ns"
        )
    if int(lowdim_muon.pion_promotion_steps) > int(lowdim_muon.pion_steps):
        raise ValueError(
            "zo.lowdim_muon.pion_promotion_steps must be less than or equal to "
            "zo.lowdim_muon.pion_steps"
        )
    if str(lowdim_muon.perturbation_mode) not in {"one_side", "two_side"}:
        raise ValueError(
            "zo.lowdim_muon.perturbation_mode must be one of: one_side, two_side"
        )
    if str(lowdim_muon.refresh_indexing) not in {"aimzo", "source"}:
        raise ValueError(
            "zo.lowdim_muon.refresh_indexing must be one of: source, aimzo"
        )
    if (
        not lowdim_muon.multiple_sample
        and str(lowdim_muon.perturbation_mode) != "two_side"
    ):
        raise ValueError(
            "zo.lowdim_muon multiple_sample=False requires perturbation_mode='two_side'"
        )

    _validate_finite("zo.curvzo.sample_ratio", curvzo.sample_ratio)
    if not (0.0 < float(curvzo.sample_ratio) <= 1.0):
        raise ValueError("zo.curvzo.sample_ratio must be > 0 and <= 1")
    _validate_finite("zo.curvzo.sgs_alpha", curvzo.sgs_alpha)
    if not (0.0 <= float(curvzo.sgs_alpha) <= 1.0):
        raise ValueError("zo.curvzo.sgs_alpha must be >= 0 and <= 1")
    _validate_finite("zo.curvzo.sensitivity_init", curvzo.sensitivity_init)
    if float(curvzo.sensitivity_init) <= 0.0:
        raise ValueError("zo.curvzo.sensitivity_init must be > 0")
    _validate_finite("zo.curvzo.sensitivity_beta", curvzo.sensitivity_beta)
    if not (0.0 < float(curvzo.sensitivity_beta) <= 1.0):
        raise ValueError("zo.curvzo.sensitivity_beta must be > 0 and <= 1")
    _validate_finite(
        "zo.curvzo.sensitivity_global_beta", curvzo.sensitivity_global_beta
    )
    if not (0.0 < float(curvzo.sensitivity_global_beta) <= 1.0):
        raise ValueError("zo.curvzo.sensitivity_global_beta must be > 0 and <= 1")
    _validate_at_least("zo.curvzo.adaptive_every", curvzo.adaptive_every, 0)
    _validate_string("zo.curvzo.sampling_mode", curvzo.sampling_mode)
    if str(curvzo.sampling_mode) not in {"poisson_pps", "multinomial", "uniform"}:
        raise ValueError(
            "zo.curvzo.sampling_mode must be one of: multinomial, poisson_pps, uniform"
        )
    _validate_bool(
        "zo.curvzo.normalize_energy_by_numel",
        curvzo.normalize_energy_by_numel,
    )


def _validate_agzo_like_config(
    prefix: str,
    config: AGZOConfig,
    *,
    allowed_forms: set[str] | None = None,
) -> None:
    if config.source_scalar_dtype not in {"bfloat16", "float32"}:
        raise ValueError(f"{prefix}.source_scalar_dtype must be bfloat16 or float32")
    forms = allowed_forms or {"basis", "basis_momentum", "abh", "abh_oja"}
    _validate_at_least(f"{prefix}.rank", config.rank, 1)
    _validate_string(f"{prefix}.target_module_regex", config.target_module_regex)
    _validate_string(f"{prefix}.fallback", config.fallback)
    if config.fallback != "gaussian":
        raise ValueError(f"{prefix}.fallback must be exactly 'gaussian'")
    _validate_at_least(
        f"{prefix}.max_activation_tokens", config.max_activation_tokens, 0
    )
    _validate_string(f"{prefix}.estimator_mode", config.estimator_mode)
    if config.estimator_mode not in {"two_side", "one_side", "loren_population"}:
        raise ValueError(
            f"{prefix}.estimator_mode must be one of: "
            "loren_population, one_side, two_side"
        )
    _validate_string(f"{prefix}.subspace_backend", config.subspace_backend)
    if config.subspace_backend not in {"power_iteration", "svd"}:
        raise ValueError(
            f"{prefix}.subspace_backend must be one of: power_iteration, svd"
        )
    _validate_at_least(f"{prefix}.power_iterations", config.power_iterations, 1)
    _validate_string(f"{prefix}.basis_seed_mode", config.basis_seed_mode)
    if config.basis_seed_mode not in AGZO_BASIS_SEED_MODES:
        raise ValueError(
            f"{prefix}.basis_seed_mode must be one of: ambient, perturbation_seed"
        )
    _validate_string(f"{prefix}.perturbation_form", config.perturbation_form)
    if config.perturbation_form not in forms:
        allowed = ", ".join(sorted(forms))
        raise ValueError(f"{prefix}.perturbation_form must be one of: {allowed}")
    _validate_string(f"{prefix}.abh_normalization", config.abh_normalization)
    if config.abh_normalization not in {"none", "rank_span", "layer_fro"}:
        raise ValueError(
            f"{prefix}.abh_normalization must be one of: layer_fro, none, rank_span"
        )
    _validate_finite(f"{prefix}.abh_layer_fro_scale", config.abh_layer_fro_scale)
    if float(config.abh_layer_fro_scale) <= 0.0:
        raise ValueError(f"{prefix}.abh_layer_fro_scale must be finite and > 0")
    _validate_at_least(f"{prefix}.abh_warmup_steps", config.abh_warmup_steps, 0)
    _validate_bool(f"{prefix}.abh_warmup_dense_noise", config.abh_warmup_dense_noise)
    _validate_finite(f"{prefix}.abh_momentum_beta", config.abh_momentum_beta)
    if not 0.0 <= float(config.abh_momentum_beta) < 1.0:
        raise ValueError(f"{prefix}.abh_momentum_beta must be in [0, 1)")
    _validate_at_least(
        f"{prefix}.abh_a_refresh_interval", config.abh_a_refresh_interval, 1
    )
    _validate_at_least(f"{prefix}.abh_right_rank", config.abh_right_rank, 1)
    _validate_at_least(
        f"{prefix}.abh_oja_wide_right_rank",
        config.abh_oja_wide_right_rank,
        0,
    )
    _validate_at_least(
        f"{prefix}.abh_oja_active_top_count",
        config.abh_oja_active_top_count,
        0,
    )
    _validate_at_least(
        f"{prefix}.abh_oja_active_tail_count",
        config.abh_oja_active_tail_count,
        0,
    )
    _validate_string(
        f"{prefix}.abh_oja_active_sampling_policy",
        config.abh_oja_active_sampling_policy,
    )
    if str(config.abh_oja_active_sampling_policy) not in {
        "top_tail",
        "top48_fixed_last16",
        "tail64",
        "top48_random_orth16",
        "plumage_all64",
        "plumage_top48_tail16",
        "fixed16_weighted48_floor020_tau64_a07",
        "fixed32_weighted32_floor025_tau64_a05",
        "pure_hi16_w5_mid12_tail10",
        "pure_hi16_w6_mid12_tail10",
        "top64_tail64_energy002",
        "population_3top64_1top64tail16",
    }:
        raise ValueError(
            f"{prefix}.abh_oja_active_sampling_policy must be one of: "
            "top_tail, top48_fixed_last16, tail64, top48_random_orth16, "
            "plumage_all64, "
            "plumage_top48_tail16, "
            "fixed16_weighted48_floor020_tau64_a07, "
            "fixed32_weighted32_floor025_tau64_a05, pure_hi16_w5_mid12_tail10, "
            "pure_hi16_w6_mid12_tail10, top64_tail64_energy002, "
            "population_3top64_1top64tail16"
        )
    _validate_string(f"{prefix}.abh_oja_basis_source", config.abh_oja_basis_source)
    if str(config.abh_oja_basis_source) not in {"center", "random_probe"}:
        raise ValueError(
            f"{prefix}.abh_oja_basis_source must be one of: center, random_probe"
        )
    if (
        int(config.abh_oja_active_top_count) > 0
        or int(config.abh_oja_active_tail_count) > 0
    ) and (
        int(config.abh_oja_active_top_count) + int(config.abh_oja_active_tail_count)
        != int(config.abh_right_rank)
    ):
        raise ValueError(
            f"{prefix}.abh_oja_active_top_count + "
            f"{prefix}.abh_oja_active_tail_count must equal "
            f"{prefix}.abh_right_rank when active slicing is enabled"
        )
    _validate_string(f"{prefix}.abh_right_rank_policy", config.abh_right_rank_policy)
    if str(config.abh_right_rank_policy) not in {"fixed", "proportional"}:
        raise ValueError(
            f"{prefix}.abh_right_rank_policy must be one of: fixed, proportional"
        )
    _validate_finite(f"{prefix}.abh_right_rank_ratio", config.abh_right_rank_ratio)
    if float(config.abh_right_rank_ratio) < 0.0:
        raise ValueError(f"{prefix}.abh_right_rank_ratio must be finite and >= 0")
    if (
        str(config.abh_right_rank_policy) == "proportional"
        and float(config.abh_right_rank_ratio) <= 0.0
    ):
        raise ValueError(
            f"{prefix}.abh_right_rank_ratio must be finite and > 0 when "
            f"{prefix}.abh_right_rank_policy is proportional"
        )
    _validate_at_least(f"{prefix}.abh_right_rank_min", config.abh_right_rank_min, 1)
    _validate_string(
        f"{prefix}.abh_activation_token_policy",
        config.abh_activation_token_policy,
    )
    if str(config.abh_activation_token_policy) not in {"fixed", "rank_multiple"}:
        raise ValueError(
            f"{prefix}.abh_activation_token_policy must be one of: fixed, rank_multiple"
        )
    _validate_finite(
        f"{prefix}.abh_activation_token_multiplier",
        config.abh_activation_token_multiplier,
    )
    if float(config.abh_activation_token_multiplier) <= 0.0:
        raise ValueError(
            f"{prefix}.abh_activation_token_multiplier must be finite and > 0"
        )
    _validate_finite(f"{prefix}.abh_oja_eta", config.abh_oja_eta)
    if float(config.abh_oja_eta) < 0.0:
        raise ValueError(f"{prefix}.abh_oja_eta must be finite and >= 0")
    _validate_at_least(
        f"{prefix}.abh_oja_eta_decay_interval",
        config.abh_oja_eta_decay_interval,
        0,
    )
    _validate_finite(
        f"{prefix}.abh_oja_eta_decay_factor", config.abh_oja_eta_decay_factor
    )
    if not 0.0 < float(config.abh_oja_eta_decay_factor) <= 1.0:
        raise ValueError(
            f"{prefix}.abh_oja_eta_decay_factor must be finite and in (0, 1]"
        )
    _validate_at_least(
        f"{prefix}.abh_oja_update_interval", config.abh_oja_update_interval, 1
    )
    _validate_at_least(
        f"{prefix}.abh_oja_late_update_start_step",
        config.abh_oja_late_update_start_step,
        0,
    )
    _validate_at_least(
        f"{prefix}.abh_oja_late_update_interval",
        config.abh_oja_late_update_interval,
        1,
    )
    _validate_bool(f"{prefix}.abh_oja_lagged_basis", config.abh_oja_lagged_basis)
    _validate_string(
        f"{prefix}.abh_oja_lagged_update_source",
        config.abh_oja_lagged_update_source,
    )
    if str(config.abh_oja_lagged_update_source) not in {"minus", "plus", "random"}:
        raise ValueError(
            f"{prefix}.abh_oja_lagged_update_source must be one of: minus, plus, random"
        )
    _validate_string(f"{prefix}.abh_oja_q_update_rule", config.abh_oja_q_update_rule)
    if str(config.abh_oja_q_update_rule) not in {
        "qr_oja",
        "osd",
        "ema_oja",
        "tangent_ema_oja",
    }:
        raise ValueError(
            f"{prefix}.abh_oja_q_update_rule must be one of: "
            "ema_oja, osd, qr_oja, tangent_ema_oja"
        )
    _validate_finite(f"{prefix}.abh_oja_q_ema_beta", config.abh_oja_q_ema_beta)
    if not 0.0 <= float(config.abh_oja_q_ema_beta) < 1.0:
        raise ValueError(f"{prefix}.abh_oja_q_ema_beta must be in [0, 1)")
    _validate_string(f"{prefix}.abh_oja_q_optimizer", config.abh_oja_q_optimizer)
    if str(config.abh_oja_q_optimizer) not in {"adamw", "sgd"}:
        raise ValueError(f"{prefix}.abh_oja_q_optimizer must be one of: adamw, sgd")
    _validate_finite(f"{prefix}.abh_oja_q_lr", config.abh_oja_q_lr)
    if float(config.abh_oja_q_lr) < 0.0:
        raise ValueError(f"{prefix}.abh_oja_q_lr must be finite and >= 0")
    _validate_finite(f"{prefix}.abh_oja_q_orth_lambda", config.abh_oja_q_orth_lambda)
    if float(config.abh_oja_q_orth_lambda) < 0.0:
        raise ValueError(f"{prefix}.abh_oja_q_orth_lambda must be finite and >= 0")
    _validate_finite(
        f"{prefix}.abh_q_dropout_keep_fraction",
        config.abh_q_dropout_keep_fraction,
    )
    if not 0.0 < float(config.abh_q_dropout_keep_fraction) <= 1.0:
        raise ValueError(
            f"{prefix}.abh_q_dropout_keep_fraction must be finite and in (0, 1]"
        )
    _validate_string(f"{prefix}.abh_noise_cache", config.abh_noise_cache)
    if str(config.abh_noise_cache) not in {"none", "step_full"}:
        raise ValueError(f"{prefix}.abh_noise_cache must be one of: none, step_full")
    _validate_string(f"{prefix}.abh_left_factor", config.abh_left_factor)
    if str(config.abh_left_factor) not in {
        "dense_left",
        "orthogonal_a",
        "orthogonal_ab_random",
        "gaussian_ab_random",
        "gaussian_a_random_b",
        "orthogonal_b_random_a",
    }:
        raise ValueError(
            f"{prefix}.abh_left_factor must be one of: dense_left, "
            "orthogonal_a, orthogonal_ab_random, gaussian_ab_random, "
            "gaussian_a_random_b, "
            "orthogonal_b_random_a"
        )
    if hasattr(config, "abh_num_noise"):
        _validate_at_least(f"{prefix}.abh_num_noise", config.abh_num_noise, 1)
    if hasattr(config, "abh_population_dense_noise_count"):
        _validate_at_least(
            f"{prefix}.abh_population_dense_noise_count",
            config.abh_population_dense_noise_count,
            0,
        )
        if int(config.abh_population_dense_noise_count) > int(config.abh_num_noise):
            raise ValueError(
                f"{prefix}.abh_population_dense_noise_count must be <= "
                f"{prefix}.abh_num_noise"
            )
    _validate_string(f"{prefix}.abh_update_transform", config.abh_update_transform)
    if str(config.abh_update_transform) not in {
        "ab_adam",
        "ab_momentum",
        "muon_ns_r",
        "muon_ns_r_momentum",
        "none",
        "muon_ns_small_b",
        "muon_ns_small_b_momentum",
    }:
        raise ValueError(
            f"{prefix}.abh_update_transform must be one of: "
            "ab_adam, ab_momentum, muon_ns_r, muon_ns_r_momentum, muon_ns_small_b, "
            "muon_ns_small_b_momentum, none"
        )
    _validate_string(f"{prefix}.abh_probe_transform", config.abh_probe_transform)
    if str(config.abh_probe_transform) not in {
        "none",
        "adamu_b",
        "loren_q",
        "loren_r_flat",
    }:
        raise ValueError(
            f"{prefix}.abh_probe_transform must be one of: adamu_b, loren_q, "
            "loren_r_flat, none"
        )
    _validate_finite(f"{prefix}.abh_adamu_alpha", config.abh_adamu_alpha)
    if not 0.0 <= float(config.abh_adamu_alpha) <= 1.0:
        raise ValueError(f"{prefix}.abh_adamu_alpha must be finite and in [0, 1]")
    _validate_finite(f"{prefix}.abh_adamu_beta1", config.abh_adamu_beta1)
    if not 0.0 <= float(config.abh_adamu_beta1) <= 1.0:
        raise ValueError(f"{prefix}.abh_adamu_beta1 must be finite and in [0, 1]")
    _validate_finite(f"{prefix}.abh_adamu_beta2", config.abh_adamu_beta2)
    if not 0.0 <= float(config.abh_adamu_beta2) <= 1.0:
        raise ValueError(f"{prefix}.abh_adamu_beta2 must be finite and in [0, 1]")
    _validate_finite(f"{prefix}.abh_adamu_eps", config.abh_adamu_eps)
    if float(config.abh_adamu_eps) <= 0.0:
        raise ValueError(f"{prefix}.abh_adamu_eps must be finite and > 0")
    _validate_finite(f"{prefix}.abh_loren_q_damping", config.abh_loren_q_damping)
    if float(config.abh_loren_q_damping) <= 0.0:
        raise ValueError(f"{prefix}.abh_loren_q_damping must be finite and > 0")
    _validate_finite(
        f"{prefix}.abh_loren_q_a_init_std",
        config.abh_loren_q_a_init_std,
    )
    if float(config.abh_loren_q_a_init_std) < 0.0:
        raise ValueError(f"{prefix}.abh_loren_q_a_init_std must be finite and >= 0")
    _validate_finite(f"{prefix}.abh_loren_q_lr_cov", config.abh_loren_q_lr_cov)
    if float(config.abh_loren_q_lr_cov) < 0.0:
        raise ValueError(f"{prefix}.abh_loren_q_lr_cov must be finite and >= 0")
    _validate_finite(
        f"{prefix}.abh_loren_q_a_eps_power", config.abh_loren_q_a_eps_power
    )
    if float(config.abh_loren_q_a_eps_power) < 0.0:
        raise ValueError(f"{prefix}.abh_loren_q_a_eps_power must be finite and >= 0")
    _validate_string(
        f"{prefix}.abh_loren_population_baseline",
        config.abh_loren_population_baseline,
    )
    if config.abh_loren_population_baseline not in {
        "center",
        "population_mean",
    }:
        raise ValueError(
            f"{prefix}.abh_loren_population_baseline must be one of: "
            "center, population_mean"
        )
    _validate_bool(
        f"{prefix}.abh_loren_population_normalize_std",
        config.abh_loren_population_normalize_std,
    )
    _validate_finite(
        f"{prefix}.abh_loren_population_std_eps",
        config.abh_loren_population_std_eps,
    )
    if float(config.abh_loren_population_std_eps) < 0.0:
        raise ValueError(
            f"{prefix}.abh_loren_population_std_eps must be finite and >= 0"
        )
    _validate_finite(
        f"{prefix}.abh_loren_population_clip_std",
        config.abh_loren_population_clip_std,
    )
    if float(config.abh_loren_population_clip_std) < 0.0:
        raise ValueError(
            f"{prefix}.abh_loren_population_clip_std must be finite and >= 0"
        )
    if int(config.abh_loren_population_trim_extremes) < 0:
        raise ValueError(f"{prefix}.abh_loren_population_trim_extremes must be >= 0")
    _validate_finite(
        f"{prefix}.abh_loren_population_deferred_residual_lambda",
        config.abh_loren_population_deferred_residual_lambda,
    )
    deferred_residual_lambda = float(
        config.abh_loren_population_deferred_residual_lambda
    )
    if deferred_residual_lambda != -1.0 and not 0.0 <= deferred_residual_lambda <= 1.0:
        raise ValueError(
            f"{prefix}.abh_loren_population_deferred_residual_lambda must be "
            "-1 or in [0, 1]"
        )
    _validate_finite(
        f"{prefix}.abh_two_side_residual_lambda",
        config.abh_two_side_residual_lambda,
    )
    residual_lambda = float(config.abh_two_side_residual_lambda)
    if residual_lambda != -1.0 and not 0.0 <= residual_lambda <= 1.0:
        raise ValueError(
            f"{prefix}.abh_two_side_residual_lambda must be -1 or in [0, 1]"
        )
    _validate_finite(f"{prefix}.abh_adam_beta2", config.abh_adam_beta2)
    if not 0.0 <= float(config.abh_adam_beta2) < 1.0:
        raise ValueError(f"{prefix}.abh_adam_beta2 must be finite and in [0, 1)")
    _validate_finite(f"{prefix}.abh_adam_eps", config.abh_adam_eps)
    if float(config.abh_adam_eps) <= 0.0:
        raise ValueError(f"{prefix}.abh_adam_eps must be finite and > 0")
    _validate_finite(
        f"{prefix}.abh_projected_grad_clip", config.abh_projected_grad_clip
    )
    if float(config.abh_projected_grad_clip) < 0.0:
        raise ValueError(f"{prefix}.abh_projected_grad_clip must be finite and >= 0")
    _validate_finite(
        f"{prefix}.abh_curvature_damping_lambda",
        config.abh_curvature_damping_lambda,
    )
    if float(config.abh_curvature_damping_lambda) < 0.0:
        raise ValueError(
            f"{prefix}.abh_curvature_damping_lambda must be finite and >= 0"
        )
    _validate_finite(f"{prefix}.abh_meazo_scale_beta", config.abh_meazo_scale_beta)
    if not 0.0 <= float(config.abh_meazo_scale_beta) < 1.0:
        raise ValueError(f"{prefix}.abh_meazo_scale_beta must be finite and in [0, 1)")
    _validate_finite(f"{prefix}.abh_meazo_scale_eps", config.abh_meazo_scale_eps)
    if float(config.abh_meazo_scale_eps) <= 0.0:
        raise ValueError(f"{prefix}.abh_meazo_scale_eps must be finite and > 0")
    _validate_bool(
        f"{prefix}.abh_meazo_scale_bias_correction",
        config.abh_meazo_scale_bias_correction,
    )
    _validate_bool(f"{prefix}.abh_seed_pool_enabled", config.abh_seed_pool_enabled)
    _validate_at_least(f"{prefix}.abh_seed_pool_size", config.abh_seed_pool_size, 1)
    _validate_at_least(
        f"{prefix}.abh_seed_pool_warmup_steps",
        config.abh_seed_pool_warmup_steps,
        0,
    )
    _validate_finite(f"{prefix}.abh_seed_pool_gamma", config.abh_seed_pool_gamma)
    if not 0.0 < float(config.abh_seed_pool_gamma) <= 1.0:
        raise ValueError(f"{prefix}.abh_seed_pool_gamma must be finite and in (0, 1]")
    _validate_finite(f"{prefix}.abh_seed_pool_rho", config.abh_seed_pool_rho)
    if not 0.0 < float(config.abh_seed_pool_rho) <= 1.0:
        raise ValueError(f"{prefix}.abh_seed_pool_rho must be finite and in (0, 1]")
    _validate_string(
        f"{prefix}.abh_seed_pool_score_mode",
        config.abh_seed_pool_score_mode,
    )
    if str(config.abh_seed_pool_score_mode) not in {
        "abs_s_minus_curvature",
        "negative_curvature",
        "signal_ratio_damped",
        "taylor_expected_gain",
        "rma_zscore",
        "rma_signal_ratio",
        "population_rma",
    }:
        raise ValueError(
            f"{prefix}.abh_seed_pool_score_mode must be one of: "
            "abs_s_minus_curvature, negative_curvature, signal_ratio_damped, "
            "taylor_expected_gain, rma_signal_ratio, rma_zscore, population_rma"
        )
    _validate_finite(
        f"{prefix}.abh_seed_pool_curvature_lambda",
        config.abh_seed_pool_curvature_lambda,
    )
    if float(config.abh_seed_pool_curvature_lambda) < 0.0:
        raise ValueError(
            f"{prefix}.abh_seed_pool_curvature_lambda must be finite and >= 0"
        )
    _validate_finite(
        f"{prefix}.abh_seed_pool_abs_s_lambda",
        config.abh_seed_pool_abs_s_lambda,
    )
    if float(config.abh_seed_pool_abs_s_lambda) < 0.0:
        raise ValueError(f"{prefix}.abh_seed_pool_abs_s_lambda must be finite and >= 0")
    _validate_finite(
        f"{prefix}.abh_seed_pool_signed_s_lambda",
        config.abh_seed_pool_signed_s_lambda,
    )
    if float(config.abh_seed_pool_signed_s_lambda) < 0.0:
        raise ValueError(
            f"{prefix}.abh_seed_pool_signed_s_lambda must be finite and >= 0"
        )
    _validate_finite(
        f"{prefix}.abh_seed_pool_noise_lambda",
        config.abh_seed_pool_noise_lambda,
    )
    if float(config.abh_seed_pool_noise_lambda) < 0.0:
        raise ValueError(f"{prefix}.abh_seed_pool_noise_lambda must be finite and >= 0")
    _validate_finite(
        f"{prefix}.abh_seed_pool_count_lambda",
        config.abh_seed_pool_count_lambda,
    )
    if float(config.abh_seed_pool_count_lambda) < 0.0:
        raise ValueError(f"{prefix}.abh_seed_pool_count_lambda must be finite and >= 0")
    _validate_finite(
        f"{prefix}.abh_seed_pool_ucb_lambda", config.abh_seed_pool_ucb_lambda
    )
    if float(config.abh_seed_pool_ucb_lambda) < 0.0:
        raise ValueError(f"{prefix}.abh_seed_pool_ucb_lambda must be finite and >= 0")
    _validate_at_least(f"{prefix}.abh_seed_pool_top_k", config.abh_seed_pool_top_k, 1)
    _validate_at_least(
        f"{prefix}.abh_seed_pool_top_select_count",
        config.abh_seed_pool_top_select_count,
        0,
    )
    _validate_at_least(
        f"{prefix}.abh_seed_pool_ucb_select_count",
        config.abh_seed_pool_ucb_select_count,
        0,
    )
    _validate_at_least(
        f"{prefix}.abh_seed_pool_fresh_select_count",
        config.abh_seed_pool_fresh_select_count,
        0,
    )
    _validate_finite(
        f"{prefix}.abh_seed_pool_top_fraction",
        config.abh_seed_pool_top_fraction,
    )
    if not 0.0 < float(config.abh_seed_pool_top_fraction) <= 1.0:
        raise ValueError(
            f"{prefix}.abh_seed_pool_top_fraction must be finite and in (0, 1]"
        )
    _validate_finite(
        f"{prefix}.abh_seed_pool_temperature",
        config.abh_seed_pool_temperature,
    )
    if float(config.abh_seed_pool_temperature) <= 0.0:
        raise ValueError(f"{prefix}.abh_seed_pool_temperature must be finite and > 0")
    _validate_at_least(
        f"{prefix}.abh_seed_pool_screen_until_step",
        config.abh_seed_pool_screen_until_step,
        0,
    )
    _validate_finite(
        f"{prefix}.abh_seed_pool_update_c_over_eps_threshold",
        config.abh_seed_pool_update_c_over_eps_threshold,
    )
    if float(config.abh_seed_pool_update_c_over_eps_threshold) < 0.0:
        raise ValueError(
            f"{prefix}.abh_seed_pool_update_c_over_eps_threshold must be "
            "finite and >= 0"
        )
    _validate_finite(
        f"{prefix}.abh_seed_pool_update_curvature_signal_ratio_threshold",
        config.abh_seed_pool_update_curvature_signal_ratio_threshold,
    )
    if float(config.abh_seed_pool_update_curvature_signal_ratio_threshold) < 0.0:
        raise ValueError(
            f"{prefix}.abh_seed_pool_update_curvature_signal_ratio_threshold "
            "must be finite and >= 0"
        )
    _validate_at_least(
        f"{prefix}.abh_seed_pool_update_min_count",
        config.abh_seed_pool_update_min_count,
        0,
    )
    _validate_at_least(
        f"{prefix}.abh_seed_pool_min_survival_count",
        config.abh_seed_pool_min_survival_count,
        0,
    )
    _validate_at_least(
        f"{prefix}.abh_seed_pool_bad_count_threshold",
        config.abh_seed_pool_bad_count_threshold,
        0,
    )
    _validate_finite(
        f"{prefix}.abh_seed_pool_bad_score_quantile",
        config.abh_seed_pool_bad_score_quantile,
    )
    if not 0.0 <= float(config.abh_seed_pool_bad_score_quantile) <= 1.0:
        raise ValueError(
            f"{prefix}.abh_seed_pool_bad_score_quantile must be finite and in [0, 1]"
        )
    _validate_string(
        f"{prefix}.abh_seed_pool_update_filter_roles",
        config.abh_seed_pool_update_filter_roles,
    )
    _validate_string(
        f"{prefix}.abh_seed_pool_fixed_seeds",
        config.abh_seed_pool_fixed_seeds,
    )
    _validate_string(
        f"{prefix}.abh_multi_update_weighting",
        config.abh_multi_update_weighting,
    )
    if str(config.abh_multi_update_weighting) not in {
        "active_best_current_score",
        "active_max_abs_s",
        "active_min_abs_s",
        "active_signal_ratio",
        "active_min_curvature",
        "best_score",
        "score_softmax",
        "uniform",
    }:
        raise ValueError(
            f"{prefix}.abh_multi_update_weighting must be one of: "
            "active_best_current_score, active_max_abs_s, active_min_abs_s, "
            "active_min_curvature, active_signal_ratio, best_score, "
            "score_softmax, uniform"
        )
    _validate_finite(
        f"{prefix}.abh_multi_update_curvature_lambda",
        config.abh_multi_update_curvature_lambda,
    )
    if float(config.abh_multi_update_curvature_lambda) < 0.0:
        raise ValueError(
            f"{prefix}.abh_multi_update_curvature_lambda must be finite and >= 0"
        )
    _validate_string(f"{prefix}.abh_a_seed_mode", config.abh_a_seed_mode)
    if str(config.abh_a_seed_mode) not in {"block", "probe"}:
        raise ValueError(f"{prefix}.abh_a_seed_mode must be one of: block, probe")
    _validate_finite(
        f"{prefix}.abh_multi_update_softmax_tau",
        config.abh_multi_update_softmax_tau,
    )
    if float(config.abh_multi_update_softmax_tau) <= 0.0:
        raise ValueError(
            f"{prefix}.abh_multi_update_softmax_tau must be finite and > 0"
        )
    _validate_at_least(
        f"{prefix}.abh_mezo_mix_interval", config.abh_mezo_mix_interval, 0
    )
    _validate_at_least(
        f"{prefix}.abh_mezo_mix_dense_steps", config.abh_mezo_mix_dense_steps, 0
    )
    if int(config.abh_mezo_mix_interval) == 0:
        if int(config.abh_mezo_mix_dense_steps) != 0:
            raise ValueError(
                f"{prefix}.abh_mezo_mix_dense_steps must be 0 when "
                f"{prefix}.abh_mezo_mix_interval is 0"
            )
    elif int(config.abh_mezo_mix_dense_steps) > int(config.abh_mezo_mix_interval):
        raise ValueError(
            f"{prefix}.abh_mezo_mix_dense_steps must be <= "
            f"{prefix}.abh_mezo_mix_interval"
        )
    _validate_finite(
        f"{prefix}.abh_dense_residual_ratio", config.abh_dense_residual_ratio
    )
    if not 0.0 <= float(config.abh_dense_residual_ratio) <= 1.0:
        raise ValueError(
            f"{prefix}.abh_dense_residual_ratio must be finite and in [0, 1]"
        )


def _validate_vllm_worker_mode(worker_mode: Any) -> None:
    _validate_string("backend.vllm_worker_mode", worker_mode)
    if worker_mode not in {"fake", "real"}:
        raise ValueError("backend.vllm_worker_mode must be one of: fake, real")


def _validate_objective_rollout_matrix(
    *,
    objective_name: str,
    rollout_protocol: str,
    loss_aggregation: str = "sequence_sum_then_sample_mean",
    reference_policy: str = "initial_adapter",
    logprob_granularity: str = "sequence",
) -> None:
    if objective_name == "source_sst2_candidate_scoring":
        if rollout_protocol not in ALLOWED_ROLLOUT_PROTOCOLS:
            supported = ", ".join(sorted(ALLOWED_ROLLOUT_PROTOCOLS))
            raise ValueError(
                f"unknown rollout protocol {rollout_protocol!r}; "
                f"expected one of: {supported}"
            )
        if rollout_protocol != FIXED_ROLLOUT_PROTOCOL:
            raise ValueError(
                "source_sst2_candidate_scoring objective currently only supports "
                "fixed_rollout_per_step"
            )
        objective_name = "zoregular_classification_ce"
    build_objective_protocol(
        objective_name,
        rollout_protocol,
        loss_aggregation=loss_aggregation,
        reference_policy=reference_policy,
        logprob_granularity=logprob_granularity,
    )


def _has_checkable_logprob_support(logprob_backend: Any) -> bool:
    supports = getattr(logprob_backend, "supports_logprobs", None)
    if supports is not None:
        return bool(supports)
    return callable(getattr(logprob_backend, "compute_logprobs", None)) or callable(
        getattr(logprob_backend, "logprobs", None)
    )


def _section_payload(payload: dict[str, Any], section: str) -> dict[str, Any]:
    values = payload.get(section, {})
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"config section {section!r} must be a mapping")
    return values


def normalize_hizoo_config(value: HiZOOConfig | dict[str, Any]) -> HiZOOConfig:
    if isinstance(value, HiZOOConfig):
        normalized = {
            name: getattr(value, name) for name in HiZOOConfig.__dataclass_fields__
        }
        smooth_provided = bool(getattr(value, "_hessian_smooth_explicit", True))
        smooth_type_provided = bool(
            getattr(value, "_hessian_smooth_type_explicit", True)
        )
    elif isinstance(value, dict):
        known_fields = set(HiZOOConfig.__dataclass_fields__)
        unknown = sorted(set(value) - known_fields)
        if unknown:
            raise ValueError(f"unknown config keys for zo.hizoo: {', '.join(unknown)}")
        normalized = dict(value)
        smooth_provided = "hessian_smooth" in normalized
        smooth_type_provided = "hessian_smooth_type" in normalized
    else:
        raise ValueError("zo.hizoo must be a mapping")

    smooth_type = normalized.get("hessian_smooth_type", HiZOOConfig.hessian_smooth_type)
    if smooth_type_provided and not smooth_provided:
        if smooth_type in HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES:
            normalized["hessian_smooth"] = HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES[
                smooth_type
            ]
        elif smooth_type == "custom":
            raise ValueError(
                "zo.hizoo.hessian_smooth must be provided when "
                "hessian_smooth_type='custom'"
            )
    elif smooth_provided and not smooth_type_provided:
        smooth = float(normalized["hessian_smooth"])
        normalized["hessian_smooth_type"] = _hizoo_smooth_type_for_value(smooth)
    elif smooth_provided and smooth_type_provided:
        expected = HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES.get(smooth_type)
        if expected is not None and not math.isclose(
            float(normalized["hessian_smooth"]),
            expected,
            rel_tol=0.0,
            abs_tol=max(1e-15, abs(expected) * 1e-12),
        ):
            raise ValueError(
                "zo.hizoo.hessian_smooth must match "
                f"zo.hizoo.hessian_smooth_type={smooth_type!r} ({expected:g})"
            )
    return HiZOOConfig(**normalized)


def _hizoo_smooth_type_for_value(smooth: float) -> str:
    for smooth_type, expected in HIZOO_CONSTANT_HESSIAN_SMOOTH_TYPES.items():
        if math.isclose(
            smooth,
            expected,
            rel_tol=0.0,
            abs_tol=max(1e-15, abs(expected) * 1e-12),
        ):
            return smooth_type
    return "custom"


def _normalize_nested_config(name: str, cls, value: Any):
    if cls is HiZOOConfig:
        return normalize_hizoo_config(value)
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return _dataclass_from_dict(cls, value, section=f"zo.{name}")
    raise ValueError(f"zo.{name} must be a mapping")


def _normalize_zo_config(values: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    if "hizoo" in normalized:
        normalized["hizoo"] = _normalize_nested_config(
            "hizoo", HiZOOConfig, normalized["hizoo"]
        )
    if "agzo" in normalized:
        normalized["agzo"] = _normalize_nested_config(
            "agzo", AGZOConfig, normalized["agzo"]
        )
    if "aimzo" in normalized:
        normalized["aimzo"] = _normalize_nested_config(
            "aimzo", AIMZOConfig, normalized["aimzo"]
        )
    if "oja_abq" in normalized:
        normalized["oja_abq"] = _normalize_nested_config(
            "oja_abq", OjaABQConfig, normalized["oja_abq"]
        )
    if "lozo" in normalized:
        normalized["lozo"] = _normalize_nested_config(
            "lozo", LOZOConfig, normalized["lozo"]
        )
    if "svd0" in normalized:
        normalized["svd0"] = _normalize_nested_config(
            "svd0", SVD0Config, normalized["svd0"]
        )
    if "pgap" in normalized:
        normalized["pgap"] = _normalize_nested_config(
            "pgap", PGAPConfig, normalized["pgap"]
        )
    if "lowdim_muon" in normalized:
        normalized["lowdim_muon"] = _normalize_nested_config(
            "lowdim_muon", LowDimMuonConfig, normalized["lowdim_muon"]
        )
    if "curvzo" in normalized:
        normalized["curvzo"] = _normalize_nested_config(
            "curvzo", CurvZOConfig, normalized["curvzo"]
        )
    return normalized


def _dataclass_from_dict(cls, values: dict[str, Any], *, section: str):
    known_fields = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - known_fields)
    if unknown:
        raise ValueError(f"unknown config keys for {section}: {', '.join(unknown)}")

    normalized = dict(values)
    if cls is BackendConfig and "lora_target_modules" in normalized:
        modules = normalized["lora_target_modules"]
        _validate_lora_target_modules(modules)
        normalized["lora_target_modules"] = tuple(modules)
    if (
        cls is ObjectiveConfig
        and normalized.get("name") == "reward"
        and "rollout_protocol" not in normalized
    ):
        normalized["rollout_protocol"] = FRESH_ROLLOUT_PROTOCOL
    if cls is TrainerConfig and "checkpoint_milestones" in normalized:
        milestones = tuple(normalized["checkpoint_milestones"])
        max_steps = normalized.get("max_steps", 1)
        if any(type(step) is not int or not 1 <= step <= max_steps for step in milestones):
            raise ValueError("checkpoint_milestones must be integers within max_steps")
        if tuple(sorted(set(milestones))) != milestones:
            raise ValueError("checkpoint_milestones must be sorted and unique")
        normalized["checkpoint_milestones"] = milestones
    return cls(**normalized)


def load_config(path: str | Path) -> ExperimentConfig:
    import yaml

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("config file must contain a YAML mapping")

    known_sections = {
        "backend",
        "data",
        "generation",
        "logging",
        "objective",
        "trainer",
        "zo",
    }
    unknown_sections = sorted(set(payload) - known_sections)
    if unknown_sections:
        raise ValueError(f"unknown config sections: {', '.join(unknown_sections)}")

    config = ExperimentConfig(
        data=_dataclass_from_dict(
            DataConfig, _section_payload(payload, "data"), section="data"
        ),
        backend=_dataclass_from_dict(
            BackendConfig, _section_payload(payload, "backend"), section="backend"
        ),
        generation=_dataclass_from_dict(
            GenerationConfig,
            _section_payload(payload, "generation"),
            section="generation",
        ),
        objective=_dataclass_from_dict(
            ObjectiveConfig,
            _section_payload(payload, "objective"),
            section="objective",
        ),
        zo=_dataclass_from_dict(
            ZOConfig,
            _normalize_zo_config(_section_payload(payload, "zo")),
            section="zo",
        ),
        trainer=_dataclass_from_dict(
            TrainerConfig,
            _section_payload(payload, "trainer"),
            section="trainer",
        ),
        logging=_dataclass_from_dict(
            LoggingConfig,
            _section_payload(payload, "logging"),
            section="logging",
        ),
    )
    validate_config(config)
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return load_config(path)
