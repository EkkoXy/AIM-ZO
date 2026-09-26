from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from aimzo.artifacts import RunArtifactWriter
from aimzo.config import (
    ExperimentConfig,
    build_zo_step_seed_sequence,
    validate_config,
)
from aimzo.logging import make_json_safe, write_json_atomic
from aimzo.protocols import ZOREGULAR_SUPERVISED_PROTOCOL, build_objective_protocol
from aimzo.runtime_metrics import collect_cuda_memory
from aimzo.tasks.zoregular import is_zoregular_task
from aimzo.tasks.zoregular_scoring import (
    classify_from_candidate_scores,
    collate_zoregular_classification,
    collate_zoregular_generation,
    encode_zoregular_classification_sample,
    encode_zoregular_generation_sample,
    zoregular_classification_loss,
    zoregular_generation_loss,
)
from aimzo.zo.methods import build_hf_zo_method
from aimzo.zo.params import parameter_selection_metadata


class HFZORegularTrainer:
    """Sequential full-parameter MeZO trainer for ZORegular CE on HF models."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        task: Any,
        backend: Any,
        artifact_writer: RunArtifactWriter | None = None,
    ) -> None:
        validate_config(config)
        _validate_hf_zoregular_train_config(config)
        _validate_zoregular_task(task)
        _validate_objective_matches_task(config, task)
        self.config = config
        self.task = task
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.device = torch.device(getattr(backend, "device", "cpu"))
        self.model.to(self.device)
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            # ZO objectives score complete sequences and never reuse decoded
            # key/value states. Keeping the cache disabled avoids retaining a
            # full layer-by-layer KV copy without changing logits.
            model_config.use_cache = False
        self.output_dir = config.output_path
        self.artifact_writer = artifact_writer or RunArtifactWriter(
            self.output_dir,
            history_snapshot_interval=config.logging.history_snapshot_interval,
        )
        self.objective_name = config.objective.name
        self.objective_protocol = build_objective_protocol(
            self.objective_name,
            ZOREGULAR_SUPERVISED_PROTOCOL,
        ).to_dict()
        self.zo_method = build_hf_zo_method(config.zo)
        self.history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self._source_train_rows: list[Any] | None = None
        self._source_train_batch_iterator: Iterator[list[Any]] | None = None
        self._source_train_epoch = 0
        self._source_train_sampler: RandomSampler | None = None
        self._source_train_seeded = False
        self._last_objective_input_trace: dict[str, Any] | None = None
        self.start_step = 0
        self.resume_info: dict[str, Any] | None = None

    def train(self) -> dict[str, Any]:
        try:
            return self._train_impl()
        finally:
            self.artifact_writer.close()

    def _train_impl(self) -> dict[str, Any]:
        self.artifact_writer.ensure_output_dir()
        self.resume_info = self._resume_if_requested()
        if self.resume_info is None:
            self._reset_owned_artifacts()
        else:
            self._load_existing_run_history()
        self.artifact_writer.write_config(self._config_artifact_payload())
        self.artifact_writer.write_manifest(self._manifest())
        zo_step_seeds = self._build_zo_step_seed_sequence(
            self.config,
            steps=int(self.config.trainer.max_steps),
        )

        final_eval: dict[str, Any] | None = None
        best_metric_name = str(self.config.trainer.best_checkpoint_metric)
        best_metric_mode = str(self.config.trainer.best_checkpoint_mode)
        best_eval, best_eval_value = _best_finite_metric_eval(
            self.eval_history,
            metric=best_metric_name,
            mode=best_metric_mode,
        )
        best_checkpoint_path = self.output_dir / "checkpoints" / "best"
        best_checkpoint: Path | None = (
            best_checkpoint_path if best_checkpoint_path.exists() else None
        )
        for step in range(int(self.start_step), int(self.config.trainer.max_steps)):
            step_number = step + 1
            started_at = time.time()
            samples = self._sample_train_batch(step=step)
            seed = zo_step_seeds[step]
            learning_rate = _scheduled_learning_rate(
                self.config,
                step_number=step_number,
            )
            eps = _scheduled_eps(
                self.config,
                step_number=step_number,
            )
            _set_zo_method_learning_rate(self.zo_method, learning_rate)
            _set_zo_method_eps(self.zo_method, eps)

            method_result = self.zo_method.step(
                self.model,
                lambda samples=samples: self._objective(samples),
                seed=seed,
                step=step_number,
            )
            candidate_results = [
                _candidate_result_dict(
                    result,
                    protocol=self.objective_protocol,
                )
                for result in method_result.candidate_results
            ]
            for result in candidate_results:
                self.artifact_writer.append_candidate_result(
                    {"step": step_number, **result}
                )

            f_plus = method_result.f_plus
            f_minus = method_result.f_minus
            objective_delta = method_result.objective_delta
            projected_grad = method_result.projected_grad
            objective_calls = _objective_calls(
                method_result.method_diagnostics,
                candidate_results,
            )
            update_skipped = any(
                not bool(result.get("ok", False)) for result in candidate_results
            )
            history_entry: dict[str, Any] = {
                "step": step_number,
                "seed": int(method_result.seed),
                "trainer_seed": int(seed),
                "zo_method": method_result.zo_method,
                "f_plus": f_plus,
                "f_minus": f_minus,
                "objective_delta": 0.0
                if objective_delta is None
                else float(objective_delta),
                "projected_grad": projected_grad,
                "objective_calls": objective_calls,
                "update_norm": float(method_result.update_norm),
                "parameter_delta_checksum": str(method_result.parameter_delta_checksum),
                "learning_rate": float(learning_rate),
                "lr": float(learning_rate),
                "base_learning_rate": float(self.config.zo.learning_rate),
                "lr_schedule": str(self.config.zo.lr_schedule),
                "eps": float(eps),
                "base_eps": float(self.config.zo.eps),
                "eps_schedule": str(self.config.zo.eps_schedule),
                "objective": self.objective_name,
                "objective_direction": "minimize",
                "objective_protocol": self.objective_protocol["name"],
                "train_batch_checksum": self._samples_checksum(samples),
                "train_batch_size": len(samples),
                "train_batch_indices": self._sample_indices(samples),
                "objective_input_trace": self._last_objective_input_trace,
                "objective_variance": _variance(
                    [value for value in (f_plus, f_minus) if value is not None]
                ),
                "update_skipped": update_skipped,
                "update_skip_reason": (
                    "candidate_objective_failed" if update_skipped else None
                ),
                "candidate_results": candidate_results,
                "method_diagnostics": dict(method_result.method_diagnostics),
                "eval": None,
                "runtime_metrics": collect_cuda_memory(),
                "step_time": time.time() - started_at,
            }
            for key in (
                "abh_seed_pool_enabled",
                "abh_seed_pool_selection_mode",
                "abh_seed_pool_selected_seeds",
                "abh_seed_pool_selected_roles",
                "abh_seed_pool_score_rows",
                "abh_seed_pool_snapshot",
                "abh_seed_pool_score_mean",
                "abh_seed_pool_abs_s_mean",
                "abh_seed_pool_c_over_eps_mean",
                "abh_seed_pool_m_abs_s_mean",
                "abh_seed_pool_m_h_mean",
                "abh_seed_pool_rma_damping_factor_mean",
                "abh_multi_update_weighting",
                "abh_multi_update_weights",
                "abh_multi_projected_grad_raw_abs_mean",
                "abh_multi_projected_grad_abs_mean",
                "abh_multi_projected_grad_count",
            ):
                if key in method_result.method_diagnostics:
                    history_entry[key] = method_result.method_diagnostics[key]
            if update_skipped:
                error = _candidate_failure_error(
                    step=step_number,
                    candidate_results=candidate_results,
                )
                history_entry["error"] = error
                self.history.append(history_entry)
                self.artifact_writer.write_history(self.history, force_snapshot=True)
                self._write_failure_results(error)
                raise RuntimeError(_candidate_failure_message(error))

            eval_entry: dict[str, Any] | None = None
            if (
                step_number in self.config.trainer.checkpoint_milestones
                if self.config.trainer.checkpoint_milestones
                else self.config.trainer.eval_every > 0
                and step_number % self.config.trainer.eval_every == 0
            ):
                eval_entry = self._evaluate(step=step_number)
                final_eval = eval_entry
                self.eval_history.append(eval_entry)
                self.artifact_writer.write_eval_history(self.eval_history)
                eval_value = _finite_metric(eval_entry, best_metric_name)
                if eval_value is not None and _metric_is_better(
                    eval_entry,
                    value=eval_value,
                    best_eval=best_eval,
                    best_value=best_eval_value,
                    mode=best_metric_mode,
                ):
                    best_eval_value = float(eval_value)
                    best_eval = dict(eval_entry)
                    if self.config.trainer.save_checkpoints:
                        best_checkpoint = best_checkpoint_path
                        self._save_checkpoint(
                            best_checkpoint,
                            current_step=step_number,
                            f_plus=f_plus,
                            f_minus=f_minus,
                            perturbation_seed=int(method_result.seed),
                        )

            history_entry["eval"] = eval_entry
            history_entry["step_time"] = time.time() - started_at
            self.history.append(history_entry)
            self.artifact_writer.write_history(
                self.history, force_snapshot=eval_entry is not None
            )

            if (
                self.config.trainer.save_checkpoints
                and (
                    step_number in self.config.trainer.checkpoint_milestones
                    if self.config.trainer.checkpoint_milestones
                    else self.config.trainer.save_every > 0
                    and step_number % self.config.trainer.save_every == 0
                )
            ):
                self._save_checkpoint(
                    self.artifact_writer.checkpoint_path(step=step_number),
                    current_step=step_number,
                    f_plus=f_plus,
                    f_minus=f_minus,
                    perturbation_seed=int(method_result.seed),
                )

        final_step = int(self.history[-1]["step"]) if self.history else 0
        last = self.history[-1] if self.history else {}
        final_checkpoint: Path | None = None
        if self.config.trainer.save_checkpoints:
            final_checkpoint = self.artifact_writer.final_checkpoint_path()
            self._save_checkpoint(
                final_checkpoint,
                current_step=final_step,
                f_plus=last.get("f_plus"),
                f_minus=last.get("f_minus"),
                perturbation_seed=last.get("seed"),
            )
        report_checkpoint = best_checkpoint or final_checkpoint
        if best_eval is None:
            best_eval = final_eval
        results = {
            "ok": True,
            "steps": int(self.config.trainer.max_steps),
            "completed_steps": final_step,
            "final_eval": final_eval,
            "best_eval": best_eval,
            "final_checkpoint": (
                str(final_checkpoint) if final_checkpoint is not None else None
            ),
            "report_checkpoint": (
                str(report_checkpoint) if report_checkpoint is not None else None
            ),
            "report_checkpoint_policy": str(
                self.config.trainer.source_checkpoint_policy
            ),
            "history": self.history,
        }
        if self.resume_info is not None:
            results["resume"] = dict(self.resume_info)
        self.artifact_writer.write_results(results)
        self.artifact_writer.write_runtime_summary(_summarize_runtime(self.history))
        if self.eval_history:
            self.artifact_writer.write_eval_history(self.eval_history)
        return results

    @staticmethod
    def _build_zo_step_seed_sequence(
        config: ExperimentConfig,
        steps: int,
    ) -> list[int]:
        if (
            config.zo.method == "agzo"
            and config.zo.seed_mode == "source_numpy_randint"
            and config.zo.source_compatibility_profile == "agzo_source_and_paper_gap"
        ):
            # AGZO source draws sparse_grad_random_seed once before the first
            # optimization step of every batch, then draws zo_random_seed
            # inside agzo_step().
            source_draws = build_zo_step_seed_sequence(
                seed=int(config.zo.seed),
                seed_mode=config.zo.seed_mode,
                steps=int(steps) * 2,
            )
            return source_draws[1::2]
        return build_zo_step_seed_sequence(
            seed=int(config.zo.seed),
            seed_mode=config.zo.seed_mode,
            steps=int(steps),
        )

    def _config_artifact_payload(self) -> dict[str, Any]:
        payload = asdict(self.config)
        if self.zo_method.name in {"zomuon", "zomopi"} and hasattr(
            self.zo_method, "lowdim_config"
        ):
            payload["zo"]["lowdim_muon"] = asdict(self.zo_method.lowdim_config)
        return payload

    def _write_failure_results(self, error: dict[str, Any]) -> None:
        completed_steps = max(0, int(error.get("step", 1)) - 1)
        results = {
            "ok": False,
            "steps": int(self.config.trainer.max_steps),
            "completed_steps": completed_steps,
            "error": error,
            "history": self.history,
        }
        if self.resume_info is not None:
            results["resume"] = dict(self.resume_info)
        self.artifact_writer.write_results(results)
        self.artifact_writer.write_runtime_summary(_summarize_runtime(self.history))

    def _objective(self, samples: list[Any]) -> tuple[float, dict[str, float]]:
        if self.objective_name == "zoregular_generation_ce":
            return self._generation_objective(samples)
        if self.objective_name in {
            "zoregular_classification_ce",
            "source_sst2_candidate_scoring",
        }:
            return self._classification_objective(samples)
        raise ValueError(f"Unsupported HF ZORegular objective: {self.objective_name!r}")

    @torch.no_grad()
    def _classification_objective(
        self,
        samples: list[Any],
    ) -> tuple[float, dict[str, float]]:
        features = [
            encode_zoregular_classification_sample(
                self._template(),
                sample,
                self.tokenizer,
                max_length=int(
                    getattr(self.config.backend, "max_model_len", None) or 2048
                ),
                source_style_tokenization=(
                    self.objective_name == "source_sst2_candidate_scoring"
                ),
            )
            for sample in samples
        ]
        batch = collate_zoregular_classification(
            features,
            pad_token_id=_pad_token_id(self.tokenizer),
            padding_side=(
                "left"
                if self.objective_name == "source_sst2_candidate_scoring"
                else "right"
            ),
            source_style_labels=(
                self.objective_name == "source_sst2_candidate_scoring"
            ),
        )
        self._last_objective_input_trace = self._input_trace(batch)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        candidate_scores = _candidate_suffix_logprob_scores(
            outputs.logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            option_len=batch["option_len"].to(self.device),
            average_by_option_len=(
                self.objective_name == "source_sst2_candidate_scoring"
            ),
            padding_side=(
                "left"
                if self.objective_name == "source_sst2_candidate_scoring"
                else "right"
            ),
        )
        labels = batch["labels"].to(self.device)
        num_options = batch["num_options"].to(self.device)
        gold_mask = batch["gold_mask"].to(self.device)
        loss = zoregular_classification_loss(
            candidate_scores,
            labels,
            num_options,
            gold_mask=gold_mask,
        )
        classified = classify_from_candidate_scores(
            candidate_scores,
            labels,
            num_options,
            gold_mask=gold_mask,
        )
        self._last_classification_predictions = classified["predictions"]
        loss_value = float(loss.item())
        metrics = {
            "classification_ce": loss_value,
            "loss": loss_value,
            "accuracy": float(classified["accuracy"]),
            "num_examples": float(len(samples)),
            "num_candidates": float(candidate_scores.numel()),
            "mean_candidate_logprob": float(candidate_scores.mean().item())
            if candidate_scores.numel()
            else 0.0,
        }
        return loss_value, metrics

    @torch.no_grad()
    def _generation_objective(
        self, samples: list[Any]
    ) -> tuple[float, dict[str, float]]:
        features = [
            encode_zoregular_generation_sample(
                self._template(),
                sample,
                self.tokenizer,
                max_length=int(
                    getattr(self.config.backend, "max_model_len", None) or 1024
                ),
            )
            for sample in samples
        ]
        batch = collate_zoregular_generation(
            features,
            pad_token_id=_pad_token_id(self.tokenizer),
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = zoregular_generation_loss(outputs.logits, labels)
        response_tokens = int((labels[:, 1:] != -100).sum().item())
        metrics = {
            "generation_ce": float(loss.item()),
            "loss": float(loss.item()),
            "token_nll_mean": float(loss.item()),
            "num_examples": float(len(samples)),
            "num_response_tokens": float(response_tokens),
        }
        return float(loss.item()), metrics

    def _evaluate(self, *, step: int) -> dict[str, Any]:
        return self.evaluate(step=step)

    def evaluate(self, *, step: int = 0) -> dict[str, Any]:
        """Evaluate the current model on the task's configured eval split."""
        return self.evaluate_rows(self.task.load_eval(), step=step)

    def evaluate_rows(
        self, rows: list[Any], *, step: int, include_multirc_grouped: bool = False
    ) -> dict[str, Any]:
        eval_batch_size = _eval_batch_size(
            self.config.trainer.eval_batch_size,
            len(rows),
        )
        weighted_loss = 0.0
        weighted_metrics: dict[str, float] = {}
        total_examples = 0.0
        batches = 0
        predictions: list[int] = []
        try:
            for batch in _batched(rows, eval_batch_size):
                batches += 1
                value, metrics = self._objective(batch)
                if include_multirc_grouped:
                    predictions.extend(self._last_classification_predictions)
                num_examples = float(metrics.get("num_examples", len(batch)))
                weighted_loss += float(value) * num_examples
                total_examples += num_examples
                for key, metric_value in metrics.items():
                    if isinstance(metric_value, int | float) and math.isfinite(
                        float(metric_value)
                    ):
                        weighted_metrics[key] = weighted_metrics.get(key, 0.0) + (
                            float(metric_value) * num_examples
                        )
            loss = weighted_loss / total_examples if total_examples else 0.0
            averaged = {
                key: value / total_examples
                for key, value in weighted_metrics.items()
                if total_examples
            }
            if include_multirc_grouped:
                from aimzo.tasks.qa_metrics import multirc_grouped_metrics
                averaged.update(multirc_grouped_metrics(rows, predictions))
            return {
                "step": int(step),
                "loss": float(loss),
                **averaged,
                "num_eval_examples": len(rows),
                "num_eval_batches": batches,
                "payload_checksum": self._samples_checksum(rows),
            }
        except Exception as exc:  # noqa: BLE001 - eval failures are recorded.
            return {
                "step": int(step),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "num_eval_examples": len(rows),
                "num_eval_batches": batches,
            }

    def _sample_train_batch(self, *, step: int) -> list[Any]:
        if (
            self.objective_name == "source_sst2_candidate_scoring"
            or self.config.data.zoregular_batch_sampler_policy == "accelerate_seedable"
        ):
            return self._sample_source_train_batch(step=step)
        sampler = getattr(self.task, "sample_train_batch", None)
        if callable(sampler):
            return list(
                sampler(
                    batch_size=int(self.config.trainer.train_batch_size),
                    seed=int(self.config.data.seed),
                    step=int(step),
                )
            )
        return list(self.task.load_train())[: int(self.config.trainer.train_batch_size)]

    def _sample_source_train_batch(self, *, step: int) -> list[Any]:
        del step
        rows = self._source_training_rows()
        if not rows:
            return []
        while True:
            if self._source_train_batch_iterator is None:
                self._source_train_batch_iterator = self._iter_source_train_batches(
                    rows
                )
            try:
                return next(self._source_train_batch_iterator)
            except StopIteration:
                self._source_train_batch_iterator = self._iter_source_train_batches(
                    rows
                )

    def _source_training_rows(self) -> list[Any]:
        if self._source_train_rows is None:
            self._source_train_rows = list(self.task.load_train())
        return self._source_train_rows

    def _iter_source_train_batches(self, rows: list[Any]) -> Iterator[list[Any]]:
        sampler_policy = getattr(
            self.config.data,
            "zoregular_batch_sampler_policy",
            "torch_random",
        )
        if (
            sampler_policy
            in {
                "torch_random",
                "torch_random_hf_dataloader",
                "hf_trainer_persistent_generator",
            }
            and not self._source_train_seeded
        ):
            torch.manual_seed(int(self.config.zo.seed))
            self._source_train_seeded = True
        batch_size = int(self.config.trainer.train_batch_size)
        epoch = self._source_train_epoch
        self._source_train_epoch += 1
        if sampler_policy == "accelerate_seedable":
            from accelerate.data_loader import SeedableRandomSampler

            sampler = SeedableRandomSampler(
                range(len(rows)),
                data_seed=int(self.config.zo.seed),
            )
            sampler.set_epoch(epoch)
            order = [int(index) for index in sampler]
        elif sampler_policy == "torch_random_hf_dataloader":
            loader = DataLoader(
                rows,
                batch_size=batch_size,
                sampler=RandomSampler(range(len(rows))),
                collate_fn=list,
            )
            yield from loader
            return
        elif sampler_policy == "hf_trainer_persistent_generator":
            if self._source_train_sampler is None:
                sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(sampler_seed)
                self._source_train_sampler = RandomSampler(
                    range(len(rows)),
                    generator=generator,
                )
            order = [int(index) for index in self._source_train_sampler]
        else:
            order = [int(index) for index in RandomSampler(range(len(rows)))]
        for start in range(0, len(order), batch_size):
            yield [rows[index] for index in order[start : start + batch_size]]

    def _template(self) -> Any:
        template = getattr(self.task, "template")
        if callable(template) and not hasattr(template, "encode"):
            return template()
        return template

    def _save_checkpoint(
        self,
        path: Path,
        *,
        current_step: int,
        f_plus: float | None,
        f_minus: float | None,
        perturbation_seed: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        backup = path.with_name(f".{path.name}.previous")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)
        temporary.mkdir(parents=True)
        try:
            save_checkpoint = getattr(self.backend, "save_checkpoint", None)
            if callable(save_checkpoint):
                save_checkpoint(temporary)
            elif callable(getattr(self.model, "save_pretrained", None)):
                self.model.save_pretrained(temporary)
            else:
                torch.save(self.model.state_dict(), temporary / "model.pt")
            write_json_atomic(
                temporary / "zo_state.json",
                self._zo_state(
                    current_step=current_step,
                    f_plus=f_plus,
                    f_minus=f_minus,
                    perturbation_seed=perturbation_seed,
                ),
            )
            self._save_method_state(temporary)
            if path.exists():
                path.rename(backup)
            temporary.rename(path)
            shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            if backup.exists() and not path.exists():
                backup.rename(path)
            raise

        if path.name.startswith("step_"):
            self._prune_periodic_checkpoints(path.parent)

    def _prune_periodic_checkpoints(self, checkpoint_root: Path) -> None:
        keep = int(self.config.trainer.max_periodic_checkpoints)
        if keep <= 0:
            return
        periodic: list[tuple[int, Path]] = []
        for candidate in checkpoint_root.glob("step_*"):
            if not candidate.is_dir():
                continue
            try:
                step = int(candidate.name.removeprefix("step_"))
            except ValueError:
                continue
            periodic.append((step, candidate))
        for _, candidate in sorted(periodic)[:-keep]:
            shutil.rmtree(candidate)

    def _save_method_state(self, path: Path) -> None:
        state = self.zo_method.state_dict()
        if _contains_tensor(state):
            torch.save(state, path / "method_state.pt")
            return
        write_json_atomic(path / "method_state.json", make_json_safe(state))

    def _resume_if_requested(self) -> dict[str, Any] | None:
        checkpoint_value = self.config.trainer.resume_from_checkpoint
        if checkpoint_value is None:
            return None
        checkpoint = Path(checkpoint_value).expanduser()
        zo_state_path = checkpoint / "zo_state.json"
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"resume checkpoint directory does not exist: {checkpoint}"
            )
        if not zo_state_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint is missing required zo_state.json: {zo_state_path}"
            )
        zo_state = _read_json_object(zo_state_path)
        current_step = int(zo_state.get("current_step", 0))
        if current_step < 0:
            raise ValueError(
                f"resume checkpoint has invalid current_step={current_step}"
            )
        max_steps = int(self.config.trainer.max_steps)
        if current_step > max_steps:
            raise ValueError(
                "resume checkpoint current_step exceeds trainer.max_steps: "
                f"{current_step} > {max_steps}"
            )
        self._load_backend_checkpoint(checkpoint)
        self._load_method_state(checkpoint)
        self.start_step = current_step
        self._advance_train_batch_stream(steps=current_step)
        info = {
            "enabled": True,
            "checkpoint": str(checkpoint),
            "start_step": int(current_step),
            "next_step": int(current_step) + 1,
            "max_steps": max_steps,
            "loaded_model": True,
            "loaded_method_state": True,
            "batch_stream_advanced": int(current_step),
        }
        self.artifact_writer.warnings.append(
            "resuming HF ZORegular run from "
            f"{checkpoint} at completed step {current_step}; next step is "
            f"{current_step + 1}"
        )
        return info

    def _load_backend_checkpoint(self, checkpoint: Path) -> None:
        load_checkpoint = getattr(self.backend, "load_checkpoint", None)
        if callable(load_checkpoint):
            load_checkpoint(checkpoint)
            self.model = self.backend.model
            self.tokenizer = self.backend.tokenizer
            self.model.to(self.device)
            return
        model_state = checkpoint / "model.pt"
        if not model_state.is_file():
            raise FileNotFoundError(
                "resume checkpoint cannot be loaded by this backend and is "
                f"missing fallback model.pt: {model_state}"
            )
        state_dict = torch.load(model_state, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def _load_method_state(self, checkpoint: Path) -> None:
        tensor_state = checkpoint / "method_state.pt"
        json_state = checkpoint / "method_state.json"
        if tensor_state.is_file():
            state = torch.load(tensor_state, map_location=self.device)
        elif json_state.is_file():
            state = _read_json_object(json_state)
        else:
            raise FileNotFoundError(
                "resume checkpoint is missing method state; expected "
                f"{tensor_state} or {json_state}"
            )
        self.zo_method.load_state_dict(state)

    def _load_existing_run_history(self) -> None:
        history_json = self.output_dir / "history.json"
        eval_json = self.output_dir / "eval_history.json"
        if history_json.is_file():
            history = json.loads(history_json.read_text(encoding="utf-8"))
            if isinstance(history, list):
                self.history = [dict(row) for row in history if isinstance(row, dict)]
        elif (self.output_dir / "history.jsonl").is_file():
            rows: list[dict[str, Any]] = []
            for line in (
                (self.output_dir / "history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            self.history = rows
        if eval_json.is_file():
            eval_history = json.loads(eval_json.read_text(encoding="utf-8"))
            if isinstance(eval_history, list):
                self.eval_history = [
                    dict(row) for row in eval_history if isinstance(row, dict)
                ]
        if self.resume_info is not None:
            checkpoint_step = int(self.resume_info["start_step"])
            self.history = [
                row
                for row in self.history
                if int(row.get("step", checkpoint_step + 1)) <= checkpoint_step
            ]
            self.eval_history = [
                row
                for row in self.eval_history
                if int(row.get("step", checkpoint_step + 1)) <= checkpoint_step
            ]

    def _advance_train_batch_stream(self, *, steps: int) -> None:
        if steps <= 0 or self.objective_name != "source_sst2_candidate_scoring":
            return
        for _ in range(int(steps)):
            self._sample_source_train_batch(step=0)

    def _zo_state(
        self,
        *,
        current_step: int,
        f_plus: float | None,
        f_minus: float | None,
        perturbation_seed: int | None = None,
    ) -> dict[str, Any]:
        if perturbation_seed is None:
            perturbation_seed = self._build_zo_step_seed_sequence(
                self.config,
                steps=int(current_step),
            )[int(current_step) - 1]
        learning_rate = _scheduled_learning_rate(
            self.config,
            step_number=max(1, int(current_step)),
        )
        return {
            "current_step": int(current_step),
            "base_model_name": self.config.backend.model_name,
            "objective": self.objective_name,
            "zo_method": self.zo_method.name,
            "estimator": self.config.zo.estimator,
            "learning_rate": float(learning_rate),
            "base_learning_rate": float(self.config.zo.learning_rate),
            "lr_schedule": str(self.config.zo.lr_schedule),
            "eps": float(self.config.zo.eps),
            "data_seed": int(self.config.data.seed),
            "perturbation_seed": int(perturbation_seed),
            "last_candidate_objective_values": {
                "plus": f_plus,
                "minus": f_minus,
            },
        }

    def _manifest(self) -> dict[str, Any]:
        return {
            "task": self.config.data.task,
            "backend": self.config.backend.name,
            "model": self.config.backend.model_name,
            "objective": self.objective_name,
            "zo_method": self.zo_method.name,
            "estimator": self.config.zo.estimator,
            "parameter_scope": self.config.zo.parameter_scope,
            "logprob_backend": "hf",
            "trainer": "hf_zoregular",
            "objective_protocol": self.objective_protocol["name"],
            "objective_direction": "minimize",
            "logprob_granularity": self.objective_protocol["logprob_granularity"],
            "protocol": dict(self.objective_protocol),
            "parameter_metadata": parameter_selection_metadata(
                self.model,
                self.config.zo.parameter_scope,
            ),
        }

    def _samples_checksum(self, samples: list[Any]) -> str:
        payload = [
            {
                "data": getattr(sample, "data", {}),
                "candidates": getattr(sample, "candidates", None),
                "correct_candidate": getattr(sample, "correct_candidate", None),
            }
            for sample in samples
        ]
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _sample_indices(self, samples: list[Any]) -> list[Any]:
        indices: list[Any] = []
        for sample in samples:
            data = getattr(sample, "data", {})
            if isinstance(data, dict):
                indices.append(data.get("idx"))
            else:
                indices.append(None)
        return indices

    def _input_trace(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): self._tensor_trace(value)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

    @staticmethod
    def _tensor_trace(value: torch.Tensor) -> dict[str, Any]:
        tensor = value.detach().cpu().contiguous()
        metadata = {
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        flat = tensor.reshape(-1)
        if tensor.is_floating_point():
            total: int | float = float(tensor.float().sum().item())
        else:
            total = int(tensor.long().sum().item())
        return {
            **metadata,
            "checksum": digest.hexdigest(),
            "sum": total,
            "first_values": flat[:16].tolist(),
        }

    def _reset_owned_artifacts(self) -> None:
        for filename in (
            "candidate_results.jsonl",
            "config.json",
            "eval_history.json",
            "history.json",
            "manifest.json",
            "results.json",
            "runtime_summary.json",
            "warnings.json",
        ):
            path = self.output_dir / filename
            if path.exists():
                path.unlink()
        checkpoints = self.output_dir / "checkpoints"
        if checkpoints.exists():
            shutil.rmtree(checkpoints)


def _candidate_suffix_logprob_scores(
    logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    option_len: torch.Tensor,
    average_by_option_len: bool = False,
    padding_side: str = "right",
) -> torch.Tensor:
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    next_tokens = input_ids[:, 1:]
    if padding_side == "left":
        token_positions = torch.arange(next_tokens.shape[1], device=input_ids.device)
        suffix_start = next_tokens.shape[1] - option_len
        suffix_mask = token_positions.unsqueeze(0) >= suffix_start.unsqueeze(1)
        suffix_mask &= attention_mask[:, 1:].bool()
    else:
        actual_lengths = attention_mask.sum(dim=1)
        token_positions = (
            torch.arange(next_tokens.shape[1], device=input_ids.device) + 1
        )
        suffix_start = actual_lengths - option_len
        suffix_mask = (token_positions.unsqueeze(0) >= suffix_start.unsqueeze(1)) & (
            token_positions.unsqueeze(0) < actual_lengths.unsqueeze(1)
        )

    # Only answer-suffix positions contribute to candidate scores. Selecting
    # those rows before log_softmax avoids materializing an FP32
    # [batch, sequence, vocabulary] tensor for prompt positions whose scores
    # are discarded. The per-token vocabulary reduction remains unchanged.
    selected_logits = logits[:, :-1, :][suffix_mask]
    selected_tokens = next_tokens[suffix_mask]
    selected_token_logprobs = (
        F.log_softmax(selected_logits, dim=-1)
        .gather(-1, selected_tokens.unsqueeze(-1))
        .squeeze(-1)
    )
    token_logprobs = selected_token_logprobs.new_zeros(next_tokens.shape)
    token_logprobs[suffix_mask] = selected_token_logprobs
    scores = (token_logprobs * suffix_mask.to(token_logprobs.dtype)).sum(dim=1)
    if average_by_option_len:
        return scores / suffix_mask.sum(dim=1).clamp_min(1).to(scores.dtype)
    return scores


def _validate_hf_zoregular_train_config(config: ExperimentConfig) -> None:
    if config.backend.name != "hf":
        raise ValueError("HF ZORegular training requires backend.name='hf'")
    if not is_zoregular_task(config.data.task):
        raise ValueError("HF ZORegular training requires a ZORegular data.task")
    if str(config.zo.estimator) not in {"mezo_two_point", "mezo_population"}:
        raise ValueError(
            "HF ZORegular training supports only zo.estimator='mezo_two_point' "
            "or 'mezo_population'"
        )
    if config.zo.parameter_scope != "full_parameters":
        method = str(getattr(config.zo, "method", "mezo"))
        if method in {"hizoo", "agzo"}:
            raise ValueError(
                "HF ZORegular "
                f"zo.method='{method}' requires "
                "zo.parameter_scope='full_parameters'"
            )
        raise ValueError(
            "HF ZORegular training requires zo.parameter_scope='full_parameters'"
        )


def _validate_zoregular_task(task: Any) -> None:
    missing = [
        name
        for name in ("template", "load_eval", "is_generation")
        if not hasattr(task, name)
    ]
    if missing:
        raise TypeError(
            "HFZORegularTrainer requires a ZORegular-like task with attributes: "
            f"template, load_eval, is_generation; missing {', '.join(missing)}"
        )
    if not (
        callable(getattr(task, "sample_train_batch", None))
        or callable(getattr(task, "load_train", None))
    ):
        raise TypeError(
            "HFZORegularTrainer requires sample_train_batch() or load_train()"
        )
    if not callable(getattr(task, "load_eval")):
        raise TypeError("ZORegular-like task load_eval must be callable")


def _validate_objective_matches_task(config: ExperimentConfig, task: Any) -> None:
    if (
        config.objective.name == "source_sst2_candidate_scoring"
        and config.data.task != "sst2"
    ):
        raise ValueError(
            "HF ZORegular objective.name='source_sst2_candidate_scoring' is "
            "supported only for SST-2 data.task='sst2'"
        )
    expected_names = _objective_names(task)
    if (
        config.data.task in {"copa", "record"}
        and "zoregular_generation_ce" not in expected_names
    ):
        expected_names = (*expected_names, "zoregular_generation_ce")
    if config.objective.name not in expected_names:
        expected = "' or '".join(expected_names)
        raise ValueError(
            f"HF ZORegular objective.name must be '{expected}' for "
            f"data.task={config.data.task!r}; got {config.objective.name!r}"
        )


def _objective_names(task: Any) -> tuple[str, ...]:
    if _is_generation_task(task):
        return ("zoregular_generation_ce",)
    return ("zoregular_classification_ce", "source_sst2_candidate_scoring")


def _is_generation_task(task: Any) -> bool:
    is_generation = getattr(task, "is_generation")
    if callable(is_generation):
        return bool(is_generation())
    return bool(is_generation)


def _candidate_result_dict(
    result: dict[str, Any],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    value = _objective_value(result)
    candidate_protocol = dict(protocol)
    if candidate_protocol.get("name") == "zoregular_supervised_ce":
        candidate_protocol["name"] = "zoregular_supervised"
    return {
        "candidate_id": str(result.get("candidate_id", "unknown")),
        "objective_value": value,
        "ok": value is not None and result.get("error_state") is None,
        "finite": value is not None,
        "error_state": result.get("error_state"),
        "metrics": dict(result.get("metrics", {})),
        "protocol": candidate_protocol,
        "perturbation_seed": int(result.get("perturbation_seed", 0)),
        "eps": float(result.get("eps", 0.0)),
        "scaling": float(result.get("scaling", 0.0)),
        "elapsed_time": float(result.get("elapsed_time", 0.0)),
    }


def _objective_calls(
    method_diagnostics: dict[str, Any],
    candidate_results: list[dict[str, Any]],
) -> int:
    value = method_diagnostics.get("objective_calls")
    if value is None:
        return len(candidate_results)
    return int(value)


def _scheduled_learning_rate(
    config: ExperimentConfig,
    *,
    step_number: int,
) -> float:
    base_lr = float(config.zo.learning_rate)
    schedule = str(config.zo.lr_schedule)
    if schedule == "constant":
        return base_lr

    max_steps = max(1, int(config.trainer.max_steps))
    warmup_steps = max(0, int(config.zo.lr_warmup_steps))
    step_number = max(1, int(step_number))
    min_ratio = float(config.zo.lr_schedule_min_ratio)

    if warmup_steps > 0 and step_number <= warmup_steps:
        return base_lr * (float(step_number) / float(warmup_steps))

    decay_start = warmup_steps + 1
    decay_steps = max(1, max_steps - warmup_steps)
    decay_position = max(0, step_number - decay_start)
    progress = min(1.0, float(decay_position) / float(max(1, decay_steps - 1)))
    if schedule == "linear_decay":
        multiplier = min_ratio + (1.0 - min_ratio) * (1.0 - progress)
    elif schedule == "cosine_decay":
        multiplier = min_ratio + (1.0 - min_ratio) * (
            0.5 * (1.0 + math.cos(math.pi * progress))
        )
    elif schedule == "step_decay":
        interval = max(1, int(config.zo.lr_schedule_step_interval))
        factor = float(config.zo.lr_schedule_step_factor)
        decay_count = max(0, (step_number - 1) // interval)
        multiplier = max(min_ratio, factor**decay_count)
    elif schedule == "milestone_step_decay":
        factor = float(config.zo.lr_schedule_step_factor)
        milestones = [
            int(value.strip())
            for value in str(config.zo.lr_schedule_milestones).split(",")
            if value.strip()
        ]
        decay_count = sum(1 for milestone in milestones if step_number >= milestone)
        multiplier = max(min_ratio, factor**decay_count)
    else:
        raise ValueError(f"Unsupported zo.lr_schedule: {schedule!r}")
    return base_lr * multiplier


def _scheduled_eps(
    config: ExperimentConfig,
    *,
    step_number: int,
) -> float:
    base_eps = float(config.zo.eps)
    schedule = str(config.zo.eps_schedule)
    if schedule == "constant":
        return base_eps

    schedule_total_steps = config.zo.eps_schedule_total_steps
    max_steps = max(
        1,
        int(
            config.trainer.max_steps
            if schedule_total_steps is None
            else schedule_total_steps
        ),
    )
    step_number = max(1, int(step_number))
    min_ratio = float(config.zo.eps_schedule_min_ratio)
    progress = min(
        1.0,
        float(step_number - 1) / float(max(1, max_steps - 1)),
    )
    if schedule == "linear_decay":
        multiplier = min_ratio + (1.0 - min_ratio) * (1.0 - progress)
    elif schedule == "cosine_decay":
        multiplier = min_ratio + (1.0 - min_ratio) * (
            0.5 * (1.0 + math.cos(math.pi * progress))
        )
    elif schedule == "step_decay":
        interval = max(1, int(config.zo.eps_schedule_step_interval))
        factor = float(config.zo.eps_schedule_step_factor)
        decay_count = max(0, (step_number - 1) // interval)
        multiplier = max(min_ratio, factor**decay_count)
    else:
        raise ValueError(f"Unsupported zo.eps_schedule: {schedule!r}")
    return base_eps * multiplier


def _set_zo_method_learning_rate(method: Any, learning_rate: float) -> None:
    config = getattr(method, "config", None)
    if config is None:
        return
    method.config = replace(config, learning_rate=float(learning_rate))


def _set_zo_method_eps(method: Any, eps: float) -> None:
    config = getattr(method, "config", None)
    if config is None:
        return
    method.config = replace(config, eps=float(eps))


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _candidate_failure_error(
    *,
    step: int,
    candidate_results: list[dict[str, Any]],
) -> dict[str, Any]:
    failed = [
        result for result in candidate_results if not bool(result.get("ok", False))
    ]
    return {
        "type": "CandidateObjectiveError",
        "message": "candidate objective failed",
        "step": int(step),
        "failed_candidates": [
            str(result.get("candidate_id", "unknown")) for result in failed
        ],
        "candidate_results": failed,
    }


def _candidate_failure_message(error: dict[str, Any]) -> str:
    candidates = ", ".join(error.get("failed_candidates", [])) or "unknown"
    return (
        f"candidate objective failed at step {int(error.get('step', 0))}: {candidates}"
    )


def _objective_value(result: dict[str, Any] | None) -> float | None:
    if result is None:
        return None
    value = result.get("objective_value")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _best_finite_metric_eval(
    eval_history: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
) -> tuple[dict[str, Any] | None, float | None]:
    best_eval: dict[str, Any] | None = None
    best_value: float | None = None
    for row in eval_history:
        value = _finite_metric(row, metric)
        if value is None:
            continue
        if _metric_is_better(
            row,
            value=value,
            best_eval=best_eval,
            best_value=best_value,
            mode=mode,
        ):
            best_value = value
            best_eval = dict(row)
    return best_eval, best_value


def _finite_metric(row: dict[str, Any], metric: str) -> float | None:
    value = row.get(metric)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return None
    return float(value)


def _metric_is_better(
    row: dict[str, Any],
    *,
    value: float,
    best_eval: dict[str, Any] | None,
    best_value: float | None,
    mode: str,
) -> bool:
    if best_eval is None or best_value is None:
        return True
    if mode == "max" and value > best_value:
        return True
    if mode == "min" and value < best_value:
        return True
    if value != best_value:
        return False
    loss = _finite_metric(row, "loss")
    best_loss = _finite_metric(best_eval, "loss")
    return loss is not None and (best_loss is None or loss < best_loss)


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return float(sum((value - mean_value) ** 2 for value in values) / len(values))


def _pad_token_id(tokenizer: Any) -> int:
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", 0)
    return int(token_id if token_id is not None else 0)


def _eval_batch_size(batch_size: int, num_rows: int) -> int:
    size = int(batch_size)
    if size <= 0:
        return max(1, int(num_rows))
    return size


def _batched(rows: list[Any], batch_size: int) -> Iterator[list[Any]]:
    size = max(1, int(batch_size))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _summarize_runtime(history: list[dict[str, Any]]) -> dict[str, Any]:
    total_step_time = sum(float(row.get("step_time", 0.0)) for row in history)
    return {
        "candidate_objective_calls": sum(
            len(row.get("candidate_results", [])) for row in history
        ),
        "total_step_time": total_step_time,
        "mean_step_time": total_step_time / len(history) if history else 0.0,
    }
