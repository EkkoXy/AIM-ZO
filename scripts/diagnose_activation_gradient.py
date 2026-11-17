"""Offline activation/gradient diagnostic; never updates model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch


def split_indices(count: int, basis_count: int, gradient_count: int, seed: int):
    if min(basis_count, gradient_count) < 1 or count < basis_count + gradient_count:
        raise ValueError("insufficient rows for positive, disjoint reference pools")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return indices[:basis_count], indices[basis_count:basis_count + gradient_count]


def activation_basis(tokens: torch.Tensor, rank: int) -> torch.Tensor:
    x = tokens.detach().float()
    if x.ndim != 2 or not torch.isfinite(x).all() or rank < 1:
        raise ValueError("expected finite activation matrix and positive rank")
    if min(x.shape) < rank:
        raise ValueError("activation matrix is smaller than requested rank")
    options = {"driver": "gesvd"} if x.is_cuda else {}
    _, singular, vh = torch.linalg.svd(x, full_matrices=False, **options)
    tolerance = singular[0] * max(x.shape) * torch.finfo(x.dtype).eps
    if int((singular > tolerance).sum()) < rank:
        raise ValueError("activation numerical rank is below requested rank")
    return vh[:rank].T.contiguous()


def coverage(gradient: torch.Tensor, basis: torch.Tensor) -> float | None:
    g, q = gradient.detach().float(), basis.detach().float()
    if g.ndim != 2 or q.ndim != 2 or g.shape[1] != q.shape[0]:
        raise ValueError("gradient and right basis have incompatible shapes")
    if not torch.isfinite(g).all() or not torch.isfinite(q).all():
        raise ValueError("nonfinite gradient or basis")
    energy = g.square().sum()
    if energy.item() == 0:
        return None
    return float(((g @ q).square().sum() / energy).item())


def random_basis(dimension: int, rank: int, seed: int) -> torch.Tensor:
    if not 1 <= rank <= dimension:
        raise ValueError("random basis rank must be within input dimension")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.linalg.qr(torch.randn(dimension, rank, generator=generator), mode="reduced").Q


def valid_activation_tokens(activation, mask, quota):
    if quota < 1 or mask is None or activation.numel() // activation.shape[-1] != mask.numel():
        raise ValueError("invalid quota or activation/attention mask shape mismatch")
    x = activation.detach().reshape(-1, activation.shape[-1])[mask.reshape(-1).bool()]
    if len(x) > quota:
        positions = torch.linspace(0, len(x) - 1, quota, device=x.device).long()
        x = x[positions]
    return x.float().cpu()


def reference_indices(samples, task, basis_count, gradient_count, seed):
    if task != "squad":
        return split_indices(len(samples), basis_count, gradient_count, seed)
    order = list(range(len(samples)))
    random.Random(seed).shuffle(order)
    a = order[:basis_count]
    contexts = {samples[i].data["context"] for i in a}
    b = [i for i in order[basis_count:] if samples[i].data["context"] not in contexts][:gradient_count]
    if len(a) != basis_count or len(b) != gradient_count or min(basis_count, gradient_count) < 1:
        raise ValueError("insufficient disjoint SQuAD context groups")
    return a, b


def reconstruct_bases(tokens, rank, device):
    bases, counts, reused = {}, {}, 0
    for name, rows in tokens.items():
        canonical = name
        if name.endswith((".k_proj", ".v_proj")):
            candidate = name.rsplit(".", 1)[0] + ".q_proj"
            if candidate in bases and len(tokens[candidate]) == len(rows):
                if all(torch.equal(a, b) for a, b in zip(tokens[candidate], rows)):
                    canonical = candidate
        counts[name] = sum(len(row) for row in rows)
        if canonical != name:
            bases[name] = bases[canonical]
            reused += 1
        else:
            bases[name] = activation_basis(torch.cat(rows).to(device), rank).cpu()
        if len(bases) % 16 == 0:
            print(f"SVD modules {len(bases)}/{len(tokens)}, reused={reused}", flush=True)
    return bases, counts, reused


def sample_loss(backend, task, sample, max_length: int, hook_mask):
    from aimzo.tasks.zoregular_scoring import (
        collate_zoregular_classification, collate_zoregular_generation,
        encode_zoregular_classification_sample, encode_zoregular_generation_sample,
        zoregular_classification_loss, zoregular_generation_loss,
    )
    from aimzo.trainers.hf_zoregular import _candidate_suffix_logprob_scores, _pad_token_id

    generation = task.is_generation
    encode = encode_zoregular_generation_sample if generation else encode_zoregular_classification_sample
    collate = collate_zoregular_generation if generation else collate_zoregular_classification
    feature = encode(task.template, sample, backend.tokenizer, max_length=max_length)
    batch = collate([feature], pad_token_id=_pad_token_id(backend.tokenizer))
    ids = batch["input_ids"].to(backend.device)
    mask = batch["attention_mask"].to(backend.device)
    hook_mask[0] = mask
    logits = backend.model(input_ids=ids, attention_mask=mask).logits
    if generation:
        return zoregular_generation_loss(logits, batch["labels"].to(backend.device))
    scores = _candidate_suffix_logprob_scores(
        logits, input_ids=ids, attention_mask=mask,
        option_len=batch["option_len"].to(backend.device),
    )
    return zoregular_classification_loss(
        scores, batch["labels"].to(backend.device),
        batch["num_options"].to(backend.device), gold_mask=batch["gold_mask"].to(backend.device),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task", required=True, choices=["rte", "sst2", "squad"])
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--modules", required=True, nargs="+")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--basis-samples", type=int, default=32)
    parser.add_argument("--gradient-samples", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", default="activation-gradient")
    parser.add_argument("--compute-device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is not authorized")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    if args.max_tokens < args.basis_samples or args.max_length < 2:
        raise ValueError("invalid token budget or sequence length")
    from aimzo.backends.hf import HFBackend
    from aimzo.tasks.zoregular import ZORegularTask

    task = ZORegularTask(task_name=args.task, data_root=args.data_root, seed=args.seed)
    samples = task.load_train()
    a, b = reference_indices(samples, args.task, args.basis_samples, args.gradient_samples, args.seed)
    sample_payload = [vars(sample) for sample in samples]
    data_hash = hashlib.sha256(json.dumps(sample_payload, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".references.json").open("x") as file:
        json.dump(dict(data_sha256=data_hash, basis_indices=a, gradient_indices=b), file, indent=2)
    start = time.monotonic()
    print(f"loading {args.stage} {args.task}", flush=True)
    backend = HFBackend(model_name=str(args.checkpoint), dtype="bfloat16", parameter_scope="full_parameters")
    model = backend.model.eval()
    model.config.use_cache = False
    modules = dict(model.named_modules())
    if len(set(args.modules)) != len(args.modules):
        raise ValueError("duplicate modules")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name in args.modules:
        if name not in modules or not isinstance(modules[name], torch.nn.Linear):
            raise ValueError(f"expected linear module: {name}")
        modules[name].weight.requires_grad_(True)
    tokens = {name: [] for name in args.modules}
    hook_mask = [None]
    quota = args.max_tokens // args.basis_samples

    def hook_for(name):
        def hook(module, inputs, output):
            del module, output
            tokens[name].append(valid_activation_tokens(inputs[0], hook_mask[0], quota))
        return hook

    handles = [modules[name].register_forward_hook(hook_for(name)) for name in args.modules]
    try:
        with torch.no_grad():
            for index in a:
                sample_loss(backend, task, samples[index], args.max_length, hook_mask)
    finally:
        for handle in handles:
            handle.remove()
    print("activation forwards complete; constructing SVD bases", flush=True)
    bases, token_counts, reused = reconstruct_bases(tokens, args.rank, args.compute_device)
    print("SVD bases complete; measuring gradients", flush=True)
    del tokens
    gradients = {name: torch.zeros_like(modules[name].weight, dtype=torch.float32, device="cpu") for name in args.modules}
    losses = []
    for index in b:
        model.zero_grad(set_to_none=True)
        loss = sample_loss(backend, task, samples[index], args.max_length, hook_mask)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite CE")
        loss.backward()
        losses.append(float(loss.detach()))
        for name in args.modules:
            gradient = modules[name].weight.grad
            if gradient is None:
                raise ValueError(f"missing gradient: {name}")
            gradients[name].add_(gradient.detach().float().cpu(), alpha=1 / len(b))
        if len(losses) % 8 == 0:
            print(f"gradient samples {len(losses)}/{len(b)} CE={sum(losses)/len(losses):.6f}", flush=True)
    rows, control_cache = [], {}
    for name, gradient in gradients.items():
        dimension = gradient.shape[1]
        if dimension not in control_cache:
            control_cache[dimension] = [random_basis(dimension, args.rank, args.seed + 1000 + i).to(args.compute_device) for i in range(5)]
        g = gradient.to(args.compute_device)
        actual = coverage(g, bases[name].to(args.compute_device))
        controls = [coverage(g, q) for q in control_cache[dimension]]
        mean_random = None if controls[0] is None else sum(controls) / len(controls)
        rows.append(dict(module=name, input_dimension=gradient.shape[1], rank=args.rank,
                         activation_tokens=token_counts[name], gradient_energy=float(gradient.square().sum()),
                         activation_coverage=actual, random_coverage=controls,
                         excess_coverage=None if actual is None else actual - mean_random))
        if len(rows) % 16 == 0:
            print(f"coverage modules {len(rows)}/{len(gradients)}", flush=True)
    result = dict(status="complete", study="activation-gradient", protocol=args.protocol, checkpoint=str(args.checkpoint.resolve()),
                  stage=args.stage, task=args.task, seed=args.seed, data_sha256=data_hash,
                  entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reused_attention_bases=reused,
                  basis_indices=a, gradient_indices=b, definition="coverage of mean gradient by fresh uncentered activation SVD",
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  rows=rows, mean_ce=sum(losses) / len(losses), training_steps=0,
                  sample_objective_calls=len(a) + len(b), backward_calls=len(b),
                  wall_seconds=time.monotonic() - start,
                  peak_memory_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
