from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ZO_REGULAR_CLASSIFICATION_TASKS = (
    "sst2",
    "rte",
    "boolq",
    "cb",
    "wic",
    "wsc",
    "multirc",
    "copa",
    "record",
)
ZO_REGULAR_GENERATION_TASKS = ("squad", "drop")
ZO_REGULAR_TASKS = ZO_REGULAR_CLASSIFICATION_TASKS + ZO_REGULAR_GENERATION_TASKS
ZO_REGULAR_SAMPLING_POLICIES = {
    "split",
    "mezo_train_dev",
    "multirc_grouped_shuffled",
}


@dataclass(frozen=True)
class ZORegularSample:
    data: dict[str, Any]
    candidates: list[Any] | None
    correct_candidate: Any


@dataclass(frozen=True)
class ZORegularTaskSpec:
    name: str
    data_subdir: str
    is_generation: bool
    build_sample: Callable[[dict[str, Any]], ZORegularSample]
    template_cls: type


def _require_answers(task_name: str, answers: list[Any]) -> list[Any]:
    if not answers:
        raise ValueError(f"{task_name} example must include non-empty answers")
    return answers


def _first_answer(candidate: Any) -> Any:
    if isinstance(candidate, list):
        return candidate[0]
    return candidate


def _bs_sst2(ex: dict[str, Any]) -> ZORegularSample:
    return ZORegularSample(
        data=dict(ex),
        candidates=[0, 1],
        correct_candidate=int(ex["label"]),
    )


def _bs_rte(ex: dict[str, Any]) -> ZORegularSample:
    data = dict(ex)
    if "premise" not in data and "sentence1" in data:
        data["premise"] = data["sentence1"]
    if "hypothesis" not in data and "sentence2" in data:
        data["hypothesis"] = data["sentence2"]
    return ZORegularSample(
        data=data,
        candidates=[0, 1],
        correct_candidate=int(ex["label"]),
    )


def _bs_boolq(ex: dict[str, Any]) -> ZORegularSample:
    raw_answer = ex["answer"] if "answer" in ex else ex["label"]
    answer = "Yes" if bool(raw_answer) else "No"
    return ZORegularSample(
        data=dict(ex),
        candidates=["Yes", "No"],
        correct_candidate=answer,
    )


def _bs_cb(ex: dict[str, Any]) -> ZORegularSample:
    return ZORegularSample(
        data=dict(ex),
        candidates=[0, 1, 2],
        correct_candidate=int(ex["label"]),
    )


def _bs_wic(ex: dict[str, Any]) -> ZORegularSample:
    return ZORegularSample(
        data=dict(ex),
        candidates=[0, 1],
        correct_candidate=int(ex["label"]),
    )


def _bs_wsc(ex: dict[str, Any]) -> ZORegularSample:
    return ZORegularSample(
        data=dict(ex),
        candidates=[0, 1],
        correct_candidate=int(ex["label"]),
    )


def _bs_multirc(ex: dict[str, Any]) -> ZORegularSample:
    return ZORegularSample(
        data=dict(ex),
        candidates=[0, 1],
        correct_candidate=int(ex["label"]),
    )


def _bs_copa(ex: dict[str, Any]) -> ZORegularSample:
    candidates = [ex["choice1"], ex["choice2"]]
    return ZORegularSample(
        data=dict(ex),
        candidates=candidates,
        correct_candidate=ex[f"choice{int(ex['label']) + 1}"],
    )


def _bs_record(ex: dict[str, Any]) -> ZORegularSample:
    entities = list(ex["entities"])
    answers = list(ex["answers"])
    if not entities:
        raise ValueError(
            f"ReCoRD record requires non-empty entities; "
            f"answers={answers!r}, entities={entities!r}"
        )
    if not any(answer in entities for answer in answers):
        raise ValueError(
            f"ReCoRD record answers must appear in entities; "
            f"answers={answers!r}, entities={entities!r}"
        )
    return ZORegularSample(
        data=dict(ex),
        candidates=entities,
        correct_candidate=answers,
    )


def _bs_squad(ex: dict[str, Any]) -> ZORegularSample:
    answers = _require_answers("SQuAD", list(ex["answers"]["text"]))
    return ZORegularSample(
        data={
            "title": ex["title"],
            "context": ex["context"],
            "question": ex["question"],
            "answers": answers,
        },
        candidates=None,
        correct_candidate=answers,
    )


def _bs_drop(ex: dict[str, Any]) -> ZORegularSample:
    answers = _require_answers("DROP", list(ex["answers_spans"]["spans"]))
    return ZORegularSample(
        data={
            "context": ex["passage"],
            "question": ex["question"],
            "answers": answers,
        },
        candidates=None,
        correct_candidate=answers,
    )


class SST2Template:
    verbalizer = {0: "terrible", 1: "great"}

    def encode(self, sample: ZORegularSample) -> str:
        return f"{sample.data['sentence'].strip()} It was"

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return f"{self.encode(sample)} {self.verbalizer[candidate]}"


class RTETemplate:
    verbalizer = {0: "Yes", 1: "No"}

    def encode(self, sample: ZORegularSample) -> str:
        return (
            f'{sample.data["premise"]}\nDoes this mean that '
            f'"{sample.data["hypothesis"]}" is true? Yes or No?\n'
        )

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return self.encode(sample) + self.verbalizer[candidate]


class BoolQTemplate:
    verbalizer = None

    def encode(self, sample: ZORegularSample) -> str:
        question = sample.data["question"]
        if not question.endswith("?"):
            question += "?"
        question = question[0].upper() + question[1:]
        return f"{sample.data['passage']} {question}\n"

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return self.encode(sample) + str(candidate)


class CBTemplate:
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def encode(self, sample: ZORegularSample) -> str:
        return (
            f'Suppose {sample.data["premise"]} Can we infer that '
            f'"{sample.data["hypothesis"]}"? Yes, No, or Maybe?\n'
        )

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return self.encode(sample) + self.verbalizer[candidate]


class WICTemplate:
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample: ZORegularSample) -> str:
        return (
            f'Does the word "{sample.data["word"]}" have the same meaning in these '
            f"two sentences? Yes, No?\n{sample.data['sentence1']}\n"
            f"{sample.data['sentence2']}\n"
        )

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return self.encode(sample) + self.verbalizer[candidate]


class WSCTemplate:
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample: ZORegularSample) -> str:
        return (
            f'{sample.data["text"]}\nIn the previous sentence, does the pronoun '
            f'"{sample.data["span2_text"].lower()}" refer to '
            f"{sample.data['span1_text']}? Yes or No?\n"
        )

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return self.encode(sample) + self.verbalizer[candidate]


class MultiRCTemplate:
    verbalizer = {0: "No", 1: "Yes"}

    def encode(self, sample: ZORegularSample) -> str:
        return (
            f'{sample.data["paragraph"]}\nQuestion: {sample.data["question"]}\n'
            f'I found this answer "{sample.data["answer"]}". Is that correct? '
            f"Yes or No?\n"
        )

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return self.encode(sample) + self.verbalizer[candidate]


class CopaTemplate:
    verbalizer = None

    def _conj(self, sample: ZORegularSample) -> str:
        if sample.data["question"] == "effect":
            return " so "
        if sample.data["question"] == "cause":
            return " because "
        raise ValueError(f"unknown COPA question type: {sample.data['question']}")

    def encode(self, sample: ZORegularSample) -> str:
        premise = sample.data["premise"].rstrip()
        if premise.endswith("."):
            premise = premise[:-1]
        return premise + self._conj(sample)

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        words = str(candidate).split(" ")
        if words and words[0] != "I":
            words[0] = words[0].lower()
        return self.encode(sample) + " ".join(words)


class ReCoRDTemplate:
    verbalizer = None

    def _passage(self, sample: ZORegularSample) -> str:
        return sample.data["passage"].replace("@highlight\n", "- ")

    def encode(self, sample: ZORegularSample) -> str:
        return f"{self._passage(sample)}\n-"

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        candidate_text = _first_answer(candidate)
        query = sample.data["query"].replace("@placeholder", str(candidate_text))
        return f"{self._passage(sample)}\n- {query}"


class SQuADTemplate:
    verbalizer = None
    generation = True

    def encode(self, sample: ZORegularSample) -> str:
        return (
            f"Title: {sample.data['title']}\nContext: {sample.data['context']}\n"
            f"Question: {sample.data['question'].strip()}\nAnswer:"
        )

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return f"{self.encode(sample)} {_first_answer(candidate)}\n"


class DROPTemplate:
    verbalizer = None
    generation = True

    def encode(self, sample: ZORegularSample) -> str:
        return (
            f"Passage: {sample.data['context']}\n"
            f"Question: {sample.data['question'].strip()}\nAnswer:"
        )

    def verbalize(self, sample: ZORegularSample, candidate: Any) -> str:
        return f"{self.encode(sample)} {_first_answer(candidate)}\n"


ZO_REGULAR_SPECS = {
    "sst2": ZORegularTaskSpec("sst2", "sst2", False, _bs_sst2, SST2Template),
    "rte": ZORegularTaskSpec("rte", "rte", False, _bs_rte, RTETemplate),
    "boolq": ZORegularTaskSpec("boolq", "boolq", False, _bs_boolq, BoolQTemplate),
    "cb": ZORegularTaskSpec("cb", "cb", False, _bs_cb, CBTemplate),
    "wic": ZORegularTaskSpec("wic", "wic", False, _bs_wic, WICTemplate),
    "wsc": ZORegularTaskSpec("wsc", "wsc", False, _bs_wsc, WSCTemplate),
    "multirc": ZORegularTaskSpec(
        "multirc",
        "multirc",
        False,
        _bs_multirc,
        MultiRCTemplate,
    ),
    "copa": ZORegularTaskSpec("copa", "copa", False, _bs_copa, CopaTemplate),
    "record": ZORegularTaskSpec("record", "record", False, _bs_record, ReCoRDTemplate),
    "squad": ZORegularTaskSpec("squad", "squad", True, _bs_squad, SQuADTemplate),
    "drop": ZORegularTaskSpec("drop", "drop", True, _bs_drop, DROPTemplate),
}


def is_zoregular_task(task_name: str) -> bool:
    return task_name in ZO_REGULAR_SPECS


def get_zoregular_spec(task_name: str) -> ZORegularTaskSpec:
    try:
        return ZO_REGULAR_SPECS[task_name]
    except KeyError as exc:
        raise ValueError(f"unknown ZORegular task {task_name!r}") from exc


def get_zoregular_template(task_name: str):
    return get_zoregular_spec(task_name).template_cls()


def build_zoregular_sample(task_name: str, example: dict[str, Any]) -> ZORegularSample:
    return get_zoregular_spec(task_name).build_sample(example)


def _cap_rows(
    rows: list[ZORegularSample],
    max_samples: int | None,
) -> list[ZORegularSample]:
    if max_samples is None:
        return rows
    return rows[: max(0, int(max_samples))]


def _shuffle_and_cap_rows(
    rows: list[ZORegularSample],
    max_samples: int | None,
    *,
    seed: int,
) -> list[ZORegularSample]:
    if max_samples is None:
        return rows
    shuffled = list(rows)
    random.Random(int(seed)).shuffle(shuffled)
    return shuffled[: max(0, int(max_samples))]


def _multirc_question_key(sample: ZORegularSample) -> tuple[int, int]:
    idx = sample.data.get("idx")
    if not isinstance(idx, dict) or "paragraph" not in idx or "question" not in idx:
        raise ValueError(
            "multirc_grouped_shuffled requires each MultiRC row to contain "
            "idx.paragraph and idx.question"
        )
    return int(idx["paragraph"]), int(idx["question"])


def _multirc_grouped_eval_rows(
    rows: list[ZORegularSample],
    max_samples: int | None,
    *,
    seed: int,
) -> list[ZORegularSample]:
    if max_samples is None:
        return rows
    limit = max(0, int(max_samples))
    if limit == 0:
        return []

    by_paragraph: dict[int, dict[tuple[int, int], list[ZORegularSample]]] = {}
    for sample in rows:
        key = _multirc_question_key(sample)
        by_paragraph.setdefault(key[0], {}).setdefault(key, []).append(sample)

    rng = random.Random(int(seed))
    paragraph_ids = list(by_paragraph)
    rng.shuffle(paragraph_ids)
    question_queues: dict[int, list[list[ZORegularSample]]] = {}
    for paragraph_id in paragraph_ids:
        groups = list(by_paragraph[paragraph_id].values())
        rng.shuffle(groups)
        question_queues[paragraph_id] = groups

    # Round-robin across paragraphs before taking a second question from any one
    # paragraph. This avoids recreating MultiRC's contiguous source ordering.
    ordered_groups: list[list[ZORegularSample]] = []
    while any(question_queues.values()):
        for paragraph_id in paragraph_ids:
            groups = question_queues[paragraph_id]
            if groups:
                ordered_groups.append(groups.pop())

    selected: list[ZORegularSample] = []
    for group in ordered_groups:
        if len(selected) + len(group) <= limit:
            selected.extend(group)
    return selected


class ZORegularTask:
    benchmark = "zoregular"

    def __init__(
        self,
        *,
        task_name: str,
        data_root: str | Path,
        max_train_samples: int | None = None,
        max_eval_samples: int | None = None,
        zoregular_train_dev_samples: int | None = None,
        seed: int = 42,
        sampling_policy: str = "split",
    ) -> None:
        if sampling_policy not in ZO_REGULAR_SAMPLING_POLICIES:
            allowed = ", ".join(sorted(ZO_REGULAR_SAMPLING_POLICIES))
            raise ValueError(
                f"ZORegularTask sampling_policy must be one of: {allowed}"
            )
        if sampling_policy == "multirc_grouped_shuffled" and task_name != "multirc":
            raise ValueError(
                "multirc_grouped_shuffled sampling_policy is only valid for multirc"
            )
        self.spec = get_zoregular_spec(task_name)
        self.task_name = task_name
        self.data_root = Path(data_root).expanduser()
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.zoregular_train_dev_samples = zoregular_train_dev_samples
        self.seed = seed
        self.sampling_policy = sampling_policy
        self._mezo_train_dev_rows: tuple[ZORegularSample, ...] | None = None

    @property
    def is_generation(self) -> bool:
        return self.spec.is_generation

    @property
    def template(self):
        return self.spec.template_cls()

    def load_train(self) -> list[ZORegularSample]:
        if self.sampling_policy == "mezo_train_dev":
            include_eval = (
                self.max_eval_samples is not None
                or self.zoregular_train_dev_samples is not None
            )
            return self._load_mezo_train_dev_rows(include_eval=include_eval)[
                : int(self.max_train_samples)
            ]
        if self.sampling_policy == "multirc_grouped_shuffled":
            return _shuffle_and_cap_rows(
                load_zoregular_split(self.task_name, self.data_root, "train"),
                self.max_train_samples,
                seed=self.seed,
            )
        return _cap_rows(
            load_zoregular_split(self.task_name, self.data_root, "train"),
            self.max_train_samples,
        )

    def load_eval(self) -> list[ZORegularSample]:
        if self.sampling_policy == "mezo_train_dev":
            if self.max_eval_samples is None:
                return load_zoregular_split(
                    self.task_name,
                    self.data_root,
                    "validation",
                )
            rows = self._load_mezo_train_dev_rows()
            train_count = int(self.max_train_samples)
            eval_count = int(self.max_eval_samples)
            return rows[train_count : train_count + eval_count]
        if self.sampling_policy == "multirc_grouped_shuffled":
            return _multirc_grouped_eval_rows(
                load_zoregular_split(self.task_name, self.data_root, "validation"),
                self.max_eval_samples,
                seed=self.seed + 1_000_003,
            )
        return _cap_rows(
            load_zoregular_split(self.task_name, self.data_root, "validation"),
            self.max_eval_samples,
        )

    def load_train_dev(self) -> list[ZORegularSample]:
        if self.sampling_policy != "mezo_train_dev":
            raise ValueError("train-dev rows require mezo_train_dev sampling_policy")
        if self.max_train_samples is None:
            raise ValueError("train-dev rows require max_train_samples")
        if self.zoregular_train_dev_samples is None and self.max_eval_samples is None:
            raise ValueError(
                "train-dev rows require zoregular_train_dev_samples or max_eval_samples"
            )
        train_count = int(self.max_train_samples)
        eval_count = (
            int(self.zoregular_train_dev_samples)
            if self.zoregular_train_dev_samples is not None
            else int(self.max_eval_samples)
        )
        rows = self._load_mezo_train_dev_rows(include_eval=True)
        return rows[train_count : train_count + eval_count]

    def _load_mezo_train_dev_rows(
        self,
        *,
        include_eval: bool = True,
    ) -> list[ZORegularSample]:
        if self._mezo_train_dev_rows is not None:
            return list(self._mezo_train_dev_rows)
        if self.max_train_samples is None:
            raise ValueError(
                "mezo_train_dev sampling_policy requires max_train_samples"
            )
        if (
            include_eval
            and self.max_eval_samples is None
            and self.zoregular_train_dev_samples is None
        ):
            raise ValueError(
                "mezo_train_dev sampling_policy requires max_eval_samples when "
                "slicing eval rows from the permuted train split; "
                "max_eval_samples=None uses the full validation split for eval"
            )
        train_count = int(self.max_train_samples)
        eval_count = (
            int(self.zoregular_train_dev_samples)
            if self.zoregular_train_dev_samples is not None
            else int(self.max_eval_samples)
            if include_eval
            else 0
        )
        total_count = train_count + eval_count
        rows = load_zoregular_split(self.task_name, self.data_root, "train")
        if total_count > len(rows):
            if not include_eval:
                raise ValueError(
                    "mezo_train_dev sampling_policy requires "
                    f"max_train_samples <= train split size ({len(rows)})"
                )
            raise ValueError(
                "mezo_train_dev sampling_policy requires max_train_samples + "
                f"max_eval_samples <= train split size ({len(rows)})"
            )
        selected = (
            np.random.RandomState(self.seed).permutation(len(rows)).tolist()[:total_count]
        )
        self._mezo_train_dev_rows = tuple(rows[index] for index in selected)
        return list(self._mezo_train_dev_rows)

    def sample_train_batch(
        self,
        *,
        batch_size: int,
        seed: int,
        step: int,
    ) -> list[ZORegularSample]:
        rows = self.load_train()
        if not rows:
            return []
        rng = random.Random(int(seed) + int(step))
        if batch_size <= len(rows):
            return rng.sample(rows, int(batch_size))
        return [rng.choice(rows) for _ in range(int(batch_size))]


def load_zoregular_split(
    task_name: str,
    data_root: str | Path,
    split: str,
) -> list[ZORegularSample]:
    spec = get_zoregular_spec(task_name)
    path = Path(data_root).expanduser() / spec.data_subdir
    if not path.exists():
        raise FileNotFoundError(
            f"missing ZORegular data directory for {task_name}: {path}"
        )
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            f"datasets is required to load ZORegular task {task_name} at {path}"
        ) from exc

    dataset = load_from_disk(str(path))
    if split not in dataset:
        raise ValueError(
            f"missing split {split!r} for ZORegular task {task_name} at {path}"
        )
    return [spec.build_sample(dict(example)) for example in dataset[split]]
