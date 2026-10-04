# semantic_guided_answer_finetune_v097.py
#
# LLM_SEM v0.9.7
# Semantic-Guided Answer Fine-Tuning
#
# Objective:
#   answer-only LM loss
# + semantic preservation loss on benchmark vectors
#
# Trainable:
#   final Transformer block
#   final_norm
#   lm_head (lower learning rate)
#
# Frozen:
#   embedding
#   earlier Transformer blocks

from __future__ import annotations

import argparse
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torch.nn.functional as F

from chat import build_semantic_generation_prompt
from model import LanguageModel
from semantic_eval import load_benchmark
from semantic_router import SemanticRouter
from semantic_intent_v034 import extract_purpose_intent
from semantic_proposition_v036 import extract_propositions, proposition_concepts
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-sem-consolidation-v081.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_DATASET = "data/semantic_guided_qa_v097.json"
DEFAULT_BENCHMARK = "my_benchmark.csv"
DEFAULT_OUTPUT = "model/model-sem-guided-answer-v097.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.9.7 Semantic-Guided Answer Fine-Tuning"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--lm-head-lr", type=float, default=1e-5)
    p.add_argument("--preserve-weight", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--holdout", type=float, default=0.20)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--train-blocks", type=int, default=1)
    p.add_argument("--min-generation-sim", type=float, default=0.0)
    p.add_argument("--result-json", default="")
    p.add_argument("--prefer-final-state", action="store_true")
    p.add_argument("--concept-balanced", action="store_true")
    p.add_argument("--protected-distill-weight", type=float, default=0.0)
    p.add_argument("--new-knowledge-weight", type=float, default=1.0)
    return p.parse_args()


def load_dataset(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for row in obj.get("samples", []):
        query = str(row.get("query", "")).strip()
        answer = str(row.get("answer", "")).strip()
        label = str(row.get("label", "")).strip()
        intent = str(row.get("intent", "")).strip()
        concepts = [str(x).strip() for x in row.get("concepts", []) if str(x).strip()]
        truth = str(row.get("truth_status", "UNVERIFIED")).strip().upper()
        if query and answer and label:
            rows.append({
                "query": query,
                "answer": answer,
                "label": label,
                "intent": intent or None,
                "concepts": concepts,
                "truth_status": truth,
                "sleep_source": str(row.get("sleep_source", row.get("source", "base"))),
                "sleep_weight": float(row.get("sleep_weight", 1.0)),
                "must_train": bool(row.get("must_train", False)),
                "protected": bool(row.get("protected", False)),
                "runtime_prompt": str(row.get("runtime_prompt", "")),
                "runtime_selected_label": str(row.get("runtime_selected_label", "")),
            })
    if not rows:
        raise RuntimeError("No valid QA rows.")
    return rows


def row_concept(row: dict) -> str:
    concepts = [str(x).strip() for x in row.get("concepts", []) if str(x).strip()]
    return concepts[0] if concepts else str(row.get("query", "")).strip()


def concept_balanced_qa_loss(model, tokenizer, rows, device):
    mandatory = [row for row in rows if bool(row.get("must_train", False))]
    optional = [
        row for row in rows
        if not bool(row.get("must_train", False))
        and not bool(row.get("protected", False))
    ]

    concept_groups: dict[str, list[torch.Tensor]] = {}
    for row in mandatory:
        concept_groups.setdefault(row_concept(row), []).append(
            answer_lm_loss(model, tokenizer, row, device)
        )

    concept_losses = [
        torch.stack(losses).mean()
        for losses in concept_groups.values()
        if losses
    ]
    mandatory_loss = (
        torch.stack(concept_losses).mean()
        if concept_losses
        else None
    )

    optional_losses = [
        answer_lm_loss(model, tokenizer, row, device)
        for row in optional
    ]
    optional_loss = (
        torch.stack(optional_losses).mean()
        if optional_losses
        else None
    )

    if mandatory_loss is not None and optional_loss is not None:
        return 0.8 * mandatory_loss + 0.2 * optional_loss
    if mandatory_loss is not None:
        return mandatory_loss
    if optional_loss is not None:
        return optional_loss
    return torch.tensor(0.0, device=device)


def concept_generation_report(model, tokenizer, rows, device):
    mandatory = [row for row in rows if bool(row.get("must_train", False))]
    grouped: dict[str, list[dict]] = {}
    for row in mandatory:
        grouped.setdefault(row_concept(row), []).append(row)

    details = []
    concept_scores = []
    for concept, concept_rows in grouped.items():
        sims = []
        generated_rows = []
        for row in concept_rows:
            sim, generated = generation_similarity(
                model, tokenizer, row, device
            )
            sims.append(sim)
            generated_rows.append((str(row["query"]), sim, generated))
        concept_sim = sum(sims) / len(sims) if sims else 0.0
        concept_scores.append(concept_sim)
        details.append({
            "concept": concept,
            "similarity": concept_sim,
            "rows": generated_rows,
        })

    mean_sim = sum(concept_scores) / len(concept_scores) if concept_scores else 1.0
    min_sim = min(concept_scores, default=1.0)
    return mean_sim, min_sim, details


def split_rows(rows: list[dict], holdout: float, seed: int):
    rng = random.Random(seed)
    mandatory = [row for row in rows if bool(row.get("must_train", False))]
    protected = [row for row in rows if bool(row.get("protected", False))]
    optional = [
        row for row in rows
        if not bool(row.get("must_train", False))
        and not bool(row.get("protected", False))
    ]
    rng.shuffle(optional)

    if len(optional) <= 1:
        return mandatory + protected + optional, []

    n_test = max(1, int(round(len(optional) * holdout)))
    n_test = min(n_test, len(optional) - 1)
    test_rows = optional[:n_test]
    train_rows = mandatory + protected + optional[n_test:]
    return train_rows, test_rows


def configure_trainable(model: LanguageModel, train_blocks: int = 1):
    for p in model.parameters():
        p.requires_grad = False

    n_blocks = max(1, min(int(train_blocks), len(model.blocks)))
    for block in model.blocks[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    semantic_params = [
        p
        for block in model.blocks[-n_blocks:]
        for p in block.parameters()
        if p.requires_grad
    ] + [
        p for p in model.final_norm.parameters() if p.requires_grad
    ]
    lm_head_params = [p for p in model.lm_head.parameters() if p.requires_grad]
    return semantic_params, lm_head_params


def build_prompt(row: dict) -> str:
    runtime_prompt = str(row.get("runtime_prompt", "")).strip()
    if runtime_prompt:
        return runtime_prompt
    return build_semantic_generation_prompt(
        row["query"],
        selected_label=row["label"],
        gate="ACCEPT",
        intent=row["intent"],
        concepts=row["concepts"],
        truth_record={"truth_status": row["truth_status"]},
    )


def context_window_answer_loss(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
) -> torch.Tensor:
    """Teacher-force answer tokens with the same context window as generate().

    v0.10.44 keeps exact runtime-visible prefixes but batches positions that
    have the same prefix length. Transformer samples are independent across
    the batch, so this is mathematically equivalent to the old per-token loop
    while requiring far fewer forward passes.
    """
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    full = prompt_ids + answer_ids

    first_target = len(prompt_ids)
    context_length = max(1, int(model.context_length))

    groups: dict[int, list[tuple[list[int], int]]] = {}
    for target_pos in range(first_target, len(full)):
        start = max(0, target_pos - context_length)
        prefix = full[start:target_pos]
        if not prefix:
            continue
        groups.setdefault(len(prefix), []).append(
            (prefix, int(full[target_pos]))
        )

    if not groups:
        return torch.tensor(0.0, device=device, requires_grad=True)

    summed_losses = []
    total_targets = 0

    for items in groups.values():
        x = torch.tensor(
            [prefix for prefix, _ in items],
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(
            [target for _, target in items],
            dtype=torch.long,
            device=device,
        )

        logits = model(x)[:, -1, :]
        summed_losses.append(
            F.cross_entropy(
                logits,
                targets,
                reduction="sum",
            )
        )
        total_targets += len(items)

    return torch.stack(summed_losses).sum() / max(1, total_targets)


def answer_lm_loss(
    model: LanguageModel,
    tokenizer: Tokenizer,
    row: dict,
    device: torch.device,
) -> torch.Tensor:
    return context_window_answer_loss(
        model,
        tokenizer,
        build_prompt(row),
        str(row["answer"]),
        device,
    )



@torch.no_grad()
def mandatory_qa_nll(model, tokenizer, rows, device):
    mandatory = [row for row in rows if bool(row.get("must_train", False))]
    if not mandatory:
        return 0.0, []
    details = []
    vals = []
    for row in mandatory:
        value = float(answer_lm_loss(model, tokenizer, row, device).item())
        vals.append(value)
        details.append({
            "query": str(row["query"]),
            "nll": value,
            "answer": str(row["answer"]),
        })
    return sum(vals) / len(vals), details

@torch.no_grad()
def semantic_vector(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    alpha: float,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=alpha,
        normalize_hybrid=False,
    )[0]



@torch.no_grad()
def align_must_train_runtime_prompts(
    teacher: LanguageModel,
    tokenizer: Tokenizer,
    benchmark,
    rows: list[dict],
    alpha: float,
) -> list[dict]:
    router = SemanticRouter(teacher, tokenizer, alpha=alpha)
    router.fit(benchmark)
    aligned = []

    for row in rows:
        if not bool(row.get("must_train", False)):
            continue

        query = str(row["query"])
        ranked = router.route(query)
        if not ranked:
            continue
        top = ranked[0]

        extracted = extract_purpose_intent(query)
        props = extract_propositions(
            extracted.concept_texts[0]
            if extracted.concept_texts
            else query
        )
        concepts = proposition_concepts(
            extracted.concept_texts,
            props,
        )

        runtime_prompt = build_semantic_generation_prompt(
            query,
            selected_label=top.label,
            gate="INTERNAL_PROBE",
            intent=extracted.intent,
            concepts=concepts,
            truth_record=None,
        )
        row["runtime_prompt"] = runtime_prompt
        row["runtime_selected_label"] = str(top.label)
        aligned.append(row)

    return aligned

@torch.no_grad()
def build_runtime_replay_rows(
    teacher: LanguageModel,
    tokenizer: Tokenizer,
    benchmark,
    rows: list[dict],
    device: torch.device,
    alpha: float,
) -> list[dict]:
    router = SemanticRouter(teacher, tokenizer, alpha=alpha)
    router.fit(benchmark)
    replay_rows = []

    for row in rows:
        query = str(row["query"])
        ranked = router.route(query)
        if not ranked:
            continue
        top = ranked[0]

        extracted = extract_purpose_intent(query)
        props = extract_propositions(
            extracted.concept_texts[0]
            if extracted.concept_texts
            else query
        )
        concepts = proposition_concepts(
            extracted.concept_texts,
            props,
        )

        prompt = build_semantic_generation_prompt(
            query,
            selected_label=top.label,
            gate="INTERNAL_PROBE",
            intent=extracted.intent,
            concepts=concepts,
            truth_record=None,
        )
        prompt_ids = tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )
        generated = teacher.generate(
            prompt_ids,
            max_new_tokens=96,
            eos_id=tokenizer.eos_id,
            temperature=0.2,
            top_k=1,
            repetition_penalty=1.10,
        )
        continuation = generated[len(prompt_ids):]
        answer = tokenizer.decode(
            continuation,
            skip_special_tokens=True,
        ).strip()
        answer = stabilize_generated_answer(answer)

        canonical_answer = str(row.get("answer", "")).strip()
        if canonical_answer:
            replay_rows.append({
                "query": query,
                "prompt": prompt,
                "answer": canonical_answer,
                "source_answer": answer,
                "selected_label": str(top.label),
                "target_kind": "TRUSTED_CANONICAL",
            })

    return replay_rows


def protected_distillation_loss(
    student: LanguageModel,
    teacher: LanguageModel,
    tokenizer: Tokenizer,
    replay_row: dict,
    device: torch.device,
) -> torch.Tensor:
    # Keep the exact runtime /internal prompt and the trusted canonical answer.
    # Use the same rolling context window as runtime generation.
    return context_window_answer_loss(
        student,
        tokenizer,
        str(replay_row["prompt"]),
        str(replay_row["answer"]),
        device,
    )

def semantic_vector_grad(
    model: LanguageModel,
    tokenizer: Tokenizer,
    text: str,
    device: torch.device,
    alpha: float,
) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return model.encode_semantic(
        x,
        pooling="hybrid",
        hybrid_alpha=alpha,
        normalize_hybrid=False,
    )[0]


@torch.no_grad()
def mean_qa_loss(model, tokenizer, rows, device):
    model.eval()
    vals = []
    for row in rows:
        vals.append(float(answer_lm_loss(model, tokenizer, row, device).item()))
    return sum(vals) / len(vals) if vals else 0.0


def stabilize_generated_answer(text: str) -> str:
    """Keep one complete Japanese answer sentence and discard runaway tails."""
    text = str(text).strip()
    if not text:
        return text
    end = text.find("。")
    if end >= 0:
        return text[: end + 1].strip()
    return text


def generation_quality(text: str) -> dict[str, float | bool]:
    raw = str(text)
    stable = stabilize_generated_answer(raw)
    terminated = stable.endswith("。")
    if not stable:
        return {
            "terminated": False,
            "abnormal_ratio": 1.0,
            "repetition_ratio": 1.0,
        }

    abnormal = sum(
        1 for ch in stable
        if not (
            ch.isalnum()
            or ch.isspace()
            or "\u3040" <= ch <= "\u30ff"
            or "\u3400" <= ch <= "\u9fff"
            or ch in "。、・「」『』（）()：:！？!?ー〜～,%％+-=/"
        )
    )
    abnormal_ratio = abnormal / max(1, len(stable))

    chunks = [
        stable[i:i+4]
        for i in range(max(0, len(stable) - 3))
    ]
    repetition_ratio = (
        1.0 - len(set(chunks)) / len(chunks)
        if chunks else 0.0
    )
    return {
        "terminated": terminated,
        "abnormal_ratio": abnormal_ratio,
        "repetition_ratio": repetition_ratio,
    }


@torch.no_grad()
def generation_similarity(
    model: LanguageModel,
    tokenizer: Tokenizer,
    row: dict,
    device: torch.device,
    *,
    max_new_tokens: int = 96,
) -> tuple[float, str]:
    model.eval()
    prompt = build_prompt(row)
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=0.2,
        top_k=1,
        repetition_penalty=1.10,
    )
    continuation = generated[len(prompt_ids):]
    answer_raw = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    answer = stabilize_generated_answer(answer_raw)

    expected = "".join(stabilize_generated_answer(str(row["answer"])).split())
    actual = "".join(answer.split())
    ratio = SequenceMatcher(None, expected, actual).ratio() if expected or actual else 1.0
    return ratio, answer


@torch.no_grad()
def mandatory_generation_report(
    model: LanguageModel,
    tokenizer: Tokenizer,
    rows: list[dict],
    device: torch.device,
) -> tuple[float, list[tuple[str, float, str, str]]]:
    mandatory = [row for row in rows if bool(row.get("must_train", False))]
    if not mandatory:
        return 1.0, []
    details = []
    vals = []
    for row in mandatory:
        ratio, generated = generation_similarity(
            model,
            tokenizer,
            row,
            device,
        )
        vals.append(ratio)
        details.append(
            (
                str(row["query"]),
                ratio,
                str(row["answer"]),
                generated,
            )
        )
    return sum(vals) / len(vals), details


@torch.no_grad()
def mean_semantic_cosine(student, teacher, tokenizer, benchmark, device, alpha):
    vals = []
    for row in benchmark:
        a = semantic_vector(student, tokenizer, row.text, device, alpha)
        b = semantic_vector(teacher, tokenizer, row.text, device, alpha)
        vals.append(float(F.cosine_similarity(a, b, dim=0).item()))
    return sum(vals) / len(vals) if vals else 1.0


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Use --allow-cpu only for testing.")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = Tokenizer.load(args.tokenizer)
    rows = load_dataset(Path(args.dataset))
    train_rows, test_rows = split_rows(rows, args.holdout, args.seed)
    benchmark = load_benchmark(args.benchmark)

    teacher, source_checkpoint = LanguageModel.load_checkpoint(args.model, device=device)
    student, _ = LanguageModel.load_checkpoint(args.model, device=device)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    semantic_params, lm_head_params = configure_trainable(student, args.train_blocks)

    initial_trainable = {
        name: tensor.detach().cpu().clone()
        for name, tensor in student.state_dict().items()
        if (
            name.startswith("blocks.")
            or name.startswith("final_norm.")
            or name.startswith("lm_head.")
        )
    }

    teacher_vectors = {
        row.text: semantic_vector(
            teacher, tokenizer, row.text, device, args.alpha
        ).detach()
        for row in benchmark
    }

    runtime_aligned_new_rows = align_must_train_runtime_prompts(
        teacher,
        tokenizer,
        benchmark,
        train_rows,
        args.alpha,
    )

    protected_source_rows = [
        row for row in train_rows if bool(row.get("protected", False))
    ]
    runtime_replay_rows = build_runtime_replay_rows(
        teacher,
        tokenizer,
        benchmark,
        protected_source_rows,
        device,
        args.alpha,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": semantic_params, "lr": args.learning_rate},
            {"params": lm_head_params, "lr": args.lm_head_lr},
        ],
        weight_decay=0.01,
    )

    before_train = mean_qa_loss(teacher, tokenizer, train_rows, device)
    before_test = mean_qa_loss(teacher, tokenizer, test_rows, device)
    mandatory_nll_before, mandatory_nll_before_details = mandatory_qa_nll(
        teacher, tokenizer, train_rows, device
    )

    print("=" * 100)
    print(" LLM_SEM v0.10.44 Batched Context-Aligned Fine-Tuning")
    print("=" * 100)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Source checkpoint   :", args.model)
    print("Source loss         :", source_checkpoint.get("loss"))
    print("Output checkpoint   :", args.output)
    print("QA samples          :", len(rows))
    print("Train samples       :", len(train_rows))
    print("Mandatory sleep rows:", sum(int(bool(r.get("must_train", False))) for r in train_rows))
    print("Holdout samples     :", len(test_rows))
    print("Benchmark samples   :", len(benchmark))
    print("Trainable semantic  :", f"last {args.train_blocks} block(s) + final_norm")
    print("Trainable LM head   : True")
    print("Semantic LR         :", args.learning_rate)
    print("LM-head LR          :", args.lm_head_lr)
    print("Preserve weight     :", args.preserve_weight)
    print("Concept balanced    :", args.concept_balanced)
    print("Protected rows      :", sum(int(bool(r.get("protected", False))) for r in train_rows))
    print("Protected distill wt:", args.protected_distill_weight)
    print("New knowledge weight:", args.new_knowledge_weight)
    print("Runtime replay rows :", len(runtime_replay_rows))
    print("Runtime-aligned NEW :", len(runtime_aligned_new_rows))
    print("LM context length    :", student.context_length)
    print("Teacher-force window :", "MATCHES generate()")
    print("Teacher-force batching:", "GROUPED BY PREFIX LENGTH")
    print("Before train QA NLL :", f"{before_train:.6f}")
    print("Before holdout NLL  :", f"{before_test:.6f}")
    print()

    mandatory_trace_rows = [
        row for row in train_rows if bool(row.get("must_train", False))
    ]
    optional_trace_rows = [
        row for row in train_rows
        if not bool(row.get("must_train", False))
        and not bool(row.get("protected", False))
    ]

    print("=" * 100)
    print(" SLEEP TRAINING TEXT TRACE")
    print("=" * 100)

    print(f"[NEW / MUST_TRAIN] rows={len(mandatory_trace_rows)}")
    if mandatory_trace_rows:
        for index, row in enumerate(mandatory_trace_rows, 1):
            print(
                f"  {index:02d}. concept={row_concept(row)!r} "
                f"label={row.get('label')!r} "
                f"weight={float(row.get('sleep_weight', 1.0)):.3f}"
            )
            print("      query :", row["query"])
            print("      answer:", row["answer"])
            if str(row.get("runtime_prompt", "")).strip():
                print(
                    "      runtime label:",
                    row.get("runtime_selected_label", ""),
                )
                print("      runtime prompt:")
                for line in str(row["runtime_prompt"]).splitlines():
                    print("        " + line)
    else:
        print("  (none)")

    print()
    print(f"[PROTECTED / CANONICAL_RUNTIME_REPLAY] rows={len(runtime_replay_rows)}")
    if runtime_replay_rows:
        for index, row in enumerate(runtime_replay_rows, 1):
            print(
                f"  {index:02d}. label={row.get('selected_label')!r}"
            )
            print("      query          :", row["query"])
            print("      source generated:", row.get("source_answer", ""))
            print("      canonical target:", row["answer"])
            print("      target kind     :", row.get("target_kind", "TRUSTED_CANONICAL"))
            print("      prompt:")
            for line in str(row["prompt"]).splitlines():
                print("        " + line)
    else:
        print("  (none)")

    print()
    print(f"[OPTIONAL / BASE_STABILIZATION] rows={len(optional_trace_rows)}")
    if optional_trace_rows:
        for index, row in enumerate(optional_trace_rows, 1):
            print(
                f"  {index:02d}. label={row.get('label')!r} "
                f"source={row.get('sleep_source')!r} "
                f"weight={float(row.get('sleep_weight', 1.0)):.3f}"
            )
            print("      query :", row["query"])
            print("      answer:", row["answer"])
    else:
        print("  (none)")
    print("=" * 100)
    print()

    last_total = None
    best_holdout = float("inf")
    best_sem_cos = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, max(1, args.epochs) + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)

        if args.concept_balanced:
            qa_loss = concept_balanced_qa_loss(
                student, tokenizer, train_rows, device
            )
        else:
            qa_losses = []
            qa_weights = []
            for row in train_rows:
                qa_losses.append(answer_lm_loss(student, tokenizer, row, device))
                qa_weights.append(max(0.01, float(row.get("sleep_weight", 1.0))))
            weighted = [
                loss * weight
                for loss, weight in zip(qa_losses, qa_weights)
            ]
            qa_loss = torch.stack(weighted).sum() / sum(qa_weights)

        preserve_losses = []
        for row in benchmark:
            student_vec = semantic_vector_grad(
                student, tokenizer, row.text, device, args.alpha
            )
            teacher_vec = teacher_vectors[row.text]
            preserve_losses.append(
                1.0 - F.cosine_similarity(student_vec, teacher_vec, dim=0)
            )
        preserve_loss = torch.stack(preserve_losses).mean()

        if runtime_replay_rows and args.protected_distill_weight > 0.0:
            protected_losses = [
                protected_distillation_loss(
                    student,
                    teacher,
                    tokenizer,
                    replay_row,
                    device,
                )
                for replay_row in runtime_replay_rows
            ]
            protected_loss = torch.stack(protected_losses).mean()
        else:
            protected_loss = torch.tensor(0.0, device=device)

        total = (
            args.new_knowledge_weight * qa_loss
            + args.preserve_weight * preserve_loss
            + args.protected_distill_weight * protected_loss
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            semantic_params + lm_head_params,
            1.0,
        )
        optimizer.step()
        last_total = float(total.item())

        holdout_nll = mean_qa_loss(student, tokenizer, test_rows, device)
        sem_cos = mean_semantic_cosine(
            student, teacher, tokenizer, benchmark, device, args.alpha
        )

        # LLM_TRY-style quality-aware checkpoint selection:
        # accept a candidate only while semantic retention remains strong,
        # then choose the lowest holdout answer NLL.
        if sem_cos >= 0.98 and holdout_nll < best_holdout:
            best_holdout = holdout_nll
            best_sem_cos = sem_cos
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in student.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            marker = " *BEST" if best_epoch == epoch else ""
            print(
                f"Epoch {epoch:>3}/{args.epochs} "
                f"total={float(total.item()):.6f} "
                f"qa={float(qa_loss.item()):.6f} "
                f"preserve={float(preserve_loss.item()):.6f} "
                f"protect={float(protected_loss.item()):.6f} "
                f"holdout={holdout_nll:.6f} "
                f"sem_cos={sem_cos:.6f}{marker}"
            )

    if args.prefer_final_state:
        print("Quality-selected checkpoint: final state preferred for iterative sleep.")
    elif best_state is not None:
        student.load_state_dict(best_state)
        student.to(device)
        print(
            f"Quality-selected checkpoint: epoch={best_epoch} "
            f"holdout={best_holdout:.6f} sem_cos={best_sem_cos:.6f}"
        )
    else:
        print("Quality-selected checkpoint: none met semantic cosine >= 0.98; using final state.")

    after_train = mean_qa_loss(student, tokenizer, train_rows, device)
    after_test = mean_qa_loss(student, tokenizer, test_rows, device)
    mandatory_nll_after, mandatory_nll_after_details = mandatory_qa_nll(
        student, tokenizer, train_rows, device
    )

    delta_sq = 0.0
    base_sq = 0.0
    for name, tensor in student.state_dict().items():
        if name not in initial_trainable:
            continue
        current = tensor.detach().cpu().float()
        initial = initial_trainable[name].float()
        diff = current - initial
        delta_sq += float((diff * diff).sum().item())
        base_sq += float((initial * initial).sum().item())
    trainable_param_delta_l2 = delta_sq ** 0.5
    trainable_param_relative_delta = (
        trainable_param_delta_l2 / (base_sq ** 0.5)
        if base_sq > 0.0
        else 0.0
    )

    sem_cos = mean_semantic_cosine(
        student, teacher, tokenizer, benchmark, device, args.alpha
    )
    generation_sim, generation_details = mandatory_generation_report(
        student,
        tokenizer,
        train_rows,
        device,
    )
    generation_min = (
        min((ratio for _, ratio, _, _ in generation_details), default=1.0)
    )
    concept_generation_mean, concept_generation_min, concept_generation_details = (
        concept_generation_report(
            student,
            tokenizer,
            train_rows,
            device,
        )
    )
    quality_details = [
        generation_quality(generated)
        for _, _, _, generated in generation_details
    ]
    termination_rate = (
        sum(int(bool(q["terminated"])) for q in quality_details) / len(quality_details)
        if quality_details else 1.0
    )
    abnormal_ratio_max = max(
        (float(q["abnormal_ratio"]) for q in quality_details),
        default=0.0,
    )
    repetition_ratio_max = max(
        (float(q["repetition_ratio"]) for q in quality_details),
        default=0.0,
    )

    # Recompute the objective for the actually selected model state so
    # checkpoint metadata describes the saved weights rather than the final
    # training iteration that may have been rolled back.
    with torch.no_grad():
        if args.concept_balanced:
            selected_qa_loss = concept_balanced_qa_loss(
                student, tokenizer, train_rows, device
            )
        else:
            selected_losses = []
            selected_weights = []
            for row in train_rows:
                selected_losses.append(
                    answer_lm_loss(student, tokenizer, row, device)
                )
                selected_weights.append(
                    max(0.01, float(row.get("sleep_weight", 1.0)))
                )
            selected_qa_loss = (
                torch.stack([
                    loss * weight
                    for loss, weight in zip(
                        selected_losses, selected_weights
                    )
                ]).sum() / sum(selected_weights)
                if selected_losses
                else torch.tensor(0.0, device=device)
            )

        selected_preserve = []
        for row in benchmark:
            selected_vec = semantic_vector(
                student, tokenizer, row.text, device, args.alpha
            )
            selected_preserve.append(
                1.0 - F.cosine_similarity(
                    selected_vec,
                    teacher_vectors[row.text],
                    dim=0,
                )
            )
        selected_preserve_loss = (
            torch.stack(selected_preserve).mean()
            if selected_preserve
            else torch.tensor(0.0, device=device)
        )

        if runtime_replay_rows and args.protected_distill_weight > 0.0:
            selected_protected_loss = torch.stack([
                protected_distillation_loss(
                    student,
                    teacher,
                    tokenizer,
                    replay_row,
                    device,
                )
                for replay_row in runtime_replay_rows
            ]).mean()
        else:
            selected_protected_loss = torch.tensor(0.0, device=device)

        selected_total = float((
            args.new_knowledge_weight * selected_qa_loss
            + args.preserve_weight * selected_preserve_loss
            + args.protected_distill_weight * selected_protected_loss
        ).item())

    selected_epoch = (
        max(1, args.epochs)
        if args.prefer_final_state or best_state is None
        else best_epoch
    )
    restored_best_state = (
        not args.prefer_final_state
        and best_state is not None
        and selected_epoch != max(1, args.epochs)
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    student.save_checkpoint(
        str(output),
        optimizer=None if restored_best_state else optimizer,
        epoch=selected_epoch,
        loss=selected_total,
    )

    print()
    print("Final evaluation")
    print("----------------")
    print("Train QA NLL    :", f"{before_train:.6f} -> {after_train:.6f}")
    print("Holdout QA NLL  :", f"{before_test:.6f} -> {after_test:.6f}")
    print("Target QA NLL   :", f"{mandatory_nll_before:.6f} -> {mandatory_nll_after:.6f}")
    print("Trainable delta :", f"L2={trainable_param_delta_l2:.6f} rel={trainable_param_relative_delta:.9f}")
    print("Semantic cosine :", f"{sem_cos:.6f}")
    print("Mandatory generation similarity:", f"{generation_sim:.6f}")
    print("Mandatory minimum similarity   :", f"{generation_min:.6f}")
    print("Concept generation mean        :", f"{concept_generation_mean:.6f}")
    print("Concept generation minimum     :", f"{concept_generation_min:.6f}")
    print("Natural termination rate       :", f"{termination_rate:.6f}")
    print("Maximum abnormal-char ratio    :", f"{abnormal_ratio_max:.6f}")
    print("Maximum repetition ratio       :", f"{repetition_ratio_max:.6f}")
    print("Saved checkpoint:", output)
    print("Selected epoch  :", selected_epoch)
    print("Selected objective:", f"{selected_total:.6f}")
    if generation_details:
        print()
        print("Generation probe details")
        print("------------------------")
        for query, ratio, expected, generated in generation_details:
            print(f"[{ratio:.3f}] {query}")
            print("  expected :", expected)
            print("  generated:", generated)
    print()
    passed = (
        (not test_rows or after_test < before_test)
        and sem_cos >= 0.98
        and generation_sim >= args.min_generation_sim
    )
    if passed:
        print("RESULT: PASS")
        print("Answer supervision passed NLL, semantic retention, and generation checks.")
    else:
        print("RESULT: REVIEW")
        print("Inspect holdout improvement and semantic retention before promotion.")

    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {
                    "source": str(args.model),
                    "output": str(args.output),
                    "train_nll_before": before_train,
                    "train_nll_after": after_train,
                    "holdout_nll_before": before_test,
                    "holdout_nll_after": after_test,
                    "target_nll_before": mandatory_nll_before,
                    "target_nll_after": mandatory_nll_after,
                    "target_nll_before_details": mandatory_nll_before_details,
                    "target_nll_after_details": mandatory_nll_after_details,
                    "trainable_param_delta_l2": trainable_param_delta_l2,
                    "trainable_param_relative_delta": trainable_param_relative_delta,
                    "semantic_cosine": sem_cos,
                    "protected_distillation_loss": float(protected_loss.item()),
                    "new_knowledge_weight": float(args.new_knowledge_weight),
                    "generation_similarity_mean": generation_sim,
                    "generation_similarity_min": generation_min,
                    "concept_generation_mean": concept_generation_mean,
                    "concept_generation_min": concept_generation_min,
                    "concept_generation_details": concept_generation_details,
                    "termination_rate": termination_rate,
                    "abnormal_ratio_max": abnormal_ratio_max,
                    "repetition_ratio_max": repetition_ratio_max,
                    "generation_details": [
                        {
                            "query": query,
                            "similarity": ratio,
                            "expected": expected,
                            "generated": generated,
                        }
                        for query, ratio, expected, generated in generation_details
                    ],
                    "passed": passed,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    if args.require_pass and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
