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
from pathlib import Path

import torch
import torch.nn.functional as F

from chat import build_semantic_generation_prompt
from model import LanguageModel
from semantic_eval import load_benchmark
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
            })
    if not rows:
        raise RuntimeError("No valid QA rows.")
    return rows


def split_rows(rows: list[dict], holdout: float, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_test = max(1, int(round(len(rows) * holdout)))
    n_test = min(n_test, len(rows) - 1)
    return rows[n_test:], rows[:n_test]


def configure_trainable(model: LanguageModel):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    semantic_params = [
        p for p in list(model.blocks[-1].parameters()) + list(model.final_norm.parameters())
        if p.requires_grad
    ]
    lm_head_params = [p for p in model.lm_head.parameters() if p.requires_grad]
    return semantic_params, lm_head_params


def build_prompt(row: dict) -> str:
    return build_semantic_generation_prompt(
        row["query"],
        selected_label=row["label"],
        gate="ACCEPT",
        intent=row["intent"],
        concepts=row["concepts"],
        truth_record={"truth_status": row["truth_status"]},
    )


def answer_lm_loss(
    model: LanguageModel,
    tokenizer: Tokenizer,
    row: dict,
    device: torch.device,
) -> torch.Tensor:
    prompt = build_prompt(row)
    answer = row["answer"]

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    full = prompt_ids + answer_ids

    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full[1:]], dtype=torch.long, device=device)
    logits = model(x)

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
    ).view(1, -1)

    # Predictions whose target token belongs to the answer region.
    first_answer_target = max(0, len(prompt_ids) - 1)
    mask = torch.zeros_like(token_loss)
    mask[:, first_answer_target:] = 1.0
    denom = mask.sum().clamp_min(1.0)
    return (token_loss * mask).sum() / denom


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

    semantic_params, lm_head_params = configure_trainable(student)

    teacher_vectors = {
        row.text: semantic_vector(
            teacher, tokenizer, row.text, device, args.alpha
        ).detach()
        for row in benchmark
    }

    optimizer = torch.optim.AdamW(
        [
            {"params": semantic_params, "lr": args.learning_rate},
            {"params": lm_head_params, "lr": args.lm_head_lr},
        ],
        weight_decay=0.01,
    )

    before_train = mean_qa_loss(teacher, tokenizer, train_rows, device)
    before_test = mean_qa_loss(teacher, tokenizer, test_rows, device)

    print("=" * 100)
    print(" LLM_SEM v0.10.0 Semantic-Guided Fine-Tuning + Quality Selection")
    print("=" * 100)
    print("Device              :", device)
    if device.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("Source checkpoint   :", args.model)
    print("Source loss         :", source_checkpoint.get("loss"))
    print("Output checkpoint   :", args.output)
    print("QA samples          :", len(rows))
    print("Train samples       :", len(train_rows))
    print("Holdout samples     :", len(test_rows))
    print("Benchmark samples   :", len(benchmark))
    print("Trainable semantic  : final block + final_norm")
    print("Trainable LM head   : True")
    print("Semantic LR         :", args.learning_rate)
    print("LM-head LR          :", args.lm_head_lr)
    print("Preserve weight     :", args.preserve_weight)
    print("Before train QA NLL :", f"{before_train:.6f}")
    print("Before holdout NLL  :", f"{before_test:.6f}")
    print()

    last_total = None
    best_holdout = float("inf")
    best_sem_cos = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, max(1, args.epochs) + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)

        qa_losses = [
            answer_lm_loss(student, tokenizer, row, device)
            for row in train_rows
        ]
        qa_loss = torch.stack(qa_losses).mean()

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

        total = qa_loss + args.preserve_weight * preserve_loss
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
                f"holdout={holdout_nll:.6f} "
                f"sem_cos={sem_cos:.6f}{marker}"
            )

    if best_state is not None:
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
    sem_cos = mean_semantic_cosine(
        student, teacher, tokenizer, benchmark, device, args.alpha
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    student.save_checkpoint(
        str(output),
        optimizer=optimizer,
        epoch=max(1, args.epochs),
        loss=last_total,
    )

    print()
    print("Final evaluation")
    print("----------------")
    print("Train QA NLL    :", f"{before_train:.6f} -> {after_train:.6f}")
    print("Holdout QA NLL  :", f"{before_test:.6f} -> {after_test:.6f}")
    print("Semantic cosine :", f"{sem_cos:.6f}")
    print("Saved checkpoint:", output)
    print("Selected epoch  :", best_epoch if best_state is not None else args.epochs)
    print()
    passed = after_test < before_test and sem_cos >= 0.98
    if passed:
        print("RESULT: PASS")
        print("Answer supervision improved holdout NLL while preserving semantic geometry.")
    else:
        print("RESULT: REVIEW")
        print("Inspect holdout improvement and semantic retention before promotion.")

    if args.require_pass and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
