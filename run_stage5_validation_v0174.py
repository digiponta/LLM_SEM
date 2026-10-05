#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.17.4 Stage 5 Validation

Validates that an internalized checkpoint:
  A) answers the newly internalized fact at runtime,
  B) preserves the 14 validated protected facts,
  C) tolerates harmless surface variants,
  D) does not falsely activate the new fact for nearby/unrelated concepts,
  E) reproduces the same behavior after checkpoint reload,
  F) performs all generation directly from checkpoint weights (no Semantic Memory lookup).

This validator intentionally does NOT read data/semantic_memory_v0160.jsonl.
"""

from __future__ import annotations

import argparse
import difflib
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from model import LanguageModel
from tokenizer import Tokenizer


DEFAULT_MODEL = "model/model-sem-sleep-v0172.pt"
DEFAULT_TOKENIZER = "model/tokenizer.json"
DEFAULT_PROTECTED = "data/protected_knowledge_v0167.jsonl"
DEFAULT_RESULTS = "results/stage5_validation_v0174.json"

TRAILING_PUNCTUATION = "、，,。．.!！?？:：;；"

NEW_FACT_PROMPT = "量子センサーとは"
NEW_FACT_PARAPHRASE = "量子センサとは"
NEW_FACT_ANSWER = "原子や電子などの量子的性質を利用する高感度な計測技術である。"

SURFACE_VARIANTS = [
    "CPUとは、",
    "GPUとは。",
    "Pythonとは？",
    "量子センサーとは。",
    "量子センサとは？",
]

NEAR_NEGATIVES = [
    "量子通信とは",
    "量子コンピュータとは",
    "量子暗号とは",
]

UNRELATED_NEGATIVES = [
    "暗号",
    "文学とは",
    "ブラックホールとは",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM_SEM v0.17.4 Stage 5 Validation"
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--protected", default=DEFAULT_PROTECTED)
    p.add_argument("--results", default=DEFAULT_RESULTS)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--repetition-penalty", type=float, default=1.10)
    p.add_argument(
        "--false-activation-similarity",
        type=float,
        default=0.70,
        help="FAIL a negative probe when output similarity to the new fact is >= this value.",
    )
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def normalize_runtime_prompt(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return normalized.rstrip(TRAILING_PUNCTUATION).strip()


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def load_protected(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            prompt = str(item.get("prompt", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not prompt or not answer:
                raise ValueError(
                    f"Invalid protected row at line {line_no}: prompt/answer required"
                )
            rows.append({"prompt": prompt, "answer": answer})
    return rows


@torch.no_grad()
def generate_answer(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    runtime_prompt = normalize_runtime_prompt(prompt)
    prefix = tokenizer.encode(
        runtime_prompt,
        add_bos=True,
        add_eos=False,
    )
    generated = model.generate(
        prefix,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    return tokenizer.decode(
        generated[len(prefix):],
        skip_special_tokens=True,
    ).strip()


def exact_match(actual: str, expected: str) -> bool:
    return normalize_text(actual) == normalize_text(expected)


def similarity_to_new_fact(text: str) -> float:
    actual = normalize_text(text)
    expected = normalize_text(NEW_FACT_ANSWER)
    if not actual:
        return 0.0
    if actual == expected:
        return 1.0
    if actual in expected or expected in actual:
        return 1.0
    return difflib.SequenceMatcher(
        None,
        actual,
        expected,
        autojunk=False,
    ).ratio()


def validate_once(
    model: LanguageModel,
    tokenizer: Tokenizer,
    protected_rows: List[Dict[str, str]],
    args,
) -> Tuple[Dict[str, object], Dict[str, str]]:
    outputs: Dict[str, str] = {}
    checks: List[Dict[str, object]] = []

    def run_exact(category: str, prompt: str, expected: str):
        actual = generate_answer(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.repetition_penalty,
        )
        outputs[prompt] = actual
        passed = exact_match(actual, expected)
        checks.append({
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "actual": actual,
            "pass": passed,
        })

    # A) New internalized fact, exact and spelling variant.
    run_exact("new_fact", NEW_FACT_PROMPT, NEW_FACT_ANSWER)
    run_exact("new_fact_paraphrase", NEW_FACT_PARAPHRASE, NEW_FACT_ANSWER)

    # B) Protected knowledge regression.
    protected_map = {
        row["prompt"]: row["answer"]
        for row in protected_rows
    }
    for row in protected_rows:
        run_exact("protected", row["prompt"], row["answer"])

    # C) Harmless punctuation surface variants.
    for prompt in SURFACE_VARIANTS:
        canonical = normalize_runtime_prompt(prompt)
        if canonical == NEW_FACT_PARAPHRASE:
            expected = NEW_FACT_ANSWER
        elif canonical == NEW_FACT_PROMPT:
            expected = NEW_FACT_ANSWER
        else:
            expected = protected_map.get(canonical)
        if expected is None:
            checks.append({
                "category": "surface_variant",
                "prompt": prompt,
                "expected": None,
                "actual": "",
                "pass": False,
                "reason": f"no canonical target for normalized prompt {canonical!r}",
            })
        else:
            run_exact("surface_variant", prompt, expected)

    # D) False-activation tests. We do not require a particular answer; we
    # require that the learned quantum-sensor answer NOT be spuriously emitted.
    for category, prompts in (
        ("near_negative", NEAR_NEGATIVES),
        ("unrelated_negative", UNRELATED_NEGATIVES),
    ):
        for prompt in prompts:
            actual = generate_answer(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.repetition_penalty,
            )
            outputs[prompt] = actual
            similarity = similarity_to_new_fact(actual)
            passed = similarity < args.false_activation_similarity
            checks.append({
                "category": category,
                "prompt": prompt,
                "actual": actual,
                "similarity_to_new_fact": similarity,
                "threshold": args.false_activation_similarity,
                "pass": passed,
            })

    category_summary: Dict[str, Dict[str, int]] = {}
    for item in checks:
        category = str(item["category"])
        bucket = category_summary.setdefault(
            category,
            {"pass": 0, "fail": 0},
        )
        bucket["pass" if item["pass"] else "fail"] += 1

    passed = all(bool(item["pass"]) for item in checks)
    return {
        "pass": passed,
        "checks": checks,
        "category_summary": category_summary,
    }, outputs


def main():
    args = parse_args()
    device = choose_device(args.device)

    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    protected_path = Path(args.protected)
    results_path = Path(args.results)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)
    if not protected_path.exists():
        raise FileNotFoundError(protected_path)

    tokenizer = Tokenizer.load(str(tokenizer_path))
    protected_rows = load_protected(protected_path)

    print("=" * 88)
    print(" LLM_SEM v0.17.4 Stage 5 Validation")
    print("=" * 88)
    print("Model                  :", model_path)
    print("Tokenizer              :", tokenizer_path)
    print("Protected file         :", protected_path)
    print("Protected entries      :", len(protected_rows))
    print("Semantic Memory lookup : DISABLED / NOT USED")
    print("False-activation th    :", args.false_activation_similarity)
    print("Device                 :", device)
    if device.type == "cuda":
        print("GPU                    :", torch.cuda.get_device_name(device))
    print()

    # First direct checkpoint load.
    model_a, checkpoint_a = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    model_a.eval()
    first, outputs_a = validate_once(
        model_a,
        tokenizer,
        protected_rows,
        args,
    )

    print("FIRST LOAD")
    print("-" * 88)
    for item in first["checks"]:
        status = "PASS" if item["pass"] else "FAIL"
        extra = ""
        if "similarity_to_new_fact" in item:
            extra = (
                f" sim={item['similarity_to_new_fact']:.3f}"
                f" th={item['threshold']:.3f}"
            )
        print(
            f"[{status}] {item['category']:20s} "
            f"{item['prompt']!r}{extra}"
        )
        print("       actual:", item.get("actual", ""))
    print()

    # E) Reload reproducibility: construct a fresh model from the same checkpoint
    # and require byte-for-byte-equivalent decoded outputs for every probe.
    del model_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_b, checkpoint_b = LanguageModel.load_checkpoint(
        str(model_path),
        device=device,
    )
    model_b.eval()
    second, outputs_b = validate_once(
        model_b,
        tokenizer,
        protected_rows,
        args,
    )

    reload_mismatches = {
        prompt: {
            "first": outputs_a.get(prompt, ""),
            "second": outputs_b.get(prompt, ""),
        }
        for prompt in sorted(set(outputs_a) | set(outputs_b))
        if outputs_a.get(prompt, "") != outputs_b.get(prompt, "")
    }
    reload_pass = not reload_mismatches

    first_pass = bool(first["pass"])
    second_pass = bool(second["pass"])
    overall_pass = first_pass and second_pass and reload_pass

    print("RELOAD REPRODUCIBILITY")
    print("-" * 88)
    print("First validation      :", "PASS" if first_pass else "FAIL")
    print("Second validation     :", "PASS" if second_pass else "FAIL")
    print("Reload output equality:", "PASS" if reload_pass else "FAIL")
    if reload_mismatches:
        for prompt, values in reload_mismatches.items():
            print(f"[FAIL] reload mismatch {prompt!r}")
            print("       first :", values["first"])
            print("       second:", values["second"])
    print()

    # F) Explicitly record that no Semantic Memory file was opened by this script.
    result = {
        "version": "v0.17.4",
        "model": str(model_path),
        "checkpoint_sleep": checkpoint_b.get("sleep"),
        "checkpoint_repair": checkpoint_b.get("repair"),
        "semantic_memory_lookup": False,
        "new_fact": {
            "prompt": NEW_FACT_PROMPT,
            "paraphrase": NEW_FACT_PARAPHRASE,
            "answer": NEW_FACT_ANSWER,
        },
        "negative_probes": {
            "near": NEAR_NEGATIVES,
            "unrelated": UNRELATED_NEGATIVES,
            "similarity_threshold": args.false_activation_similarity,
        },
        "first_load": first,
        "second_load": second,
        "reload_reproducibility": {
            "pass": reload_pass,
            "mismatches": reload_mismatches,
        },
        "overall_pass": overall_pass,
        "status": "STAGE5_VALIDATED" if overall_pass else "STAGE5_FAIL",
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 88)
    print(" STAGE 5 RESULT")
    print("=" * 88)
    print("New fact/runtime      :", (
        "PASS"
        if first["category_summary"].get("new_fact", {}).get("fail", 0) == 0
        else "FAIL"
    ))
    print("Paraphrase            :", (
        "PASS"
        if first["category_summary"].get("new_fact_paraphrase", {}).get("fail", 0) == 0
        else "FAIL"
    ))
    print("Protected 14          :", (
        "PASS"
        if first["category_summary"].get("protected", {}).get("fail", 0) == 0
        else "FAIL"
    ))
    print("Surface variants      :", (
        "PASS"
        if first["category_summary"].get("surface_variant", {}).get("fail", 0) == 0
        else "FAIL"
    ))
    print("Near-negative reject  :", (
        "PASS"
        if first["category_summary"].get("near_negative", {}).get("fail", 0) == 0
        else "FAIL"
    ))
    print("Unrelated reject      :", (
        "PASS"
        if first["category_summary"].get("unrelated_negative", {}).get("fail", 0) == 0
        else "FAIL"
    ))
    print("Reload reproducibility:", "PASS" if reload_pass else "FAIL")
    print("Semantic Memory lookup: NONE")
    print("Results               :", results_path)
    print("STATUS                :", result["status"])

    raise SystemExit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
