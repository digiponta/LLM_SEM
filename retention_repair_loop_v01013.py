from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--memory", default="data/semantic_memory.jsonl")
    p.add_argument("--tokenizer", default="model/tokenizer.json")
    p.add_argument("--benchmark", default="my_benchmark.csv")
    p.add_argument("--manifest", default="model/active-model.json")
    p.add_argument("--sleep-dataset", default="data/semantic_sleep_qa_v0106.json")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()

    source = Path(args.source)
    candidate = Path(args.candidate)
    current = candidate
    answer_baseline = candidate

    for i in range(args.rounds + 1):
        check = subprocess.run([
            sys.executable, "consolidated_retention_v094.py",
            "--source", str(source),
            "--candidate", str(current),
            "--tokenizer", args.tokenizer,
            "--benchmark", args.benchmark,
            "--memory", args.memory,
        ], check=False)
        if check.returncode == 0:
            answer_check = subprocess.run([
                sys.executable, "answer_retention_v01014.py",
                "--source", str(answer_baseline),
                "--candidate", str(current),
                "--dataset", args.sleep_dataset,
                "--tokenizer", args.tokenizer,
            ], check=False)

            if answer_check.returncode != 0:
                print(
                    "REPAIR> semantic retention passed but answer retention failed; "
                    "candidate will not be promoted."
                )
                raise SystemExit(1)

            runtime_check = subprocess.run([
                sys.executable, "runtime_answer_retention_v01015.py",
                "--source", str(source),
                "--candidate", str(current),
                "--dataset", args.sleep_dataset,
                "--benchmark", args.benchmark,
                "--tokenizer", args.tokenizer,
            ], check=False)

            if runtime_check.returncode != 0:
                print(
                    "REPAIR> semantic and dataset-answer retention passed, "
                    "but actual /internal runtime retention failed; "
                    "candidate will not be promoted."
                )
                raise SystemExit(1)

            if current != candidate:
                candidate.write_bytes(current.read_bytes())
            promote = subprocess.run([
                sys.executable, "promote_active_model_v094.py",
                "--candidate", str(candidate),
                "--manifest", args.manifest,
                "--retention-pass",
                "--note", "v0.10.15 runtime-answer retention repair",
            ], check=False)
            raise SystemExit(promote.returncode)

        if i >= args.rounds:
            break

        repaired = candidate.with_name(
            f"{candidate.stem}.repair{i+1}{candidate.suffix}"
        )
        repair = subprocess.run([
            sys.executable, "retention_repair_v01013.py",
            "--model", str(current),
            "--output", str(repaired),
            "--memory", args.memory,
            "--tokenizer", args.tokenizer,
            "--benchmark", args.benchmark,
            "--sleep-dataset", args.sleep_dataset,
            "--epochs", str(args.epochs),
        ], check=False)
        if repair.returncode != 0:
            raise SystemExit(repair.returncode)
        current = repaired

    raise SystemExit(1)


if __name__ == "__main__":
    main()
