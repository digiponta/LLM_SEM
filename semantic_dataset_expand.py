# semantic_dataset_expand.py
#
# Build a larger balanced semantic dataset for LLM_SEM v0.2.
#
# Generates 6 known classes x 50 Japanese sentences = 300 samples,
# then performs a deterministic stratified split:
#   train      : 30/class = 180
#   validation : 10/class =  60
#   test       : 10/class =  60
#
# NOTE:
# This is a deterministic synthetic expansion intended to reduce the extremely
# small-sample problem in projection training. The generated validation/test
# sets are useful for regression testing, but a separately authored holdout
# should still be kept for final independent evaluation.

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_OUTPUT_DIR = "data_semantic"
DEFAULT_SEED = 42

CLASS_TOPICS: Dict[str, List[str]] = {
    "animal": [
        "猫", "犬", "鳥", "馬", "イルカ",
        "象", "ライオン", "ウサギ", "ペンギン", "サル",
    ],
    "computer": [
        "CPU", "GPU", "メモリ", "データベース", "ネットワーク",
        "Linux", "Python", "クラウド", "ストレージ", "アルゴリズム",
    ],
    "food": [
        "カレー", "寿司", "ラーメン", "パン", "野菜",
        "果物", "味噌汁", "パスタ", "豆腐", "チーズ",
    ],
    "science": [
        "重力", "電磁気", "量子力学", "化学反応", "DNA",
        "惑星", "光", "温度", "原子", "生態系",
    ],
    "transport": [
        "電車", "自動車", "飛行機", "バス", "自転車",
        "新幹線", "船", "地下鉄", "タクシー", "道路",
    ],
    "weather": [
        "雨", "雪", "台風", "気温", "湿度",
        "風", "雲", "雷", "天気予報", "気圧",
    ],
}

CLASS_TEMPLATES: Dict[str, List[str]] = {
    "animal": [
        "{topic}の特徴について知りたいです。",
        "{topic}はどのような生き物ですか。",
        "{topic}の行動や習性を説明してください。",
        "{topic}が暮らす環境について教えてください。",
        "{topic}の生態を調べたいです。",
    ],
    "computer": [
        "{topic}の仕組みについて知りたいです。",
        "{topic}はコンピュータでどのような役割を持ちますか。",
        "{topic}の性能や動作を説明してください。",
        "{topic}を使うときの基本を教えてください。",
        "{topic}に関する技術を調べたいです。",
    ],
    "food": [
        "{topic}について知りたいです。",
        "{topic}はどのような料理や食品ですか。",
        "{topic}の作り方や特徴を説明してください。",
        "{topic}の味や材料について教えてください。",
        "{topic}を食べるときのポイントを調べたいです。",
    ],
    "science": [
        "{topic}の科学的な仕組みを知りたいです。",
        "{topic}は科学ではどのように説明されますか。",
        "{topic}の原理を説明してください。",
        "{topic}に関する現象について教えてください。",
        "{topic}を科学的に調べたいです。",
    ],
    "transport": [
        "{topic}について知りたいです。",
        "{topic}はどのように移動に使われますか。",
        "{topic}の仕組みや特徴を説明してください。",
        "{topic}を利用するときの情報を教えてください。",
        "{topic}に関する交通の仕組みを調べたいです。",
    ],
    "weather": [
        "{topic}について知りたいです。",
        "{topic}は天候にどのような影響がありますか。",
        "{topic}が発生する仕組みを説明してください。",
        "{topic}に関する気象情報を教えてください。",
        "{topic}の変化を天気の観点から調べたいです。",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and stratify an expanded LLM_SEM dataset."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def generate_samples() -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for label in sorted(CLASS_TOPICS):
        topics = CLASS_TOPICS[label]
        templates = CLASS_TEMPLATES[label]
        if len(topics) != 10 or len(templates) != 5:
            raise ValueError(
                f"{label}: expected 10 topics and 5 templates."
            )
        for topic in topics:
            for template in templates:
                rows.append((label, template.format(topic=topic)))

    texts = [text for _, text in rows]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate generated text detected.")
    return rows


def stratified_split(
    rows: Sequence[Tuple[str, str]],
    seed: int,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)

    rng = random.Random(seed)
    train: List[Tuple[str, str]] = []
    validation: List[Tuple[str, str]] = []
    test: List[Tuple[str, str]] = []

    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        if len(items) != 50:
            raise ValueError(f"{label}: expected 50 samples.")
        train.extend(items[:30])
        validation.extend(items[30:40])
        test.extend(items[40:50])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def write_csv(path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(rows)


def print_distribution(name: str, rows: Sequence[Tuple[str, str]]) -> None:
    counts = Counter(label for label, _ in rows)
    print(f"{name:<12}: {len(rows):>3} samples", end="")
    details = ", ".join(
        f"{label}={counts[label]}"
        for label in sorted(counts)
    )
    print("  (" + details + ")")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    rows = generate_samples()
    train, validation, test = stratified_split(rows, args.seed)

    write_csv(output_dir / "semantic_all_300.csv", rows)
    write_csv(output_dir / "semantic_train_180.csv", train)
    write_csv(output_dir / "semantic_validation_60.csv", validation)
    write_csv(output_dir / "semantic_test_60.csv", test)

    print()
    print("LLM_SEM Semantic Dataset Expansion")
    print("----------------------------------")
    print("Seed        :", args.seed)
    print("Output dir  :", output_dir)
    print()
    print_distribution("All", rows)
    print_distribution("Train", train)
    print_distribution("Validation", validation)
    print_distribution("Test", test)
    print()
    print("Files")
    print("-----")
    print(output_dir / "semantic_all_300.csv")
    print(output_dir / "semantic_train_180.csv")
    print(output_dir / "semantic_validation_60.csv")
    print(output_dir / "semantic_test_60.csv")
    print()
    print(
        "Note: validation/test are generated from the same synthetic "
        "template family. Keep a separately authored holdout for final "
        "independent evaluation."
    )


if __name__ == "__main__":
    main()
