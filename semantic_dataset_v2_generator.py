# semantic_dataset_v2_generator.py
#
# Generate Semantic Dataset v2 for LLM_SEM v0.2.
#
# Goals:
#   - preserve the existing six semantic classes
#   - increase Japanese utterance diversity
#   - include direct, indirect, conversational, short and implicit expressions
#   - keep label,text compatibility with existing LLM_SEM tools
#   - add pattern,difficulty metadata for analysis
#
# Output:
#   data_semantic/semantic_v2_all_300.csv
#   data_semantic/semantic_v2_train_180.csv
#   data_semantic/semantic_v2_validation_60.csv
#   data_semantic/semantic_v2_test_60.csv

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_OUTPUT_DIR = "data_semantic"
DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_CLASS = 50

LABELS = (
    "animal",
    "computer",
    "food",
    "science",
    "transport",
    "weather",
)

TOPICS: Dict[str, Sequence[str]] = {
    "animal": (
        "猫", "犬", "鳥", "馬", "イルカ",
        "象", "ライオン", "ウサギ", "ペンギン", "サル",
    ),
    "computer": (
        "CPU", "GPU", "メモリ", "データベース", "ネットワーク",
        "Linux", "Python", "クラウド", "ストレージ", "アルゴリズム",
    ),
    "food": (
        "カレー", "寿司", "ラーメン", "パン", "野菜",
        "果物", "味噌汁", "パスタ", "豆腐", "チーズ",
    ),
    "science": (
        "重力", "電磁気", "量子力学", "化学反応", "DNA",
        "惑星", "光", "温度", "原子", "生態系",
    ),
    "transport": (
        "電車", "自動車", "飛行機", "バス", "自転車",
        "新幹線", "船", "地下鉄", "タクシー", "道路",
    ),
    "weather": (
        "雨", "雪", "台風", "気温", "湿度",
        "風", "雲", "雷", "天気予報", "気圧",
    ),
}

IMPLICIT_INTENTS: Dict[str, Sequence[str]] = {
    "animal": (
        "この生き物はどんな場所で暮らす",
        "人と一緒に暮らせる",
        "何を食べて生活する",
        "どんな行動をすることが多い",
        "野生ではどんな特徴がある",
    ),
    "computer": (
        "処理をもっと速くしたい",
        "メモリ使用量を減らしたい",
        "このエラーの原因を知りたい",
        "複数の計算機で処理を分けたい",
        "データを安全に保存したい",
    ),
    "food": (
        "今日の夕食を何にするか迷っている",
        "簡単に作れるものを知りたい",
        "材料の組み合わせを考えたい",
        "どんな味になるのか知りたい",
        "栄養のある食べ方を考えたい",
    ),
    "science": (
        "なぜ物体が落ちるのか知りたい",
        "光がどう伝わるのか気になる",
        "物質が変化する理由を知りたい",
        "小さな粒子の振る舞いを理解したい",
        "生命の仕組みを科学的に考えたい",
    ),
    "transport": (
        "そこまでどうやって行けばいい",
        "できるだけ早く移動したい",
        "乗り換えを少なくしたい",
        "荷物が多いけれど移動したい",
        "朝早く出発する方法を知りたい",
    ),
    "weather": (
        "傘を持って行ったほうがいい",
        "コートを着たほうがいい",
        "洗濯物を外に干して大丈夫",
        "今日は外で活動できそう",
        "窓を開けたままで大丈夫",
    ),
}

EXPLICIT_PATTERNS: Sequence[Tuple[str, str, str]] = (
    ("direct", "easy", "{topic}について教えてください。"),
    ("question", "easy", "{topic}はどういうものですか？"),
    ("command", "easy", "{topic}について調べてください。"),
    ("conversational", "medium", "ちょっと気になっているんだけど、{topic}ってどう？"),
    ("polite", "medium", "{topic}について教えていただけますか？"),
    ("casual", "medium", "{topic}ってどんな感じ？"),
    ("short", "hard", "{topic}は？"),
)

IMPLICIT_PATTERNS: Sequence[Tuple[str, str, str]] = (
    ("indirect", "medium", "{intent}ですか？"),
    ("contextual", "hard", "予定を考えているんだけど、{intent}？"),
    ("implicit", "hard", "{intent}かな。"),
)


@dataclass(frozen=True)
class SemanticSample:
    label: str
    text: str
    pattern: str
    difficulty: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diverse Japanese Semantic Dataset v2."
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Currently fixed to 50 to preserve 180/60/60 split compatibility.",
    )
    return parser.parse_args()


def _take_five(values: Sequence[str], offset: int) -> List[str]:
    if len(values) < 5:
        raise ValueError("At least five source phrases are required.")
    result = []
    for i in range(5):
        result.append(values[(offset + i) % len(values)])
    return result


def generate_samples(samples_per_class: int = 50) -> List[SemanticSample]:
    if samples_per_class != 50:
        raise ValueError(
            "Semantic Dataset v2 currently requires --samples-per-class 50 "
            "to preserve the 300/180/60/60 dataset layout."
        )

    rows: List[SemanticSample] = []

    for label in LABELS:
        topics = TOPICS[label]
        intents = IMPLICIT_INTENTS[label]

        # 7 explicit pattern families x 5 utterances = 35 samples.
        for pattern_index, (pattern, difficulty, template) in enumerate(
            EXPLICIT_PATTERNS
        ):
            for topic in _take_five(topics, pattern_index):
                rows.append(
                    SemanticSample(
                        label=label,
                        text=template.format(topic=topic),
                        pattern=pattern,
                        difficulty=difficulty,
                    )
                )

        # 3 semantic/implicit pattern families x 5 utterances = 15 samples.
        for pattern_index, (pattern, difficulty, template) in enumerate(
            IMPLICIT_PATTERNS
        ):
            for intent in _take_five(intents, pattern_index):
                rows.append(
                    SemanticSample(
                        label=label,
                        text=template.format(intent=intent),
                        pattern=pattern,
                        difficulty=difficulty,
                    )
                )

    texts = [row.text for row in rows]
    if len(texts) != len(set(texts)):
        duplicates = [
            text for text, count in Counter(texts).items() if count > 1
        ]
        raise ValueError(
            "Duplicate generated text detected: " + repr(duplicates[:5])
        )

    counts = Counter(row.label for row in rows)
    for label in LABELS:
        if counts[label] != samples_per_class:
            raise ValueError(
                f"{label}: expected {samples_per_class}, got {counts[label]}"
            )

    return rows


def stratified_pattern_split(
    rows: Sequence[SemanticSample],
    seed: int,
) -> Tuple[
    List[SemanticSample],
    List[SemanticSample],
    List[SemanticSample],
]:
    grouped: Dict[str, List[SemanticSample]] = defaultdict(list)
    for row in rows:
        grouped[row.label].append(row)

    rng = random.Random(seed)
    train: List[SemanticSample] = []
    validation: List[SemanticSample] = []
    test: List[SemanticSample] = []

    # Split within each label while mixing pattern families.
    # 30 / 10 / 10 per class -> 180 / 60 / 60 overall.
    for label in LABELS:
        by_pattern: Dict[str, List[SemanticSample]] = defaultdict(list)
        for row in grouped[label]:
            by_pattern[row.pattern].append(row)

        train_label: List[SemanticSample] = []
        validation_label: List[SemanticSample] = []
        test_label: List[SemanticSample] = []

        # Round-robin from shuffled pattern buckets prevents one split from
        # accidentally containing only a narrow expression style.
        buckets = []
        for pattern in sorted(by_pattern):
            bucket = list(by_pattern[pattern])
            rng.shuffle(bucket)
            buckets.append(bucket)

        merged: List[SemanticSample] = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    merged.append(bucket.pop())

        train_label.extend(merged[:30])
        validation_label.extend(merged[30:40])
        test_label.extend(merged[40:50])

        train.extend(train_label)
        validation.extend(validation_label)
        test.extend(test_label)

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def write_csv(path: Path, rows: Sequence[SemanticSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text", "pattern", "difficulty"])
        for row in rows:
            writer.writerow(
                [row.label, row.text, row.pattern, row.difficulty]
            )


def print_distribution(name: str, rows: Sequence[SemanticSample]) -> None:
    label_counts = Counter(row.label for row in rows)
    pattern_counts = Counter(row.pattern for row in rows)
    print(f"{name:<12}: {len(rows):>3} samples")
    print(
        "  labels  : "
        + ", ".join(
            f"{label}={label_counts[label]}" for label in LABELS
        )
    )
    print(
        "  patterns: "
        + ", ".join(
            f"{pattern}={pattern_counts[pattern]}"
            for pattern in sorted(pattern_counts)
        )
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    rows = generate_samples(args.samples_per_class)
    train, validation, test = stratified_pattern_split(rows, args.seed)

    outputs = {
        "all": output_dir / "semantic_v2_all_300.csv",
        "train": output_dir / "semantic_v2_train_180.csv",
        "validation": output_dir / "semantic_v2_validation_60.csv",
        "test": output_dir / "semantic_v2_test_60.csv",
    }

    write_csv(outputs["all"], rows)
    write_csv(outputs["train"], train)
    write_csv(outputs["validation"], validation)
    write_csv(outputs["test"], test)

    print()
    print("LLM_SEM Semantic Dataset v2 Generator")
    print("-------------------------------------")
    print("Seed       :", args.seed)
    print("Output dir :", output_dir)
    print()
    print_distribution("All", rows)
    print_distribution("Train", train)
    print_distribution("Validation", validation)
    print_distribution("Test", test)
    print()
    print("Files")
    print("-----")
    for path in outputs.values():
        print(path)
    print()
    print(
        "CSV compatibility: existing LLM_SEM tools only require label,text; "
        "pattern,difficulty are optional metadata."
    )


if __name__ == "__main__":
    main()
