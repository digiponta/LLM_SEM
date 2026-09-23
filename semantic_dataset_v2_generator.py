# semantic_dataset_v2_generator.py
#
# Generate Semantic Dataset v2 for LLM_SEM v0.2.
#
# Supported sizes:
#   50 samples/class  -> 300 total  -> 180/60/60
#   100 samples/class -> 600 total  -> 360/120/120
#
# Each class uses ten utterance patterns:
#   direct, question, command, conversational, polite,
#   casual, short, indirect, contextual, implicit
#
# The CSV keeps label,text compatibility and adds:
#   pattern,difficulty

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
DEFAULT_SAMPLES_PER_CLASS = 100

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
        "群れで生活することが多い",
        "夜に活動することが多い",
        "どれくらい長く生きる",
        "子どもはどのように育つ",
        "人に慣れやすい",
    ),
    "computer": (
        "処理をもっと速くしたい",
        "メモリ使用量を減らしたい",
        "このエラーの原因を知りたい",
        "複数の計算機で処理を分けたい",
        "データを安全に保存したい",
        "計算を並列化したい",
        "通信の遅延を減らしたい",
        "プログラムを自動化したい",
        "大量のデータを検索したい",
        "障害時にも処理を続けたい",
    ),
    "food": (
        "今日の夕食を何にするか迷っている",
        "簡単に作れるものを知りたい",
        "材料の組み合わせを考えたい",
        "どんな味になるのか知りたい",
        "栄養のある食べ方を考えたい",
        "温かいものを食べたい",
        "短時間で用意できるものがいい",
        "朝に食べやすいものを探している",
        "野菜を多く取れるものがいい",
        "家にある材料で作りたい",
    ),
    "science": (
        "なぜ物体が落ちるのか知りたい",
        "光がどう伝わるのか気になる",
        "物質が変化する理由を知りたい",
        "小さな粒子の振る舞いを理解したい",
        "生命の仕組みを科学的に考えたい",
        "熱がどのように移動するか知りたい",
        "電気が流れる仕組みを理解したい",
        "宇宙の天体が動く理由を知りたい",
        "遺伝情報が伝わる仕組みを知りたい",
        "化学反応が起こる条件を知りたい",
    ),
    "transport": (
        "そこまでどうやって行けばいい",
        "できるだけ早く移動したい",
        "乗り換えを少なくしたい",
        "荷物が多いけれど移動したい",
        "朝早く出発する方法を知りたい",
        "渋滞を避けて移動したい",
        "料金を抑えて移動したい",
        "駅から目的地まで行きたい",
        "遠くまで安全に移動したい",
        "夜でも使える移動手段を知りたい",
    ),
    "weather": (
        "傘を持って行ったほうがいい",
        "コートを着たほうがいい",
        "洗濯物を外に干して大丈夫",
        "今日は外で活動できそう",
        "窓を開けたままで大丈夫",
        "風が強くなりそう",
        "夜は冷え込みそう",
        "明日は暑くなりそう",
        "雷に注意したほうがいい",
        "週末に外出できそう",
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
        choices=(50, 100),
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="50 -> 300 total, 100 -> 600 total (default: 100).",
    )
    return parser.parse_args()


def _select(values: Sequence[str], count: int, offset: int) -> List[str]:
    if len(values) < count:
        raise ValueError(
            f"Need at least {count} source phrases, got {len(values)}."
        )
    ordered = list(values)
    if count == len(ordered):
        return ordered[offset:] + ordered[:offset]
    return [
        ordered[(offset + i) % len(ordered)]
        for i in range(count)
    ]


def generate_samples(samples_per_class: int) -> List[SemanticSample]:
    if samples_per_class not in (50, 100):
        raise ValueError("samples-per-class must be 50 or 100.")

    per_pattern = samples_per_class // 10
    rows: List[SemanticSample] = []

    for label in LABELS:
        topics = TOPICS[label]
        intents = IMPLICIT_INTENTS[label]

        for pattern_index, (pattern, difficulty, template) in enumerate(
            EXPLICIT_PATTERNS
        ):
            for topic in _select(topics, per_pattern, pattern_index):
                rows.append(
                    SemanticSample(
                        label=label,
                        text=template.format(topic=topic),
                        pattern=pattern,
                        difficulty=difficulty,
                    )
                )

        for pattern_index, (pattern, difficulty, template) in enumerate(
            IMPLICIT_PATTERNS
        ):
            for intent in _select(intents, per_pattern, pattern_index):
                rows.append(
                    SemanticSample(
                        label=label,
                        text=template.format(intent=intent),
                        pattern=pattern,
                        difficulty=difficulty,
                    )
                )

    texts = [row.text for row in rows]
    duplicates = [
        text for text, count in Counter(texts).items() if count > 1
    ]
    if duplicates:
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
    samples_per_class: int,
    seed: int,
) -> Tuple[
    List[SemanticSample],
    List[SemanticSample],
    List[SemanticSample],
]:
    grouped: Dict[str, List[SemanticSample]] = defaultdict(list)
    for row in rows:
        grouped[row.label].append(row)

    if samples_per_class == 50:
        train_per_class, validation_per_class, test_per_class = 30, 10, 10
    elif samples_per_class == 100:
        train_per_class, validation_per_class, test_per_class = 60, 20, 20
    else:
        raise ValueError("samples-per-class must be 50 or 100.")

    rng = random.Random(seed)
    train: List[SemanticSample] = []
    validation: List[SemanticSample] = []
    test: List[SemanticSample] = []

    for label in LABELS:
        by_pattern: Dict[str, List[SemanticSample]] = defaultdict(list)
        for row in grouped[label]:
            by_pattern[row.pattern].append(row)

        train_label: List[SemanticSample] = []
        validation_label: List[SemanticSample] = []
        test_label: List[SemanticSample] = []

        # Split each pattern independently so every split contains every
        # utterance pattern for every semantic class.
        for pattern in sorted(by_pattern):
            bucket = list(by_pattern[pattern])
            rng.shuffle(bucket)

            if samples_per_class == 50:
                train_n, validation_n = 3, 1
            else:
                train_n, validation_n = 6, 2

            train_label.extend(bucket[:train_n])
            validation_label.extend(
                bucket[train_n:train_n + validation_n]
            )
            test_label.extend(bucket[train_n + validation_n:])

        if len(train_label) != train_per_class:
            raise ValueError(f"{label}: unexpected train split size.")
        if len(validation_label) != validation_per_class:
            raise ValueError(f"{label}: unexpected validation split size.")
        if len(test_label) != test_per_class:
            raise ValueError(f"{label}: unexpected test split size.")

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
    difficulty_counts = Counter(row.difficulty for row in rows)

    print(f"{name:<12}: {len(rows):>3} samples")
    print(
        "  labels    : "
        + ", ".join(
            f"{label}={label_counts[label]}" for label in LABELS
        )
    )
    print(
        "  patterns  : "
        + ", ".join(
            f"{pattern}={pattern_counts[pattern]}"
            for pattern in sorted(pattern_counts)
        )
    )
    print(
        "  difficulty: "
        + ", ".join(
            f"{name}={difficulty_counts[name]}"
            for name in sorted(difficulty_counts)
        )
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    rows = generate_samples(args.samples_per_class)
    train, validation, test = stratified_pattern_split(
        rows,
        args.samples_per_class,
        args.seed,
    )

    total = len(rows)
    outputs = {
        "all": output_dir / f"semantic_v2_all_{total}.csv",
        "train": output_dir / f"semantic_v2_train_{len(train)}.csv",
        "validation": output_dir / (
            f"semantic_v2_validation_{len(validation)}.csv"
        ),
        "test": output_dir / f"semantic_v2_test_{len(test)}.csv",
    }

    write_csv(outputs["all"], rows)
    write_csv(outputs["train"], train)
    write_csv(outputs["validation"], validation)
    write_csv(outputs["test"], test)

    print()
    print("LLM_SEM Semantic Dataset v2 Generator")
    print("-------------------------------------")
    print("Seed              :", args.seed)
    print("Samples per class :", args.samples_per_class)
    print("Output dir        :", output_dir)
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
        "Each pattern is represented in every class and every split. "
        "Existing LLM_SEM tools remain compatible with label,text."
    )


if __name__ == "__main__":
    main()
