#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NDC-based domain classification for LLM_SEM.

The classifier intentionally keeps UNKNOWN outside NDC.  NDC 000 is a real
classification ("General works"), so unknown/unclassified items use code=None.

This module provides:
  - 10 main NDC classes
  - selected 3-digit refinements useful to LLM_SEM
  - compatibility mapping from the historical six labels
  - deterministic keyword classification for Semantic Memory metadata

The classifier is deliberately lightweight and dependency-free.  It is a domain
metadata layer, not a replacement for semantic-vector routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Dict, Iterable, Optional, Tuple


NDC_MAIN: Dict[str, str] = {
    "0": "総記",
    "1": "哲学",
    "2": "歴史",
    "3": "社会科学",
    "4": "自然科学",
    "5": "技術・工学",
    "6": "産業",
    "7": "芸術",
    "8": "言語",
    "9": "文学",
}

NDC_CODES: Dict[str, str] = {
    "000": "総記",
    "002": "知識・学問・学術",
    "007": "情報科学",
    "100": "哲学",
    "140": "心理学",
    "150": "倫理学・道徳",
    "200": "歴史",
    "280": "伝記",
    "290": "地理・地誌・紀行",
    "300": "社会科学",
    "320": "法律",
    "330": "経済",
    "360": "社会",
    "370": "教育",
    "400": "自然科学",
    "410": "数学",
    "420": "物理学",
    "430": "化学",
    "440": "天文学・宇宙科学",
    "450": "地球科学・地学",
    "451": "気象学",
    "460": "生物科学・一般生物学",
    "480": "動物学",
    "490": "医学",
    "500": "技術・工学",
    "501": "工業基礎学",
    "530": "機械工学",
    "540": "電気工学",
    "547": "通信工学",
    "548": "情報工学",
    "590": "家政学・生活科学",
    "596": "食品・料理",
    "600": "産業",
    "610": "農業",
    "670": "商業",
    "680": "運輸・交通",
    "690": "通信事業",
    "700": "芸術・美術",
    "760": "音楽",
    "780": "スポーツ・体育",
    "800": "言語",
    "810": "日本語",
    "830": "英語",
    "900": "文学",
    "910": "日本文学",
    "930": "英米文学",
}

LEGACY_LABEL_TO_NDC: Dict[str, str] = {
    "computer": "007",
    "science": "400",
    "animal": "480",
    "weather": "451",
    "food": "596",
    "transport": "680",
}


@dataclass(frozen=True)
class NDCResult:
    code: Optional[str]
    main: Optional[str]
    name: str
    state: str
    source: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


_KEYWORD_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("007", (
        "ai", "人工知能", "llm", "大規模言語モデル", "機械学習", "深層学習",
        "python", "プログラム", "プログラミング", "ソフトウェア", "アルゴリズム",
        "データベース", "データ構造", "オペレーティングシステム", "os",
        "コンピュータ", "計算機", "cpu", "gpu", "cuda", "semantic",
    )),
    ("548", ("情報工学", "コンピュータ工学", "計算機工学")),
    ("547", ("通信工学", "ネットワーク工学", "無線工学")),
    ("420", (
        "物理", "量子力学", "量子", "相対性理論", "力学", "電磁気", "熱力学",
    )),
    ("410", ("数学", "代数", "幾何", "解析学", "微分", "積分", "確率", "統計")),
    ("430", ("化学", "元素", "分子", "化合物", "化学反応")),
    ("440", ("宇宙", "天文", "銀河", "恒星", "惑星", "ブラックホール")),
    ("451", ("天気", "気象", "台風", "降水", "気温", "気圧", "天候")),
    ("480", ("動物", "哺乳類", "鳥類", "魚類", "昆虫", "犬", "猫")),
    ("460", ("生物", "生命", "細胞", "遺伝", "dna", "進化")),
    ("490", ("医学", "医療", "病気", "疾患", "治療", "薬")),
    ("400", ("科学", "自然科学")),
    ("596", ("料理", "食品", "食べ物", "レシピ", "調理", "食事")),
    ("590", ("家政", "生活科学")),
    ("540", ("電気工学", "電子工学", "回路", "半導体")),
    ("530", ("機械工学", "機械", "ロボット")),
    ("500", ("工学", "技術")),
    ("680", ("交通", "運輸", "鉄道", "自動車", "航空", "船舶")),
    ("610", ("農業", "農作", "作物", "園芸")),
    ("670", ("商業", "流通", "小売", "マーケティング")),
    ("600", ("産業",)),
    ("330", ("経済", "金融", "投資", "市場", "景気")),
    ("320", ("法律", "法学", "憲法", "民法", "刑法")),
    ("370", ("教育", "学校", "学習指導", "授業")),
    ("360", ("社会", "福祉", "社会問題")),
    ("300", ("社会科学",)),
    ("140", ("心理", "心理学", "認知")),
    ("150", ("倫理", "道徳")),
    ("100", ("哲学", "思想", "論理学")),
    ("290", ("地理", "地誌", "旅行", "地域")),
    ("280", ("伝記", "人物史")),
    ("200", ("歴史", "史学")),
    ("780", ("スポーツ", "体育", "運動競技")),
    ("760", ("音楽", "作曲", "演奏", "楽器")),
    ("700", ("芸術", "美術", "絵画", "彫刻", "写真")),
    ("810", ("日本語", "国語", "日本語文法")),
    ("830", ("英語", "英文", "英会話", "英語文法")),
    ("800", ("言語", "言語学", "文法")),
    ("910", ("日本文学", "和歌", "俳句")),
    ("930", ("英文学", "英米文学")),
    ("900", ("文学", "小説", "詩", "物語")),
    ("002", ("学問", "学術", "知識")),
)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def describe(code: str) -> NDCResult:
    code = str(code).zfill(3)
    if not re.fullmatch(r"\d{3}", code):
        raise ValueError(f"invalid NDC code: {code!r}")
    main = code[0]
    name = NDC_CODES.get(code, NDC_MAIN.get(main, ""))
    if not name:
        raise ValueError(f"unsupported NDC code: {code}")
    return NDCResult(
        code=code,
        main=main,
        name=name,
        state="CLASSIFIED",
        source="explicit",
    )


def from_legacy_label(label: str) -> NDCResult:
    code = LEGACY_LABEL_TO_NDC.get(normalize_text(label))
    if code is None:
        return unknown("legacy")
    item = describe(code)
    return NDCResult(
        code=item.code,
        main=item.main,
        name=item.name,
        state="CLASSIFIED",
        source="legacy",
    )


def unknown(source: str = "keyword") -> NDCResult:
    return NDCResult(
        code=None,
        main=None,
        name="未分類",
        state="UNKNOWN",
        source=source,
    )


def classify(text: str, *, legacy_label: Optional[str] = None) -> NDCResult:
    if legacy_label:
        legacy = from_legacy_label(legacy_label)
        if legacy.state == "CLASSIFIED":
            return legacy

    normalized = normalize_text(text)
    if not normalized:
        return unknown()

    for code, keywords in _KEYWORD_RULES:
        if any(normalize_text(keyword) in normalized for keyword in keywords):
            item = describe(code)
            return NDCResult(
                code=item.code,
                main=item.main,
                name=item.name,
                state="CLASSIFIED",
                source="keyword",
            )
    return unknown()


def classify_memory(prompt: str, answer: str = "", *, legacy_label: Optional[str] = None) -> NDCResult:
    return classify(f"{prompt}\n{answer}", legacy_label=legacy_label)


def enrich_memory_item(item: Dict[str, object]) -> Dict[str, object]:
    row = dict(item)
    prompt = str(row.get("prompt", "")).strip()
    answer = str(row.get("answer", "")).strip()

    existing_code = row.get("ndc_code")
    if existing_code not in (None, ""):
        try:
            result = describe(str(existing_code))
            source = str(row.get("ndc_source") or "stored")
            result = NDCResult(result.code, result.main, result.name, "CLASSIFIED", source)
        except ValueError:
            result = classify_memory(
                prompt,
                answer,
                legacy_label=str(row.get("label", "") or ""),
            )
    else:
        result = classify_memory(
            prompt,
            answer,
            legacy_label=str(row.get("label", "") or ""),
        )

    row.update({
        "ndc_code": result.code,
        "ndc_main": result.main,
        "ndc_name": result.name,
        "classification_state": result.state,
        "ndc_source": result.source,
    })
    return row
