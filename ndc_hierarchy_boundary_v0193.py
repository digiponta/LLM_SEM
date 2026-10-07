#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.3 Boundary Reinforcement Router.

Builds on v0.19.2 and reinforces only the remaining weak boundaries:
  - 150 ethics
  - 547 communication engineering
  - 548 information engineering vs 007 information science

No encoder retraining and no threshold retuning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch.nn.functional as F

from ndc_semantic_router_v0182 import encode_text
from ndc_hierarchy_unknown_v0192 import CoverageUnknownNDCRouter


BOUNDARY_EXPANSION: Dict[str, Tuple[str, ...]] = {
    "150": (
        "行為の善悪や望ましさを倫理的に考える",
        "価値判断と道徳的な行動基準を扱う",
        "正しい行為とは何かを倫理学で考察する",
    ),
    "547": (
        "無線信号を送受信する通信技術",
        "情報を電波で遠距離伝送する通信工学",
        "通信回線で信号を送る仕組みを研究する",
    ),
    "548": (
        "計算機システムの構成を情報工学として研究する",
        "コンピュータシステムを設計する情報工学",
        "情報処理システムの構成技術を扱う",
        "計算機アーキテクチャと情報工学を研究する",
    ),
}


class BoundaryReinforcedNDCRouter(CoverageUnknownNDCRouter):
    def __init__(self, model, tokenizer, **kwargs) -> None:
        super().__init__(model, tokenizer, **kwargs)

        for code, texts in BOUNDARY_EXPANSION.items():
            extra = tuple(
                F.normalize(
                    encode_text(
                        model,
                        tokenizer,
                        text,
                        pooling=self.pooling,
                    ),
                    p=2,
                    dim=-1,
                )
                for text in texts
            )
            self.code_prototypes[code] = tuple(self.code_prototypes[code]) + extra
