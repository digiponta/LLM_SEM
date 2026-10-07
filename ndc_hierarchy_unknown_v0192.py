#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM_SEM v0.19.2 Coverage + Expanded UNKNOWN Router.

Keeps:
  - v0.19.1 coverage-expanded 3-digit prototypes
  - v0.18.x stable beam / rescue / keyword-prior logic

Changes only:
  - replace the main contrastive gate with ExpandedUnknownNDCRouter
"""

from __future__ import annotations

from ndc_hierarchy_coverage_v0191 import CoverageExpandedNDCRouter
from ndc_unknown_expanded_v0192 import ExpandedUnknownNDCRouter


class CoverageUnknownNDCRouter(CoverageExpandedNDCRouter):
    def __init__(self, model, tokenizer, **kwargs) -> None:
        super().__init__(model, tokenizer, **kwargs)

        self.main_router = ExpandedUnknownNDCRouter(
            model,
            tokenizer,
            pooling=self.pooling,
        )
