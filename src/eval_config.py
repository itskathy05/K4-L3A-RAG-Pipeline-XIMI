"""Cau hinh chung cho A/B evaluation: dense-only vs hybrid+RRF.

A va B khac dung MOT nut (use_reranking). Moi thu khac -- dataset, top_k,
threshold, model, prompt, temperature, embedding -- giu nguyen giua hai
config de phep so sanh co gia tri.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    name: str
    use_reranking: bool


CONFIG_A = RunConfig(name="A_dense", use_reranking=False)
CONFIG_B = RunConfig(name="B_hybrid", use_reranking=True)

DEFAULT_TOP_K = 5
