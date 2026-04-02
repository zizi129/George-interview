"""Interview Evaluator: multi-dimension candidate scoring.

Provides two evaluation modes:
1. Per-turn lightweight (rule-based, no LLM) — fast signal for Orchestrator
2. Per-phase LLM summary (1 call per phase transition) — deeper assessment
"""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from logger import logger


class EvalDimension(Enum):
    TECHNICAL_DEPTH = "technical_depth"
    PRACTICAL_EXPERIENCE = "practical_experience"
    PROBLEM_SOLVING = "problem_solving"
    SYSTEM_THINKING = "system_thinking"
    COMMUNICATION = "communication"
    COLLABORATION = "collaboration"
    LEARNING_POTENTIAL = "learning_potential"
    STRESS_RESILIENCE = "stress_resilience"


DIMENSION_LABELS = {
    EvalDimension.TECHNICAL_DEPTH: "技术深度",
    EvalDimension.PRACTICAL_EXPERIENCE: "实战经验",
    EvalDimension.PROBLEM_SOLVING: "问题分析与解决",
    EvalDimension.SYSTEM_THINKING: "系统思维与设计",
    EvalDimension.COMMUNICATION: "表达与沟通",
    EvalDimension.COLLABORATION: "协作与推动",
    EvalDimension.LEARNING_POTENTIAL: "学习能力与潜力",
    EvalDimension.STRESS_RESILIENCE: "抗压与应变",
}

DEFAULT_SCORE = 50
MIN_SCORE = 0
MAX_SCORE = 100


@dataclass
class EvalSignal:
    """Lightweight signal returned to Orchestrator after each turn."""
    reply_quality: str = "partial"     # full | partial | empty | evasive
    phase_saturated: bool = False
    dimension_updates: dict[str, int] = field(default_factory=dict)


@dataclass
class DimensionScore:
    score: int = DEFAULT_SCORE
    evidence: list[str] = field(default_factory=list)
    turn_count: int = 0


class InterviewEvaluator:
    """Tracks multi-dimension scores across the interview."""

    def __init__(self):
        self._scores: dict[str, DimensionScore] = {
            dim.value: DimensionScore() for dim in EvalDimension
        }
        self._turn_signals: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-turn lightweight evaluation (rule-based, no LLM)
    # ------------------------------------------------------------------

    def evaluate_turn_lightweight(
        self,
        user_reply: str,
        current_card_id: str,
        knowledge: Any,
    ) -> EvalSignal:
        """Fast heuristic evaluation of a single user reply."""
        reply = (user_reply or "").strip()
        signal = EvalSignal()

        if not reply or len(reply) < 10:
            signal.reply_quality = "empty"
            return signal

        card = knowledge.get_card_by_id(current_card_id) if current_card_id else None

        has_numbers = bool(re.search(r"\d+[%％倍万亿kKmM]", reply))
        has_specifics = bool(re.search(
            r"(具体|实际|我们|我做|我负责|当时|结果|效果|提升|降低|优化|线上|上线|部署)",
            reply,
        ))
        reply_length = len(reply)

        if reply_length > 200 and has_specifics:
            signal.reply_quality = "full"
        elif reply_length > 80 and (has_numbers or has_specifics):
            signal.reply_quality = "partial"
        elif reply_length < 30 or re.search(r"(不太清楚|没做过|不了解|不知道)", reply):
            signal.reply_quality = "evasive"
        else:
            signal.reply_quality = "partial"

        if card and card.rubric:
            evaluates = card.rubric.get("evaluates", [])
            good_signals = card.rubric.get("good_signals", [])
            red_flags = card.rubric.get("red_flags", [])

            good_count = sum(1 for s in good_signals if _fuzzy_match(s, reply))
            red_count = sum(1 for f in red_flags if _fuzzy_match(f, reply))

            delta = 0
            if signal.reply_quality == "full":
                delta = 5 + good_count * 3
            elif signal.reply_quality == "partial":
                delta = 2 + good_count * 2
            elif signal.reply_quality == "evasive":
                delta = -3
            delta -= red_count * 4

            for dim_name in evaluates:
                if dim_name in self._scores:
                    ds = self._scores[dim_name]
                    ds.score = max(MIN_SCORE, min(MAX_SCORE, ds.score + delta))
                    ds.turn_count += 1
                    if good_count > 0:
                        ds.evidence.append(f"卡片{current_card_id}: 匹配{good_count}个好信号")
                    if red_count > 0:
                        ds.evidence.append(f"卡片{current_card_id}: 匹配{red_count}个风险信号")
                    signal.dimension_updates[dim_name] = ds.score

        self._turn_signals.append({
            "card_id": current_card_id,
            "quality": signal.reply_quality,
            "reply_length": reply_length,
        })

        return signal

    # ------------------------------------------------------------------
    # Snapshot for persistence in agent_state
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict[str, Any]:
        return {
            "scores": {
                dim: {"score": ds.score, "evidence": ds.evidence[-5:], "turn_count": ds.turn_count}
                for dim, ds in self._scores.items()
            },
            "turn_count": len(self._turn_signals),
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        scores_data = snapshot.get("scores", {})
        for dim_name, data in scores_data.items():
            if dim_name in self._scores:
                ds = self._scores[dim_name]
                ds.score = int(data.get("score", DEFAULT_SCORE))
                ds.evidence = list(data.get("evidence", []))
                ds.turn_count = int(data.get("turn_count", 0))

    # ------------------------------------------------------------------
    # Report generation helpers
    # ------------------------------------------------------------------

    def get_dimension_summary(self) -> dict[str, dict[str, Any]]:
        result = {}
        for dim in EvalDimension:
            ds = self._scores[dim.value]
            result[dim.value] = {
                "label": DIMENSION_LABELS.get(dim, dim.value),
                "score": ds.score,
                "evidence": ds.evidence[-5:],
                "turn_count": ds.turn_count,
            }
        return result

    def get_overall_score(self) -> int:
        scored_dims = [ds for ds in self._scores.values() if ds.turn_count > 0]
        if not scored_dims:
            return DEFAULT_SCORE
        return round(sum(ds.score for ds in scored_dims) / len(scored_dims))


def _fuzzy_match(pattern: str, text: str) -> bool:
    """Check if the core keywords from *pattern* appear in *text*."""
    keywords = re.findall(r"[\u4e00-\u9fffA-Za-z]+", pattern)
    if not keywords:
        return False
    matched = sum(1 for kw in keywords if kw.lower() in text.lower())
    return matched >= max(1, len(keywords) // 2)
