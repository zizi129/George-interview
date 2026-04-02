"""Interview Knowledge Layer: structured question card loading and retrieval.

Cards are JSON files under ``interview_knowledge/cards/<phase>/``.  The loader
indexes them for fast lookup by phase, tags, and ID.  A future ``enrich_card``
method will integrate RAG retrieval for up-to-date technical content.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from logger import logger

CARDS_ROOT = Path(__file__).resolve().parent / "interview_knowledge" / "cards"


@dataclass
class QuestionCard:
    id: str
    phase: str
    tags: list[str] = field(default_factory=list)
    difficulty: str = "mid"
    stem: str = ""
    suggested_question: str = ""
    follow_up_hooks: list[dict[str, str]] = field(default_factory=list)
    rubric: dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuestionCard:
        return cls(
            id=str(data.get("id", "")),
            phase=str(data.get("phase", "")),
            tags=[str(t) for t in data.get("tags", []) if t],
            difficulty=str(data.get("difficulty", "mid")),
            stem=str(data.get("stem", "")),
            suggested_question=str(data.get("suggested_question", "")),
            follow_up_hooks=data.get("follow_up_hooks") or [],
            rubric=data.get("rubric") or {},
            priority=int(data.get("priority", 0)),
        )


class InterviewKnowledge:
    """Loads and indexes question cards.  Provides retrieval by phase/tags."""

    def __init__(self, cards_root: Path | str | None = None):
        self._cards_root = Path(cards_root) if cards_root else CARDS_ROOT
        self._cards_by_id: dict[str, QuestionCard] = {}
        self._cards_by_phase: dict[str, list[QuestionCard]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_cards_for_role(self, job_title: str = "") -> None:
        """Load all cards from disk.  ``job_title`` is reserved for future
        per-role filtering; currently all cards are loaded."""
        if self._loaded:
            return
        self._cards_by_id.clear()
        self._cards_by_phase.clear()

        if not self._cards_root.is_dir():
            logger.warning("cards directory not found: %s", self._cards_root)
            self._loaded = True
            return

        for json_path in sorted(self._cards_root.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        self._index_card(item)
                elif isinstance(data, dict):
                    self._index_card(data)
            except Exception:
                logger.exception("failed to load card: %s", json_path)

        self._loaded = True
        logger.info(
            "knowledge loaded: %d cards across %d phases",
            len(self._cards_by_id),
            len(self._cards_by_phase),
        )

    def _index_card(self, data: dict[str, Any]) -> None:
        card = QuestionCard.from_dict(data)
        if not card.id or not card.phase:
            return
        self._cards_by_id[card.id] = card
        self._cards_by_phase.setdefault(card.phase, []).append(card)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_card_by_id(self, card_id: str) -> QuestionCard | None:
        return self._cards_by_id.get(card_id)

    def get_cards(
        self,
        phase: str,
        tags: list[str] | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[QuestionCard]:
        """Return cards for *phase*, optionally boosted by *tags*, excluding
        already-asked IDs.  Results are sorted by relevance (tag overlap then
        priority)."""
        candidates = list(self._cards_by_phase.get(phase, []))
        if exclude_ids:
            candidates = [c for c in candidates if c.id not in exclude_ids]

        if not tags:
            candidates.sort(key=lambda c: -c.priority)
            return candidates

        tag_set = {t.lower().strip() for t in tags if t}

        def _score(card: QuestionCard) -> tuple[int, int]:
            overlap = sum(1 for t in card.tags if t.lower() in tag_set)
            return (-overlap, -card.priority)

        candidates.sort(key=_score)
        return candidates

    def get_all_cards(self) -> list[QuestionCard]:
        return list(self._cards_by_id.values())

    # ------------------------------------------------------------------
    # RAG extension point (P2)
    # ------------------------------------------------------------------

    def enrich_card(self, card: QuestionCard, context: dict[str, Any]) -> QuestionCard:
        """Future: retrieve relevant technical content via RAG and attach it
        to the card for the Articulator to reference.  Currently a no-op."""
        return card
