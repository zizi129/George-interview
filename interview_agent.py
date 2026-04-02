"""Interview Agent: Orchestrator + Planner for structured interview flow control.

The Agent sits between user input (app.py) and LLM output (llm.py).  It owns
the interview phase state machine, selects question cards from the knowledge
layer, and issues structured instructions to the LLM articulation layer.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from basereal import BaseReal
from interview_evaluator import EvalDimension, EvalSignal, InterviewEvaluator
from interview_knowledge import InterviewKnowledge, QuestionCard
from logger import logger

# Module-level knowledge cache: one instance per process, loaded once from disk.
_knowledge_cache: InterviewKnowledge | None = None

def _get_knowledge() -> InterviewKnowledge:
    global _knowledge_cache
    if _knowledge_cache is None:
        _knowledge_cache = InterviewKnowledge()
        _knowledge_cache.load_cards_for_role("")
    return _knowledge_cache


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

class InterviewPhase(Enum):
    OPENING = "opening"
    TECHNICAL = "technical"
    PROJECT = "project"
    STRESS = "stress"
    SOFT_SKILLS = "soft_skills"
    CANDIDATE_QA = "candidate_qa"
    CLOSING = "closing"


@dataclass
class PhaseConfig:
    phase: InterviewPhase
    time_budget_seconds: int
    max_main_questions: int
    max_follow_ups_per_card: int


DEFAULT_PHASE_SEQUENCE: list[PhaseConfig] = [
    PhaseConfig(InterviewPhase.OPENING,      time_budget_seconds=150,  max_main_questions=1, max_follow_ups_per_card=0),
    PhaseConfig(InterviewPhase.TECHNICAL,     time_budget_seconds=540,  max_main_questions=4, max_follow_ups_per_card=2),
    PhaseConfig(InterviewPhase.PROJECT,       time_budget_seconds=540,  max_main_questions=3, max_follow_ups_per_card=2),
    PhaseConfig(InterviewPhase.STRESS,        time_budget_seconds=270,  max_main_questions=2, max_follow_ups_per_card=1),
    PhaseConfig(InterviewPhase.SOFT_SKILLS,   time_budget_seconds=270,  max_main_questions=2, max_follow_ups_per_card=1),
    PhaseConfig(InterviewPhase.CANDIDATE_QA,  time_budget_seconds=150,  max_main_questions=1, max_follow_ups_per_card=0),
]

TOTAL_INTERVIEW_SECONDS = 1800


# ---------------------------------------------------------------------------
# Agent instruction — passed to Articulator (LLM speaking layer)
# ---------------------------------------------------------------------------

@dataclass
class AgentInstruction:
    action: str                        # ask_main | follow_up | transition | open | invite_qa | close
    card: QuestionCard | None = None
    follow_up_hint: str = ""
    phase: InterviewPhase = InterviewPhase.OPENING
    context_summary: str = ""
    constraint: str = ""


# ---------------------------------------------------------------------------
# Agent state — persisted in BaseReal._interview_context["agent_state"]
# ---------------------------------------------------------------------------

def build_default_agent_state() -> dict[str, Any]:
    return {
        "current_phase": InterviewPhase.OPENING.value,
        "current_phase_idx": 0,
        "phase_turn_count": 0,
        "phase_main_question_count": 0,
        "current_card_id": "",
        "follow_up_budget": 0,
        "asked_card_ids": [],
        "phase_history": [],
        "evaluation_snapshot": {},
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class InterviewOrchestrator:
    """Finite state machine that tracks the current interview phase and decides
    when to transition based on time, turn count, and evaluator signals."""

    def __init__(self, phase_sequence: list[PhaseConfig] | None = None):
        self.phases = phase_sequence or list(DEFAULT_PHASE_SEQUENCE)

    def get_current_config(self, state: dict) -> PhaseConfig:
        idx = max(0, min(int(state.get("current_phase_idx", 0)), len(self.phases) - 1))
        return self.phases[idx]

    def should_transition(
        self,
        state: dict,
        elapsed_seconds: float,
        eval_signal: EvalSignal | None = None,
    ) -> bool:
        cfg = self.get_current_config(state)
        phase_turns = int(state.get("phase_turn_count", 0))
        phase_main_qs = int(state.get("phase_main_question_count", 0))
        follow_budget = int(state.get("follow_up_budget", 0))

        phase_start = self._phase_start_time(state)
        phase_elapsed = max(0.0, elapsed_seconds - phase_start)

        if phase_elapsed >= cfg.time_budget_seconds:
            return True

        if phase_main_qs >= cfg.max_main_questions and follow_budget <= 0:
            return True

        if elapsed_seconds >= TOTAL_INTERVIEW_SECONDS - 150:
            return True

        if eval_signal and eval_signal.phase_saturated:
            return True

        return False

    def advance_phase(self, state: dict) -> dict:
        """Move to the next phase. Returns updated state."""
        state = dict(state)
        idx = int(state.get("current_phase_idx", 0))
        current_phase = state.get("current_phase", "opening")
        history = list(state.get("phase_history", []))
        history.append({
            "phase": current_phase,
            "turns": int(state.get("phase_turn_count", 0)),
            "main_questions": int(state.get("phase_main_question_count", 0)),
            "cards_asked": list(state.get("asked_card_ids", [])),
        })

        next_idx = idx + 1
        if next_idx >= len(self.phases):
            state["current_phase"] = InterviewPhase.CLOSING.value
            state["current_phase_idx"] = len(self.phases) - 1
        else:
            state["current_phase"] = self.phases[next_idx].phase.value
            state["current_phase_idx"] = next_idx

        state["phase_turn_count"] = 0
        state["phase_main_question_count"] = 0
        state["current_card_id"] = ""
        state["follow_up_budget"] = 0
        state["phase_history"] = history
        return state

    def is_final_phase(self, state: dict) -> bool:
        idx = int(state.get("current_phase_idx", 0))
        return idx >= len(self.phases) - 1

    def should_close(self, state: dict, elapsed_seconds: float) -> bool:
        if elapsed_seconds >= TOTAL_INTERVIEW_SECONDS:
            return True
        current = state.get("current_phase", "")
        if current == InterviewPhase.CLOSING.value:
            return True
        if current == InterviewPhase.CANDIDATE_QA.value and self.should_transition(state, elapsed_seconds):
            return True
        return False

    def _phase_start_time(self, state: dict) -> float:
        """Estimate when the current phase started, in seconds from interview start.
        We use the sum of budgets of all previous phases as a proxy.
        Since we start at technical (skipping opening), phase_idx=1 means
        we skip the opening budget in the accumulation."""
        idx = int(state.get("current_phase_idx", 0))
        # phases[0] is opening (already done by interview_start, not counted)
        # Sum budgets from phase 1 up to (but not including) current idx
        elapsed_budget = sum(
            self.phases[i].time_budget_seconds
            for i in range(1, min(idx, len(self.phases)))
        )
        return float(elapsed_budget)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class InterviewPlanner:
    """Selects the next question card and manages follow-up budget."""

    def __init__(self, knowledge: InterviewKnowledge):
        self.knowledge = knowledge

    def decide(
        self,
        state: dict,
        phase_config: PhaseConfig,
        context: dict[str, Any],
        eval_signal: EvalSignal | None = None,
    ) -> AgentInstruction:
        phase = InterviewPhase(state.get("current_phase", "opening"))

        if phase == InterviewPhase.OPENING:
            return AgentInstruction(action="open", phase=phase)

        if phase == InterviewPhase.CANDIDATE_QA:
            return AgentInstruction(action="invite_qa", phase=phase)

        if phase == InterviewPhase.CLOSING:
            return AgentInstruction(action="close", phase=phase)

        follow_budget = int(state.get("follow_up_budget", 0))
        current_card_id = state.get("current_card_id", "")

        if follow_budget > 0 and current_card_id:
            card = self.knowledge.get_card_by_id(current_card_id)
            if card:
                hook = self._pick_follow_up_hook(card, context)
                return AgentInstruction(
                    action="follow_up",
                    card=card,
                    follow_up_hint=hook,
                    phase=phase,
                )

        asked_ids = set(state.get("asked_card_ids", []))
        candidates = self.knowledge.get_cards(
            phase=phase.value,
            tags=self._extract_priority_tags(context),
            exclude_ids=asked_ids,
        )

        if not candidates:
            candidates = self.knowledge.get_cards(
                phase=phase.value,
                tags=[],
                exclude_ids=asked_ids,
            )

        if not candidates:
            return AgentInstruction(action="transition", phase=phase)

        card = candidates[0]
        return AgentInstruction(
            action="ask_main",
            card=card,
            phase=phase,
        )

    def update_state_after_decision(
        self,
        state: dict,
        decision: AgentInstruction,
        phase_config: PhaseConfig,
    ) -> dict:
        state = dict(state)

        if decision.action == "ask_main" and decision.card:
            asked = list(state.get("asked_card_ids", []))
            asked.append(decision.card.id)
            state["asked_card_ids"] = asked
            state["current_card_id"] = decision.card.id
            state["follow_up_budget"] = phase_config.max_follow_ups_per_card
            state["phase_main_question_count"] = int(state.get("phase_main_question_count", 0)) + 1

        elif decision.action == "follow_up":
            state["follow_up_budget"] = max(0, int(state.get("follow_up_budget", 0)) - 1)

        state["phase_turn_count"] = int(state.get("phase_turn_count", 0)) + 1
        return state

    def _extract_priority_tags(self, context: dict[str, Any]) -> list[str]:
        """Extract tags from JD and resume to prioritize card selection."""
        tags: list[str] = []
        profile = context.get("resume_profile") or {}
        if isinstance(profile, dict):
            skills = profile.get("skills", [])
            if isinstance(skills, list):
                tags.extend(str(s).lower().strip() for s in skills[:5] if s)
        return tags

    def _pick_follow_up_hook(self, card: QuestionCard, context: dict[str, Any]) -> str:
        if card.follow_up_hooks:
            for hook in card.follow_up_hooks:
                return hook.get("question", hook.get("follow_up", ""))
        return ""


# ---------------------------------------------------------------------------
# Main entry point — replaces direct llm_response call
# ---------------------------------------------------------------------------

def interview_agent_process_turn(message: str, nerfreal: BaseReal) -> None:
    """Called from app.py instead of llm_response. Orchestrates the full
    agent pipeline: state check -> plan -> knowledge -> articulate."""
    from llm import (
        agent_llm_response,
        build_interview_closing,
        _estimate_speech_duration_seconds,
    )

    generation_id = nerfreal.begin_generation()
    start = time.perf_counter()

    try:
        context = nerfreal.get_interview_context()
        elapsed = nerfreal.get_interview_elapsed_seconds()
        agent_state = dict(context.get("agent_state") or build_default_agent_state())

        orchestrator = InterviewOrchestrator()
        # Use module-level cached knowledge — loaded once, reused every turn.
        knowledge = _get_knowledge()
        evaluator = InterviewEvaluator()

        eval_snapshot = agent_state.get("evaluation_snapshot", {})
        if eval_snapshot:
            evaluator.load_snapshot(eval_snapshot)

        eval_signal = evaluator.evaluate_turn_lightweight(
            message,
            agent_state.get("current_card_id", ""),
            knowledge,
        )

        if orchestrator.should_close(agent_state, elapsed):
            closing = build_interview_closing(nerfreal, "面试时间已到，感谢参与")
            if nerfreal.is_generation_current(generation_id):
                nerfreal.add_interview_message("assistant", closing, {"source": "closing"})
                nerfreal.put_msg_txt(closing, {"source": "closing"})
                nerfreal.mark_interview_finished(
                    "agent_time_limit",
                    time.time() + _estimate_speech_duration_seconds(closing),
                )
            return

        if orchestrator.should_transition(agent_state, elapsed, eval_signal):
            if orchestrator.is_final_phase(agent_state):
                agent_state["current_phase"] = InterviewPhase.CLOSING.value
            else:
                agent_state = orchestrator.advance_phase(agent_state)
                logger.info(
                    "agent phase transition: sessionid=%s new_phase=%s elapsed=%ss",
                    nerfreal.sessionid,
                    agent_state["current_phase"],
                    round(elapsed),
                )

        phase_config = orchestrator.get_current_config(agent_state)
        planner = InterviewPlanner(knowledge)
        decision = planner.decide(agent_state, phase_config, context, eval_signal)

        if decision.action == "transition":
            if orchestrator.is_final_phase(agent_state):
                decision = AgentInstruction(action="close", phase=decision.phase)
            else:
                next_phase_value = (
                    orchestrator.phases[agent_state["current_phase_idx"] + 1].phase.value
                    if agent_state["current_phase_idx"] + 1 < len(orchestrator.phases)
                    else "closing"
                )
                decision.context_summary = next_phase_value
                agent_state = orchestrator.advance_phase(agent_state)
                phase_config = orchestrator.get_current_config(agent_state)
                decision_after = planner.decide(agent_state, phase_config, context, eval_signal)
                # If we can immediately ask a card, do so rather than a transition message
                if decision_after.action == "ask_main":
                    decision = decision_after

        agent_state = planner.update_state_after_decision(agent_state, decision, phase_config)
        agent_state["evaluation_snapshot"] = evaluator.get_snapshot()
        # Clear first-turn flag after first processing
        agent_state["is_first_turn"] = False

        nerfreal.set_interview_context(agent_state=agent_state)

        logger.info(
            "agent decision: sessionid=%s phase=%s action=%s card=%s elapsed=%ss",
            nerfreal.sessionid,
            agent_state.get("current_phase"),
            decision.action,
            decision.card.id if decision.card else "none",
            round(elapsed),
        )

        agent_llm_response(decision, nerfreal, generation_id)

    except Exception:
        logger.exception("interview_agent_process_turn")
        if nerfreal.is_generation_current(generation_id):
            fallback = "当前面试官连接异常，请稍后重试，或换个问题继续。"
            nerfreal.add_interview_message("assistant", fallback, {"source": "agent_error"})
            nerfreal.put_msg_txt(fallback, {"source": "agent_error"})
