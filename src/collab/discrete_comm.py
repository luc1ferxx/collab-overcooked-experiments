from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TOKEN_DICTIONARY = (
    "Token dictionary:\n"
    "S1(target) -> clarify ambiguous assignment (state who acts where).\n"
    "B1(resource) -> resolve conflict by yielding or reassigning.\n"
    "H1(object) -> initiate/confirm handoff of the named object.\n"
    "F1(context) -> failure or error recovery instructions.\n"
)

def _hash(items: List[str]) -> str:
    joined = "|".join(items)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


@dataclass
class TriggerDecision:
    mode: str
    tokens: List[str] = field(default_factory=list)
    reason: str = ""
    fallback: bool = False


class TriggerDetector:
    """Heuristic trigger detector combining ideas from the cited papers."""

    def __init__(self, agent):
        self.agent = agent
        self._last_error_count = 0

    def detect(self, state) -> Tuple[List[str], Dict[str, bool]]:  # pragma: no cover
        tokens: List[str] = []
        flags: Dict[str, bool] = {
            "ambiguity": False,
            "conflict": False,
            "handoff": False,
            "failure": False,
        }

        teammate = getattr(self.agent, "teammate", None)

        # Ambiguity: both queues empty -> need direction
        if (
            self.agent.action_wait_parse.qsize() == 0
            and not self.agent.current_ml_action
            and self.agent.teammate.action_wait_parse.qsize() == 0
            and not self.agent.teammate.current_ml_action
        ):
            tokens.append("S1(idle)")
            flags["ambiguity"] = True

        # First turn: proactively send instruction request if nothing queued
        if (
            getattr(self.agent, "current_timestep", 0) == 0
            and not tokens
        ):
            tokens.append("S1(need_instruction)")
            flags["ambiguity"] = True

        # Additional ambiguity: both agents have different first actions but share goal tile
        if teammate is not None:
            my_next = self._peek_action(self.agent)
            teammate_next = self._peek_action(teammate)
            if my_next and teammate_next and my_next != teammate_next:
                if self._same_target(my_next, teammate_next):
                    tokens.append(f"S1({my_next})")
                    flags["ambiguity"] = True

        # Conflict: both agents pursuing same high-level action
        if (
            self.agent.current_ml_action
            and self.agent.current_ml_action
            == self.agent.teammate.current_ml_action
            and self.agent.current_ml_action is not None
        ):
            tokens.append(f"B1({self.agent.current_ml_action})")
            flags["conflict"] = True

        # Handoff: agent holds object, teammate empty, adjacent tiles
        try:
            players = state.players
            me = players[self.agent.agent_index]
            teammate = players[1 - self.agent.agent_index]
            if getattr(me, "held_object", None) and not getattr(
                teammate, "held_object", None
            ):
                dist = abs(me.position[0] - teammate.position[0]) + abs(
                    me.position[1] - teammate.position[1]
                )
                if dist <= 1:
                    obj = getattr(me.held_object, "name", "object")
                    tokens.append(f"H1({obj})")
                    flags["handoff"] = True
        except Exception:
            pass

        # Failure trigger based on validator errors
        try:
            err = self.agent.turn_statistics_dict["statistical_data"]["error"][
                self.agent.agent_index
            ]["validator_error"]["error_num"]
            if err > self._last_error_count:
                tokens.append("F1(latest)")
                self._last_error_count = err
                flags["failure"] = True
        except Exception:
            pass

        return tokens, flags

    def _peek_action(self, agent) -> Optional[str]:
        if agent.current_ml_action:
            return agent.current_ml_action
        if agent.action_wait_parse.qsize() > 0:
            return agent.action_wait_parse.queue[0]
        return None

    def _same_target(self, action_a: str, action_b: str) -> bool:
        def _extract(action: str) -> str:
            if "(" in action:
                return action.split("(", 1)[1].rstrip(")")
            return action

        return _extract(action_a) == _extract(action_b)


class MessagePruner:
    def __init__(self, window: int = 3):
        self.window = window
        self.deque: deque[str] = deque(maxlen=window)

    def prune(self, tokens: List[str]) -> List[str]:
        pruned: List[str] = []
        for token in tokens:
            if token not in self.deque:
                pruned.append(token)
                self.deque.append(token)
        return pruned

    def reset(self):
        self.deque.clear()


class NoProgressWatchdog:
    def __init__(self, window: int = 3):
        self.window = window
        self.history: deque[str] = deque(maxlen=window)

    def observe(self, state, agent) -> bool:  # pragma: no cover - env heavy
        fingerprint = self._fingerprint(state, agent)
        self.history.append(fingerprint)
        if len(self.history) < self.window:
            return False
        return len(set(self.history)) == 1

    def reset(self):
        self.history.clear()

    def _fingerprint(self, state, agent) -> str:
        try:
            players = state.players
            me = players[agent.agent_index]
            teammate = players[1 - agent.agent_index]
            held = getattr(me.held_object, "name", "none") if me.held_object else "none"
            teem = (
                getattr(teammate.held_object, "name", "none")
                if teammate.held_object
                else "none"
            )
            queue = list(agent.action_wait_parse.queue)
            return _hash(
                [
                    str(me.position),
                    held,
                    str(teammate.position),
                    teem,
                    ";".join(queue),
                ]
            )
        except Exception:
            return "unknown"


class DiscreteCommManager:
    """Implements event-triggered discrete communication pipeline."""

    def __init__(self, agent, enable_trigger: bool = True, force_always_llm: bool = False):
        self.agent = agent
        self.teammate_name: Optional[str] = None
        self.detector = TriggerDetector(agent)
        self.pruner = MessagePruner()
        self.watchdog = NoProgressWatchdog()
        self.current_decision: Optional[TriggerDecision] = None
        self.metrics: Dict[str, int] = {"silent_turns": 0, "fallback_count": 0}
        self.enable_trigger = enable_trigger
        self.force_always_llm = force_always_llm

    def set_teammate(self, name: str):
        self.teammate_name = name

    def reset(self):
        self.pruner.reset()
        self.watchdog.reset()
        self.current_decision = None

    # -- Turn lifecycle -------------------------------------------------

    def begin_turn(self, state) -> TriggerDecision:
        if self.force_always_llm:
            decision = TriggerDecision(mode="llm", tokens=[], reason="always_force")
            self.current_decision = decision
            self._log_decision(decision, {"force": True})
            return decision

        if not self.enable_trigger:
            decision = TriggerDecision(mode="llm", tokens=[], reason="trigger_disabled")
            self.current_decision = decision
            self._log_decision(decision, {"trigger": False})
            return decision

        tokens, flags = self.detector.detect(state)
        tokens = self.pruner.prune(tokens)

        fallback = False
        if not tokens and self.watchdog.observe(state, self.agent):
            tokens.append(f"S2({self.agent.name.lower()})")
            fallback = True

        if not tokens:
            decision = TriggerDecision(mode="silent", tokens=[], reason="no_trigger")
            self.metrics["silent_turns"] += 1
        else:
            decision = TriggerDecision(
                mode="llm", tokens=tokens, reason="trigger", fallback=fallback
            )
            if fallback:
                self.metrics["fallback_count"] += 1

        self.current_decision = decision
        self._log_decision(decision, flags)
        return decision

    def prepare_prompt(self, state_prompt: str, base_prompt: str, role: str) -> str:
        if not self.current_decision:
            return state_prompt + base_prompt

        summary = self._summarize_state(state_prompt)
        dialogue = self._truncate_dialog(base_prompt)

        if self.current_decision.mode == "silent":
            header = (
                "<DiscreteProtocol>\nTokens: NONE\n"
                "Respond with 'analysis: [NOTHING]', keep the existing plan, and 'say: [NOTHING]'.\n"
                "</DiscreteProtocol>\n"
            )
            return header + summary

        token_line = (
            "Tokens: " + " ".join(self.current_decision.tokens)
            if self.current_decision.tokens
            else "Tokens: NONE"
        )
        header = (
            "<DiscreteProtocol>\n"
            f"{token_line}\n"
            "Use the tokens exactly in the 'say' field. Keep analysis <= 1 sentence and keep the plan to <=2 actions.\n"
        )
        if self.current_decision.fallback:
            header += "Fallback requested: add one short clarification sentence.\n"
        header += "</DiscreteProtocol>\n"

        response_hint = (
            "Respond with three lines: 'analysis:', 'plan:' (at most two actions), and 'say:' (tokens first)."
        )

        parts = [header, summary]
        if dialogue:
            parts.append("<Dialogue>\n" + dialogue + "\n</Dialogue>")
        parts.append(response_hint)
        return "\n".join(parts)

    def build_silent_response(self) -> str:
        plan = self._plan_snapshot()
        name = self.agent.name
        return f"{name} analysis: [NOTHING]\n{name} plan: {plan}\n{name} say: [NOTHING]"

    def build_token_response(self) -> str:
        tokens = " ".join(self.current_decision.tokens) if self.current_decision else "[NOTHING]"
        plan = self._plan_snapshot()
        name = self.agent.name
        return f"{name} analysis: [NOTHING]\n{name} plan: {plan}\n{name} say: {tokens}"

    def record_response(self, state, response: str):  # pragma: no cover
        # Feed the response back into watchdog so it can reset if progress occurs
        if self.current_decision and self.current_decision.mode == "llm":
            self.watchdog.history.clear()
        self.current_decision = None

    def force_llm(self, reason: str) -> TriggerDecision:
        decision = TriggerDecision(mode="llm", tokens=[], reason=reason, fallback=False)
        self.current_decision = decision
        self._log_decision(decision, {"force": True})
        return decision

    def _log_decision(self, decision: TriggerDecision, flags: Dict[str, bool]):
        if hasattr(self.agent, "discrete_comm_trace"):
            self.agent.discrete_comm_trace.append({
                "t": getattr(self.agent, "current_timestep", -1),
                "decision": decision.mode,
                "tokens": decision.tokens,
                "reason": decision.reason,
                "fallback": decision.fallback,
                "flags": flags,
            })

    def _plan_snapshot(self) -> str:
        actions: List[str] = []
        if getattr(self.agent, "current_ml_action", None):
            actions.append(self.agent.current_ml_action)
        queue_actions = list(self.agent.action_wait_parse.queue)
        for act in queue_actions:
            if len(actions) >= 2:
                break
            actions.append(act)
        if not actions:
            actions = ["wait(1)"]
        return ";".join(actions)

    def _summarize_state(self, state_prompt: str) -> str:
        lines = [line for line in state_prompt.splitlines() if line.strip()]
        if not lines:
            return ""
        keep = lines[:6]
        return "\n".join(keep)

    def _truncate_dialog(self, dialogue: str) -> str:
        lines = [line for line in dialogue.splitlines() if line.strip()]
        if not lines:
            return ""
        return "\n".join(lines[:6])


