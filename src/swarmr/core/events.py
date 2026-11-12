"""Interpreting a LangGraph update stream as delegation events.

One place decides what a stream chunk *means*; the terminal renderer and the MCP
job log both consume the result. Without this, each would re-derive subagent
attribution from raw messages and the two would drift.

LangGraph's subgraph namespace does not carry the subagent's name, so
attribution is recovered by mapping each `task` tool_call_id to the
subagent_type it was dispatched with. The matching ToolMessage then identifies
the verdict exactly, even with several subagents running concurrently.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from swarmr.core.attribution import Attribution
from swarmr.core.briefing import effective_briefing
from swarmr.core.harness import (
    DESCRIPTION_KEY,
    PLAN_TOOL,
    PROMPT_KEY,
    SUBAGENT_KEY,
    TASK_TOOL,
)
from swarmr.core.text import clip, content_text, flatten

__all__ = ["Event", "EventKind", "EventReader"]


class EventKind(StrEnum):
    PLAN = "plan"
    DISPATCH = "dispatch"
    AUDIT_INPUT = "audit_input"
    CALL = "call"
    RESULT = "result"
    VERDICT = "verdict"
    REPORT = "report"
    FILED_REPORT = "filed_report"


@dataclass(frozen=True, slots=True)
class Event:
    """One observable thing that happened during a run.

    Attributes:
        kind: What happened.
        who: Subagent name for DISPATCH/VERDICT/AUDIT_INPUT, tool name for
            CALL/RESULT, empty otherwise.
        text: Human-readable detail, already trimmed of raw payloads.
        root: True when produced by the root agent rather than a subagent.
        stream: Short id of the LangGraph subgraph that produced it. Subagents
            run concurrently, so their tool calls interleave in one event
            sequence; this is the only way to tell them apart.
    """

    kind: EventKind
    who: str
    text: str
    root: bool
    stream: str = ""


_ARG_LIMIT = 70


def _argument(value: Any) -> str:
    """Render one tool argument.

    A file body or manifest is measured rather than clipped: "3000 chars" tells a
    reader what happened, where the first seventy characters of it do not.
    """
    text = flatten(value)
    if len(text) <= _ARG_LIMIT:
        return text
    if " " not in text[:_ARG_LIMIT] or len(text) > 400:
        return f"<{len(text)} chars>"
    return clip(text, _ARG_LIMIT)


class EventReader:
    """Stateful reader: feed it stream chunks, get events.

    Stateful because verdict attribution needs the dispatch that preceded it.
    """

    def __init__(
        self,
        audit_agents: tuple[str, ...] = (),
        report_tool: str = "",
        render_report: Any = None,
        request: str = "",
        attribution: Attribution | None = None,
        briefed_agents: tuple[str, ...] = (),
    ) -> None:
        self.audit_agents = audit_agents
        # Everyone whose first briefing the middleware passes through untouched.
        self._exempt = audit_agents + briefed_agents
        # The run's tag -> subagent map, shared with the middleware that fills
        # it. Owned by the run, so two concurrent runs never see each other's
        # names; a reader built without one simply shows raw stream tags.
        self.attribution = attribution or Attribution()
        # The caller's request, so a first-round dispatch is displayed as the
        # briefing the subagent actually receives rather than the prose the
        # orchestrator wrote and the harness replaced.
        self.request = request
        self._briefed: set[str] = set()
        # A team may nominate a tool whose arguments *are* the final report.
        # Capturing it here means the report is taken from a deliberate action
        # rather than from whichever prose happened to arrive last.
        self.report_tool = report_tool
        self.render_report = render_report
        # Call ids of report filings, so their receipts are not echoed as
        # ordinary tool results. Pending state: an id is discarded once its
        # receipt arrives, so this set cannot answer "has the report been
        # filed?" — `_report_is_filed` is separate for exactly that reason.
        self._filed: set[str] = set()
        self._dispatched: dict[str, str] = {}
        self._calls: dict[str, str] = {}
        # Latched, never cleared. The filing IS the deliverable, so prose after
        # it is at best a duplicate, and it is where junk lands: one provider
        # returned a literal placeholder, thinking signature and all, as
        # assistant text for a thinking-only final message. Provider-agnostic.
        self._report_is_filed = False

    def read(self, namespace: tuple[str, ...], update: dict[Any, Any]) -> Iterator[Event]:
        root = not namespace
        stream = self.attribution.label(namespace)
        for payload in update.values():
            if not isinstance(payload, dict):
                continue
            for message in payload.get("messages", []):
                kind = getattr(message, "type", "")
                if kind == "ai":
                    yield from self._ai(message, root, stream)
                elif kind == "tool":
                    yield from self._tool(message, root, stream)

    def _ai(self, message: Any, root: bool, stream: str = "") -> Iterator[Event]:
        for call in getattr(message, "tool_calls", None) or []:
            yield from self._call(call, root, stream)
        # A subagent's closing prose IS its task return value, surfaced as a
        # VERDICT when the ToolMessage arrives. Emitting it here too would
        # duplicate every verdict verbatim.
        if not root:
            return
        if self._report_is_filed:
            return
        if text := content_text(message.content):
            yield Event(EventKind.REPORT, "", text, root, stream)

    def _call(
        self, call: dict[str, Any], root: bool, stream: str = ""
    ) -> Iterator[Event]:
        args = call.get("args", {}) or {}
        name = call["name"]
        call_id = call.get("id", "")

        if name == TASK_TOOL:
            who = args.get(SUBAGENT_KEY, "?")
            self._dispatched[call_id] = who
            authored = args.get(DESCRIPTION_KEY) or args.get(PROMPT_KEY, "")
            normalised = effective_briefing(
                self.request, who, self._exempt, self._briefed
            )
            self._briefed.add(who)
            summary = normalised if normalised is not None else authored
            yield Event(EventKind.DISPATCH, who, flatten(summary), root, stream)
            if who in self.audit_agents:
                body = args.get(PROMPT_KEY) or args.get(DESCRIPTION_KEY) or ""
                yield Event(EventKind.AUDIT_INPUT, who, body.strip(), root, stream)
            return

        if self.report_tool and name == self.report_tool:
            self._filed.add(call_id)
            self._report_is_filed = True
            rendered = self.render_report(args) if callable(self.render_report) else args
            yield Event(EventKind.FILED_REPORT, "", str(rendered), root, stream)
            return

        if name == PLAN_TOOL:
            for todo in args.get("todos", []):
                yield Event(
                    EventKind.PLAN, "", flatten(todo.get("content")), root, stream
                )
            return

        shown = ", ".join(
            f"{key}={_argument(value)}"
            for key, value in args.items()
            if value not in (None, "", False)
        )
        self._calls[call_id] = name
        yield Event(EventKind.CALL, name, f"{name}({shown})", root, stream)

    def _tool(self, message: Any, root: bool, stream: str = "") -> Iterator[Event]:
        # Flattened, not str()'d: a tool result may arrive as content blocks, and
        # repr'ing that produces Python syntax where a verdict should be.
        content = message.content
        text = content_text(content) or (content if isinstance(content, str) else "")
        call_id = getattr(message, "tool_call_id", "")
        if call_id in self._filed:
            # The filing receipt carries no information; the report was already
            # emitted from the call arguments.
            self._filed.discard(call_id)
            return
        if who := self._dispatched.get(call_id, ""):
            yield Event(EventKind.VERDICT, who, text.strip(), root, stream)
            return
        tool_name = self._calls.pop(call_id, "")
        yield Event(EventKind.RESULT, tool_name, text, root, stream)
