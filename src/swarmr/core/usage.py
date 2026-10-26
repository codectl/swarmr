"""Exact per-agent token accounting.

Every provider response carries `usage_metadata`, and Deep Agents tags each
subagent's model calls with `lc_agent_name`. Joining those two gives real token
counts per agent with no external service and no estimation.

LangSmith is complementary, not required: set LANGSMITH_TRACING=true and
LANGSMITH_API_KEY and the same runs also appear there with full traces. This
module keeps working either way, which matters because the numbers are useful
offline and in CI.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

__all__ = ["AgentUsage", "UsageTracker", "langsmith_enabled"]


def langsmith_enabled() -> bool:
    """Whether LangSmith tracing is configured for this process."""
    flag = os.environ.get("LANGSMITH_TRACING", os.environ.get("LANGCHAIN_TRACING_V2", ""))
    return flag.strip().lower() in {"1", "true", "yes"} and bool(
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    )


@dataclass(slots=True)
class AgentUsage:
    """Token totals for one agent."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True, eq=False)
class UsageTracker(BaseCallbackHandler):
    """Aggregates token usage per subagent.

    Thread-safe because subagents run concurrently on their own threads inside
    one graph.

    `eq=False` keeps identity hashing: langchain merges callback handlers into
    a `set` when the summarization middleware invokes the summary model, and a
    dataclass with the generated `__eq__` is unhashable there.
    """

    # Label for the root agent, whose model calls carry no lc_agent_name. The
    # team names it; `core` must not assume "commander" or any other role.
    root_label: str = "orchestrator"
    per_agent: dict[str, AgentUsage] = field(default_factory=dict)
    _owners: dict[UUID, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # BaseCallbackHandler opts out of tracing internals it does not need.
    @property
    def ignore_chain(self) -> bool:
        return True

    @property
    def ignore_agent(self) -> bool:
        return True

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Remember which agent owns this model call.

        `lc_agent_name` is absent exactly for the root agent, so the fallback is
        meaningful rather than a catch-all.
        """
        owner = str((metadata or {}).get("lc_agent_name") or self.root_label)
        with self._lock:
            self._owners[run_id] = owner

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            owner = self._owners.pop(run_id, self.root_label)
            entry = self.per_agent.setdefault(owner, AgentUsage())
            entry.calls += 1
            usage = _usage_of(response)
            entry.input_tokens += int(usage.get("input_tokens") or 0)
            entry.output_tokens += int(usage.get("output_tokens") or 0)
            details = usage.get("output_token_details") or {}
            entry.reasoning_tokens += int(details.get("reasoning") or 0)
            input_details = usage.get("input_token_details") or {}
            entry.cached_tokens += int(input_details.get("cache_read") or 0)

    def totals(self) -> AgentUsage:
        with self._lock:
            total = AgentUsage()
            for entry in self.per_agent.values():
                total.calls += entry.calls
                total.input_tokens += entry.input_tokens
                total.output_tokens += entry.output_tokens
                total.reasoning_tokens += entry.reasoning_tokens
                total.cached_tokens += entry.cached_tokens
            return total

    def snapshot(self) -> dict[str, Any]:
        """Serialisable usage report, biggest consumer first."""
        with self._lock:
            rows = {
                name: {
                    "calls": u.calls,
                    "input": u.input_tokens,
                    "output": u.output_tokens,
                    "reasoning": u.reasoning_tokens,
                    "cached": u.cached_tokens,
                    "total": u.total_tokens,
                }
                for name, u in sorted(
                    self.per_agent.items(), key=lambda kv: -kv[1].total_tokens
                )
            }
        total = self.totals()
        return {
            "per_agent": rows,
            "total_tokens": total.total_tokens,
            "model_calls": total.calls,
            "langsmith": langsmith_enabled(),
        }


def _usage_of(response: LLMResult) -> dict[str, Any]:
    """Pull usage_metadata off a response, whichever shape the provider used."""
    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            if usage := getattr(message, "usage_metadata", None):
                return dict(usage)
    output = response.llm_output or {}
    if usage := output.get("usage_metadata") or output.get("token_usage"):
        # OpenAI-compatible providers name these differently.
        return {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get(
                "output_tokens", usage.get("completion_tokens", 0)
            ),
        }
    return {}
