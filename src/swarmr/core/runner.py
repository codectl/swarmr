"""Run a team's graph, streamed to a terminal or collected for a caller.

Both entrypoints walk the same stream through the same `EventReader`, so a run
behaves and reports identically whether it was started from the CLI or over MCP.
The only difference is what the observer does with each event.

A run owns its own `RunContext`, which is what lets two investigations proceed at
once: subagent attribution used to live in module state that every run cleared on
entry, so a second run silently wiped the first one's names.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from swarmr.core.events import Event, EventKind, EventReader
from swarmr.core.render import Renderer
from swarmr.core.report import format_usage
from swarmr.core.team import Team
from swarmr.core.text import content_text
from swarmr.core.usage import UsageTracker

__all__ = ["Observer", "run", "run_streamed"]

Observer = Callable[[Event], None]
TargetHook = Callable[[str], None]


def run(
    team: Team,
    request: str,
    observe: Observer | None = None,
    on_target: TargetHook | None = None,
    usage: UsageTracker | None = None,
    params: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Run the team once. Returns (report, banner).

    Args:
        team: The team to run.
        request: The symptom or task, as prose.
        observe: Called for every event as it happens. The CLI passes a
            renderer; the MCP job passes a progress recorder.
        on_target: Called with the target banner as soon as the team has
            profiled it, before the first token. Lets a caller show what is
            being investigated while the run is still going.
        usage: Token accountant. Pass one to get exact per-agent counts;
            it is filled in as the run proceeds, so a caller may read it while
            the run is still going.
        params: The caller's values for the team's declared `Param`s. Checked
            by `Team.context` before anything is built.
    """
    context = team.context(params)
    build = team.build(context)
    if on_target is not None:
        on_target(build.banner)

    reader = EventReader(
        audit_agents=team.audit_agents,
        report_tool=team.report_tool,
        render_report=team.render_report,
        request=request,
        attribution=context.attribution,
    )
    streamed_report = ""
    filed_report = ""
    final_state: dict[str, Any] | None = None
    config: dict[str, Any] = {"recursion_limit": team.recursion_limit}
    if usage is not None:
        config["callbacks"] = [usage]

    for chunk in build.graph.stream(
        {"messages": [{"role": "user", "content": request}]},
        config,
        stream_mode=["updates", "values"],
        subgraphs=True,
    ):
        namespace, mode, payload = cast(
            tuple[tuple[str, ...], str, dict[Any, Any]], chunk
        )
        if mode == "values":
            if not namespace and isinstance(payload, dict):
                final_state = payload
            continue
        for event in reader.read(namespace, payload):
            if observe is not None:
                observe(event)
            if event.kind is EventKind.FILED_REPORT:
                filed_report = event.text
            elif event.kind is EventKind.REPORT:
                streamed_report = event.text

    report = (
        filed_report
        or _structured_report(final_state)
        or _final_report(final_state)
        or streamed_report
    )
    return report or "the run produced no final report", build.banner


def _structured_report(state: dict[str, Any] | None) -> str:
    """The team's structured answer, when it declared a response_format.

    Preferred over prose: the harness requires the model to emit it, so it
    cannot be skipped the way a closing message can. Teams supply a `render`
    method; anything else falls back to its string form.
    """
    if not state:
        return ""
    response = state.get("structured_response")
    if response is None:
        return ""
    render = getattr(response, "render", None)
    if callable(render):
        return str(render()).strip()
    return str(response).strip()


def _final_report(state: dict[str, Any] | None) -> str:
    """The last assistant answer in the finished state.

    Skips messages that only carry tool calls, and flattens content blocks:
    providers may return a list of blocks rather than a string, and treating
    that as "no text" is what silently drops the real report.
    """
    if not state:
        return ""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue
        if text := content_text(getattr(message, "content", "")):
            return text
    return ""


def run_streamed(
    team: Team, request: str, colour: bool = True, params: Mapping[str, str] | None = None
) -> str:
    """Run with live terminal output. Returns the final report."""
    renderer = Renderer(
        colour=colour,
        orchestrator=team.orchestrator,
        digest=team.digest,
        is_error=team.is_error,
    )
    tracker = UsageTracker(root_label=team.orchestrator)
    renderer.field("team", team.name)
    renderer.field("request", request)
    report, _ = run(
        team,
        request,
        observe=renderer.show,
        on_target=lambda banner: renderer.field("target", banner),
        usage=tracker,
        params=params,
    )
    renderer.note(format_usage(tracker.snapshot()))
    return report
