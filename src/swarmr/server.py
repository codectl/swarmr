"""MCP server: one tool per registered team, plus a shared job poller.

Composability boundary for omp, pi, Claude Code and anything else speaking MCP.
The server iterates `REGISTRY`, so adding or deleting a team never touches this
file.

Runs take minutes and MCP is request/response with a client-side timeout, so the
surface is a job API: `start_<team>` returns an id immediately, `check_task`
polls. Each poll carries the delegation trail — which specialist was dispatched,
what it called, what it concluded — so the caller can watch the team work instead
of staring at an opaque "running".
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Mapping
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from swarmr.core.jobs import Job, JobState, JobStore
from swarmr.core.report import (
    format_roster,
    format_snapshot,
    format_snapshot_line,
)
from swarmr.core.runner import run
from swarmr.core.team import Team
from swarmr.core.usage import UsageTracker
from swarmr.teams import get, names

__all__ = ["build_server", "main"]

# A long-poll must return well inside any client's own request timeout, and a
# short slice keeps the loop responsive without busy-spinning.
_MAX_WAIT_SECONDS = 120
_POLL_SLICE_SECONDS = 0.5

_JOBS = JobStore()


def _work(
    team: Team, request: str, params: Mapping[str, str]
) -> Callable[[Job], tuple[str, str]]:
    """Bind a team run to a job so progress lands on the job as it happens."""

    def job_work(job: Job) -> tuple[str, str]:
        tracker = UsageTracker(root_label=team.orchestrator)
        try:
            return run(
                team,
                request,
                observe=job.record,
                on_target=job.set_target,
                usage=tracker,
                params=params,
            )
        finally:
            # Published even on failure: a run that burned tokens and then blew
            # up is exactly when you want the numbers.
            job.usage = tracker.snapshot()

    return job_work


def _poll_hint(snap: dict[str, Any]) -> str:
    """How to poll again, in this server's own vocabulary.

    Written here rather than in `core.report`: the tool name and its arguments
    are this file's API, and a renamed tool must not leave the shared layer
    printing instructions that no longer exist.
    """
    return (
        f"Still running. Poll again with check_task(job={snap['job']!r}, "
        f"wait_seconds=15, since_cursor={snap.get('trail_cursor', 0)}, "
        f"since_milestone={snap.get('milestone', 0)}). since_cursor returns only "
        "steps you have not seen; since_milestone decides when the wait ends. "
        "Narrate each new batch as it arrives. Never sleep in a shell, and never "
        "start a second run for the same request."
    )


def _start_tool(team: Team, store: JobStore) -> Any:
    """Build the per-team start tool.

    A named tool per team, rather than one generic `run(team, task)`, because the
    calling model routes on tool descriptions. Collapsing them forces it to
    guess.

    The team's declared params become arguments of the tool. The server derives
    the input schema from the function signature, so the signature is built
    rather than written: `request`, then one keyword argument per `Param`,
    optional ones defaulting to empty.
    """

    def start(request: str, **params: str) -> str:
        # Checked here, before a job exists: a missing required param is the
        # caller's mistake and belongs in the tool error, not in a job that
        # fails a poll later. The run gets the checked mapping, never the raw one.
        context = team.context(params)
        job = store.start(
            team.name,
            request,
            _work(team, request, context.params),
            orchestrator=team.orchestrator,
            digest=team.digest,
        )
        return (
            f"{team.name} job {job.id} started."
            + format_roster(team.name, team.roster())
            + "\n\n"
            f"Follow it with repeated check_task(job={job.id!r}, wait_seconds=15) "
            "calls. Each returns as soon as there is news, with the delegation trail "
            "showing which specialist was dispatched, what it queried and what it "
            "concluded. Show the user each new batch of DELEGATION lines rather than "
            "waiting silently. A full investigation takes 3-6 minutes. Never sleep in "
            "a shell between polls, and never start a second run for this incident."
        )

    request_param = inspect.Parameter(
        "request",
        inspect.Parameter.KEYWORD_ONLY,
        annotation=Annotated[
            str,
            Field(
                description="The symptom or task, in prose. Include what you know "
                f'about where and when it started. Example: "{team.prompt_hint}"'
            ),
        ],
    )
    declared = [
        inspect.Parameter(
            p.name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Annotated[str, Field(description=p.description)],
            default=inspect.Parameter.empty if p.required else "",
        )
        for p in team.params
    ]
    start.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [request_param, *declared], return_annotation=str
    )
    start.__name__ = team.tool_name
    start.__doc__ = (
        f"{team.description}\n\n"
        "Starts the investigation in the background and returns a job id "
        "immediately. Poll check_task with that id to follow the delegation trail "
        "and collect the report. Do not start a second run while one is still "
        "running."
    )
    return start


def _check_task_tool(store: JobStore) -> Any:
    """Build the poller. Takes its store, so a test can drive it in-process."""

    async def check_task(
        job: str,
        wait_seconds: int = 0,
        since_milestone: int | None = None,
        since_cursor: int | None = None,
        trail: bool = True,
    ) -> str:
        """Check a running team task, follow its delegation, and collect the report.

        While running, returns which specialists are in flight, how many calls
        each tool has made, any verdicts already returned, and the recent trail
        of dispatches, tool calls and results. When finished, returns the full
        report.

        Poll this repeatedly and show the user each new DELEGATION batch: that
        trail is how they watch the specialists work. Keep wait_seconds SHORT
        (15 is right) so every call returns promptly with whatever is new. A
        long wait hides the team behind a spinner, which defeats the purpose.
        Use a longer wait only when nobody is watching and you just want the
        finished report.

        NEVER run a shell sleep between polls; wait_seconds is the wait
        mechanism.

        Args:
            job: The job id returned by a start_* tool.
            wait_seconds: Block up to this many seconds waiting for new
                activity, returning early the moment something happens. Use 15
                while narrating progress, 0 for an instant snapshot, up to 120
                only when you simply want to wait out the run unattended.
            since_milestone: Return as soon as the job's `milestone` exceeds
                this. Omit it and the current milestone is captured on entry, so
                a plain wait means "wait for the NEXT thing to happen". Pass an
                older value to catch up on activity you have not seen yet.
                Milestones are dispatches and verdicts, not individual tool
                calls, so a wait wakes on real progress rather than on every
                cluster read.
            since_cursor: Return only trail steps after this cursor. Pass the
                `trail_cursor` from your previous poll; omit it for the whole
                trail. This is separate from since_milestone because the critic
                can work for minutes inside one milestone, and without a cursor
                those steps would be resent on every timeout.
            trail: Include the step-by-step trail. Set false for a compact poll.
        """
        found = store.get(job)
        if found is None:
            known = ", ".join(j.id for j in store.list()) or "none"
            return f"unknown job {job!r}. Known jobs: {known}"

        if wait_seconds > 0:
            await _await_progress(found, wait_seconds, since_milestone)
        snap = found.snapshot(trail=trail, since_cursor=since_cursor)
        return format_snapshot(snap, running_footer=_poll_hint(snap))

    return check_task


async def _await_progress(
    job: Job, wait_seconds: int, since_milestone: int | None
) -> None:
    """Sleep in small slices until the job advances, settles, or time runs out.

    A caller that omits since_milestone means "tell me when something new
    happens", so the baseline is the milestone at entry. Defaulting it to 0
    would make every poll after the first milestone return instantly, turning a
    long-poll into a busy-poll.

    asyncio.sleep rather than a blocking wait: the run owns a worker thread, and
    the event loop must stay free to serve other tools meanwhile.
    """
    baseline = job.milestone if since_milestone is None else since_milestone
    deadline = time.monotonic() + min(wait_seconds, _MAX_WAIT_SECONDS)
    while time.monotonic() < deadline:
        if job.state is not JobState.RUNNING or job.milestone > baseline:
            return
        await asyncio.sleep(_POLL_SLICE_SECONDS)


def _list_tasks_tool(store: JobStore) -> Any:
    """Build the listing tool over the same store."""

    def list_tasks() -> str:
        """List registered teams and recent tasks with their states."""
        rows = [format_snapshot_line(job.snapshot(trail=False)) for job in store.list()]
        return "\n".join(
            [
                f"teams: {', '.join(names())}",
                "",
                "tasks:" if rows else "tasks: none",
                *rows,
            ]
        )

    return list_tasks


def build_server(store: JobStore | None = None) -> MCPServer:
    """The MCP surface. Pass a store to observe or pre-seed jobs in a test."""
    jobs = store if store is not None else _JOBS
    server = MCPServer(
        name="swarmr",
        instructions=(
            "Teams of Deep Agents domain specialists. Each start_* tool launches a "
            "multi-agent investigation as a background job and returns a job id; "
            "poll check_task with that id to watch the specialists work and to "
            "collect the report once state is 'done'. Investigations are read-only: "
            "they diagnose and recommend, and never modify the target."
        ),
    )
    for name in names():
        team = get(name)
        server.add_tool(
            _start_tool(team, jobs),
            name=team.tool_name,
            title=team.summary,
            description=team.description,
        )
    server.add_tool(_check_task_tool(jobs), name="check_task")
    server.add_tool(_list_tasks_tool(jobs), name="list_tasks")
    return server


def main() -> int:
    build_server().run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
