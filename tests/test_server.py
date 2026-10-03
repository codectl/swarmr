"""The MCP surface, driven in-process: no stdio, no subprocess, no model.

Every tool is built over an injected `JobStore`, which is what makes this
possible. The long-poll semantics get the most attention: an omitted
`since_milestone` means "wake me on the NEXT thing", and getting that wrong
turns a long-poll into a busy-poll that spends a client's whole budget on
snapshots of a run that has not moved.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from swarmr.core.events import Event, EventKind
from swarmr.core.jobs import Job, JobStore
from swarmr.core.team import Member, Param, TeamError
from swarmr.server import (
    _await_progress,
    _check_task_tool,
    _list_tasks_tool,
    _poll_hint,
    _start_tool,
    build_server,
)

REPORT_TEXT = "SYMPTOM\n502\nROOT CAUSE\ntargetPort 8081"
CHUNKS = [((), "values", {"messages": [AIMessage(content=REPORT_TEXT)]})]


@pytest.fixture(autouse=True)
def prompt_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the long-poll slice: these tests assert on waiting, and the
    production slice would spend half a second proving each of them."""
    monkeypatch.setattr("swarmr.server._POLL_SLICE_SECONDS", 0.01)


@pytest.fixture
def store() -> JobStore:
    return JobStore()


@pytest.fixture
def running_job(store: JobStore) -> Iterator[Job]:
    """A job parked mid-run, so a poll observes state=running deterministically."""
    release = threading.Event()

    def work(job: Job) -> tuple[str, str]:
        job.set_target("kind-demo, 3 nodes")
        job.record(Event(EventKind.DISPATCH, "network", "look at routing", True, "root"))
        release.wait(5)
        return "THE REPORT", "kind-demo, 3 nodes"

    job = store.start("stub", "why 502", work, orchestrator="commander")
    _wait_until(lambda: job.milestone > 0)
    yield job
    release.set()


def _wait_until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    """Wait for a worker thread to reach a point, without a fixed sleep."""
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)


def poll(store: JobStore, **kwargs: Any) -> str:
    return asyncio.run(_check_task_tool(store)(**kwargs))


def test_build_server_registers_one_start_tool_per_team_plus_the_shared_pollers(
    store: JobStore, monkeypatch: pytest.MonkeyPatch, stub_team: Any
) -> None:
    """The registry is monkeypatched so this stays a wiring test: loading the
    real team would pull in the Kubernetes client and the whole agent stack."""
    team = stub_team(name="example")
    monkeypatch.setattr("swarmr.server.names", lambda: ["example"])
    monkeypatch.setattr("swarmr.server.get", lambda _name: team)

    tools = asyncio.run(build_server(store).list_tools())

    assert {tool.name for tool in tools} == {
        "start_example",
        "check_task",
        "list_tasks",
    }


def test_a_start_tool_returns_the_job_id_and_the_roster_before_any_work_lands(
    store: JobStore, stub_team: Any, settled: Any
) -> None:
    """The caller has to be able to poll immediately, and to show the user who
    is on the case while the run is still going."""
    team = stub_team(CHUNKS, members=(Member("network", "routing"),))
    answer = _start_tool(team, store)(request="why does payments return 502")

    job = store.list()[0]
    assert f"stub job {job.id} started." in answer
    assert "network  routing" in answer
    assert settled(job).snapshot()["report"] == REPORT_TEXT


class TestDeclaredParams:
    """A team's `Param`s are arguments of its start tool, in the published schema.

    The server derives the schema from the function signature, so a team with
    params needs a built signature — a generic `start(request)` would publish a
    tool the calling model cannot fill in.
    """

    PARAMS = (Param("repo", "the checkout"), Param("setup", "prep", required=False))

    def test_schema_lists_each_param_with_its_description_and_requiredness(
        self, store: JobStore, monkeypatch: pytest.MonkeyPatch, stub_team: Any
    ) -> None:
        team = stub_team(name="example", params=self.PARAMS)
        monkeypatch.setattr("swarmr.server.names", lambda: ["example"])
        monkeypatch.setattr("swarmr.server.get", lambda _name: team)

        tools = asyncio.run(build_server(store).list_tools())
        schema = next(t for t in tools if t.name == "start_example").input_schema

        assert schema["required"] == ["request", "repo"]
        assert schema["properties"]["repo"]["description"] == "the checkout"
        assert schema["properties"]["setup"]["default"] == ""

    def test_the_values_reach_the_build(
        self, store: JobStore, stub_team: Any, settled: Any
    ) -> None:
        team = stub_team(CHUNKS, params=self.PARAMS)
        _start_tool(team, store)(request="why", repo="/r")
        job = settled(store.list()[0])
        assert job.snapshot()["report"] == REPORT_TEXT
        assert dict(stub_team.runs[0].params) == {"repo": "/r"}

    def test_a_missing_required_param_fails_the_call_and_starts_no_job(
        self, store: JobStore, stub_team: Any
    ) -> None:
        """The caller's mistake belongs in the tool error, not in a job that
        fails a poll later."""
        team = stub_team(CHUNKS, params=self.PARAMS)
        with pytest.raises(TeamError, match="requires repo"):
            _start_tool(team, store)(request="why")
        assert store.list() == []


def test_an_unknown_job_names_the_jobs_that_do_exist(
    store: JobStore, running_job: Job
) -> None:
    """A mistyped id is the common case; a bare "not found" leaves the caller
    with no way back to the run it just started."""
    answer = poll(store, job="nope")
    assert "unknown job 'nope'" in answer
    assert running_job.id in answer


def test_polling_a_running_job_ends_with_instructions_for_the_next_poll(
    store: JobStore, running_job: Job
) -> None:
    answer = poll(store, job=running_job.id)
    assert "kind-demo, 3 nodes" in answer
    assert "-> network: look at routing" in answer
    footer = answer.splitlines()[-1]
    assert footer.startswith("Still running. Poll again with check_task(")


def test_polling_a_settled_job_returns_the_report_and_stops_asking_for_polls(
    store: JobStore, stub_team: Any, settled: Any
) -> None:
    job = settled(store.start("stub", "why 502", lambda _job: ("THE REPORT", "banner")))
    answer = poll(store, job=job.id)
    assert "REPORT\nTHE REPORT" in answer
    assert "Poll again" not in answer


def test_poll_hint_names_the_job_and_both_resume_arguments() -> None:
    """The two arguments do different jobs — the cursor decides what is sent,
    the milestone decides when the wait ends — so the hint must offer both."""
    hint = _poll_hint({"job": "abc123", "trail_cursor": 7, "milestone": 3})
    assert "check_task(job='abc123'" in hint
    assert "since_cursor=7" in hint
    assert "since_milestone=3" in hint


def test_list_tasks_names_the_teams_and_one_line_per_job(
    store: JobStore, monkeypatch: pytest.MonkeyPatch, running_job: Job
) -> None:
    monkeypatch.setattr("swarmr.server.names", lambda: ["example"])
    answer = _list_tasks_tool(store)()
    assert "teams: example" in answer
    assert running_job.id in answer
    assert "why 502" in answer


def test_await_progress_returns_at_once_for_a_settled_job(job: Any) -> None:
    """Nothing more will ever happen, so waiting out the timeout is dead time."""
    finished = job()
    finished.settle("THE REPORT", "banner")

    started = time.monotonic()
    asyncio.run(_await_progress(finished, wait_seconds=5, since_milestone=None))

    assert time.monotonic() - started < 0.5


def test_await_progress_returns_at_once_when_the_baseline_is_already_behind(
    job: Any,
) -> None:
    """Catching up on activity a caller has not seen must not cost a wait."""
    stale = job()
    stale.record(Event(EventKind.DISPATCH, "network", "go", True, "root"))

    started = time.monotonic()
    asyncio.run(_await_progress(stale, wait_seconds=5, since_milestone=0))

    assert time.monotonic() - started < 0.5


def test_await_progress_without_a_baseline_waits_for_the_next_milestone(
    job: Any,
) -> None:
    """The documented subtlety: an omitted since_milestone means "the next
    thing", so it is taken at entry. Defaulting it to 0 would make every poll
    after the first dispatch return instantly."""
    current = job()
    current.record(Event(EventKind.DISPATCH, "network", "go", True, "root"))

    def advance() -> None:
        time.sleep(0.05)
        current.record(Event(EventKind.VERDICT, "network", "implicated", True, "root"))

    waker = threading.Thread(target=advance)
    started = time.monotonic()
    waker.start()
    asyncio.run(_await_progress(current, wait_seconds=5, since_milestone=None))
    elapsed = time.monotonic() - started
    waker.join()

    assert elapsed >= 0.05, "it must not have returned on the milestone it entered with"
    assert elapsed < 1.0, "and it must return on the next one, not on the timeout"
    assert current.milestone == 2


def test_await_progress_never_waits_longer_than_the_servers_own_cap(
    job: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poll must return well inside any client's request timeout, however long
    the caller asked to wait, so the cap is applied to the deadline."""
    monkeypatch.setattr("swarmr.server._MAX_WAIT_SECONDS", 0.05)
    stuck = job()

    started = time.monotonic()
    asyncio.run(_await_progress(stuck, wait_seconds=120, since_milestone=None))

    assert 0.05 <= time.monotonic() - started < 1.0
