"""The team contract: the invariants a misdeclared team used to violate silently.

Two of these are regressions rather than hypotheticals. A half-declared report
pair produced closing prose instead of the filed report, with nothing anywhere
saying so. And `target()` without a profiler builds the graph, which constructs a
model client — so merely asking which cluster you are pointed at needed an API
key.
"""

from __future__ import annotations

from typing import Any

import pytest

from swarmr.core.team import Member, Param, TeamError


@pytest.mark.parametrize(
    ("report_tool", "render_report"),
    [("file_report", None), ("", lambda args: str(args))],
    ids=["tool-without-renderer", "renderer-without-tool"],
)
def test_team_half_declared_report_pair_is_rejected(
    stub_team: Any, report_tool: str, render_report: Any
) -> None:
    """Half of the pair fails silently: the reader captures a filing only when
    `report_tool` is set, so a renderer alone means the run falls back to prose."""
    with pytest.raises(ValueError, match="must be set together"):
        stub_team(report_tool=report_tool, render_report=render_report)


@pytest.mark.parametrize(
    ("report_tool", "render_report"),
    [("file_report", lambda args: str(args)), ("", None)],
    ids=["both-set", "neither-set"],
)
def test_team_accepts_a_whole_report_pair_or_none(
    stub_team: Any, report_tool: str, render_report: Any
) -> None:
    team = stub_team(report_tool=report_tool, render_report=render_report)
    assert team.report_tool == report_tool


@pytest.mark.parametrize("limit", [0, -1])
def test_team_rejects_a_non_positive_recursion_limit(stub_team: Any, limit: int) -> None:
    """A budget below one cannot run a single step, so it is a declaration error."""
    with pytest.raises(ValueError, match="recursion_limit must be positive"):
        stub_team(recursion_limit=limit)


def test_team_accepts_a_recursion_limit_of_one(stub_team: Any) -> None:
    assert stub_team(recursion_limit=1).recursion_limit == 1


def test_target_uses_the_profile_without_building_the_graph(stub_team: Any) -> None:
    """The bug: asking which target you point at needed a model API key, because
    the only way to get the banner was to build the graph."""
    team = stub_team(profile=lambda run: "kind-demo, 3 nodes")
    assert team.target() == "kind-demo, 3 nodes"
    assert stub_team.runs == [], "profiling must not build the agent graph"


def test_target_falls_back_to_the_build_banner(stub_team: Any) -> None:
    """A team without a profiler still answers, by building and reading the banner."""
    team = stub_team()
    assert team.target() == "the target"
    assert len(stub_team.runs) == 1


@pytest.mark.parametrize(
    ("request_text", "expected"),
    [
        ("look at demo", "look at demo"),
        ("  look at demo  ", "look at demo"),
        ("", "the default sweep"),
        ("   \n ", "the default sweep"),
    ],
    ids=["caller-wins", "stripped", "empty-falls-back", "blank-falls-back"],
)
def test_request_or_default_prefers_the_caller(
    stub_team: Any, request_text: str, expected: str
) -> None:
    team = stub_team(default_request="the default sweep")
    assert team.request_or_default(request_text) == expected


def test_request_or_default_is_empty_when_the_team_declares_no_default(
    stub_team: Any,
) -> None:
    """The caller decides what to do about it; the team must not invent a task."""
    assert stub_team().request_or_default("  ") == ""


def test_tool_name_is_derived_from_the_team_name(stub_team: Any) -> None:
    """The MCP tool name is API surface, so it is fixed by the team name."""
    assert stub_team(name="gitops_sync").tool_name == "start_gitops_sync"


def test_roster_aligns_roles_under_the_longest_name(stub_team: Any) -> None:
    team = stub_team(
        members=(
            Member("network", "routing and policy"),
            Member("io", "disks and volumes"),
        )
    )
    lines = team.roster().splitlines()
    pairs = zip(lines, team.members, strict=True)
    columns = [line.index(member.role) for line, member in pairs]
    assert len(set(columns)) == 1, "roles must line up in one column"
    assert lines[0] == "  network  routing and policy"


def test_roster_is_empty_for_a_team_with_no_declared_members(stub_team: Any) -> None:
    """Empty rather than a stray heading: callers concatenate this into output."""
    assert stub_team().roster() == ""


PARAMS = (Param("repo", "the checkout"), Param("setup", "prep command", required=False))


@pytest.mark.parametrize("name", ["Repo", "my-param", "2nd", "class", "request"])
def test_param_rejects_names_that_cannot_be_a_flag_or_a_tool_argument(name: str) -> None:
    """The name is API surface on both surfaces: a `--flag` and a JSON key next
    to `request`, which is already taken."""
    with pytest.raises(ValueError):
        Param(name, "d")


def test_team_rejects_duplicate_param_names(stub_team: Any) -> None:
    with pytest.raises(ValueError, match="duplicate param names"):
        stub_team(params=(Param("repo", "a"), Param("repo", "b")))


def test_context_passes_checked_params_through_to_the_build(stub_team: Any) -> None:
    team = stub_team(params=PARAMS)
    run = team.context({"repo": "/r", "setup": "make deps"})
    assert dict(run.params) == {"repo": "/r", "setup": "make deps"}


def test_context_refuses_a_missing_required_param_by_name_and_description(
    stub_team: Any,
) -> None:
    """The sentence names what to supply; the operator should not need the docs."""
    with pytest.raises(TeamError, match="requires repo: repo — the checkout"):
        stub_team(params=PARAMS).context({"setup": "x"})


@pytest.mark.parametrize("value", ["", "   "], ids=["empty", "blank"])
def test_context_treats_a_blank_value_as_absent(stub_team: Any, value: str) -> None:
    """An unfilled CLI flag arrives as "", an omitted tool argument as nothing;
    they must be the same case, or a required param could be satisfied by ""."""
    team = stub_team(params=PARAMS)
    with pytest.raises(TeamError, match="requires repo"):
        team.context({"repo": value})
    assert "setup" not in team.context({"repo": "/r", "setup": value}).params


def test_context_refuses_a_param_the_team_did_not_declare(stub_team: Any) -> None:
    """Silently dropping it would let a typo (`rpeo`) run against nothing."""
    with pytest.raises(TeamError, match="takes no parameter rpeo"):
        stub_team(params=PARAMS).context({"repo": "/r", "rpeo": "/x"})


def test_a_team_without_params_refuses_any(stub_team: Any) -> None:
    with pytest.raises(TeamError, match="declared: none"):
        stub_team().context({"repo": "/r"})


def test_target_hands_the_params_to_the_profiler(stub_team: Any) -> None:
    """A team whose target arrives as params has nothing to profile without them."""
    team = stub_team(params=PARAMS, profile=lambda run: f"repo {run.params['repo']}")
    assert team.target({"repo": "/r"}) == "repo /r"
    assert stub_team.runs == []
