"""The terminal entrypoint: exit codes, and what actually gets run.

The registry and the runner are both monkeypatched, so nothing here loads a real
team or starts an agent. What is being pinned is the argument contract: the two
paths that must never reach a model (`--list`, `--target`), which request wins
when both a default and a caller request exist, and that a team with no task to
run says so rather than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from swarmr import cli
from swarmr.core.team import Param, TeamError


@dataclass(slots=True)
class Runs:
    """What `run_streamed` was asked to do, in place of doing it."""

    requests: list[str] = field(default_factory=list)
    teams: list[str] = field(default_factory=list)
    params: list[dict[str, str]] = field(default_factory=list)

    def __call__(
        self, team: Any, request: str, colour: bool = True, params: Any = None
    ) -> str:
        self.teams.append(team.name)
        self.requests.append(request)
        self.params.append(dict(params or {}))
        return "THE REPORT"


@pytest.fixture
def runs(monkeypatch: pytest.MonkeyPatch) -> Runs:
    recorder = Runs()
    monkeypatch.setattr(cli, "run_streamed", recorder)
    return recorder


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, stub_team: Any) -> dict[str, Any]:
    """A stand-in registry with the same lookup contract as the real one."""
    teams: dict[str, Any] = {
        "stub": stub_team(
            summary="a stub team",
            prompt_hint="payments returns 502, namespace demo",
            default_request="the default sweep",
            profile=lambda run: "kind-demo, 3 nodes",
        )
    }

    def get(name: str) -> Any:
        if name not in teams:
            raise ValueError(
                f"unknown team {name!r}. Available teams: {', '.join(sorted(teams))}"
            )
        return teams[name]

    monkeypatch.setattr(cli, "names", lambda: sorted(teams))
    monkeypatch.setattr(cli, "get", get)
    return teams


def test_list_prints_each_team_with_its_summary_and_an_example_invocation(
    registry: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """The listing is the discovery surface: a name alone does not tell a reader
    what the team is for or how to invoke it."""
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "stub\n  a stub team\n" in out
    assert '  e.g. teams stub "payments returns 502, namespace demo"' in out


def test_no_arguments_lists_the_teams_rather_than_failing(
    registry: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main([]) == 0
    listed = capsys.readouterr().out
    assert cli.main(["--list"]) == 0
    assert capsys.readouterr().out == listed


def test_an_unknown_team_reports_the_available_names_on_stderr(
    registry: dict[str, Any], runs: Runs, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo must produce an actionable error and a non-zero exit, not a
    traceback that hides the list of teams that do exist."""
    assert cli.main(["nope", "why 502"]) == 2
    captured = capsys.readouterr()
    assert "unknown team 'nope'" in captured.err
    assert "Available teams: stub" in captured.err
    assert captured.out == ""
    assert runs.requests == []


def test_target_prints_the_profile_and_never_runs_the_agent(
    registry: dict[str, Any], runs: Runs, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of this flag: answering "what am I pointed at" without a
    model API key."""
    assert cli.main(["--target", "stub"]) == 0
    assert capsys.readouterr().out == "kind-demo, 3 nodes\n"
    assert runs.requests == []


def test_a_bare_team_name_runs_the_teams_own_default_request(
    registry: dict[str, Any], runs: Runs
) -> None:
    """Only the team knows what a useful sweep of its own target looks like."""
    assert cli.main(["stub"]) == 0
    assert runs.requests == ["the default sweep"]
    assert runs.teams == ["stub"]


def test_an_explicit_request_wins_over_the_default(
    registry: dict[str, Any], runs: Runs
) -> None:
    """Prose arrives as several argv words and must be reassembled, not indexed."""
    assert cli.main(["stub", "payments", "returns", "502"]) == 0
    assert runs.requests == ["payments returns 502"]


def test_a_team_with_no_default_falls_back_to_its_example_invocation(
    registry: dict[str, Any], runs: Runs, stub_team: Any
) -> None:
    """Better than refusing: the example is a real, runnable request."""
    registry["stub"] = stub_team(prompt_hint="payments returns 502")
    assert cli.main(["stub"]) == 0
    assert runs.requests == ["payments returns 502"]


def test_a_team_with_nothing_to_run_says_so_instead_of_inventing_a_task(
    registry: dict[str, Any],
    runs: Runs,
    stub_team: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry["stub"] = stub_team()
    assert cli.main(["stub"]) == 2
    assert "declares no default request" in capsys.readouterr().err
    assert runs.requests == []


class TestDeclaredParams:
    """A team's `Param`s are `--flags`, and the prose stays the prose.

    The trap is argparse's: with `request` swallowing every stray word, a flag
    it was not told about leaves its value in the request. So the flags must be
    registered before the parse, from the team argv names.
    """

    PARAMS = (Param("repo", "the checkout"), Param("setup", "prep", required=False))

    @pytest.fixture
    def param_team(self, registry: dict[str, Any], stub_team: Any) -> Any:
        registry["stub"] = stub_team(
            params=self.PARAMS,
            default_request="the default sweep",
            profile=lambda run: f"repo {run.params['repo']}",
        )
        return registry["stub"]

    def test_flag_values_reach_the_run_and_never_the_request(
        self, param_team: Any, runs: Runs
    ) -> None:
        assert cli.main(["stub", "--repo", "/r", "the", "symptom"]) == 0
        assert runs.requests == ["the symptom"]
        assert runs.params == [{"repo": "/r", "setup": ""}]

    def test_flags_may_follow_the_request(self, param_team: Any, runs: Runs) -> None:
        assert cli.main(["stub", "the", "symptom", "--repo", "/r"]) == 0
        assert runs.requests == ["the symptom"]
        assert runs.params[0]["repo"] == "/r"

    def test_target_profiles_with_the_flags(
        self, param_team: Any, runs: Runs, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--target", "stub", "--repo", "/r"]) == 0
        assert capsys.readouterr().out == "repo /r\n"
        assert runs.requests == []

    def test_a_missing_required_flag_is_one_line_on_stderr(
        self, param_team: Any, runs: Runs, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--target", "stub"]) == 2
        assert "requires repo" in capsys.readouterr().err
        assert runs.requests == []

    def test_a_flag_the_team_did_not_declare_is_a_usage_error(
        self, param_team: Any, runs: Runs
    ) -> None:
        with pytest.raises(SystemExit):
            cli.main(["stub", "--rpeo", "/r"])
        assert runs.requests == []

    def test_list_shows_the_flags(
        self, param_team: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--list"]) == 0
        assert "  params: --repo <repo> [--setup <setup>]" in capsys.readouterr().out


class TestEnvironmentFailures:
    """A team failing on its environment is news, not a crash.

    An operator saw sixty frames of generated Kubernetes client for an expired
    8h token. `TeamError` is the contract that makes any team's environmental
    failure — credential, connectivity, ambiguity — one actionable line, while
    a genuine bug still keeps its traceback.
    """

    def test_a_team_error_while_profiling_prints_one_line(
        self,
        registry: dict[str, Any],
        runs: Runs,
        stub_team: Any,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def refuse(run: Any) -> str:
            raise TeamError("credential expired 3h ago. Run `mint --now`.")

        registry["stub"] = stub_team(profile=refuse)
        assert cli.main(["--target", "stub"]) == 2
        captured = capsys.readouterr()
        assert captured.err == ("error: credential expired 3h ago. Run `mint --now`.\n")
        assert "Traceback" not in captured.err
        assert runs.requests == []

    def test_a_team_error_while_running_prints_one_line(
        self,
        registry: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The failure surfaces from inside the run, after argument handling."""

        def refuse(*_: Any, **__: Any) -> str:
            raise TeamError("the cluster rejected the token (401).")

        monkeypatch.setattr(cli, "run_streamed", refuse)
        assert cli.main(["stub", "payments returns 502"]) == 2
        assert capsys.readouterr().err == (
            "error: the cluster rejected the token (401).\n"
        )

    def test_an_unexpected_exception_keeps_its_traceback(
        self, registry: dict[str, Any], stub_team: Any
    ) -> None:
        """Swallowing a bug would turn it into a mystery."""

        def crash(run: Any) -> str:
            raise KeyError("nodes")

        registry["stub"] = stub_team(profile=crash)
        with pytest.raises(KeyError):
            cli.main(["--target", "stub"])
