"""Terminal entrypoint: run a team with live streaming output.

This is the demo and debugging surface. The MCP server in `server.py` is the
integration surface; both share `core.runner`, so a run behaves identically
either way.

    teams --list                  # what is installed
    teams <team> "the symptom, as prose"
    teams <team>                  # the team's own default request
    teams --target <team>         # profile the target, then exit
    teams <team> --repo P --test C "…"   # a team's declared params, as flags
"""

from __future__ import annotations

import argparse
import sys

from swarmr.core.runner import run_streamed
from swarmr.core.team import Param, Team, TeamError
from swarmr.teams import get, names

__all__ = ["main"]


def _parser(team: Team | None = None) -> argparse.ArgumentParser:
    """The argument contract, plus the named team's declared params as flags.

    One parser rather than a second pass over leftovers: with `request` taking
    every stray word, a flag argparse does not know (`--repo X`) would leave
    `X` in the request. So the team is read off argv first and its flags are
    registered before the real parse. Presence of required ones is
    `Team.context`'s check, so an omission is the same one-line `TeamError`
    the MCP surface gives.
    """
    parser = argparse.ArgumentParser(
        prog="teams",
        description="Run a Deep Agents team of domain specialists.",
    )
    parser.add_argument("team", nargs="?", help=f"one of: {', '.join(names())}")
    parser.add_argument("request", nargs="*", help="the symptom or task, as prose")
    parser.add_argument(
        "--list", action="store_true", help="list registered teams and exit"
    )
    parser.add_argument(
        "--target",
        action="store_true",
        help="profile the team's target and exit without running the agent",
    )
    parser.add_argument("--no-colour", action="store_true", help="disable ANSI colour")
    for param in team.params if team else ():
        parser.add_argument(f"--{param.name}", default="", help=param.description)
    return parser


def _team_name(argv: list[str]) -> str:
    """The first word that is not a flag. Every core flag is a bare switch, so
    the first non-flag word can only be the team."""
    return next((a for a in argv if not a.startswith("-")), "")


def _flags(params: tuple[Param, ...]) -> str:
    shown = (f"--{p.name} <{p.name}>" for p in params)
    return " ".join(
        s if p.required else f"[{s}]" for p, s in zip(params, shown, strict=True)
    )


def _list_teams() -> int:
    for name in names():
        team = get(name)
        print(f"{team.name}\n  {team.summary}")
        if team.params:
            print(f"  params: {_flags(team.params)}")
        if team.prompt_hint:
            words = ["teams", team.name, _flags(team.params), f'"{team.prompt_hint}"']
            print(f"  e.g. {' '.join(w for w in words if w)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    name = _team_name(argv)
    if not name or "--list" in argv:
        _parser().parse_args(argv)
        return _list_teams()

    try:
        team = get(name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args = _parser(team).parse_args(argv)
    params = {p.name: getattr(args, p.name) for p in team.params}

    # Both remaining paths reach into the team's environment — a cluster, a
    # credential, an API — so both can fail for reasons that are the operator's
    # to fix. `TeamError` carries the sentence; anything else is a bug and keeps
    # its traceback.
    try:
        if args.target:
            # `Team.target()` uses the team's own profiler when it has one, so
            # this path talks to the target and nothing else — no model, no API
            # key.
            print(team.target(params))
            return 0
        return _run(team, args, params)
    except TeamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(team: Team, args: argparse.Namespace, params: dict[str, str]) -> int:
    """Stream one investigation."""
    # The team owns its default: only it knows what a useful sweep of its own
    # target looks like. A team that declares none gets its example invocation.
    request = team.request_or_default(" ".join(args.request)) or team.prompt_hint
    if not request:
        print(
            f"error: team {team.name!r} declares no default request; "
            "pass the symptom or task as prose",
            file=sys.stderr,
        )
        return 2

    colour = sys.stdout.isatty() and not args.no_colour
    run_streamed(team, request, colour=colour, params=params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
