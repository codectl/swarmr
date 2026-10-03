"""Deep Agents teams of domain specialists.

Layout:
    core/    domain-free machinery: model wiring, renderer, job store, runner
    teams/   discovery; core ships no team of its own
    cli.py   terminal entrypoint with live streaming
    server.py MCP server exposing one tool per installed team

Dependencies flow one way: teams -> core. Teams never import each other, which
is what makes any team deletable in one edit: discovery reads the
`swarmr.teams` entry-point group, so no shared file names a team.

This module exports the authoring contract and nothing else. It deliberately
does not touch discovery: a package initializer runs before any submodule, so
importing `swarmr.core.text` would otherwise load every registered
team and its dependencies. Reach for `swarmr.teams.get(name)` to load
one.
"""

from __future__ import annotations

from importlib.metadata import version

from swarmr.core.team import (
    Lazy,
    Member,
    Param,
    RunContext,
    Team,
    TeamBuild,
    TeamBuilder,
)

__all__ = [
    "Lazy",
    "Member",
    "Param",
    "RunContext",
    "Team",
    "TeamBuild",
    "TeamBuilder",
    "__version__",
]

__version__ = version("swarmr")
