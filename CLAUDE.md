# CLAUDE.md — swarmr

## Overview

Core distribution for teams of [Deep Agents](https://github.com/langchain-ai/deepagents) domain specialists, exposed two ways: a streaming terminal CLI (`teams`) and an MCP server (`teams-mcp`).

Ships **no team**. Teams live in their own distributions, depend on this one, and are found through the `swarmr.teams` entry-point group. Dependencies flow one way — teams → core — and `core` is domain-free: it knows no role names, no payload shapes, not even what a failed tool result looks like. Everything it would otherwise assume is a field on `Team`.

- **Distribution:** `swarmr` (`src/swarmr`, hatchling)
- **Python:** >=3.13
- **Runtime deps:** `deepagents>=0.7.9,<0.8`, `langchain-openai>=1.6`, `mcp>=2`, `pydantic>=2.10`
- **Model:** Kimi coding endpoint (`api.kimi.com/coding/v1`, `kimi-for-coding`)

## Entry Points

### CLI (`src/swarmr/cli.py` → `teams`)

```
teams --list                  # what is installed
teams --target <team>         # profile the target, then exit (no model needed)
teams <team> "the symptom, as prose"
teams <team>                  # the team's own default_request
teams <team> --repo P --test C "…"   # a team's declared params, as flags
```

`--no-colour` disables ANSI; colour is on only when stdout is a tty. `--target` goes through `Team.target()`, which prefers the team's `profile` — building the graph would construct a model client and so require an API key just to ask what the target is.

A team's `params` become `--<name>` flags. The parser is built *after* the team name is read off argv, because `request` takes every stray word: a flag argparse was not told about (`--repo X`) would leave `X` in the request. `--list` shows each team's flags; a missing required one is a `TeamError` from `Team.context`, the same sentence the MCP surface gives.

Exit codes: `0` ok, `2` unknown team / `TeamError` / no request and no default. `TeamError` prints one sentence on stderr; anything else keeps its traceback, because anything else is a bug.

### MCP server (`src/swarmr/server.py` → `teams-mcp`)

Stdio transport, built on `mcp.server.mcpserver.MCPServer`.

```
{
  "mcpServers": {
    "swarmr": {
      "type": "stdio",
      "command": "teams-mcp",
      "cwd": "/abs/path/to/your/working/dir",
      "timeout": 0
    }
  }
}
```

`cwd` is where `load_env` starts walking for `.env`, so it must be the directory you would otherwise `cd` into.

**Tools:** `start_<team>` for every installed team, plus `check_task` and `list_tasks`. The server iterates discovery, so installing a team and restarting adds its tool with no configuration change. One named tool per team rather than a generic `run(team, task)`, because the calling model routes on tool descriptions — collapsing them forces it to guess.

**Environment (`core/model.py`):** `KIMI_API_KEY` (required), `KIMI_MODEL`, `KIMI_BASE_URL`, `KIMI_ENV_FILE`. `load_env` walks up from cwd looking for `.env`, then falls back to `~/.env`; `KIMI_ENV_FILE` overrides both. Only `KIMI_*` keys are read, and `os.environ.setdefault` means a real environment variable always wins over the file. No dotenv dependency. Temperature is deliberately never passed: `kimi-for-coding` rejects any value but 1 with HTTP 400.

**Optional:** `LANGSMITH_TRACING` + `LANGSMITH_API_KEY` (or the `LANGCHAIN_*` spellings) add full traces; the token accounting in `core/usage.py` works either way.

## Architecture

**Package root**\
`src/swarmr/__init__.py` — exports the authoring contract (`Team`, `TeamBuild`, `TeamBuilder`, `RunContext`, `Member`, `Lazy`, `__version__`) and deliberately never touches discovery: a package initializer runs before any submodule, so naming `swarmr.teams` here would make `import swarmr.core.text` load every installed team.

**Discovery**\
`teams/__init__.py` — `GROUP = "swarmr.teams"`, `names()`, `get(name)`. `names()` reads installed distribution metadata and imports nothing; `get` resolves the entry point and asserts `isinstance(team, Team)` — the one thing an out-of-tree package can get wrong, so the error names the offending entry.

**Contract**\
`core/team.py` — `Team`, `TeamBuild`, `TeamBuilder`, `RunContext`, `Member`, `Lazy`, `TeamError`. `core/harness.py` — the deepagents vocabulary read off the stream (`TASK_TOOL`, `PLAN_TOOL`, `SUBAGENT_KEY`, `DESCRIPTION_KEY`, `PROMPT_KEY`), defined once so an upstream rename is a single edit.

**Interpretation**\
`core/events.py` (`EventReader`: stream chunks → `Event`s), `core/briefing.py` (`effective_briefing`: the first-round rule, stated without importing the framework), `core/attribution.py` (`Attribution`: stream tag → subagent name, per run).

**Presentation**\
`core/render.py` (`Renderer`: terminal output, one dim label per speaker change), `core/report.py` (`format_snapshot`, `format_snapshot_line`, `format_roster`, `format_usage`: an MCP poll as plain text).

**Orchestration**\
`core/runner.py` (`run`, `run_streamed`, `Observer`), `core/jobs.py` (`Job`, `JobState`, `JobStore`: background runs, observable while they happen).

**Framework adapters**\
`core/middleware.py` (`AnnounceName`, `FirstRoundBriefing` — the only place `core` touches LangChain middleware), `core/model.py` (`build_model`, `load_env`), `core/usage.py` (`UsageTracker`, `AgentUsage`, `langsmith_enabled`).

**Bottom**\
`core/text.py` (`flatten`, `clip`, `wrap`, `content_text`), `core/digest.py` (`generic_digest`).

**Testing surface**\
`core/testing.py` — `EXAMPLE_TEAM`, `TeamFactory`, `JobFactory`, `GraphStub`, `Chunk`. Shipped in the wheel deliberately: a team in another distribution tests against the same stand-ins core does.

Dependency direction inside `core` runs one way: contract ← interpretation ← presentation ← framework adapters, with `text` and `digest` at the bottom. `jobs` depends on `digest` rather than on `render`, because it records and never prints.

## Lazy Loading

Two mechanisms, same reason: listing teams must not cost what running them costs.

**`core.Lazy`** — a team field declared as `Lazy("module:attribute")`, imported on first call and then held by `sys.modules`. Applies to `build` and `profile`, which drag in the model SDK, the agent stack and the domain client. Measured on the incident team: 4533 modules at declaration time became 133.

**`core/__init__.py` PEP 562 `__getattr__`** — re-exports resolve on attribute access against the `_EXPORTS` name → submodule map, not at import. `from swarmr.core import Renderer` still works and loads exactly what it names; an eager list here made `import swarmr.core.text` (thirty lines of stdlib string handling) pull in the model SDK.

The MCP server reads `name`, `summary` and `description` for every installed team on every start. That is the path both mechanisms protect.

## Run Flow

`team.build(RunContext())` → `TeamBuild(graph, banner)` → `on_target(banner)` → `graph.stream(..., stream_mode=["updates", "values"], subgraphs=True)` → `EventReader.read(namespace, payload)` → `observe(event)`

Both surfaces walk the same stream through the same reader; only the observer differs — the CLI passes `Renderer.show`, the MCP job passes `Job.record`. `config` carries `recursion_limit` from the team and the `UsageTracker` as a callback.

**Why both stream modes:** `updates` drives the live trail, `values` carries the authoritative message list. Reconstructing the report from update events alone is fragile — a compaction step or a non-string content block can leave an early narration as the last text seen, which is how a run once reported its plan instead of its findings.

**Report selection, in order:** filed report (the team's `report_tool` arguments) → `structured_response.render()` → last assistant message with prose in the final state → streamed closing prose → `"the run produced no final report"`. A filed report wins because the team files it deliberately, whereas closing prose is optional and has been observed missing on converged runs. `_report_is_filed` latches and is never cleared, so prose arriving after a filing is dropped — that is where junk lands.

## Events

`EventKind`: `PLAN`, `DISPATCH`, `AUDIT_INPUT`, `CALL`, `RESULT`, `VERDICT`, `REPORT`, `FILED_REPORT`.

`Event(kind, who, text, root, stream)`. `who` is a subagent name for `DISPATCH`/`VERDICT`/`AUDIT_INPUT` and a tool name for `CALL`/`RESULT`. `stream` is the short subgraph id, which is the only way to separate concurrently running subagents in one event sequence.

**Verdict attribution:** LangGraph's subgraph namespace does not carry the subagent's name, so the reader maps each `task` tool_call_id to the `subagent_type` it was dispatched with; the matching `ToolMessage` then identifies the verdict exactly, even with several subagents in flight.

**Name attribution:** `AnnounceName.wrap_model_call` binds the running graph's `checkpoint_ns` to the name the team already knows, before the subagent's first tool call. Every `|`-separated segment is registered, because `stream_mode="updates"` reports only the outer one. Best-effort: if LangGraph stops exposing the namespace, `label` returns the raw tag and the trail is merely less readable.

**Argument rendering:** an argument over 70 chars is clipped at a word boundary, but one with no space in its first 70 chars or longer than 400 is *measured* instead — `<3000 chars>` tells a reader what happened where the first seventy characters of a manifest do not.

## First-Round Briefing

On the **first** dispatch to each subagent, the `task` call's `description` is replaced with the caller's own request. Later dispatches pass through: by then the orchestrator genuinely knows something, and a targeted follow-up is the point of a second round. Teams listed in `audit_agents` are exempt — an adjudicator's payload is a finished hypothesis, which is exactly what it must receive.

The rule lives in `core/briefing.py` and the enforcement in `core/middleware.py`. `EventReader` calls the same rule, because a tool call is displayed the moment the model emits it — before any middleware runs — so without a shared rule the terminal would show text the subagent never saw.

Enforced by the harness rather than by prompt because prompt instructions did not hold: observed runs had the orchestrator restating each specialist's own checklist back to it and pre-framing the problem per domain, handing over the conclusion the specialist was supposed to reach independently. The harness's own task-tool description asks for full detail, so a prompt rule requesting one sentence is arguing with the tool schema.

## Jobs

MCP is request/response with a client-side timeout and a run takes minutes, so the surface is a job API. `JobStore.start` spawns a daemon thread per run — threads, not asyncio, because the agent graph and its clients are synchronous — and retains 32 jobs, evicting the oldest settled ones.

Every `Job` mutation takes the instance lock: the worker thread records while the MCP loop serves polls. `snapshot()` returns copies only, so a poll never iterates a mutating dict or mixes fields from two instants. `settle` writes state, result, target and finish time together, so a poll can never see `state=done` with the report unset.

**Two counters, deliberately separate:**

- `milestone` bumps only on a plan, a dispatch, a verdict, a filing, or the run settling. Long-pollers wake on this, so a client spends a handful of polls per run instead of one per cluster read.
- `trail_seq` is per trail line. A milestone can stall for minutes while one subagent works, so filtering the trail by milestone would resend the same lines on every timeout. The cursor is what makes a poll strictly incremental.

The trail is a `deque(maxlen=120)` of `(seq, author, line)`. Author is stored apart from the text so a reader can group by who acted rather than repeat an inline tag on every line.

`check_task(job, wait_seconds=0, since_milestone=None, since_cursor=None, trail=True)` long-polls in 0.5s `asyncio.sleep` slices up to 120s, keeping the event loop free. Omitting `since_milestone` captures the current milestone on entry — meaning "wake me on the NEXT thing" — because defaulting it to 0 would turn every poll after the first milestone into a busy-poll.

A `TeamError` from a run becomes the job's failure reason verbatim; any other exception is recorded as `TypeName: message`. Neither kills the server.

## Key Interfaces

**`Team`** (frozen dataclass, `core/team.py`) — the whole ABI. Required: `name`, `summary`, `description`, `build`. Optional: `profile`, `params`, `default_request`, `members`, `prompt_hint`, `report_tool`, `render_report`, `orchestrator`, `audit_agents`, `digest`, `is_error`, `recursion_limit` (160). `name` forms the MCP tool name via `tool_name`, so renaming it breaks callers. `__post_init__` rejects `report_tool` without `render_report` or vice versa — half of that pair fails silently — a non-positive `recursion_limit`, and duplicate param names.

**`Param`** — `(name, description, required=True)`. Declared on `Team.params` for everything about the target that must not come from the environment: a repository path, a test command. Each is an argument of the MCP start tool (the server builds the tool's signature from them, so the published schema carries name, description and requiredness) and a `--name` flag on the CLI. `name` must be a snake_case identifier and not `request`.

**`Team.context(params)`** — the one gate both surfaces pass through: rejects names the team did not declare, requires the required ones, treats blank values as absent (an unfilled flag and an omitted argument must mean the same thing), and returns a fresh `RunContext`. What a value points at is the team's to validate, from `build` or `profile`, as a `TeamError`. The server calls it before a job exists so a missing argument is a tool error, not a job that fails a poll later.

**`TeamBuilder`** — `(run: RunContext) -> TeamBuild`. Called once per run, never cached, so profiling reflects the target's current state rather than whatever was true at import time. `profile` has the same signature: a team whose target arrives as params has nothing to profile without them.

**`RunContext`** — carries the run's `Attribution` and `params` (a read-only mapping of the checked values). Per run, never global: attribution used to be module state cleared on entry, so a second MCP run starting mid-flight wiped the first one's names.

**`Observer`** — `Callable[[Event], None]`.

**`Lazy`** — `resolve()` imports without calling, which is what lets a test check that every declared target still points somewhere real.

## Team Vocabulary

Each field exists because `core` would otherwise have to assume a domain:

- `orchestrator` — without it, `core` invents a name for root-level activity.
- `audit_agents` — without it, an adjudicator's verbatim input gets replaced by the briefing rule.
- `digest` — without it, tool results render as shape only, via `generic_digest`.
- `is_error` — without it, nothing is ever marked failed; failure looks different in every domain.
- `report_tool` + `render_report` — without them, the deliverable falls back to whichever prose arrived last.
- `default_request` — without it, a bare `teams <team>` has nothing to run.
- `params` — without them, a team's target can only come from the environment, and one server process can serve only one target.
- `recursion_limit` — without it, six specialists and twelve share one ceiling.

A default is supplied for each so declaring a team stays short, but no default encodes a domain.

## Adding a Team

```
[project]
dependencies = ["swarmr>=1.3,<2", "whatever-sdk>=1"]

[project.entry-points."swarmr.teams"]
my_team = "my_package:TEAM"
```

```
from swarmr.core.team import Lazy, Member, Team

TEAM = Team(
    name="my_team",
    summary="One sentence, shown in tool listings.",
    description="What the calling model routes on: symptoms, not subject area.",
    build=Lazy("my_package.agent:build"),
    profile=Lazy("my_package.agent:profile"),
    members=(Member("lead", "what it investigates"),),
)
```

That is the whole coupling. Both surfaces iterate discovery, so the team gets a `start_my_team` tool and a CLI command with no further edits, and `pip uninstall` unregisters it. The one rule: anything importing an agent framework, model SDK or domain client goes behind `Lazy`.

Teams pin `swarmr>=1.3,<2` — `1.3` introduced `Param` and the `profile(run)` signature. Core's `kube`, `blame` and `all` extras pin the first-party teams the other way, so `swarmr[all]` never pairs a core with a team built against an older contract.

## Design Patterns

**Domain-free core** — nothing in `core` may name a domain. That constraint is what makes a team package deletable: delete it and nothing in the tree refers to it.

**Fields over assumptions** — every domain question `core` would ask is a `Team` field with a neutral default.

**Rule / enforcement split** — `briefing` states the rule, `middleware` applies it; `attribution` holds the map, `middleware` binds it. Both rule modules stay importable, testable and reusable without the agent framework.

**Deferred imports** — `Lazy` for team fields, PEP 562 `__getattr__` for core re-exports, function-local imports for LangGraph in `current_namespace`.

**Entry points as the registry** — no list to edit, so two teams can be built in parallel without meeting in a shared file.

**Snapshot before format** — `report.py` formats captured dicts and never touches a live `Job`.

**Locked mutation** — every `Job`, `Attribution` and `UsageTracker` mutation takes a lock; subagents run concurrently on their own threads.

**Thread per run** — the honest model for a synchronous graph, and it keeps the MCP event loop answering polls.

**Best-effort degradation** — a changed LangGraph internal costs readability, never a run.

**Single output owner** — only `Renderer` writes to stdout, which is what makes a headless run silent by construction.

**Shipped stubs** — `core/testing.py` is library code, not test code, so out-of-tree teams test against the same stand-ins.

**Frozen slotted dataclasses** — `Team`, `Event`, `Member`, `Lazy`, `RunContext`, `TeamBuild`.

## Testing

No model, no API key, no network. `GraphStub` replays a fixed chunk list through the exact `stream` signature the runner calls, so a test exercises the real stream loop, reader and report selection with nothing behind it.

**Layout:** slice tests live inside the package they cover (`src/swarmr/core/tests/`); tests of the composed whole live in `tests/` (`test_cli.py`, `test_server.py`, `test_teams_registry.py`). `testpaths = ["src", "tests"]`, importlib import mode so test modules need no `__init__.py` and duplicate basenames across slices do not collide. The wheel excludes `**/tests`.

**Fixtures:** shared fixtures live in the root `conftest.py` — the one directory that is an ancestor of both trees. `stub_team`, `job`, `settled`. The stubs themselves are in `swarmr.core.testing`, because a team in its own distribution cannot reach into this file.

**Install required:** `pytest_sessionstart` fails with a `UsageError` if the `swarmr` distribution is not installed. Discovery reads distribution metadata, so a bare `PYTHONPATH=src` tree imports fine while `entry_points` sees nothing, and the failures then read like a broken registry rather than a missing install. There is deliberately no source-tree fallback.

**The suite passes with no team installed**, which is the proof that core stands alone. Discovery is exercised against an injected entry point pointing at `core.testing:EXAMPLE_TEAM` — the same path a third-party distribution takes. `core/tests/test_render.py` renders an invented foreman-and-welder team to prove the presentation layer is domain-free. The complementary half — that a real installed team stays unimported until run — is asserted in each team's own suite, where there is a team to not-import.

`core/tests` may not import from any team.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check . && pyright
```

Activate the venv rather than calling `.venv/bin/…`: pyright takes its interpreter from PATH, so unactivated it type-checks against a Python with none of the dependencies and reports every import unresolved. Otherwise pass `--pythonpath .venv/bin/python`. No `venvPath`/`venv` in `pyproject.toml` for the same reason — those hardcode a directory that exists on one contributor's machine and in no CI.

CI (`.github/workflows/ci.yml`) runs `pytest`, `ruff check .` and `pyright` on 3.13. `e2e` is a declared marker for tests needing a live target and a model key.

## Dependencies

- `deepagents` — the agent harness. **Capped `<0.8`:** it ships breaking changes in minor releases, and the `SubAgent`/`create_deep_agent` surface this package builds on is exactly what moves. Raise the ceiling deliberately, with the suite as the gate.
- `langchain-openai` — `ChatOpenAI` against the Kimi coding endpoint. `timeout=180`, `max_retries=3`.
- `langchain` / `langchain-core` / `langgraph` — reached transitively for `AgentMiddleware`, `ToolCallRequest`, `BaseCallbackHandler`, `Command` and `langgraph.config.get_config`.
- `mcp` — `MCPServer`, stdio transport.
- `pydantic` — `SecretStr` for the API key.
- Dev: `pytest`, `pytest-cov`, `ruff` (line length 90; `E,F,I,UP,B,SIM,RUF`), `pyright` (standard mode, 3.13).
