# swarmr

Teams of deep agents domain
specialists, exposed two ways: a streaming terminal CLI and an MCP server.

This is the core distribution. It ships **no team**. Teams live in their own
repositories, depend on this one, and are found through installed metadata —
`pip install` and `pip uninstall` are the whole lifecycle.

## Install

Core and the team must land in the **same environment**; discovery reads
installed distribution metadata, not paths.

```
uv tool install "swarmr[all]" --with-executables-from swarmr-kube
```

`all` pulls in every first-party team; `swarmr[kube]` or `swarmr[blame]` picks
one. `--with-executables-from` is easy to miss: without it only core's `teams`
and `teams-mcp` reach your PATH. Name each team package that ships commands
(`swarmr-kube` ships `incident-credentials`; `swarmr-blame` ships none).

From a checkout instead of PyPI:

```
uv tool install ../swarmr --with ../swarmr-kube --with ../swarmr-blame \
  --with-executables-from swarmr-kube
```

Then a model key, in the directory you run from:

```
printf 'KIMI_API_KEY=sk-…\n' > .env && chmod 600 .env
```

Loading walks up from the current directory, then falls back to `~/.env`;
`KIMI_ENV_FILE` overrides both.

## CLI

```
teams --list
teams --target <team>
teams <team> "the symptom, as prose"
```

Output shows delegation live: the plan, parallel dispatches, each tool call with
a digest of its result, every subagent's verdict attributed by name, per-agent
token counts.

## MCP

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

`cwd` is where the server looks for `.env`. Tools: `start_<team>` for every
installed team, plus `check_task` and `list_tasks`. Runs take minutes, so
`start_*` returns a job id immediately and `check_task` long-polls — returning
the moment a specialist is dispatched or reports, with only the trail steps you
have not seen.

Install a team, restart the server, and its tool appears. No configuration
changes.

## References

- [Architecture overview](./CLAUDE.md)
- [Concept](./concept.png)
- [Langchain deepagents](https://github.com/langchain-ai/deepagents)
- [Langchain docs](https://docs.langchain.com/oss/python/deepagents/overview)
- [Langchain api reference](https://reference.langchain.com/python/deepagents)
