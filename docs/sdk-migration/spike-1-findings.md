# Spike 1 — Claude Agent SDK hello-world

**Branch:** `spike/sdk-hello`
**Artifact:** `agent/core/sdk_backend_spike.py`
**Status:** ✅ Worked first try

## Goal

Prove three things before committing to a full `litellm → claude-agent-sdk` migration:

1. `claude login` (Max/Pro subscription) can authenticate the SDK — no `ANTHROPIC_API_KEY`.
2. ml-intern's existing tool handlers can be wrapped as SDK MCP tools without rewriting.
3. The event/message stream is rich enough to reconstruct ml-intern's 9 UI events.

## Setup

```bash
git checkout -b spike/sdk-hello
uv pip install claude-agent-sdk   # v0.1.65
# Claude Code CLI (v2.1.117) already on PATH; `claude login` previously done.
```

Prerequisites confirmed:

- `which claude` → `/home/thomas/.local/bin/claude` (v2.1.117).
- No Anthropic API key set in env for this spike.

## What the spike does

`agent/core/sdk_backend_spike.py` — standalone, not wired into the CLI. It:

1. Wraps `_bash_handler` from `agent/tools/local_tools.py` with the SDK's `@tool` decorator.
2. Creates an in-process MCP server via `create_sdk_mcp_server(tools=[bash_tool])`.
3. Runs a one-shot prompt via `query(prompt=..., options=ClaudeAgentOptions(...))`.
4. Prints each streamed message (`SystemMessage`, `AssistantMessage` with `TextBlock`/`ToolUseBlock`, `UserMessage` with `ToolResultBlock`, `ResultMessage`).

Run:

```bash
uv run python -m agent.core.sdk_backend_spike "list the first 3 files in /tmp"
```

## Results

### Run 1 — default tool allowlist

```
[tool_use] Bash args={'command': 'ls /tmp | head -5', ...}
[tool_result] auth-agent768238816
check_dl.py
...
[assistant] First 5 files in /tmp: `auth-agent768238816`, ...
[result] stop=end_turn turns=2 usd=0.19831449999999998
```

The agent used **Claude Code's built-in `Bash` tool** instead of our registered `mcp__local__bash`, even though our tool was in `allowed_tools` and the system prompt asked for it.

### Run 2 — force the MCP path

Added `disallowed_tools=["Bash", "Read", "Write", "Edit"]` to `ClaudeAgentOptions`.

```
[tool_use] ToolSearch args={'query': 'select:mcp__local__bash', 'max_results': 1}
[tool_result] [{'type': 'tool_reference', 'tool_name': 'mcp__local__bash'}]
[tool_use] mcp__local__bash args={'command': 'ls /tmp | head -n 3', ...}
[tool_result] [{'type': 'text', 'text': 'auth-agent768238816\ncheck_dl.py\n...'}]
[assistant] First 3 files in /tmp: `auth-agent768238816`, ...
[result] stop=end_turn turns=3 usd=0.29774100000000003
```

Our MCP handler was invoked. One extra hop: the SDK issues a `ToolSearch` to resolve the tool name before the call (deferred MCP tool discovery).

## API shape (for the full migration)

Learned by introspecting `claude_agent_sdk`:

- **Entry points**: `query(prompt, options)` returns `AsyncIterator[SystemMessage | AssistantMessage | UserMessage | ResultMessage | StreamEvent | RateLimitEvent]`. Stateful variant: `ClaudeSDKClient(options)`.
- **Tools**: `@tool(name, desc, input_schema)` + `create_sdk_mcp_server(tools=[...])` → `McpSdkServerConfig`. In-process, no subprocess. Auto-prefix: `mcp__<server>__<tool>`.
- **Permissions**: `can_use_tool(name, args, ctx) -> PermissionResultAllow | PermissionResultDeny`. `permission_mode` values: `default` / `acceptEdits` / `plan` / `bypassPermissions` / `dontAsk` / `auto`.
- **Hooks**: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `PreCompact`, `Notification`, `SubagentStart`, `PermissionRequest`.
- **Native features worth exploiting**: `skills`, `agents` (sub-agents), `thinking`/`effort`, `task_budget`, `max_budget_usd`, `include_partial_messages`, `session_store`, `fork_session`, `resume`, `setting_sources`.

## Handler shim cost

Wrapping `_bash_handler` (`tuple[str, bool]` return) took ~15 lines:

```python
@tool("bash", "...", {"command": str, "work_dir": str, "timeout": int})
async def bash_tool(args):
    output, ok = await _bash_handler({k: v for k, v in args.items() if v})
    return {"content": [{"type": "text", "text": output}], "is_error": not ok}
```

The three ml-intern handler signature variants (minimal / session-aware / full — see `agent/core/tools.py:250-261`) all collapse to the same shim pattern: ignore unused kwargs, forward `args`, map the return tuple. Per-tool cost is trivial.

## Surprises

### 1. The SDK ships `Bash`/`Read`/`Write`/`Edit` built in — and the model prefers them

Biggest finding. Our MCP wrapper was ignored until we explicitly `disallowed_tools` the builtins. Implication:

> `agent/tools/local_tools.py` (~400 lines reimplementing bash/read/write/edit) can likely be **deleted** in the full migration. The SDK's builtins cover the same ground and are more sophisticated (file checkpointing, proper edit semantics, streaming).

This shrinks migration scope. We only need MCP shims for ml-intern-specific tools: `hf_inspect_dataset`, `hf_repo_*`, `github_*`, `research`, `plan`, `hf_jobs`, `hf_papers`, `explore_hf_docs`, `hf_docs_fetch`.

### 2. MCP tool discovery is deferred

With only MCP tools in the allowlist, the SDK prefaces each session with a `ToolSearch` call to resolve tool names. One hop per session (not per call). For ml-intern's ~15 specialized tools, negligible — may matter at much larger scale.

### 3. `total_cost_usd` is notional on Max/login sessions

`ResultMessage.total_cost_usd` is populated (~$0.20–0.30 for these toy runs) even when auth is via `claude login` with nothing actually billed to the API. Fine for telemetry; surfacing it to users as "your bill" would mislead.

### 4. Editor import resolution

Pyright reports `reportMissingImports` on `claude_agent_sdk` because the workspace interpreter isn't the uv venv. Cosmetic — runtime is fine. Fix at full-migration time via `pyproject.toml` dependency + pyright settings.

## Concerns closed or updated

Ref: `~/.claude/plans/composed-squishing-island.md`.

| # | Concern | Status after Spike 1 |
|---|---|---|
| 1 | SDK API surface stability | ✅ Confirmed — stable, well-documented, in-process MCP works |
| 2 | Tool-handler uniformity | ✅ Shim is ~15 lines per tool, handler variants collapse uniformly |
| 4 | Event mapping fidelity (partial) | ✅ Message types → ml-intern events mapping is clear (see below) |
| 10 | `claude login` UX | ✅ Works; full migration just needs a PATH + auth precheck |

Event mapping preview (informs Spike 3):

| ml-intern event | SDK source |
|---|---|
| `assistant_chunk` | `StreamEvent` (with `include_partial_messages=True`) or `AssistantMessage.content[TextBlock]` deltas |
| `assistant_message` | `AssistantMessage.content[TextBlock]` |
| `assistant_stream_end` | end of `AssistantMessage` in stream |
| `tool_call` | `AssistantMessage.content[ToolUseBlock]` |
| `tool_output` | `UserMessage.content[ToolResultBlock]` |
| `turn_complete` | `ResultMessage` (carries `stop_reason`, `num_turns`, `total_cost_usd`) |
| `approval_required` | `can_use_tool` callback invocation |
| `tool_state_change` | derived from approval callback return + hook events |
| `interrupted` | cancellation semantics TBD in Spike 6 |

## Revised migration scope

Out-of-scope items from the plan to reconsider after this spike:

- **Delete `agent/tools/local_tools.py`** (new candidate). Use SDK builtins instead.
- `research_tool` → **use `ClaudeAgentOptions.agents`** (native subagents). Strong candidate to replace custom implementation.
- `ContextManager` compaction → the SDK's own compaction + `PreCompact` hook looks expressive enough to port any custom behavior. Lean toward deleting ours.
- Doom-loop detector → port as a `PreToolUse` hook.
- Skills → `ClaudeAgentOptions.skills` parameter is native; trivial to load.

## Next

- **Spike 2**: wire `_needs_approval()` (`agent/core/agent_loop.py:48`) into `can_use_tool`. Add `hf_jobs` as a second MCP tool. Confirm the callback gets enough info (tool name + args) to replay ml-intern's approval rules.
- **Spike 3**: write the full event adapter and diff against a litellm-path golden trace.
- Decide whether to merge the spike as-is or wait until Spike 3 has a reusable adapter module.
