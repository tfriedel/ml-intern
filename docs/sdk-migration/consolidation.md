# SDKBackend consolidation

**Branch:** `spike/sdk-hello`
**Artifact:** `agent/core/sdk_backend.py` (+ updated `agent/core/sdk_event_adapter.py`)
**Supersedes:** `sdk_backend_spike.py`, `sdk_backend_spike_2.py`, `sdk_backend_spike_3.py` (deleted)
**Commit:** `3c94727`

After Spikes 1–3 established feasibility, the three throwaway drivers were merged into a single reusable module that Spikes 4–6 (and eventually the real CLI wiring) will drive. This document is the reference for what landed and why.

## Reading order for the SDK migration

1. [`spike-1-findings.md`](./spike-1-findings.md) — SDK installation, auth via `claude login`, in-process MCP tool wrapping. Surprise: SDK builtins (Bash/Read/Write/Edit) are preferred over MCP wrappers.
2. [`spike-2-findings.md`](./spike-2-findings.md) — `can_use_tool` permission callback, `_needs_approval` reuse. Gotchas: `allowed_tools` pre-approves and bypasses the callback; streaming mode is required.
3. [`spike-3-findings.md`](./spike-3-findings.md) — `SDKEventAdapter` mapping SDK messages to ml-intern's Event queue. Surprises: `ToolSearch` pollution, inflated `tool_count`, `ResultMessage` is a goldmine.
4. **This doc** — consolidation.
5. [`spike-4-findings.md`](./spike-4-findings.md) — long-run + compaction controls. Headline: SDK defaults to Opus 4.7 with 1M context, auto-compact threshold at 967k. `/compact` and `DISABLE_AUTO_COMPACT=1` verified; `ClaudeSDKClient` (not `query()`) is the right primitive for the migration.
6. [`spike-5-findings.md`](./spike-5-findings.md) — local training smoke test. Headline: end-to-end PASS. Agent trained SmolLM-135M on a local JSONL, 3 bash calls, 0 hf_jobs calls, real checkpoint on disk, 59s / $0.34.
7. [`spike-6-findings.md`](./spike-6-findings.md) — cancellation / interruption. Headline: SDK `interrupt()` mechanism is solid (aborts turns in ~5ms, session survives). BUT `_bash_handler`'s synchronous `subprocess.run` leaks orphan processes — fix in the full migration by deleting it and re-enabling the SDK Bash builtin.

## What `SDKBackend` does

Drop-in replacement for the `litellm.acompletion` call at the heart of `Handlers.run_agent` (`agent/core/agent_loop.py:250`, `:335`). It does **not** replace the submission loop, Session bookkeeping, ContextManager, or approval-batching — those stay where they are for now.

At construction time it:

1. Wraps every ml-intern `ToolSpec` as an `SdkMcpTool` via `make_sdk_tool_from_spec`.
2. Bundles them into one in-process `create_sdk_mcp_server(name="ml-intern", ...)`.
3. Builds a `can_use_tool` callback that delegates to `_needs_approval()` and emits `approval_required` + `tool_state_change` onto the shared event queue.
4. Wires an `SDKEventAdapter` to the same queue.

At `run_turn(user_message)` time it:

1. Builds `ClaudeAgentOptions` — NOT setting `allowed_tools` (Spike 2 gotcha); disabling `Bash`/`Read`/`Write`/`Edit` builtins so ml-intern's tool-name-based approval remains authoritative; enabling `include_partial_messages=True` for chunk-level streaming.
2. Wraps the prompt as a single-message async iterable (required by `can_use_tool`; Spike 2 gotcha).
3. Hands the SDK iterator to `SDKEventAdapter.consume()`, which emits ml-intern Events on the queue and returns the final `ResultMessage` summary.

## Public API

```python
backend = SDKBackend(
    tool_specs=tool_router.tools,     # ml-intern ToolSpec iterable
    event_queue=session.event_queue,  # asyncio.Queue — CLI/web UI subscribe here
    config=session.config,            # Config; feeds _needs_approval (yolo_mode etc.)
    session=session,                  # for handlers that touch session.hf_token etc.
    system_prompt=system_prompt_v2,   # optional; defaults to None (SDK-default)
    max_turns=50,
    # Scripting helpers (used by demo, not production):
    deny_all_sensitive=False,
    await_user_decision=None,         # async (name, args, tool_use_id) -> bool
)

summary = await backend.run_turn("fine-tune SmolLM on /tmp/my-data")
# summary: {stop_reason, num_turns, total_cost_usd, duration_ms,
#           session_id, is_error, permission_denials}
```

## Key design decisions

### 1. `contextvars` for tool_use_id propagation

**Problem**: the SDK calls MCP tool handlers with `args` only — no `tool_use_id`. But ml-intern's `hf_jobs_handler` needs the id to emit correlated `tool_log` events.

**Solution**: `can_use_tool` fires *before* tool execution with the id in `ctx.tool_use_id`. We stash it in a `contextvars.ContextVar` there; the tool handler reads it back. Same async context, same var.

Module: `agent/core/sdk_backend.py:_current_tool_call_id`.

### 2. Handler signature dispatch in `_call_handler`

ml-intern handlers come in three shapes:

- `async def handler(args: dict) -> tuple[str, bool]`
- `async def handler(args, session=None) -> tuple[str, bool]`
- `async def handler(args, session=None, tool_call_id=None) -> tuple[str, bool]`

…plus a `**_kw` variant in `local_tools`. We `inspect.signature()` each handler and only pass the kwargs it declares (or `**kwargs`). Mirrors `ToolRouter.call_tool()`'s approach without instantiating a ToolRouter.

### 3. Default `disallowed_tools = ["Bash", "Read", "Write", "Edit"]`

Spike 1 found that the model prefers SDK builtin tools over MCP wrappers — so if we register a `bash` MCP tool but leave the builtin `Bash` enabled, the builtin wins and our approval/event logic is bypassed.

Disabling the four builtins keeps ml-intern's approval + event model authoritative. This is a reversible choice — if we later decide to delete `agent/tools/local_tools.py` and adopt the SDK builtins (Spike 1 conclusion hinted at this), remove the entries from `DEFAULT_DISALLOWED_BUILTINS`.

### 4. `approval_required` + `tool_state_change` from one place

Spike 3 identified `tool_state_change` as the last unmapped event. It lives naturally inside `make_permission_callback`: when `_needs_approval` returns True, emit `approval_required`, ask the decider (user or scripted), emit `tool_state_change(state="approved"/"rejected")`, return the SDK verdict.

This closes 10/16 events end-to-end in `SDKBackend` alone; the remaining six (`processing`, `tool_log`, `interrupted`, `undo_complete`, `shutdown`, and the `compacted` edge case) are caller-responsibility or lifecycle events that already have homes in `agent_loop.py`.

### 5. Scoped `ready` tool counts

With the user's machine typical state the SDK reports ~130 available tools: 26 Claude Code builtins + ~104 MCP-plugin tools (Asana, Gmail, Notion, Sentry, …) + our own. The `ready` event now breaks this out:

- `tool_count` — ml-intern's own (scoped by `mcp__ml-intern__` prefix)
- `builtin_tool_count` — Claude Code builtins
- `other_mcp_tool_count` — user's other MCP plugins

Production UI shows `tool_count`; the other two are available for status bars or telemetry.

## What's NOT in `SDKBackend` yet

Deliberate omissions. Each is tracked as an open TODO that a later spike or the full migration will close:

| Omission | Reason | Owner |
|---|---|---|
| CLI wiring (`agent/main.py`) | Would commit to the SDK path before Spikes 4–6 run | Full migration |
| ContextManager integration | SDK compacts on its own; need Spike 4 to decide whether to delete ours | Spike 4 |
| Session save-to-HF upload | Decide in Spike 4 whether authoritative source is SDK's session store or ml-intern's | Spike 4 |
| `research_tool` subagent | Should become native SDK subagents; rethink at full-migration time | Later |
| Dual-backend switch (`--backend {sdk,litellm}`) | Out of spike scope | Later |
| Doom-loop detector | Spike 3 confirmed it can ride `PreToolUse` hook; porting not done | Later |
| `tool_log` pass-through | Works via `session.send_event` once real session is threaded in; needs integration test | Spike 4/5 |
| Cancellation wiring | Needs `ctx.signal` plumbing | Spike 6 |

## Smoke-test summary

`python -m agent.core.sdk_backend {bash,deny,mix}` runs the same three scenarios the individual spike drivers did. All three pass end-to-end with correct event ordering, proper approval flow, and populated result summaries.

Costs observed with Max-plan auth (notional — not billed):

| Scenario | Turns | Duration | Cost |
|---|---|---|---|
| `bash` | 4 | 15.9s | $0.249 |
| `deny` | 3 | 12.0s | $0.259 |
| `mix`  | 4 | 12.2s | $0.321 |

Cost is higher than the pre-consolidation spike runs because the consolidated module enables `include_partial_messages=True` and opens a slightly larger tool surface. Will re-measure once Spike 4 runs longer sessions.

## How this plays into the migration

`SDKBackend` is now an isolated, testable unit. The remaining spikes can measure its behavior under realistic conditions without touching the rest of ml-intern. When we're satisfied, the full migration becomes:

1. Add `--backend sdk` flag in `agent/main.py`.
2. When enabled, instantiate `SDKBackend` with the real `session.tool_router.tools` and `session.event_queue`, and route each user submission through `backend.run_turn()` instead of the litellm path.
3. Keep the litellm path in place until the SDK path has feature parity; then decide on deletion.

Estimated touch surface for step 1+2: one branch in `agent/core/agent_loop.py:Handlers.run_agent`, ~30 lines.
