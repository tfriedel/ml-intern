# Spike 2 — Permission callback parity

**Branch:** `spike/sdk-hello`
**Artifact:** `agent/core/sdk_backend_spike_2.py`
**Status:** ✅ Worked after fixing two gotchas

## Goal

Prove that ml-intern's existing approval rules (`_needs_approval()` in `agent/core/agent_loop.py:48`) can be reused verbatim as the SDK's `can_use_tool` callback — i.e. the SDK hands us enough info to make the same decisions we make today.

## Setup

Extended Spike 1 with:

1. A second MCP tool wrapping `hf_jobs_handler` (`agent/tools/jobs_tool.py:1059`).
2. A `can_use_tool` callback that strips the `mcp__<server>__` prefix, calls `_needs_approval(bare_name, args, config=None)`, and returns `PermissionResultAllow` / `PermissionResultDeny` accordingly.
3. Three scripted scenarios (`pass` / `deny` / `list`) exercised via `python -m agent.core.sdk_backend_spike_2 <scenario>`.

## Results

### Deny scenario — `hf_jobs run` on `t4-small`

```
[tool_use] mcp__local__hf_jobs args={'operation': 'run', 'script': 'print("hi")', 'hardware_flavor': 't4-small', ...}
[can_use_tool] {"raw_name": "mcp__local__hf_jobs", "bare_name": "hf_jobs",
                "args": {...}, "tool_use_id": "toolu_014916kSd98Qh...",
                "agent_id": null, "needs_approval": true}
[tool_result] [spike] Tool `hf_jobs` requires approval (operation=run,
              hardware=t4-small). Approval denied for spike purposes.
[assistant] Denied. The `hf_jobs` run operation on `t4-small` requires
            approval, which was denied for spike purposes. No job was
            submitted.
[result] stop=end_turn turns=3 usd=0.17593950000000003
```

### Pass scenario — `hf_jobs ps`

```
[tool_use] mcp__local__hf_jobs args={'operation': 'ps', ...}
[can_use_tool] {..., "needs_approval": false, "tool_use_id": "toolu_01TFjQCH..."}
[tool_result] No running jobs found. Use `{"operation": "ps", "all": true}`
              to show all jobs.
[assistant] No running HF jobs. (Pass `all: true` to include completed/failed ones.)
[result] stop=end_turn turns=3 usd=0.17349175
```

Both scenarios behave as ml-intern does today — `_needs_approval()` was the decision maker, unchanged.

## What the callback receives

From `ToolPermissionContext` + positional args:

| Field | Value | ml-intern equivalent |
|---|---|---|
| `tool_name` | `mcp__local__hf_jobs` (prefixed) | `tool_call.function.name` after prefix-strip |
| `tool_args` | full dict (SDK fills every declared schema field; empty strings for unused ones) | `tool_args` |
| `ctx.tool_use_id` | `toolu_014916kSd98Qh...` | `tool_call_id` — **maps cleanly for event correlation** |
| `ctx.agent_id` | `null` for main agent; populated for subagents | would be a new field for subagent tracking |
| `ctx.signal` | cancellation signal | map to `interrupted` event |
| `ctx.suggestions` | list of `PermissionUpdate` from the SDK | unused today |

Return shape:

- `PermissionResultAllow(updated_input=..., updated_permissions=...)` — **can mutate args** before execution (e.g. clamp `timeout` to a max) and/or install permission updates.
- `PermissionResultDeny(message=..., interrupt=bool)` — message is delivered to the model as the tool result.

## Ergonomics / port cost

The full `can_use_tool` callback — including prefix-stripping and the `_needs_approval` delegation — is **~20 lines**. Zero changes to `_needs_approval` itself. Same is true for porting ml-intern's other config-dependent rules (`yolo_mode`, `confirm_cpu_jobs`, `auto_file_upload`): just pass the real `Config` object instead of `None`.

## Gotchas (documented for the full migration)

### Gotcha 1 — `allowed_tools` bypasses `can_use_tool`

First attempt had `allowed_tools=["mcp__local__bash", "mcp__local__hf_jobs"]` and the callback never fired — the tool went straight to execution.

**Cause**: `allowed_tools` in the SDK means "pre-approved, skip permission check", not "available for the agent to see". MCP tools are already available via `mcp_servers`; putting them in the allowlist pre-approves them.

**Fix**: remove MCP tools from `allowed_tools`. They remain callable (discovered via the `mcp_servers` registration), but every call now routes through `can_use_tool`.

**Implication for migration**: the final production config should NOT use `allowed_tools` for any ml-intern tool we want governed. Either leave them out entirely or use the callback to selectively mark non-sensitive tools as allowed via `updated_permissions` on first call.

### Gotcha 2 — `can_use_tool` requires streaming mode

Passing `prompt="..."` as a plain string triggered:

```
ValueError: can_use_tool callback requires streaming mode.
Please provide prompt as an AsyncIterable instead of a string.
```

**Fix**: wrap the prompt in an async generator that yields user message dicts:

```python
async def prompt_stream():
    yield {"type": "user", "message": {"role": "user", "content": prompt}}

async for msg in query(prompt=prompt_stream(), options=options):
    ...
```

**Implication**: the real ml-intern CLI is already streaming-oriented, so this is fine. Headless mode (single-prompt, auto-approve) currently uses a one-shot string prompt via litellm; after migration it'll need the same async-iterable wrapper. Trivial.

### Gotcha 3 — extra turn for MCP tool resolution

Every MCP tool call is preceded by a builtin `ToolSearch` hop that the callback does *not* gate (it's a builtin, considered read-only). Turn count in both scenarios was 3 (search + tool + final). Not a blocker; worth measuring over long sessions.

## Concerns closed

Ref: `~/.claude/plans/composed-squishing-island.md`.

| # | Concern | Status |
|---|---|---|
| 2 | Tool-handler uniformity / MCP shim cost | ✅ Fully closed — `hf_jobs_handler` wrapped without modification |
| 4 | Approval portion of event mapping fidelity | ✅ Closed — callback gives tool name, full args, tool_use_id, agent_id. Enough to render `approval_required` events identically |
| 8 | Doom-loop detector portability (partial) | ✅ Strong signal it can ride `PreToolUse` hook — same surface as `can_use_tool` |

## Implications for `can_use_tool` design in the full migration

- Call `_needs_approval(bare, args, session.config)` with the real Config (not `None` as in spike) so `yolo_mode` / `confirm_cpu_jobs` / `auto_file_upload` flags work.
- On `needs=False`, return `PermissionResultAllow()` — nothing to override.
- On `needs=True`, emit ml-intern's `approval_required` event (containing `tool_use_id`, `name`, `args`) into the existing `event_queue`, then **await** a response from `submission_queue` (which the CLI/web UI populates from user input). The awaited result is mapped to `PermissionResultAllow` or `PermissionResultDeny`.
- `updated_input` lets us clamp args server-side (e.g. enforce `hf_jobs` `timeout` ≤ org policy) without re-prompting the model.

## Cost datapoint

Across both scenarios, ~$0.17 per 3-turn run. Broadly consistent with Spike 1's ~$0.30 for 3 turns (Spike 1's prompt did more work). Max-plan auth was still active; this cost is notional, not billed.

## Next

- **Spike 3**: event adapter — reconstruct ml-intern's 9 events from the SDK message stream, diff against a golden litellm trace.
- **Spike 4**: compaction behavior under long runs.
- Decide in Spike 3 whether we have enough to merge the three spikes together as a reusable `SDKBackend` adapter class, or keep spiking individually.
