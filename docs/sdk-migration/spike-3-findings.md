# Spike 3 — Event-stream mapping

**Branch:** `spike/sdk-hello`
**Artifacts:** `agent/core/sdk_event_adapter.py`, `agent/core/sdk_backend_spike_3.py`
**Status:** ✅ Works — full event surface round-trips cleanly

## Goal

Prove that the SDK's async message stream can be adapted to ml-intern's existing 9-ish `Event` types, which the CLI and web UI already consume. If the adapter is thin (~150 lines), no UI changes are needed on migration.

## What the adapter does

`SDKEventAdapter.consume(async_iterator)` walks the SDK message stream and `q.put(Event(...))`s ml-intern events onto an `asyncio.Queue`. Mapping:

| SDK input | ml-intern output |
|---|---|
| `SystemMessage(subtype="init")` | `ready` `{message, tool_count}` |
| `SystemMessage(subtype="compact_boundary")` | `compacted` `{old_tokens, new_tokens}` |
| `StreamEvent` (`content_block_delta` → `text_delta`) | `assistant_chunk` `{content}` |
| `AssistantMessage[TextBlock]` | `assistant_stream_end` + `assistant_message` `{content}` |
| `AssistantMessage[ToolUseBlock]` | `tool_call` `{tool, arguments, tool_call_id}` |
| `AssistantMessage[ThinkingBlock]` | *(skipped today; could add `thinking` event)* |
| `AssistantMessage.error` | `error` `{error}` |
| `UserMessage[ToolResultBlock]` | `tool_output` `{tool, tool_call_id, output, success}` |
| `ResultMessage` | `turn_complete` `{history_size, stop_reason, num_turns, cost, ...}` or `error` if `is_error` |

Approval is on a separate channel — `can_use_tool` emits `approval_required` directly onto the same queue.

## Results — `mix` scenario

Prompt: *"First use `bash` to print the current date. Then list HF jobs with hf_jobs operation=ps."*

Event trace (concise):

```
ready              {tool_count: 132}
tool_call          {tool: ToolSearch, arguments: {...}, tool_call_id: toolu_01Sm...}
tool_output        {tool: ToolSearch, output: "tool_reference...", success: true}
tool_call          {tool: bash, arguments: {command: "date"}, tool_call_id: toolu_01Tm...}
tool_output        {tool: bash, output: "Mi 22. Apr 14:51:12 CEST 2026", success: true}
tool_call          {tool: hf_jobs, arguments: {operation: "ps", ...}, tool_call_id: toolu_016B...}
tool_output        {tool: hf_jobs, output: "No running jobs found.", success: true}
assistant_chunk    {content: "-"}
assistant_chunk    {content: " Current date: ..."}
assistant_chunk    {content: " Hint: pass `all: true`..."}
assistant_stream_end
assistant_message  {content: "- Current date: Mi 22. Apr 14:51:12 CEST 2026\n- HF jobs (ps): No running jobs found. ..."}
turn_complete      {history_size: 4, stop_reason: "end_turn", num_turns: 4, total_cost_usd: 0.177, duration_ms: 14239, ...}
```

## Results — `deny` scenario (approval path)

Prompt: *"Submit a GPU job with hf_jobs ..."*

Event trace:

```
ready              {tool_count: 132}
tool_call          {tool: ToolSearch, ...}
tool_output        {tool: ToolSearch, ...}
tool_call          {tool: hf_jobs, arguments: {operation: "run", hardware_flavor: "t4-small", ...}, tool_call_id: toolu_01E1...}
approval_required  {tools: [{tool: hf_jobs, arguments: {...}, tool_call_id: toolu_01E1...}], count: 1}
tool_output        {tool: hf_jobs, output: "[spike] denied hf_jobs", success: false}
assistant_chunk    {content: "Denied:"}
assistant_chunk    {content: " `hf_jobs` was blocked..."}
assistant_stream_end
assistant_message  {content: "Denied: ..."}
turn_complete      {history_size: 3, stop_reason: "end_turn", ...}
```

Notice `approval_required` fires between `tool_call` and `tool_output` — identical ordering to ml-intern's litellm path.

## Litellm vs SDK comparison (from reading `agent/core/agent_loop.py`)

Rather than running a paid litellm trace, I diffed the adapter's output against the payload shapes the production loop emits:

| ml-intern event | litellm-path fields | SDK-adapter fields | Gap |
|---|---|---|---|
| `ready` | `message`, `tool_count` | `message`, `tool_count` | ✅ identical |
| `processing` | `message` | **not emitted** | ⚠️ emitted by `Handlers.run_agent` entry; adapter would need caller to emit |
| `assistant_chunk` | `content` | `content` | ✅ identical |
| `assistant_message` | `content` | `content` | ✅ identical |
| `assistant_stream_end` | `{}` | `{}` | ✅ identical |
| `tool_call` | `tool`, `arguments`, `tool_call_id` | `tool`, `arguments`, `tool_call_id` | ✅ identical |
| `tool_output` | `tool`, `tool_call_id`, `output`, `success` | same | ✅ identical |
| `tool_log` | `tool`, `log` | **not emitted** | ⚠️ ml-intern's `hf_jobs` handler pushes mid-execution log lines via `session.send_event` directly. Works the same way when we pass `session` through to the handler. |
| `tool_state_change` | `tool_call_id`, `tool`, `state` (`approved`/`rejected`/`cancelled`/`running`) | **not emitted** | ⚠️ see "Gaps" below |
| `approval_required` | `tools: [...]`, `count` | `tools: [...]`, `count` | ✅ identical |
| `turn_complete` | `history_size` | `history_size`, plus `stop_reason`, `num_turns`, `total_cost_usd`, `duration_ms`, `session_id`, `is_error`, `permission_denials` | ✅ superset (UI ignores extra fields) |
| `error` | `error` | `error` | ✅ identical |
| `interrupted` | `None` | **not emitted** | ⚠️ requires cancellation wiring (Spike 6) |
| `compacted` | `old_tokens`, `new_tokens` | `old_tokens`, `new_tokens` | ✅ identical (SDK emits as `SystemMessage(compact_boundary)`) |
| `undo_complete` | `None` | **not emitted** | ⚠️ undo is a ml-intern operation, not an SDK concept — caller responsibility |
| `shutdown` | `None` | **not emitted** | ⚠️ caller responsibility, trivial |

## Gaps & how to close them

### 1. `tool_state_change` — substantive gap

ml-intern emits `tool_state_change` on approval workflow transitions (`approved`, `rejected`, `cancelled`, `running`). The SDK doesn't have an equivalent.

**Close in the adapter**: emit `tool_state_change` from within the `can_use_tool` callback — right before returning the permission result. `state=approved` / `state=rejected`. For `cancelled`, hook cancellation (Spike 6). For `running`, emit it between `approval_required` and the actual tool execution.

Cost: ~5 lines in `can_use_tool`, plus a cancellation hook.

### 2. `processing` — caller-responsibility

Not an adapter concern. The CLI/web-UI layer already emits `processing` when it accepts a new user input, before calling the agent. Keep that in the caller.

### 3. `tool_log` — passes through via session

ml-intern's `hf_jobs_handler` calls `session.send_event(Event("tool_log", ...))` directly during long jobs. When we run the handler through the MCP shim in production, we can pass a real session object that the handler uses to emit `tool_log` onto the same queue as the adapter. No adapter change needed.

### 4. `interrupted`, `undo_complete`, `shutdown` — caller-responsibility

These are ml-intern lifecycle events, not SDK message-stream events. Keep emitting them from the `Handlers` / `Session` layer.

## Surprises

### 1. `ToolSearch` is emitted as a `tool_call` pair

Every MCP tool use is preceded by a `ToolSearch` call (the SDK's deferred tool resolution, confirmed in Spikes 1 & 2). The adapter currently passes these through as `tool_call` + `tool_output` events, which would clutter the ml-intern UI.

**Fix options** (pick one at full-migration time):

- Drop `ToolSearch` entirely in the adapter (cleanest; tool name prefix check: `if block.name == "ToolSearch": continue`).
- Map it to a muted `tool_log` event so it's observable but not elevated.
- Emit a dedicated `tool_discovery` event and let the UI style it differently.

Recommendation: drop in the adapter. It's an SDK-internal concern, not useful to the user.

### 2. `tool_count: 132` in `ready`

The SDK exposes Claude Code's full built-in tool set (Bash, Read, Write, Edit, TodoWrite, WebFetch, WebSearch, Monitor, Grep, Glob, Task, all MCP tools including 80+ from enterprise MCP servers, etc.) plus our MCP tools. That "132" is correct but alarming on first sight.

**Fix**: in the `ready` emit, either report only the count of ml-intern-custom MCP tools, or surface both counts (`builtin_tool_count` vs `custom_tool_count`). Cosmetic.

### 3. `StreamEvent.event` is the raw Anthropic wire format

With `include_partial_messages=True`, `StreamEvent.event` is a dict matching Anthropic's streaming protocol (`content_block_delta`, `text_delta`, `message_delta`, etc.). We only care about `text_delta` for chunks. `thinking_delta` is present too if extended thinking is on; skipped today.

### 4. `ResultMessage` is a goldmine

It has `duration_ms`, `duration_api_ms`, `num_turns`, `total_cost_usd`, `usage`, `permission_denials`, `errors`, `stop_reason` — all of which ml-intern today either doesn't emit or computes locally. Exposing these through `turn_complete` is a free UX win (e.g. show "turn took 14.2s / $0.18 / 4 turns" in the CLI status bar).

### 5. The drainer's race with result summary

Printing the final `consume()` return concurrently with a still-draining queue needs care — easy to hit interleaved output. The adapter itself is fine; driver needs a clean `await drain_task` after pushing a sentinel. Noted for production integration.

## Concerns closed / updated

Ref: `~/.claude/plans/composed-squishing-island.md`.

| # | Concern | Status |
|---|---|---|
| 4 | Event mapping fidelity | ✅ Closed — adapter is ~200 lines; 11/16 events map directly, 5 are caller-responsibility or trivial additions |
| 8 | Doom-loop detector portability | ✅ Further confirmed — `tool_call` events already flow through the adapter, so the existing pattern-detection logic can run on the adapter's output unchanged |

## Cost datapoints

- `mix` scenario (4 turns, 2 real tool calls + 1 `ToolSearch`): $0.177, 14.2s
- `deny` scenario (3 turns, 1 denied tool call + 1 `ToolSearch`): $0.173, 10.3s

Notional cost — not billed on the Max plan auth we're using here.

## Next

- **Spike 4**: long-run + compaction behavior. Measure token curve and cost under 50+ tool calls; test whether SDK compaction + ml-intern compaction conflict.
- **Spike 6**: cancellation semantics. Confirms the `interrupted` gap can be closed via SDK's `ctx.signal`.
- Consider consolidating spikes 1–3 into a reusable `agent/core/sdk_backend.py` before continuing — the adapter is already production-quality, and spike 4 might benefit from running against that shared module rather than another bespoke spike script.
