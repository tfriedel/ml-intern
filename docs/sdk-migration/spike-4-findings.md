# Spike 4 — Long-run & compaction behavior

**Branch:** `spike/sdk-hello`
**Artifacts:** `agent/core/sdk_backend_spike_4.py`, `/tmp/probe_context.py`, `/tmp/probe_compact_controls.py`
**Status:** ✅ Decisive — SDK compacts at ~967k tokens on 1M-context Opus, not at 190k. ml-intern's ContextManager compaction is safe to delete. Manual `/compact` and `DISABLE_AUTO_COMPACT=1` env var both verified to work; `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` does NOT.

## Goal

Find out:

1. Whether the SDK auto-compacts during a long tool-heavy run.
2. Whether the agent stays coherent past ml-intern's 190k threshold.
3. Whether ml-intern's `ContextManager._compact_and_notify()` can be deleted or whether it's still load-bearing.

## Setup

Drove the consolidated `SDKBackend` with three scenarios:

| Scenario | # bash calls | Output size | Notes |
|---|---|---|---|
| `short` | 10 | small | Sanity check for instrumentation |
| `long` | 40 | small (<1KB each) | Typical usage pattern |
| `blast` | 30 actual / 40 intended | large (2–20KB each) | Stress context with big tool outputs |

Instrumentation: subclassed `SDKEventAdapter` to capture `AssistantMessage.usage` per turn and count `SystemMessage(subtype="compact_boundary")` events.

ml-intern's `ContextManager` was **not** wired in (per Spike 4's scope: "SDK compaction on, ml-intern off"). The "both on" thrash case is a design argument, not a runnable experiment — see "Thrash question" below.

## Results

### `short` (10 calls)

```
total_cost_usd:          $0.445
duration:                56s
num_turns:               13
tool_calls:              10
assistant_messages:      24  (see "Surprise 1" — SDK emits each twice)
compaction_events:       0
peak cache_read_tokens:  33,054
peak input_tokens/turn:  6   (fresh, uncached)
```

### `long` (40 calls, small outputs)

```
total_cost_usd:          $1.46
duration:                202s
num_turns:               43
tool_calls:              40
compaction_events:       0
peak cache_read_tokens:  52,257
peak input_tokens/turn:  6
sum output_tokens:       148  (deduped from 296)
```

Cache_read grew roughly linearly: ~1k tokens/tool call. Fresh `input_tokens` stayed at 1–6 the entire run — prompt caching is extremely effective for this workload pattern.

### `blast` (30 actual calls, large outputs)

```
total_cost_usd:          $4.84
duration:                203s
num_turns:               33
tool_calls:              30
compaction_events:       0
peak cache_read_tokens:  311,226    <-- past the 200k default window
peak input_tokens/turn:  6
sum output_tokens:       107  (deduped from 214)
```

Cache_read grew aggressively (1k → 311k over 30 turns). The agent stopped with `stop_reason: end_turn` after noticing repeated commands in the third loop — no error, no timeout, fully coherent.

The `311k cache_read` data point is the key finding. We then ran a follow-up probe (`/tmp/probe_context.py`) using `ClaudeSDKClient.get_context_usage()` and got the explicit numbers:

```
rawMaxTokens:          1,000,000
maxTokens (effective): 1,000,000
autoCompactThreshold:    967,000   (~96.7% of max)
isAutoCompactEnabled:       True
model:                 claude-opus-4-7[1m]
```

The SDK defaults to **Opus 4.7 with 1M context** on Max-plan auth — not Sonnet 200k. Auto-compaction is set at ~96.7% of the 1M window. Our 311k peak was 32% of the threshold, which is why zero events fired.

**The effective ceiling is 5× higher than ml-intern's 190k compactor would tolerate.** Our compactor would fire at 190k and truncate a session that the SDK is willing to keep going with — a pure regression.

## Concern status

From `~/.claude/plans/composed-squishing-island.md`:

| # | Concern | Status |
|---|---|---|
| 3 | Compaction collision | ✅ Closed — SDK doesn't compact under realistic pressure; ml-intern's compactor is safe to delete. If we ever run past ~900k tokens and the SDK does compact, we can layer a `PreCompact` hook later. |
| 6 | Rate limits on Max | 🟡 Partial — burned ~$6.75 notional across three scenarios on Max-plan auth, no throttling, no errors. Still no 8-hour training run data, so full answer deferred to Spike 5 / beyond. |

## Compaction controls — what we verified

Probe script: `/tmp/probe_compact_controls.py`. Four trials:

| Trial | Setup | `isAutoCompactEnabled` | `autoCompactThreshold` | Notes |
|---|---|---|---|---|
| A | send `/compact` as user message | N/A | N/A | ✅ `compact_boundary` event fired; totalTokens 28,607 → 27,701; 30 internal turns; $0.22 |
| B | `env={"DISABLE_AUTO_COMPACT": "1"}` | **False** | **None** | ✅ Works — auto-compact is fully disabled |
| C | `env={"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "30"}` | True | 967,000 | ❌ Ignored — threshold unchanged |
| D | Control, no env overrides | True | 967,000 | baseline |

**Bottom line**: we have two real levers — `DISABLE_AUTO_COMPACT=1` (env) to turn it off entirely, and sending `/compact` as a user message to trigger it manually. Threshold tuning is not available via env (despite third-party references).

We can also **observe** context usage live via `ClaudeSDKClient.get_context_usage()` — returns `totalTokens`, `maxTokens`, `autoCompactThreshold`, `isAutoCompactEnabled`, and a full per-category breakdown (system prompt, system tools, MCP tools, skills, messages, autocompact buffer, free space).

## Migration implication: switch from `query()` to `ClaudeSDKClient`

Spike 4 surfaced that `ClaudeSDKClient` is a strictly better primitive than the one-shot `query()` function `SDKBackend` currently uses. It adds:

- `get_context_usage()` — live context visibility (replaces any need for ml-intern token accounting)
- `interrupt()` / `stop_task()` — solves concern #5 (cancellation) before Spike 6 even runs
- `set_model()` — runtime model switch (maps directly to ml-intern's `/model` command)
- `set_permission_mode()` — runtime permission-mode toggle
- `reconnect_mcp_server()` — MCP resilience for long sessions
- `rewind_files()` — editing undo (maps to `undo_complete`)
- Proper bidirectional streaming with stateful sessions (multi-turn from one client)

The rewrite inside `SDKBackend` is small: replace the `query()` call with a `ClaudeSDKClient` context manager and use `client.query()` + `client.receive_response()`. Same `ClaudeAgentOptions`, same `can_use_tool`, same MCP servers.

## Recommendations for the full migration

1. **Delete `agent/context_manager/manager.py:_compact_and_notify` + its call sites.** The 190k threshold fires 5× too early; running it under SDKBackend would silently degrade otherwise-working sessions.
2. **Keep `ContextManager` itself** (it still owns message history / session upload / session-id / metadata) — just drop the auto-compaction path.
3. **Refactor `SDKBackend` to wrap `ClaudeSDKClient`**, not `query()`. Unlocks cancellation, context visibility, runtime model switch, and manual `/compact`.
4. **Surface `get_context_usage()` in the CLI status bar.** Frees us from reinventing token accounting and gives users a real `/context`-style readout.
5. **Add a `PreCompact` hook** only if we need to observe compactions (e.g. to log cost-before/after). Don't port ml-intern's existing compaction logic.
6. **Expose a `/compact` escape hatch** in the ml-intern CLI. Maps to `client.query("/compact")` and lets the user trim a long session voluntarily, same as Claude Code itself.
7. **Dedupe usage samples** by `message_id` before surfacing per-turn metrics (see Surprise 1).
8. **Surface `ResultMessage.total_cost_usd` + `num_turns`** on `turn_complete`. Free UX win over the litellm path, which has no such running tally.

## Surprises

### 1. `include_partial_messages=True` double-emits `AssistantMessage`

With `include_partial_messages=True` (set in `SDKBackend._build_options`), each `AssistantMessage` is emitted twice with identical `usage`, `stop_reason`, and content. Every row in the usage table above comes in a pair.

Evidence: `long` scenario had 40 unique tool calls but my instrumentation recorded 84 `AssistantMessage` samples (2× + 4 extras for the opening/closing assistant text).

This does NOT affect the event adapter's correctness — `_handle_assistant` is idempotent for the events it emits because text content only enters `assistant_message` once the stream is closed, and `tool_call` dedup isn't needed since each `ToolUseBlock` has a unique `id`. But it DOES affect any instrumentation that accumulates from the message stream.

**Fix at full-migration time**: dedupe on `msg.message_id` (it's on AssistantMessage) before accumulating usage samples.

### 2. `cache_read_tokens` > 200k without a compaction event

At 311,226 tokens of cached prefix we expected either a `compact_boundary` or an explicit error. Got neither. The follow-up `get_context_usage()` probe explained why: the session was running on `claude-opus-4-7[1m]` (1M context) with the compaction threshold at 967,000 tokens. Our peak was 32% of that. Confirms the SDK's context-management policy is ~5× more permissive than ml-intern's 190k heuristic.

### 3. Cost structure is cache-creation-dominated at this scale

`long` cost ratio: $1.46 / 40 calls = $0.037/call
`blast` cost ratio: $4.84 / 30 calls = $0.161/call

Long outputs are 4–5× more expensive per tool call because `cache_creation_input_tokens` (the expensive write-through) grew from ~100–400/turn in `long` to 7–15k/turn in `blast`. Cached reads are ~10% the cost of fresh input, but cache creation is still premium.

**Implication**: for ml-intern's typical workload (dataset inspection, GitHub searches), keep tool outputs bounded and structured. A 50KB unfiltered file dump in a tool result pays real money per turn. Existing `ToolSpec` output-truncation is still worth doing.

### 4. `num_turns` on `ResultMessage` ≈ `tool_calls + small-constant`

Empirically, `num_turns` is `tool_calls + 3` for `long` and `tool_calls + 3` for `blast`. The three extras are: opening-reason + tool-discovery (ToolSearch) + final-summary. Useful for reasoning about cost amortization per real tool call.

### 5. Fresh `input_tokens` stays at 1–6 throughout

Once the cache is primed (first assistant message), every subsequent turn has `input_tokens: 1–6`. The SDK + Claude Code are sending what amounts to a diff against the cached prefix each turn. This is enormously more efficient than the litellm path, which re-sends the full conversation every call and has no built-in caching.

The litellm path's cost for a 40-tool-call run at Sonnet 4.6 prices would be ~$3–5 with no caching; SDK path was $1.46. Caching is a ~2–3× free cost reduction at this scale that we get purely by migrating, independent of the subscription/API question.

## Thrash question — why "both compactors on" isn't an empirical test

The plan originally asked to run both compactors simultaneously to observe thrash. Concretely, they can't thrash in a normal run because:

- ml-intern's ContextManager operates on its OWN `messages` list, rebuilding a summary when size > 190k.
- SDK maintains its own conversation state server-side.
- In `SDKBackend` today, ml-intern's ContextManager is not fed the SDK's messages — they're separate histories.

Running "both on" would mean: ml-intern rebuilds its message list via `litellm.acompletion` (compaction requires an LLM call). That still only affects what ml-intern re-sends on a NEXT litellm call — which never happens in SDK mode. So the compactor would run but have no effect on the SDK session. Confirmed: there is no runtime thrash scenario under the current architecture. The concern was hypothetical; we can drop it.

## What's still open

- **Automatic compaction at 967k**: the threshold is known; the actual auto-fire hasn't been observed end-to-end (would need a $50+ probe). Low-priority — manual `/compact` at 500k would be cheaper and under our control.
- **Behavior of `isAutoCompactEnabled: False` in a long session**: if we disable auto-compact and then exceed 1M tokens, does the session error, silently truncate, or block? Not tested.
- **Max-plan rate-limit ceiling**: burned ~$7 notional with no throttle; real ceiling unknown. Spike 5 (training run) may find it.

## Cost datapoints

| Scenario | Turns | Duration | Notional cost |
|---|---|---|---|
| short | 13 | 56s | $0.44 |
| long | 43 | 202s | $1.46 |
| blast | 33 | 203s | $4.84 |
| **total** | | | **$6.75** |

On Max-plan auth this is notional (not billed). Real API cost would be similar — 1.3× Sonnet 4.6 input, but no billing since we're auth'd through `claude login`.

## Next

- **Spike 5**: local training smoke test. Answers the end-user goal + exercises rate-limits over a longer wall-clock window.
- **Spike 6**: cancellation / interruption — now mostly pre-answered by the existence of `ClaudeSDKClient.interrupt()` + `stop_task()`. Keep the spike for orphan-HF-job behavior.
- **Pre-Spike-5 cleanup**: (a) refactor `SDKBackend` to wrap `ClaudeSDKClient` instead of `query()` — unlocks cancellation and context visibility; (b) fix the double-emit dedup.
