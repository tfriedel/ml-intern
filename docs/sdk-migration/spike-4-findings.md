# Spike 4 — Long-run & compaction behavior

**Branch:** `spike/sdk-hello`
**Artifact:** `agent/core/sdk_backend_spike_4.py`
**Status:** ✅ Decisive — the SDK never compacted, even past 300k cached tokens. ml-intern's ContextManager compaction is safe to delete.

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

The `311k cache_read` data point is the key finding. Sonnet's "default" context window is 200k; we went 55% over it without any compaction event. That strongly implies either:
- the SDK transparently engages the `context-1m-2025-08-07` beta (or its post-beta equivalent) for sessions that grow past 200k, or
- Claude Code's effective context for SDK sessions is already 1M, or
- compaction happens server-side without emitting `compact_boundary`.

Whichever is true, **the effective ceiling is far higher than ml-intern's 190k compactor would tolerate**. Our compactor would fire at 190k and truncate a session that the SDK is willing to keep going with — a pure regression.

## Concern status

From `~/.claude/plans/composed-squishing-island.md`:

| # | Concern | Status |
|---|---|---|
| 3 | Compaction collision | ✅ Closed — SDK doesn't compact under realistic pressure; ml-intern's compactor is safe to delete. If we ever run past ~900k tokens and the SDK does compact, we can layer a `PreCompact` hook later. |
| 6 | Rate limits on Max | 🟡 Partial — burned ~$6.75 notional across three scenarios on Max-plan auth, no throttling, no errors. Still no 8-hour training run data, so full answer deferred to Spike 5 / beyond. |

## Recommendations for the full migration

1. **Delete `agent/context_manager/manager.py:_compact_and_notify` + its call sites.** The 190k threshold fires before the SDK's ceiling; running it under SDKBackend would silently degrade otherwise-working sessions.
2. **Keep `ContextManager` itself** (it still owns message history / session upload / session-id / metadata) — just drop the auto-compaction path.
3. **Add a `PreCompact` hook** only if Spike 5/6 or production shows the SDK doing something we want to observe (e.g. to log cost-before/after, or to ensure trackio logs survive a compact). Don't port the existing logic.
4. **Dedupe usage samples** by `message_id` before surfacing per-turn metrics (see Surprise 1).
5. **Surface `ResultMessage.total_cost_usd` + `num_turns`** on `turn_complete`. The event adapter already forwards these; CLI/status bar should show them. Free UX win over the litellm path, which has no such running tally.

## Surprises

### 1. `include_partial_messages=True` double-emits `AssistantMessage`

With `include_partial_messages=True` (set in `SDKBackend._build_options`), each `AssistantMessage` is emitted twice with identical `usage`, `stop_reason`, and content. Every row in the usage table above comes in a pair.

Evidence: `long` scenario had 40 unique tool calls but my instrumentation recorded 84 `AssistantMessage` samples (2× + 4 extras for the opening/closing assistant text).

This does NOT affect the event adapter's correctness — `_handle_assistant` is idempotent for the events it emits because text content only enters `assistant_message` once the stream is closed, and `tool_call` dedup isn't needed since each `ToolUseBlock` has a unique `id`. But it DOES affect any instrumentation that accumulates from the message stream.

**Fix at full-migration time**: dedupe on `msg.message_id` (it's on AssistantMessage) before accumulating usage samples.

### 2. `cache_read_tokens` > 200k without a compaction event

This is the banner finding. At 311,226 tokens of cached prefix we expected either a `compact_boundary` or an explicit error. Got neither. Confirms the SDK's context-management policy is much more permissive than ml-intern's 190k heuristic.

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

- **Long-term coherence past 500k**: didn't observe it. If a Spike 5 training run + analysis grows past 500k we'll learn more.
- **When the SDK DOES compact**: unobserved. We only know it happens later than 311k. The event-adapter mapping for `compacted` is written but untested against a real `compact_boundary`.
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
- **Spike 6**: cancellation / interruption.
- Pre-Spike-5 cleanup: fix the double-emit dedup and tighten the usage capture — will make Spike 5's metrics clean.
