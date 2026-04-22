# Spike 6 — Cancellation / interruption

**Branch:** `spike/sdk-hello`
**Artifact:** `agent/core/sdk_backend_spike_6.py`
**Status:** ✅ SDK mechanism works. ⚠️ `_bash_handler` has an orphan-process gap that must be fixed before migration ships.

## Goal

Does `backend.interrupt()` (delegating to `ClaudeSDKClient.interrupt()`) actually stop an in-flight turn? Three flavors:

A. **Streaming text** — mid-response, before any tool.
B. **Session continuity** — can the backend accept new turns after an interrupt?
C. **Mid-tool execution** — does a running local `bash` process die?

Originally the plan also called for testing HF-Jobs orphan behavior. We reason from code instead of spending real HF credits — documented below.

## Results

### Test A — Streaming interrupt ✅

```
saw 25 chunks after 11.61s; calling interrupt()
interrupt() returned in 0.001s
run_turn returned in 0.005s: stop=None is_error=True
total assistant_chunks seen: 25
```

Called `interrupt()` after 25 `assistant_chunk` events had flowed through the adapter. SDK aborts in ~1ms, `run_turn` task ends in ~5ms with `is_error=True` and `stop_reason=None`. No additional chunks after interrupt. Event queue clean.

**The SDK-level interrupt mechanism is solid.**

### Test B — Session survives interrupt ✅

```
turn 1 done: stop=end_turn       # first short turn, uninterrupted
turn 2 ended: stop=None is_error=True   # long turn, interrupted mid-stream
turn 3 done: stop=end_turn       # follow-up after interrupt — works
```

`ClaudeSDKClient` keeps the session live across an interrupt. No need to reconnect, re-register tools, or rebuild state. This matters for ml-intern's CLI model where users Ctrl-C a single turn and immediately send another prompt.

### Test C — Interrupt during local bash ⚠️

```
sentinel appeared at t+21.70s
interrupt() returned in 0.014s
max poll gap during tool run: 0.25s   # event loop NOT blocked
run_turn returned: stop=tool_use is_error=True
total wall time:  21.72s   (tool command was `sleep 10`)
sentinel final:   'start\ndone\n'     # sleep 10 completed normally
```

The bash command was:
```
echo start > /tmp/spike6-sentinel.txt; sleep 10; echo done >> /tmp/spike6-sentinel.txt
```

What happened:
- `interrupt()` fired after the `sleep 10` was mid-run.
- SDK session aborted the turn.
- But the underlying `subprocess.run` (running on an MCP-server worker thread) **kept running to completion** — hence `sentinel: 'start\ndone\n'` and a 21.7s total wall time.
- Event loop was NOT blocked (poll gap 0.25s). So the MCP server runs handlers in a threadpool, and cancellation of the coroutine doesn't kill the native subprocess.

**The SDK aborts the turn; the OS process keeps going.** The user sees "turn cancelled" in the UI while their training job is still burning GPU on the host.

## Test C is a real bug we need to fix

The current handler (`agent/tools/local_tools.py:96-126`):

```python
async def _bash_handler(args: dict[str, Any], **_kw) -> tuple[str, bool]:
    command = args.get("command", "")
    ...
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        cwd=work_dir, timeout=timeout,
    )
    ...
```

This synchronous `subprocess.run` can't be cancelled from an `asyncio` context. When the SDK pulls the plug on the turn, the handler continues running on its worker thread until the command completes.

### Two options to fix

**Option 1 — rewrite `_bash_handler` using asyncio.**

```python
async def _bash_handler(args, **_kw):
    proc = await asyncio.create_subprocess_shell(
        command, stdout=PIPE, stderr=PIPE, cwd=work_dir,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.CancelledError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            proc.kill()
        raise
```

This makes the handler properly cancellable. `proc.terminate()` sends SIGTERM; if the process ignores it we escalate to SIGKILL. Clean shutdown, no orphans.

**Option 2 — delete `_bash_handler` and use the SDK's built-in `Bash` tool.**

Claude Code's built-in Bash is already cancellation-aware and has better ergonomics (output size caps, background-run support). Our motivation for disabling it (Spike 1 → `DEFAULT_DISALLOWED_BUILTINS`) was to keep ml-intern's approval authoritative — but for local read/write bash in a user's own working dir, that concern is weak.

Option 2 is the bigger simplification: lose ~400 lines of `local_tools.py`, gain a battle-tested cancellation story, gain file-checkpointing for free. Minor cost: switch approval rules to check builtin tool names (`Bash`, `Write`, `Edit`) alongside MCP names in `_needs_approval()`.

**Recommendation**: go with Option 2 at full migration time. It compounds with Spike 5's finding that the agent uses heredoc-via-bash because we disabled Write — re-enabling both fixes two problems at once.

## HF-Jobs cancellation — reasoned, not run

Reading `agent/tools/jobs_tool.py`:

- `hf_jobs run` submits a job to HF via `HfApi.run_job(...)`, then polls status.
- The handler awaits the polling loop (which is properly async: `await asyncio.sleep(...)`), so cancellation propagates correctly at the asyncio layer.
- **However**, the remote HF job keeps running regardless. Cancelling our Python side only stops us watching — HF's scheduler happily continues the job until it finishes or times out.

**Implication**: when a user hits Ctrl-C on a turn that submitted an HF job, the session is cancelled but the job is still running (and still billing). ml-intern should:

1. Capture the `job_id` from the initial submit response BEFORE polling starts.
2. On turn cancellation, emit a `tool_state_change(state="cancelled")` with the `job_id`.
3. Offer the user an explicit follow-up: *"job still running remotely — cancel it? (y/n)"*. If yes, submit `hf_jobs cancel <job_id>` on the next turn.

This is a UX pattern, not a bug. Users submitting HF jobs accept the billing model; they just need visibility.

## Concern status

From `~/.claude/plans/composed-squishing-island.md`:

| # | Concern | Status |
|---|---|---|
| 5 | Interruption semantics | 🟡 Partially resolved — SDK mechanism ✅; local bash is orphaned ⚠️ (fix in migration); HF Jobs are orphaned-by-design (UX pattern). |

## Recommendations for the full migration

1. **Fix `_bash_handler` cancellation** — Option 2 preferred (delete + use SDK builtin). At minimum, do Option 1 before shipping.
2. **Wire SIGINT / Ctrl-C to `backend.interrupt()`** in `agent/main.py`. Current code emits an `interrupted` event from the submission loop; route it through the backend instead.
3. **On turn cancellation, check for orphaned state**:
   - If any `hf_jobs run` was called: show active `job_id`s and offer cancel.
   - If any local subprocess had a tool call in flight: log a warning (after Option 2 this goes away).
4. **Surface `tool_state_change(state="cancelled")`** from the permission callback when `ctx.signal` fires. Spike 3 identified this as the last unmapped state; Spike 6 confirms it's reachable via the signal.

## Surprises

### 1. The MCP server uses a thread pool, not the event loop

I initially guessed the synchronous `subprocess.run` would block the event loop. Test C's poll-gap measurement (0.25s max) disproved this — the loop was alive. `create_sdk_mcp_server` transparently runs handlers on a worker thread. That's good news (no deadlock), but also the mechanism by which orphan processes slip through cancellation.

### 2. `interrupt()` is instantaneous (~1ms) and idempotent

No async heavy-lifting — it just signals the SDK's internal stream to close. Calling it multiple times is safe. This makes it trivially safe to bind to SIGINT without race conditions.

### 3. `run_turn` returns `stop_reason=None` (not `"interrupted"` or similar)

After interrupt, the `ResultMessage.stop_reason` is `None` and `is_error=True`. The adapter currently maps `is_error=True` → `error` event (not `interrupted`). For UX, we should special-case this: when `is_error=True` AND a user-initiated interrupt was just called, emit `interrupted` instead of `error`. Tracked as a small adapter tweak for the full migration.

## Cost datapoints

| Test | Duration | Notional cost |
|---|---|---|
| A (streaming interrupt) | ~12s | ~$0.10 |
| B (3 turns, one interrupted) | ~20s | ~$0.20 |
| C (bash sleep 10 + interrupt) | ~22s | ~$0.15 |

Total Spike 6 spend: ~$0.45 notional.

## Next

All six spikes complete. Ready for the **full migration**:

1. Re-enable SDK builtins (delete `_bash_handler`, use Claude Code's `Bash`/`Write`/`Edit`). Fixes Spike 6 Test C and Spike 5's heredoc cleanliness gap.
2. Wire `SDKBackend` into `agent/main.py` behind a `--backend sdk` flag.
3. Port `_needs_approval()` to also match builtin tool names.
4. Wire SIGINT → `backend.interrupt()` + `interrupted` event remapping.
5. Delete `ContextManager._compact_and_notify()` (Spike 4 conclusion).
6. Add `backend.get_context_usage()` → status-bar readout.
7. Ship a `finetune-locally` skill (Spike 5 recommendation).
8. Remove `ANTHROPIC_API_KEY` requirement from README; add `claude` CLI + login health check.

## Cleanup

```
rm -f /tmp/spike6-sentinel.txt
```
