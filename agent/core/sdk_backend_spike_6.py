"""
Spike 6 — cancellation / interruption semantics.

Three tests:

  A. Interrupt during LLM streaming. Asks for a long essay; waits for
     the stream to start; calls `backend.interrupt()`. Verifies the SDK
     mechanism aborts the turn quickly.

  B. Session survives interrupt. After test A, send another user turn
     and confirm the session is still usable.

  C. Interrupt during a running local bash tool — documents the known
     gap. `_bash_handler` uses synchronous `subprocess.run`, which
     blocks the asyncio event loop and cannot be interrupted. We
     reproduce that and measure the damage.

Run:
    uv run python -m agent.core.sdk_backend_spike_6 [a|b|c|all]
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from agent.core.sdk_backend import SDKBackend, _build_demo_tool_specs
from agent.core.session import Event


SENTINEL = Path("/tmp/spike6-sentinel.txt")


def _clear_sentinel() -> None:
    SENTINEL.unlink(missing_ok=True)


# ── Test A — interrupt during LLM streaming ─────────────────────────────


async def test_a_streaming_interrupt() -> dict:
    print("=" * 70)
    print("A. Interrupt during LLM streaming (no tools)")
    print("=" * 70)

    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()

    CHUNK_TARGET = 25
    ready_to_interrupt = asyncio.Event()
    seen_chunks = [0]
    events: list[Event] = []

    async def local_drain():
        while not done.is_set() or not q.empty():
            try:
                event = await asyncio.wait_for(q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            events.append(event)
            if event.event_type == "assistant_chunk":
                seen_chunks[0] += 1
                if seen_chunks[0] == CHUNK_TARGET and not ready_to_interrupt.is_set():
                    ready_to_interrupt.set()

    async with SDKBackend(
        tool_specs=_build_demo_tool_specs(),
        event_queue=q,
        system_prompt="You are verbose. Write long, detailed responses.",
        max_turns=4,
    ) as backend:
        drain_task = asyncio.create_task(local_drain())

        turn = asyncio.create_task(
            backend.run_turn(
                "Write me a 2000-word essay on the history of the "
                "programming language Lisp — dialects, implementations, "
                "influential papers, and cultural impact. Be thorough."
            )
        )

        t0 = time.monotonic()
        try:
            await asyncio.wait_for(ready_to_interrupt.wait(), timeout=30)
            t_ready = time.monotonic() - t0
            print(f"  saw {CHUNK_TARGET} chunks after {t_ready:.2f}s; calling interrupt()")
        except asyncio.TimeoutError:
            print(f"  ⚠️  didn't see {CHUNK_TARGET} chunks in 30s — interrupting anyway")

        t_int = time.monotonic()
        await backend.interrupt()
        print(f"  interrupt() returned in {time.monotonic() - t_int:.3f}s")

        try:
            t_wait = time.monotonic()
            summary = await asyncio.wait_for(turn, timeout=30)
            print(
                f"  run_turn returned in {time.monotonic() - t_wait:.3f}s: "
                f"stop={summary.get('stop_reason')} "
                f"is_error={summary.get('is_error')}"
            )
        except asyncio.TimeoutError:
            print("  ❌ run_turn hung 30s after interrupt")
            turn.cancel()
            summary = {"hung": True}

        done.set()
        try:
            await asyncio.wait_for(drain_task, timeout=2)
        except asyncio.TimeoutError:
            drain_task.cancel()

    print(f"  total assistant_chunks seen: {seen_chunks[0]}")
    return {
        "chunks_before_interrupt": seen_chunks[0],
        "summary": summary,
    }


# ── Test B — session survival after interrupt ──────────────────────────


async def test_b_session_survives_interrupt() -> dict:
    print("\n" + "=" * 70)
    print("B. Session survives interrupt — send another turn after")
    print("=" * 70)

    q: asyncio.Queue = asyncio.Queue()

    async with SDKBackend(
        tool_specs=_build_demo_tool_specs(),
        event_queue=q,
        system_prompt="Be terse unless asked otherwise.",
        max_turns=6,
    ) as backend:
        # Drain queue in background, just discard
        done = asyncio.Event()
        drain_task = asyncio.create_task(_simple_drain(q, done))

        # Turn 1: small, successful
        s1 = await backend.run_turn("Reply with just the word 'first'.")
        print(f"  turn 1 done: stop={s1.get('stop_reason')}")

        # Turn 2: long, we'll interrupt
        t2 = asyncio.create_task(
            backend.run_turn(
                "Write 500 words about the history of Unix. Take your time."
            )
        )
        # Wait a bit for streaming to be active
        await asyncio.sleep(3)
        print("  interrupting turn 2 mid-stream…")
        await backend.interrupt()
        try:
            s2 = await asyncio.wait_for(t2, timeout=30)
            print(f"  turn 2 ended: stop={s2.get('stop_reason')} "
                  f"is_error={s2.get('is_error')}")
        except Exception as e:
            print(f"  turn 2 raised: {type(e).__name__}: {e}")
            s2 = {"raised": str(e)}

        # Turn 3: prove session still works
        print("  sending turn 3 after interrupt…")
        try:
            s3 = await backend.run_turn("Reply with just the word 'third'.")
            print(f"  turn 3 done: stop={s3.get('stop_reason')}")
        except Exception as e:
            print(f"  ❌ turn 3 raised: {type(e).__name__}: {e}")
            s3 = {"raised": str(e)}

        done.set()
        try:
            await asyncio.wait_for(drain_task, timeout=2)
        except asyncio.TimeoutError:
            drain_task.cancel()

    return {"t1": s1, "t2": s2, "t3": s3}


async def _simple_drain(q: asyncio.Queue, done: asyncio.Event) -> None:
    while not done.is_set() or not q.empty():
        try:
            await asyncio.wait_for(q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue


# ── Test C — local bash blocking issue ─────────────────────────────────


async def test_c_bash_handler_blocks() -> dict:
    """Demonstrate that `_bash_handler`'s synchronous subprocess.run
    blocks the asyncio event loop, making mid-tool interrupt() a no-op."""
    print("\n" + "=" * 70)
    print("C. Bash handler blocks event loop (known gap)")
    print("=" * 70)

    _clear_sentinel()

    q: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    drain_task = asyncio.create_task(_simple_drain(q, done))

    async with SDKBackend(
        tool_specs=_build_demo_tool_specs(),
        event_queue=q,
        system_prompt="Run the exact command given, in one bash call.",
        max_turns=4,
    ) as backend:
        # Background poller: while the tool runs, try to observe any
        # time the event loop is live enough to poll the sentinel.
        poll_timestamps = []

        async def poller():
            while not done.is_set():
                poll_timestamps.append(time.monotonic())
                await asyncio.sleep(0.25)

        poll_task = asyncio.create_task(poller())

        t0 = time.monotonic()
        turn = asyncio.create_task(
            backend.run_turn(
                "Call the `bash` tool with command "
                "\"echo start > /tmp/spike6-sentinel.txt; sleep 10; "
                "echo done >> /tmp/spike6-sentinel.txt\". "
                "Report the sentinel contents afterward."
            )
        )

        # Wait for sentinel to appear (proves bash started)
        appeared_at = None
        while time.monotonic() - t0 < 25 and not SENTINEL.exists():
            await asyncio.sleep(0.05)
        if SENTINEL.exists():
            appeared_at = time.monotonic() - t0
            print(f"  sentinel appeared at t+{appeared_at:.2f}s")
        else:
            print("  sentinel never appeared within 25s")

        # Immediately try to interrupt — should fire but likely can't
        # stop a blocking subprocess.run.
        t_int = time.monotonic()
        await backend.interrupt()
        print(f"  interrupt() returned in {time.monotonic() - t_int:.3f}s")

        # Check poll timestamp density around interrupt moment. If the
        # event loop was blocked, polls won't be evenly 0.25s apart.
        pre_polls = [p for p in poll_timestamps if p < t_int]
        if len(pre_polls) >= 2:
            gaps = [pre_polls[i] - pre_polls[i-1] for i in range(1, len(pre_polls))]
            max_gap = max(gaps)
            print(f"  max poll gap during tool run: {max_gap:.2f}s "
                  f"(expected ~0.25s; large means event loop was blocked)")

        summary = await turn
        print(f"  run_turn returned: stop={summary.get('stop_reason')} "
              f"is_error={summary.get('is_error')}")
        total = time.monotonic() - t0
        print(f"  total wall time:    {total:.2f}s "
              f"(bash was sleep 10; blocked handler means ≥ 10s)")

        final = SENTINEL.read_text() if SENTINEL.exists() else "<missing>"
        print(f"  sentinel final:     {final!r}")

        done.set()
        poll_task.cancel()
        try:
            await drain_task
        except asyncio.CancelledError:
            pass

    return {
        "sentinel_final": final,
        "wall_time": total,
        "sentinel_appeared_at": appeared_at,
    }


# ── Driver ──────────────────────────────────────────────────────────────


async def _run_all():
    a = await test_a_streaming_interrupt()
    b = await test_b_session_survives_interrupt()
    c = await test_c_bash_handler_blocks()
    print("\n" + "=" * 70)
    print("SPIKE 6 SUMMARY")
    print("=" * 70)
    print(f"A (streaming interrupt):     {json.dumps(a, default=str)[:200]}")
    print(f"B (session survival):        t1.stop={b['t1'].get('stop_reason')}  "
          f"t2.stop={b['t2'].get('stop_reason') if isinstance(b['t2'], dict) else 'n/a'}  "
          f"t3.stop={b['t3'].get('stop_reason') if isinstance(b['t3'], dict) else 'n/a'}")
    print(f"C (bash blocking gap):       wall={c['wall_time']:.1f}s  "
          f"sentinel={c['sentinel_final']!r}")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "a":
        asyncio.run(test_a_streaming_interrupt())
    elif which == "b":
        asyncio.run(test_b_session_survives_interrupt())
    elif which == "c":
        asyncio.run(test_c_bash_handler_blocks())
    else:
        asyncio.run(_run_all())


if __name__ == "__main__":
    main()
