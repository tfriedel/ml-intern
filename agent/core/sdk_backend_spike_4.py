"""
Spike 4 — long-run / compaction behavior.

Drives `SDKBackend` with a deliberately long task (40 bash calls, each
forced to be summarized) to measure:

  * per-turn token usage curve (from AssistantMessage.usage)
  * compaction events (SystemMessage compact_boundary)
  * total cost, turn count, duration (ResultMessage)
  * whether the SDK stays coherent past ml-intern's 190k threshold

Does NOT engage ml-intern's ContextManager compactor — `SDKBackend`
is standalone today, no ContextManager integration to run. The "both
compactors on" case is a design-level concern (two independent history
stores diverge on compaction); it is not empirically tested here.

Run:
    uv run python -m agent.core.sdk_backend_spike_4 [scenario]

Scenarios:
    long         — 40 bash calls, forced per-call summaries (default)
    short        — 10 bash calls, quick sanity check
    blast        — 40 bash calls with large outputs (stresses compaction)
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from claude_agent_sdk import ToolUseBlock

from agent.core.sdk_backend import (
    DEFAULT_DISALLOWED_BUILTINS,
    SDKBackend,
    _build_demo_tool_specs,
)
from agent.core.sdk_event_adapter import SDKEventAdapter


# ── Instrumented adapter — captures per-turn usage & compaction count ────


class _InstrumentedAdapter(SDKEventAdapter):
    def __init__(self, event_queue: asyncio.Queue):
        super().__init__(event_queue)
        self.usage_samples: list[dict[str, Any]] = []
        self.compact_events: list[dict[str, Any]] = []
        self.tool_call_count: int = 0
        self._tool_call_timestamps: list[float] = []

    async def _handle_assistant(self, msg):
        if msg.usage:
            self.usage_samples.append(
                {
                    "input": msg.usage.get("input_tokens"),
                    "output": msg.usage.get("output_tokens"),
                    "cache_read": msg.usage.get("cache_read_input_tokens"),
                    "cache_creation": msg.usage.get("cache_creation_input_tokens"),
                    "stop_reason": msg.stop_reason,
                    "t": time.monotonic(),
                }
            )
        for b in msg.content:
            if isinstance(b, ToolUseBlock) and b.name != "ToolSearch":
                self.tool_call_count += 1
                self._tool_call_timestamps.append(time.monotonic())
        await super()._handle_assistant(msg)

    async def _handle_system(self, msg):
        if msg.subtype == "compact_boundary":
            self.compact_events.append(
                {
                    "t": time.monotonic(),
                    "data": msg.data,
                }
            )
        await super()._handle_system(msg)


# ── Scenarios ────────────────────────────────────────────────────────────


LONG_COMMANDS = [
    "date",
    "uname -a",
    "cat /etc/os-release",
    "whoami",
    "pwd",
    "df -h",
    "free -h",
    "uptime",
    "hostname",
    "ls -la ~ 2>/dev/null | head -40",
    "cat /proc/cpuinfo | head -30",
    "cat /proc/meminfo | head -20",
    "lscpu | head -25",
    "ps aux | head -40",
    "ip addr show 2>/dev/null | head -40",
    "cat /etc/hostname",
    "ls /usr/bin | head -80",
    "env | head -40",
    "echo $PATH | tr ':' '\\n'",
    "cat /proc/version",
    "ls /etc | head -40",
    "cat /proc/loadavg",
    "cat /proc/uptime",
    "cat /proc/stat | head -20",
    "ls /var/log 2>/dev/null | head -30",
    "cat /etc/passwd | head -30",
    "cat /etc/group | head -30",
    "dpkg -l 2>/dev/null | head -40",
    "which python3",
    "python3 --version",
    "which uv",
    "uv --version",
    "pip --version 2>/dev/null || echo no-pip",
    "git --version",
    "node --version 2>/dev/null || echo no-node",
    "go version 2>/dev/null || echo no-go",
    "rustc --version 2>/dev/null || echo no-rust",
    "docker --version 2>/dev/null || echo no-docker",
    "cat /proc/self/status | head -30",
    "ls -la /tmp | head -30",
]

BLAST_COMMANDS = [
    "ls -laR /etc 2>/dev/null | head -400",
    "ls -laR /usr/share/doc 2>/dev/null | head -400",
    "ls -laR /var/log 2>/dev/null | head -400",
    "cat /proc/cpuinfo",
    "cat /proc/meminfo",
    "cat /proc/self/maps | head -200",
    "dpkg -l 2>/dev/null",
    "find /etc -type f 2>/dev/null | head -400",
    "find /usr/include -type f 2>/dev/null | head -400",
    "find /usr/lib -maxdepth 3 -type d 2>/dev/null | head -400",
] * 4  # repeat so each output is chunky and different contexts build up


def _build_task(commands: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(commands))
    return (
        "Run EACH of the following bash commands, one at a time, in order, "
        "via the `bash` tool. After each command, write ONE short sentence "
        "(<20 words) describing what you learned. Do NOT skip commands; do "
        "NOT combine them; do NOT repeat commands you already ran.\n\n"
        "After all commands have been run, write a final one-paragraph "
        "summary of what the system looks like.\n\n"
        "Commands:\n" + numbered
    )


SCENARIOS: dict[str, str] = {
    "short": _build_task(LONG_COMMANDS[:10]),
    "long": _build_task(LONG_COMMANDS),
    "blast": _build_task(BLAST_COMMANDS),
}


# ── Driver ───────────────────────────────────────────────────────────────


async def _drain_events(q: asyncio.Queue, done: asyncio.Event, brief: bool) -> list:
    """Consume events. If brief, only print summary line events; else dump all."""
    captured = []
    while not (done.is_set() and q.empty()):
        try:
            event = await asyncio.wait_for(q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        captured.append(event)
        if brief:
            if event.event_type in {
                "ready",
                "approval_required",
                "tool_state_change",
                "turn_complete",
                "compacted",
                "error",
            }:
                print(
                    f"[event] {event.event_type} "
                    f"{json.dumps(event.data, default=str)[:200]}",
                    flush=True,
                )
            elif event.event_type == "tool_call":
                args = event.data.get("arguments", {}) if event.data else {}
                cmd = args.get("command", "")
                print(
                    f"[tool_call] {event.data.get('tool')} "
                    f"{str(cmd)[:90]}",
                    flush=True,
                )
        else:
            print(
                f"[event] {event.event_type} "
                f"{json.dumps(event.data, default=str)[:200]}",
                flush=True,
            )
    return captured


async def _run(scenario: str, brief: bool) -> None:
    prompt = SCENARIOS.get(scenario)
    if prompt is None:
        print(f"unknown scenario: {scenario}. Choose from {list(SCENARIOS)}")
        sys.exit(2)

    print(f"=== spike 4 — scenario={scenario} ===", flush=True)
    print(f"(prompt: {len(prompt)} chars, commands: {prompt.count(chr(10) + '1. ') + prompt.count(chr(10) + '2. ')}…)")

    q: asyncio.Queue = asyncio.Queue()
    backend = SDKBackend(
        tool_specs=_build_demo_tool_specs(),
        event_queue=q,
        config=None,
        session=None,
        system_prompt=(
            "You run shell commands diligently and summarize results "
            "concisely. Never combine commands. Never skip commands."
        ),
        max_turns=200,  # long run: lots of ToolSearch+tool+thought hops
        deny_all_sensitive=False,
    )
    # Swap in the instrumented adapter so we capture usage + compaction.
    adapter = _InstrumentedAdapter(q)
    backend.adapter = adapter

    done = asyncio.Event()
    drain_task = asyncio.create_task(_drain_events(q, done, brief))

    t0 = time.monotonic()
    try:
        summary = await backend.run_turn(prompt)
    finally:
        done.set()
        events = await drain_task
    t1 = time.monotonic()

    # ── summarize the run ───────────────────────────────────────────
    print("\n========== spike 4 summary ==========", flush=True)
    print(json.dumps(summary, default=str, indent=2))
    print()

    print("usage curve (per assistant message):")
    print(
        f"{'turn':>4}  {'in':>8}  {'out':>6}  "
        f"{'cache_read':>10}  {'cache_new':>9}  stop_reason"
    )
    for i, u in enumerate(adapter.usage_samples):
        print(
            f"{i:>4}  {str(u['input']):>8}  {str(u['output']):>6}  "
            f"{str(u['cache_read']):>10}  {str(u['cache_creation']):>9}  "
            f"{u['stop_reason']}"
        )

    # high-level stats
    input_tokens = [u["input"] for u in adapter.usage_samples if u["input"]]
    output_tokens = [u["output"] for u in adapter.usage_samples if u["output"]]
    cache_read = [u["cache_read"] for u in adapter.usage_samples if u["cache_read"]]

    print()
    print(f"total tool_calls (ex. ToolSearch): {adapter.tool_call_count}")
    print(f"total assistant messages:          {len(adapter.usage_samples)}")
    print(f"total compaction events:           {len(adapter.compact_events)}")
    if adapter.compact_events:
        for c in adapter.compact_events:
            print(f"  compact: {c['data']}")
    if input_tokens:
        print(
            f"peak input_tokens (one turn):      {max(input_tokens):,}"
        )
    if cache_read:
        print(f"peak cache_read_tokens:            {max(cache_read):,}")
    if output_tokens:
        total_out = sum(output_tokens)
        print(f"sum output_tokens:                 {total_out:,}")
    print(f"wall time:                         {t1 - t0:.1f}s")
    print(
        f"tool events in queue:              {sum(1 for e in events if e.event_type == 'tool_call')}"
    )
    print(
        f"disallowed builtins:               {DEFAULT_DISALLOWED_BUILTINS}"
    )


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "long"
    brief = "--verbose" not in sys.argv
    asyncio.run(_run(scenario, brief))


if __name__ == "__main__":
    main()
