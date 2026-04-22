"""
Spike 3 — Event-stream mapping.

Drives `SDKEventAdapter` end-to-end. Runs a prompt that exercises all the
common event types (streaming text + tool call + tool output + result) and
prints the resulting ml-intern Event stream in order.

Approval path is bolted on too: `can_use_tool` emits `approval_required`
events directly onto the same queue so we can confirm the full surface.

Run:
    uv run python -m agent.core.sdk_backend_spike_3
    uv run python -m agent.core.sdk_backend_spike_3 deny
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
    create_sdk_mcp_server,
    query,
    tool,
)

from agent.core.agent_loop import _needs_approval
from agent.core.sdk_event_adapter import SDKEventAdapter, _strip_mcp_prefix
from agent.core.session import Event
from agent.tools.jobs_tool import hf_jobs_handler
from agent.tools.local_tools import _bash_handler


# ── MCP tool wrappers (same as Spike 2) ────────────────────────────────

@tool("bash", "Run a shell command locally.", {"command": str})
async def bash_tool(args: dict[str, Any]) -> dict[str, Any]:
    output, ok = await _bash_handler({"command": args.get("command", "")})
    return {"content": [{"type": "text", "text": output}], "is_error": not ok}


@tool(
    "hf_jobs",
    "Manage HF Jobs. Sensitive ops (run on GPU) require approval.",
    {
        "operation": str,
        "script": str,
        "command": str,
        "image": str,
        "hardware_flavor": str,
        "timeout": str,
        "job_id": str,
    },
)
async def hf_jobs_tool(args: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in args.items() if v not in (None, "", [])}
    output, ok = await hf_jobs_handler(clean, session=None, tool_call_id=None)
    return {"content": [{"type": "text", "text": output}], "is_error": not ok}


# ── Approval wiring: emit `approval_required` onto the same queue ──────

def make_approval_callback(q: asyncio.Queue, deny: bool):
    async def cb(
        tool_name: str, tool_args: dict[str, Any], ctx: ToolPermissionContext
    ):
        bare = _strip_mcp_prefix(tool_name)
        needs = _needs_approval(bare, tool_args, config=None)
        if not needs:
            return PermissionResultAllow(behavior="allow")

        await q.put(Event(
            event_type="approval_required",
            data={
                "tools": [{
                    "tool": bare,
                    "arguments": tool_args,
                    "tool_call_id": ctx.tool_use_id,
                }],
                "count": 1,
            },
        ))
        if deny:
            return PermissionResultDeny(
                behavior="deny",
                message=f"[spike] denied {bare}",
                interrupt=False,
            )
        return PermissionResultAllow(behavior="allow")

    return cb


# ── Drive a prompt and print the resulting ml-intern events ────────────

async def run(prompt: str, deny: bool = True) -> None:
    q: asyncio.Queue = asyncio.Queue()
    adapter = SDKEventAdapter(q)

    server = create_sdk_mcp_server("local", tools=[bash_tool, hf_jobs_tool])
    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a terse assistant. Use MCP tools when asked. If a "
            "tool is denied, report it without retrying."
        ),
        mcp_servers={"local": server},
        disallowed_tools=["Bash", "Read", "Write", "Edit"],
        can_use_tool=make_approval_callback(q, deny=deny),
        include_partial_messages=True,  # enable StreamEvent for chunks
        max_turns=6,
    )

    async def prompt_stream():
        yield {"type": "user", "message": {"role": "user", "content": prompt}}

    # Drain the event queue in a background task so we print events as
    # they arrive (not after the whole stream finishes).
    stopped = object()

    async def drain():
        while True:
            event = await q.get()
            if event is stopped:  # type: ignore[comparison-overlap]
                return
            print(
                f"[event] {event.event_type} "
                f"{json.dumps(event.data, default=str)[:200]}",
                flush=True,
            )

    drain_task = asyncio.create_task(drain())

    print(f"\n>>> prompt: {prompt}\n", flush=True)
    summary = await adapter.consume(query(prompt=prompt_stream(), options=options))

    # Sentinel to stop the drainer.
    await q.put(stopped)  # type: ignore[arg-type]
    await drain_task

    print("\n--- result summary ---", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "session_id"}, default=str, indent=2))


SCENARIOS = {
    "bash": ("Use `bash` to print the current date.", False),
    "deny": (
        "Submit a GPU job with hf_jobs (operation=run, "
        "hardware_flavor=t4-small, script='print(1)', timeout='5m'). "
        "If denied, just report it.",
        True,
    ),
    "mix": (
        "First use `bash` to print the current date. Then list HF jobs "
        "with hf_jobs operation=ps.",
        False,
    ),
}


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "mix"
    prompt, deny = SCENARIOS.get(key, (key, False))
    asyncio.run(run(prompt, deny=deny))


if __name__ == "__main__":
    main()
