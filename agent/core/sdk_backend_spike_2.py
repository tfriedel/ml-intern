"""
Spike 2 — Permission callback parity.

Extends Spike 1. Registers bash + hf_jobs as MCP tools, wires ml-intern's
existing `_needs_approval()` into the SDK's `can_use_tool` callback, and
confirms the callback gets enough info to replay production approval
rules.

This spike does NOT actually submit HF jobs. It programmatically denies
anything `_needs_approval()` would flag, and logs what the callback saw.
For tools that bypass approval (e.g. `hf_jobs operation=ps`), the real
handler fires — which is fine, `ps` is read-only.

Run:
    uv run python -m agent.core.sdk_backend_spike_2 pass
    uv run python -m agent.core.sdk_backend_spike_2 deny
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from agent.core.agent_loop import _needs_approval
from agent.tools.jobs_tool import hf_jobs_handler
from agent.tools.local_tools import _bash_handler


# ── MCP tool wrappers ──────────────────────────────────────────────────

@tool(
    name="bash",
    description="Execute a shell command locally. Returns stdout+stderr.",
    input_schema={"command": str},
)
async def bash_tool(args: dict[str, Any]) -> dict[str, Any]:
    output, ok = await _bash_handler({"command": args.get("command", "")})
    return {"content": [{"type": "text", "text": output}], "is_error": not ok}


@tool(
    name="hf_jobs",
    description=(
        "Submit/list/inspect/cancel HF Jobs. Operations: run, ps, logs, "
        "inspect, cancel. Sensitive operations (run with GPU) require "
        "approval."
    ),
    input_schema={
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
    # Drop empty string fields (SDK fills every declared param) before
    # forwarding, so the production handler sees the same args shape it
    # gets in ml-intern today.
    clean = {k: v for k, v in args.items() if v not in (None, "", [])}
    output, ok = await hf_jobs_handler(clean, session=None, tool_call_id=None)
    return {"content": [{"type": "text", "text": output}], "is_error": not ok}


# ── Permission callback — reuses ml-intern's production rule ───────────

CALLBACK_LOG: list[dict[str, Any]] = []


def _strip_mcp_prefix(name: str) -> str:
    # SDK presents our tools as `mcp__<server>__<tool>`. Production
    # `_needs_approval` expects the bare name.
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return name


async def can_use_tool(
    tool_name: str,
    tool_args: dict[str, Any],
    ctx: ToolPermissionContext,
) -> PermissionResultAllow | PermissionResultDeny:
    bare = _strip_mcp_prefix(tool_name)
    needs = _needs_approval(bare, tool_args, config=None)

    entry = {
        "raw_name": tool_name,
        "bare_name": bare,
        "args": tool_args,
        "tool_use_id": ctx.tool_use_id,
        "agent_id": ctx.agent_id,
        "needs_approval": needs,
    }
    CALLBACK_LOG.append(entry)
    print(f"[can_use_tool] {json.dumps(entry)}", flush=True)

    if needs:
        return PermissionResultDeny(
            behavior="deny",
            message=(
                f"[spike] Tool `{bare}` requires approval "
                f"(operation={tool_args.get('operation')}, "
                f"hardware={tool_args.get('hardware_flavor')}). "
                "Approval denied for spike purposes."
            ),
            interrupt=False,
        )
    return PermissionResultAllow(behavior="allow")


# ── Run a prompt ───────────────────────────────────────────────────────

async def run_prompt(prompt: str) -> None:
    server = create_sdk_mcp_server(
        name="local",
        version="0.0.2-spike",
        tools=[bash_tool, hf_jobs_tool],
    )

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a terse ML engineer assistant. Use `mcp__local__bash` "
            "for shell, and `mcp__local__hf_jobs` for HF Jobs operations. "
            "If a tool is denied, report what happened without retrying."
        ),
        mcp_servers={"local": server},
        # NOT using `allowed_tools` here: adding a tool to that list
        # pre-approves it and bypasses `can_use_tool`. We want every MCP
        # call to route through our callback so ml-intern's approval
        # rules stay authoritative. Learned the hard way in Spike 2.
        disallowed_tools=["Bash", "Read", "Write", "Edit"],
        can_use_tool=can_use_tool,
        max_turns=5,
    )

    print(f"\n>>> prompt: {prompt}\n", flush=True)

    # can_use_tool requires streaming mode — the SDK wants the prompt as
    # an AsyncIterable of message dicts, not a plain string.
    async def prompt_stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": prompt},
        }

    async for msg in query(prompt=prompt_stream(), options=options):
        if isinstance(msg, SystemMessage):
            print(f"[system] {msg.subtype}", flush=True)
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"[assistant] {block.text}", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"[tool_use] {block.name} args={block.input}", flush=True)
        elif isinstance(msg, ResultMessage):
            print(
                f"[result] stop={msg.stop_reason} turns={msg.num_turns} "
                f"usd={getattr(msg, 'total_cost_usd', None)}",
                flush=True,
            )
        else:
            for block in getattr(msg, "content", []) or []:
                if isinstance(block, ToolResultBlock):
                    preview = str(block.content)[:300]
                    print(f"[tool_result] {preview}", flush=True)


# ── Scripted scenarios ────────────────────────────────────────────────

SCENARIOS = {
    # Should pass `_needs_approval` → False (operation=ps is read-only).
    "pass": "List all my running HF jobs (use the hf_jobs tool with operation=ps).",
    # Should trigger `_needs_approval` → True (run + GPU). Expect denial.
    "deny": (
        "Submit a tiny GPU job using hf_jobs: operation='run', "
        "script='print(\"hi\")', hardware_flavor='t4-small', timeout='5m'. "
        "If denied, just report the denial."
    ),
    # Read-only job call that hits the real handler (needs HF_TOKEN env
    # to succeed — otherwise the handler errors, which is fine).
    "list": "Use hf_jobs operation=ps to list my jobs.",
}


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "deny"
    prompt = SCENARIOS.get(scenario, scenario)
    asyncio.run(run_prompt(prompt))
    print("\n--- callback log summary ---", flush=True)
    for i, entry in enumerate(CALLBACK_LOG):
        print(
            f"#{i} {entry['bare_name']} "
            f"needs_approval={entry['needs_approval']} "
            f"tool_use_id={entry['tool_use_id']}"
        )


if __name__ == "__main__":
    main()
