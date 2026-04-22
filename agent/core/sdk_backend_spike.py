"""
Spike 1 — minimal Claude Agent SDK integration with one local tool.

Standalone. Not wired into the CLI. Proves:
  * `claude-agent-sdk` + `claude login` auth works (no ANTHROPIC_API_KEY).
  * An SDK MCP tool can wrap ml-intern's existing `_bash_handler`.
  * The agent invokes the tool and streams text back.

Run:
    uv run python -m agent.core.sdk_backend_spike
    uv run python -m agent.core.sdk_backend_spike "write a haiku about /tmp"
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

from agent.tools.local_tools import _bash_handler


# ── Wrap the existing ml-intern bash handler as an in-process MCP tool ──

@tool(
    name="bash",
    description=(
        "Execute a shell command on the local machine. "
        "Use for listing files, running scripts, invoking python, etc. "
        "Returns combined stdout+stderr."
    ),
    input_schema={
        "command": str,
        "work_dir": str,
        "timeout": int,
    },
)
async def bash_tool(args: dict[str, Any]) -> dict[str, Any]:
    # Handler expects 'command' (required) and optional 'work_dir' / 'timeout'.
    clean_args = {"command": args.get("command", "")}
    if args.get("work_dir"):
        clean_args["work_dir"] = args["work_dir"]
    if args.get("timeout"):
        clean_args["timeout"] = args["timeout"]

    output, ok = await _bash_handler(clean_args)
    return {
        "content": [{"type": "text", "text": output}],
        "is_error": not ok,
    }


# ── Run a single prompt and stream events ──────────────────────────────

async def run_prompt(prompt: str) -> None:
    server = create_sdk_mcp_server(
        name="ml-intern-local",
        version="0.0.1-spike",
        tools=[bash_tool],
    )

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a terse CLI assistant. Use the `mcp__local__bash` tool "
            "(not the builtin Bash) to answer shell-related questions. "
            "Prefer one tool call; summarize in one sentence."
        ),
        mcp_servers={"local": server},
        # The SDK auto-prefixes MCP tools as mcp__<server>__<tool>.
        allowed_tools=["mcp__local__bash"],
        # Disable the builtin Bash so Claude must use our MCP wrapper —
        # proves the MCP handler shim path end-to-end.
        disallowed_tools=["Bash", "Read", "Write", "Edit"],
        # Don't spawn a permission prompter — auto-allow our own tool.
        permission_mode="bypassPermissions",
        max_turns=6,
    )

    print(f"\n>>> prompt: {prompt}\n", flush=True)

    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, SystemMessage):
            print(f"[system] {msg.subtype}", flush=True)
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"[assistant] {block.text}", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(
                        f"[tool_use] {block.name} args={block.input}",
                        flush=True,
                    )
        elif isinstance(msg, ResultMessage):
            print(
                f"[result] stop={msg.stop_reason} "
                f"turns={msg.num_turns} "
                f"usd={getattr(msg, 'total_cost_usd', None)}",
                flush=True,
            )
        else:
            # UserMessage (tool_result echo), StreamEvent, etc.
            if hasattr(msg, "content"):
                for block in getattr(msg, "content", []) or []:
                    if isinstance(block, ToolResultBlock):
                        preview = str(block.content)[:200]
                        print(f"[tool_result] {preview}", flush=True)


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "list the first 5 files in /tmp"
    asyncio.run(run_prompt(prompt))


if __name__ == "__main__":
    main()
