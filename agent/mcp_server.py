"""Stdio MCP server that exposes ml-intern's tools to Claude Code.

Reuses `create_builtin_tools` and the existing handlers — no tool logic
is duplicated. Bash / Read / Write / Edit are deliberately NOT exposed:
Claude Code already has built-ins for those, and the SDK backend
(`use_sdk_builtins=True`) has the same stance.

Run directly:
    uv run python -m agent.mcp_server
    uv run ml-intern-mcp                 # once the console script is installed

Register with Claude Code:
    claude mcp add ml-intern -- uv --directory /path/to/ml-intern run ml-intern-mcp
or add to `.mcp.json` in the project:
    {
      "mcpServers": {
        "ml-intern": {
          "command": "uv",
          "args": ["--directory", "/path/to/ml-intern", "run", "ml-intern-mcp"]
        }
      }
    }

Environment:
    ML_INTERN_MCP_HF_INFRA=1   expose hf_jobs / hf_repo_files / hf_repo_git
                               (off by default — these write to the Hub
                               or submit paid compute)
    HF_TOKEN=...               required for HF Hub access
    GITHUB_TOKEN=...           lifts GitHub API rate limits
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
from typing import Any

import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from agent.core.tools import ToolSpec, create_builtin_tools

logger = logging.getLogger("ml_intern.mcp_server")


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


async def _invoke(spec: ToolSpec, arguments: dict[str, Any]) -> tuple[str, bool]:
    """Call a ToolSpec handler, tolerating the three signature variants
    used by ml-intern tools (args-only, args+session, args+session+tool_call_id)."""
    handler = spec.handler
    if handler is None:
        return (
            f"Tool '{spec.name}' has no local handler. External MCP tools "
            "should be registered with Claude Code directly.",
            False,
        )
    sig = inspect.signature(handler)
    kwargs: dict[str, Any] = {}
    if "session" in sig.parameters:
        # No ml-intern session exists under Claude Code; handlers that hard-
        # depend on session state will fail loudly. That's fine — the stateless
        # ones (docs, papers, datasets, github, plan, hf_jobs status queries)
        # cover most use cases.
        kwargs["session"] = None
    if "tool_call_id" in sig.parameters:
        kwargs["tool_call_id"] = None
    clean = {k: v for k, v in arguments.items() if v not in (None, "", [])}
    return await handler(clean, **kwargs)


async def _load_openapi_spec() -> ToolSpec | None:
    """Fetch the HF OpenAPI spec and return it as a ToolSpec (`find_hf_api`).

    Parallels ToolRouter.register_openapi_tool. Returns None if the fetch
    fails — `find_hf_api` is a Hub-API escape hatch, not load-bearing.
    """
    from agent.tools.docs_tools import (
        _get_api_search_tool_spec,
        search_openapi_handler,
    )
    try:
        spec = await _get_api_search_tool_spec()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping find_hf_api (OpenAPI fetch failed): %s", exc)
        return None
    return ToolSpec(
        name=spec["name"],
        description=spec["description"],
        parameters=spec["parameters"],
        handler=search_openapi_handler,
    )


def build_server(extra_specs: list[ToolSpec] | None = None) -> tuple[Server, dict[str, ToolSpec]]:
    enable_hf_infra = _env_flag("ML_INTERN_MCP_HF_INFRA", default=False)

    specs = create_builtin_tools(
        local_mode=False,
        enable_hf_infra=enable_hf_infra,
        use_sdk_builtins=True,   # drops local bash/read/write/edit + research
    )
    # Drop the HF Space sandbox tools. They need a running sandbox session
    # (created by `sandbox_create`), which doesn't exist under Claude Code,
    # and their names — `bash`, `read`, `write`, `edit` — would collide with
    # Claude Code's own builtins.
    from agent.tools.sandbox_tool import get_sandbox_tools
    sandbox_names = {t.name for t in get_sandbox_tools()}
    specs = [s for s in specs if s.name not in sandbox_names]

    for extra in extra_specs or ():
        specs.append(extra)

    registry: dict[str, ToolSpec] = {s.name: s for s in specs}
    logger.info("Exposing %d tools: %s", len(registry), ", ".join(registry))

    server: Server = Server("ml-intern")

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.parameters,
            )
            for spec in registry.values()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        spec = registry.get(name)
        if spec is None:
            return [mcp_types.TextContent(type="text", text=f"Unknown tool: {name}")]
        try:
            output, ok = await _invoke(spec, arguments or {})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s raised", name)
            return [mcp_types.TextContent(type="text", text=f"Tool error: {exc}")]
        text = output if ok else f"Tool reported failure:\n{output}"
        return [mcp_types.TextContent(type="text", text=text)]

    return server, registry


async def _serve() -> None:
    logging.basicConfig(
        level=os.environ.get("ML_INTERN_MCP_LOG", "INFO").upper(),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    openapi_spec = await _load_openapi_spec()
    server, _ = build_server(extra_specs=[openapi_spec] if openapi_spec else [])
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def cli() -> None:
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
