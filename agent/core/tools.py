"""
Tool system for the agent
Provides ToolSpec and ToolRouter for managing both built-in and MCP tools
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import EmbeddedResource, ImageContent, TextContent

from agent.config import MCPServerConfig
from agent.tools.dataset_tools import (
    HF_INSPECT_DATASET_TOOL_SPEC,
    hf_inspect_dataset_handler,
)
from agent.tools.docs_tools import (
    EXPLORE_HF_DOCS_TOOL_SPEC,
    HF_DOCS_FETCH_TOOL_SPEC,
    explore_hf_docs_handler,
    hf_docs_fetch_handler,
)
from agent.tools.github_find_examples import (
    GITHUB_FIND_EXAMPLES_TOOL_SPEC,
    github_find_examples_handler,
)
from agent.tools.github_list_repos import (
    GITHUB_LIST_REPOS_TOOL_SPEC,
    github_list_repos_handler,
)
from agent.tools.github_read_file import (
    GITHUB_READ_FILE_TOOL_SPEC,
    github_read_file_handler,
)
from agent.tools.hf_repo_files_tool import (
    HF_REPO_FILES_TOOL_SPEC,
    hf_repo_files_handler,
)
from agent.tools.hf_repo_git_tool import (
    HF_REPO_GIT_TOOL_SPEC,
    hf_repo_git_handler,
)
from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
from agent.tools.papers_tool import HF_PAPERS_TOOL_SPEC, hf_papers_handler
from agent.tools.plan_tool import PLAN_TOOL_SPEC, plan_tool_handler
from agent.tools.research_tool import RESEARCH_TOOL_SPEC, research_handler
from agent.tools.sandbox_tool import get_sandbox_tools

# NOTE: Private HF repo tool disabled - replaced by hf_repo_files and hf_repo_git
# from agent.tools.private_hf_repo_tools import (
#     PRIVATE_HF_REPO_TOOL_SPEC,
#     private_hf_repo_handler,
# )

# Suppress aiohttp deprecation warning
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="aiohttp.connector"
)

NOT_ALLOWED_TOOL_NAMES = ["hf_jobs", "hf_doc_search", "hf_doc_fetch", "hf_whoami"]


def convert_mcp_content_to_string(content: list) -> str:
    """
    Convert MCP content blocks to a string format compatible with LLM messages.

    Based on FastMCP documentation, content can be:
    - TextContent: has .text field
    - ImageContent: has .data and .mimeType fields
    - EmbeddedResource: has .resource field with .text or .blob

    Args:
        content: List of MCP content blocks

    Returns:
        String representation of the content suitable for LLM consumption
    """
    if not content:
        return ""

    parts = []
    for item in content:
        if isinstance(item, TextContent):
            # Extract text from TextContent blocks
            parts.append(item.text)
        elif isinstance(item, ImageContent):
            # TODO: Handle images
            # For images, include a description with MIME type
            parts.append(f"[Image: {item.mimeType}]")
        elif isinstance(item, EmbeddedResource):
            # TODO: Handle embedded resources
            # For embedded resources, try to extract text
            resource = item.resource
            if hasattr(resource, "text") and resource.text:
                parts.append(resource.text)
            elif hasattr(resource, "blob") and resource.blob:
                parts.append(
                    f"[Binary data: {resource.mimeType if hasattr(resource, 'mimeType') else 'unknown'}]"
                )
            else:
                parts.append(
                    f"[Resource: {resource.uri if hasattr(resource, 'uri') else 'unknown'}]"
                )
        else:
            # Fallback: try to convert to string
            parts.append(str(item))

    return "\n".join(parts)


@dataclass
class ToolSpec:
    """Tool specification for LLM"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Optional[Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]] = None


class ToolRouter:
    """
    Routes tool calls to appropriate handlers.
    Based on codex-rs/core/src/tools/router.rs
    """

    def __init__(
        self,
        mcp_servers: dict[str, MCPServerConfig],
        hf_token: str | None = None,
        local_mode: bool = False,
        enable_hf_infra: bool = True,
        use_sdk_builtins: bool = False,
    ):
        self.tools: dict[str, ToolSpec] = {}
        self.mcp_servers: dict[str, dict[str, Any]] = {}

        for tool in create_builtin_tools(
            local_mode=local_mode,
            enable_hf_infra=enable_hf_infra,
            use_sdk_builtins=use_sdk_builtins,
        ):
            self.register_tool(tool)

        self.mcp_client: Client | None = None
        if mcp_servers:
            mcp_servers_payload = {}
            for name, server in mcp_servers.items():
                data = server.model_dump()
                if hf_token:
                    data.setdefault("headers", {})["Authorization"] = f"Bearer {hf_token}"
                mcp_servers_payload[name] = data
            self.mcp_client = Client({"mcpServers": mcp_servers_payload})
        self._mcp_initialized = False

    def register_tool(self, tool: ToolSpec) -> None:
        self.tools[tool.name] = tool

    async def register_mcp_tools(self) -> None:
        tools = await self.mcp_client.list_tools()
        registered_names = []
        skipped_count = 0
        for tool in tools:
            if tool.name in NOT_ALLOWED_TOOL_NAMES:
                skipped_count += 1
                continue
            registered_names.append(tool.name)
            self.register_tool(
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.inputSchema,
                    handler=None,
                )
            )
        logger.info(
            f"Loaded {len(registered_names)} MCP tools: {', '.join(registered_names)} ({skipped_count} disabled)"
        )

    async def register_openapi_tool(self) -> None:
        """Register the OpenAPI search tool (requires async initialization)"""
        from agent.tools.docs_tools import (
            _get_api_search_tool_spec,
            search_openapi_handler,
        )

        try:
            openapi_spec = await _get_api_search_tool_spec()
            self.register_tool(
                ToolSpec(
                    name=openapi_spec["name"],
                    description=openapi_spec["description"],
                    parameters=openapi_spec["parameters"],
                    handler=search_openapi_handler,
                )
            )
            logger.info(f"Loaded OpenAPI search tool: {openapi_spec['name']}")
        except Exception as e:
            logger.warning("Failed to load OpenAPI search tool: %s", e)

    def get_tool_specs_for_llm(self) -> list[dict[str, Any]]:
        """Get tool specifications in OpenAI format"""
        specs = []
        for tool in self.tools.values():
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return specs

    async def __aenter__(self) -> "ToolRouter":
        if self.mcp_client is not None:
            try:
                await self.mcp_client.__aenter__()
                await self.mcp_client.initialize()
                await self.register_mcp_tools()
                self._mcp_initialized = True
            except Exception as e:
                logger.warning("MCP connection failed, continuing without MCP tools: %s", e)
                self.mcp_client = None

        await self.register_openapi_tool()

        total_tools = len(self.tools)
        logger.info(f"Agent ready with {total_tools} tools total")

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.mcp_client is not None:
            await self.mcp_client.__aexit__(exc_type, exc, tb)
            self._mcp_initialized = False

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session: Any = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, bool]:
        """
        Call a tool and return (output_string, success_bool).

        For MCP tools, converts the CallToolResult content blocks to a string.
        For built-in tools, calls their handler directly.
        """
        # Check if this is a built-in tool with a handler
        tool = self.tools.get(tool_name)
        if tool and tool.handler:
            import inspect

            # Check if handler accepts session argument
            sig = inspect.signature(tool.handler)
            if "session" in sig.parameters:
                # Check if handler also accepts tool_call_id parameter
                if "tool_call_id" in sig.parameters:
                    return await tool.handler(
                        arguments, session=session, tool_call_id=tool_call_id
                    )
                return await tool.handler(arguments, session=session)
            return await tool.handler(arguments)

        # Otherwise, use MCP client
        if self._mcp_initialized:
            try:
                result = await self.mcp_client.call_tool(tool_name, arguments)
                output = convert_mcp_content_to_string(result.content)
                return output, not result.is_error
            except ToolError as e:
                # Catch MCP tool errors and return them to the agent
                error_msg = f"Tool error: {str(e)}"
                return error_msg, False

        return "MCP client not initialized", False


# ============================================================================
# BUILT-IN TOOL HANDLERS
# ============================================================================


# Tools that use HF remote infrastructure (compute or writes to the Hub).
# Filtered out when `enable_hf_infra=False` (the local-first default for the
# SDK backend). `hf_repo_files` is NOT in this set: its read-only ops
# (list/read) are load-bearing for research, so we replace it with a
# restricted variant instead of dropping it — see `_readonly_hf_repo_files`.
HF_INFRA_TOOL_NAMES: set[str] = {
    HF_JOBS_TOOL_SPEC["name"],
    HF_REPO_GIT_TOOL_SPEC["name"],
}


_READONLY_HF_REPO_FILES_OPS: frozenset[str] = frozenset({"list", "read"})


async def _readonly_hf_repo_files_handler(
    arguments: dict[str, Any], session: Any = None
) -> tuple[str, bool]:
    """Wrap hf_repo_files_handler so only read ops (list/read) reach the
    real handler. Belt-and-suspenders: the JSON-schema enum already
    narrows the LLM's view, but an unvalidated payload shouldn't get to
    upload/delete code paths either.
    """
    op = arguments.get("operation")
    if op not in _READONLY_HF_REPO_FILES_OPS:
        return (
            f"Operation '{op}' is disabled in read-only mode. "
            "Only 'list' and 'read' are available. Restart the MCP server with "
            "ML_INTERN_MCP_HF_INFRA=1 (or the agent CLI with --enable-hf-infra) "
            "to enable write operations.",
            False,
        )
    return await hf_repo_files_handler(arguments, session=session)


def _readonly_hf_repo_files() -> ToolSpec:
    """Read-only variant of hf_repo_files: list + read, no upload/delete."""
    full = HF_REPO_FILES_TOOL_SPEC
    full_props: dict[str, Any] = full["parameters"]["properties"]
    readonly_props = {
        k: v for k, v in full_props.items()
        if k not in {"content", "patterns", "create_pr", "commit_message"}
    }
    readonly_props["operation"] = {
        **full_props["operation"],
        "enum": ["list", "read"],
        "description": "Operation: list or read (write ops disabled without --enable-hf-infra)",
    }
    return ToolSpec(
        name=full["name"],
        description=(
            "Read files in HF repos (models/datasets/spaces).\n\n"
            "## Operations\n"
            "- **list**: List files with sizes and structure\n"
            "- **read**: Read file content (text files only)\n\n"
            "## Use when\n"
            "- Need to see what files exist in a repo (before downloading)\n"
            "- Want to read config.json, README.md, tokenizer_config.json, dataset cards, etc.\n\n"
            "## Notes\n"
            "- For binary files (safetensors, bin), `list` shows them but `read` won't work\n"
            "- Upload/delete are disabled in read-only mode. To enable them, restart with"
            " ML_INTERN_MCP_HF_INFRA=1 (Claude Code) or --enable-hf-infra (agent CLI).\n"
        ),
        parameters={
            "type": "object",
            "properties": readonly_props,
            "required": full["parameters"]["required"],
        },
        handler=_readonly_hf_repo_files_handler,
    )


def create_builtin_tools(
    local_mode: bool = False,
    enable_hf_infra: bool = True,
    use_sdk_builtins: bool = False,
) -> list[ToolSpec]:
    """Create built-in tool specifications.

    Args:
        local_mode: run bash/read/write/edit locally (True) vs. on an HF
            Space sandbox (False). Unchanged from before.
        enable_hf_infra: when False, drop tools that submit HF Jobs, create
            sandboxes, or write to the Hub. Read-only HF tools
            (hf_inspect_dataset, hf_papers, explore_hf_docs, …) are kept.
        use_sdk_builtins: when True, skip registering the MCP local bash /
            read / write / edit wrappers — Claude Code's builtin tools
            will handle those on the SDK backend path.
    """
    # in order of importance
    tools = [
        # Research sub-agent (delegates to read-only tools in independent context)
        ToolSpec(
            name=RESEARCH_TOOL_SPEC["name"],
            description=RESEARCH_TOOL_SPEC["description"],
            parameters=RESEARCH_TOOL_SPEC["parameters"],
            handler=research_handler,
        ),
        # Documentation search tools
        ToolSpec(
            name=EXPLORE_HF_DOCS_TOOL_SPEC["name"],
            description=EXPLORE_HF_DOCS_TOOL_SPEC["description"],
            parameters=EXPLORE_HF_DOCS_TOOL_SPEC["parameters"],
            handler=explore_hf_docs_handler,
        ),
        ToolSpec(
            name=HF_DOCS_FETCH_TOOL_SPEC["name"],
            description=HF_DOCS_FETCH_TOOL_SPEC["description"],
            parameters=HF_DOCS_FETCH_TOOL_SPEC["parameters"],
            handler=hf_docs_fetch_handler,
        ),
        # Paper discovery and reading
        ToolSpec(
            name=HF_PAPERS_TOOL_SPEC["name"],
            description=HF_PAPERS_TOOL_SPEC["description"],
            parameters=HF_PAPERS_TOOL_SPEC["parameters"],
            handler=hf_papers_handler,
        ),
        # Dataset inspection tool (unified)
        ToolSpec(
            name=HF_INSPECT_DATASET_TOOL_SPEC["name"],
            description=HF_INSPECT_DATASET_TOOL_SPEC["description"],
            parameters=HF_INSPECT_DATASET_TOOL_SPEC["parameters"],
            handler=hf_inspect_dataset_handler,
        ),
        # Planning and job management tools
        ToolSpec(
            name=PLAN_TOOL_SPEC["name"],
            description=PLAN_TOOL_SPEC["description"],
            parameters=PLAN_TOOL_SPEC["parameters"],
            handler=plan_tool_handler,
        ),
        ToolSpec(
            name=HF_JOBS_TOOL_SPEC["name"],
            description=HF_JOBS_TOOL_SPEC["description"],
            parameters=HF_JOBS_TOOL_SPEC["parameters"],
            handler=hf_jobs_handler,
        ),
        # HF Repo management tools
        ToolSpec(
            name=HF_REPO_FILES_TOOL_SPEC["name"],
            description=HF_REPO_FILES_TOOL_SPEC["description"],
            parameters=HF_REPO_FILES_TOOL_SPEC["parameters"],
            handler=hf_repo_files_handler,
        ),
        ToolSpec(
            name=HF_REPO_GIT_TOOL_SPEC["name"],
            description=HF_REPO_GIT_TOOL_SPEC["description"],
            parameters=HF_REPO_GIT_TOOL_SPEC["parameters"],
            handler=hf_repo_git_handler,
        ),
        ToolSpec(
            name=GITHUB_FIND_EXAMPLES_TOOL_SPEC["name"],
            description=GITHUB_FIND_EXAMPLES_TOOL_SPEC["description"],
            parameters=GITHUB_FIND_EXAMPLES_TOOL_SPEC["parameters"],
            handler=github_find_examples_handler,
        ),
        ToolSpec(
            name=GITHUB_LIST_REPOS_TOOL_SPEC["name"],
            description=GITHUB_LIST_REPOS_TOOL_SPEC["description"],
            parameters=GITHUB_LIST_REPOS_TOOL_SPEC["parameters"],
            handler=github_list_repos_handler,
        ),
        ToolSpec(
            name=GITHUB_READ_FILE_TOOL_SPEC["name"],
            description=GITHUB_READ_FILE_TOOL_SPEC["description"],
            parameters=GITHUB_READ_FILE_TOOL_SPEC["parameters"],
            handler=github_read_file_handler,
        ),
    ]

    # Drop HF remote-infra tools when disabled. Runs before the sandbox/local
    # choice so we also skip the sandbox branch (remote compute) below.
    # `hf_repo_files` is a special case — keep a read-only variant so research
    # can still peek at config.json / README.md / dataset cards.
    if not enable_hf_infra:
        tools = [t for t in tools if t.name not in HF_INFRA_TOOL_NAMES]
        tools = [
            _readonly_hf_repo_files() if t.name == HF_REPO_FILES_TOOL_SPEC["name"] else t
            for t in tools
        ]

    # The `research` tool spawns a subagent that calls `litellm.acompletion`
    # directly — it needs ANTHROPIC_API_KEY and bypasses SDKBackend. On the
    # SDK backend it silently hangs at 0 tokens because no API key is set.
    # Claude Code's builtin `Task` tool (inherits the main session's
    # `claude login` auth) is a drop-in replacement, so drop ours on the
    # SDK path.
    if use_sdk_builtins:
        tools = [t for t in tools if t.name != RESEARCH_TOOL_SPEC["name"]]

    # Sandbox or local tools (highest priority)
    if local_mode and not use_sdk_builtins:
        from agent.tools.local_tools import get_local_tools
        tools = get_local_tools() + tools
    elif not local_mode and enable_hf_infra:
        tools = get_sandbox_tools() + tools
    # else: SDK backend is relying on Claude Code's builtin Bash/Read/Write/Edit
    # (use_sdk_builtins=True) or the caller wants a tools-only surface with no
    # local/remote shell.

    tool_names = ", ".join([t.name for t in tools])
    logger.info(f"Loaded {len(tools)} built-in tools: {tool_names}")

    return tools
