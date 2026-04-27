"""
Claude Agent SDK backend for ml-intern.

Replaces the litellm LLM call at the heart of `Handlers.run_agent` with
the Claude Agent SDK's `query()`. Reuses:
  * ml-intern tool handlers (via an in-process SDK MCP server)
  * `_needs_approval()` (via the SDK's `can_use_tool` callback)
  * `SDKEventAdapter` (streams SDK messages onto ml-intern's event queue)

Consolidates the patterns proven in Spikes 1–3. Not yet wired into the
CLI — `demo()` at the bottom runs the spike scenarios to smoke-test the
module end-to-end.

Usage sketch (for when we wire it into `agent/main.py`):

    backend = SDKBackend(
        tool_specs=tool_router.tools,
        event_queue=session.event_queue,
        config=session.config,
        session=session,
    )
    summary = await backend.run_turn("fine-tune SmolLM on /tmp/my-data")
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from typing import Any, Awaitable, Callable, Iterable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    SdkMcpTool,
    ToolPermissionContext,
    create_sdk_mcp_server,
    tool,
)

from agent.core.agent_loop import _needs_approval
from agent.core.sdk_event_adapter import SDKEventAdapter, _strip_mcp_prefix
from agent.core.session import Event

logger = logging.getLogger(__name__)


# Claude Code's built-in Bash/Read/Write/Edit are now allowed on the SDK
# path. They have better implementations than our MCP wrappers (Spike 1
# finding) and fix the orphan-process bug in `_bash_handler` that Spike 6
# flagged (synchronous subprocess.run can't be cancelled). The model
# prefers the builtins over MCP wrappers, so the MCP local_tools become
# unused on this path — we also skip registering them in `create_builtin_tools`
# when `use_sdk_builtins=True`. `_needs_approval()` doesn't match the
# builtin names, so they execute without the extra approval gate
# (acceptable for local file/shell use on the user's own machine).
DEFAULT_DISALLOWED_BUILTINS: list[str] = []


# ── Tool factory: ToolSpec → SdkMcpTool ────────────────────────────────


def _call_handler(
    handler: Callable[..., Awaitable[tuple[str, bool]]],
    args: dict[str, Any],
    session: Any | None,
    tool_call_id: str | None,
) -> Awaitable[tuple[str, bool]]:
    """Invoke a ToolSpec handler regardless of which of the three signature
    variants it uses. Mirrors the dispatch in ToolRouter.call_tool.
    """
    sig = inspect.signature(handler)
    kwargs: dict[str, Any] = {}
    if "session" in sig.parameters or _accepts_kwargs(sig):
        kwargs["session"] = session
    if "tool_call_id" in sig.parameters or _accepts_kwargs(sig):
        kwargs["tool_call_id"] = tool_call_id
    return handler(args, **kwargs)


def _accepts_kwargs(sig: inspect.Signature) -> bool:
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def make_sdk_tool_from_spec(
    spec: Any,  # ToolSpec (avoid circular import on Session-building order)
    session_provider: Callable[[], Any | None],
    tool_router_call: Callable[..., Awaitable[tuple[str, bool]]] | None = None,
) -> SdkMcpTool:
    """Wrap one ml-intern ToolSpec as an in-process SDK MCP tool.

    The SDK gives the handler only `args` at call time (no tool_use_id,
    no session). We close over a `session_provider` so the handler can
    still get at the live session object — and we forward `tool_call_id`
    when available via a contextvar-style stash populated by
    `can_use_tool` (see `SDKBackend._set_current_tool_call_id`).

    Specs loaded from external MCP servers (e.g. huggingface.co/mcp)
    arrive with `handler=None`; the litellm path routes those through
    `ToolRouter.call_tool` which falls through to `mcp_client.call_tool`.
    On the SDK path we do the same by threading a `tool_router_call`
    callable and dispatching through it when `spec.handler` is absent.
    """
    handler = spec.handler

    @tool(
        name=spec.name,
        description=spec.description,
        input_schema=spec.parameters,
    )
    async def _wrapped(args: dict[str, Any]) -> dict[str, Any]:
        # Strip empty fields so the handler sees the same shape it does
        # in the litellm path (LLM tool-use JSON only carries populated
        # fields, but the SDK fills every declared schema key).
        clean = {k: v for k, v in args.items() if v not in (None, "", [])}
        session = session_provider()
        tool_call_id = _current_tool_call_id.get(None) if session else None
        try:
            if handler is None:
                if tool_router_call is None:
                    raise RuntimeError(
                        f"Tool '{spec.name}' has no handler and no tool_router"
                        " was provided to SDKBackend to route external MCP calls."
                    )
                output, ok = await tool_router_call(
                    spec.name, clean, session=session, tool_call_id=tool_call_id
                )
            else:
                output, ok = await _call_handler(
                    handler, clean, session, tool_call_id
                )
        except Exception as e:  # noqa: BLE001 — convert to tool error
            logger.exception("Handler %s raised", spec.name)
            return {
                "content": [{"type": "text", "text": f"Tool error: {e}"}],
                "is_error": True,
            }
        return {
            "content": [{"type": "text", "text": output}],
            "is_error": not ok,
        }

    return _wrapped


# Used by the permission callback to stash the tool_use_id so the tool
# handler can retrieve it. Simpler than threading through the SDK.
import contextvars  # noqa: E402

_current_tool_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ml_intern_sdk_current_tool_call_id", default=None
)


# ── Permission callback ───────────────────────────────────────────────


def make_permission_callback(
    event_queue: asyncio.Queue,
    config: Any | None,
    deny_all_sensitive: bool = False,
    await_user_decision: (
        Callable[[str, dict[str, Any], str | None], Awaitable[bool]] | None
    ) = None,
) -> Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResultAllow | PermissionResultDeny],
]:
    """Build a `can_use_tool` callback that enforces ml-intern's rules.

    - Stashes tool_use_id in a contextvar so the tool handler can see it.
    - Delegates approval decisions to `_needs_approval(name, args, config)`.
    - Emits `approval_required` + `tool_state_change` events onto the
      shared queue.
    - If `await_user_decision` is given, awaits it to decide; otherwise
      either allows everything (default) or denies all sensitive calls
      (if `deny_all_sensitive=True`, useful for scripted demos/tests).
    """

    async def cb(
        tool_name: str,
        tool_args: dict[str, Any],
        ctx: ToolPermissionContext,
    ) -> PermissionResultAllow | PermissionResultDeny:
        bare = _strip_mcp_prefix(tool_name)

        # Make the tool_use_id visible to the tool handler.
        if ctx.tool_use_id:
            _current_tool_call_id.set(ctx.tool_use_id)

        needs = _needs_approval(bare, tool_args, config=config)
        if not needs:
            return PermissionResultAllow(behavior="allow")

        # Ask the user (or the scripted decider).
        await event_queue.put(Event(
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

        if await_user_decision is not None:
            approved = await await_user_decision(bare, tool_args, ctx.tool_use_id)
        else:
            approved = not deny_all_sensitive

        await event_queue.put(Event(
            event_type="tool_state_change",
            data={
                "tool_call_id": ctx.tool_use_id,
                "tool": bare,
                "state": "approved" if approved else "rejected",
            },
        ))

        if approved:
            return PermissionResultAllow(behavior="allow")
        return PermissionResultDeny(
            behavior="deny",
            message=(
                f"User denied approval for `{bare}` "
                f"(operation={tool_args.get('operation')})."
            ),
            interrupt=False,
        )

    return cb


# ── Top-level façade ──────────────────────────────────────────────────


class SDKBackend:
    """Drives the Claude Agent SDK against ml-intern's tools + events.

    Owns a long-lived `ClaudeSDKClient` so prompt cache, conversation
    history, and MCP connections persist across turns. Use as an async
    context manager:

        async with SDKBackend(...) as backend:
            await backend.run_turn("hello")
            await backend.run_turn("continue")
            usage = await backend.get_context_usage()

    A single-shot `run_turn()` call without `async with` works too —
    the backend lazy-connects and the caller must invoke `close()`.

    Does NOT replace the submission loop, session bookkeeping,
    context-manager persistence, or approval-batching — those stay in
    `agent/core/agent_loop.py`.
    """

    def __init__(
        self,
        tool_specs: Iterable[Any],
        event_queue: asyncio.Queue,
        config: Any | None = None,
        session: Any | None = None,
        system_prompt: str | None = None,
        max_turns: int = 50,
        deny_all_sensitive: bool = False,
        await_user_decision: (
            Callable[[str, dict[str, Any], str | None], Awaitable[bool]] | None
        ) = None,
        env: dict[str, str] | None = None,
        tool_router: Any | None = None,
        resume_session_id: str | None = None,
        fork_on_resume: bool = False,
    ):
        self.event_queue = event_queue
        self.config = config
        self.session = session
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.env = env or {}
        self.tool_router = tool_router
        self.adapter = SDKEventAdapter(event_queue)
        # When set, the backing ClaudeSDKClient resumes the named SDK
        # session id from ~/.claude/projects/<encoded-cwd>/<id>.jsonl.
        # `fork_on_resume=True` asks the SDK to mint a fresh session id
        # rooted on the resumed history (preserves the original JSONL).
        self._resume_session_id = resume_session_id
        self._fork_on_resume = fork_on_resume

        # Build in-process MCP server from the ml-intern tool specs.
        # Specs with handler=None (external MCP tools fetched by
        # ToolRouter from huggingface.co/mcp et al.) dispatch through
        # `tool_router.call_tool` so their external transport is used.
        tr_call = tool_router.call_tool if tool_router is not None else None
        sdk_tools = [
            make_sdk_tool_from_spec(
                spec,
                session_provider=lambda: self.session,
                tool_router_call=tr_call,
            )
            for spec in tool_specs
        ]
        self._server = create_sdk_mcp_server(
            name="ml-intern",
            version="0.0.1",
            tools=sdk_tools,
        )
        self._tool_names = [spec.name for spec in tool_specs]
        self._permission_cb = make_permission_callback(
            event_queue=event_queue,
            config=config,
            deny_all_sensitive=deny_all_sensitive,
            await_user_decision=await_user_decision,
        )

        self._client: ClaudeSDKClient | None = None

    def _build_options(self) -> ClaudeAgentOptions:
        kwargs: dict[str, Any] = dict(
            system_prompt=self.system_prompt,
            mcp_servers={"ml-intern": self._server},
            # NB: NOT setting `allowed_tools` — that would pre-approve
            # and bypass `can_use_tool` (Spike 2 gotcha).
            disallowed_tools=DEFAULT_DISALLOWED_BUILTINS,
            can_use_tool=self._permission_cb,
            include_partial_messages=True,
            max_turns=self.max_turns,
            env=self.env,
        )
        if self._resume_session_id:
            kwargs["resume"] = self._resume_session_id
            kwargs["fork_session"] = self._fork_on_resume
        return ClaudeAgentOptions(**kwargs)

    @property
    def sdk_session_id(self) -> str | None:
        """The active SDK session id (set after the first turn). For new
        sessions this is whatever the SDK minted; for forked resumes this
        is the new id under which the forked history is being written."""
        return self.adapter.sdk_session_id

    # ── Lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = ClaudeSDKClient(options=self._build_options())
        await self._client.connect()

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await self._client.disconnect()
        finally:
            self._client = None

    async def __aenter__(self) -> "SDKBackend":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    # ── Turn execution ──────────────────────────────────────────────

    async def run_turn(self, user_message: str) -> dict[str, Any]:
        """Send one user message and stream events onto the queue.

        Lazy-connects if the backend isn't inside `async with` yet.
        """
        await self.connect()
        assert self._client is not None
        await self._client.query(user_message)
        return await self.adapter.consume(self._client.receive_response())

    # ── Delegated capabilities (from ClaudeSDKClient) ───────────────

    async def interrupt(self) -> None:
        """Abort the current in-flight turn (if any)."""
        if self._client is None:
            return
        # Let the adapter know an `is_error=True` ResultMessage is
        # coming because of a user-initiated interrupt, not a real
        # failure — so it can remap to `interrupted`.
        self.adapter.mark_interrupted()
        await self._client.interrupt()

    async def get_context_usage(self) -> dict[str, Any]:
        """Current context-window usage breakdown. Returns `{}` if
        the backend isn't connected yet."""
        if self._client is None:
            return {}
        return await self._client.get_context_usage()

    async def set_model(self, model: str) -> None:
        if self._client is not None:
            await self._client.set_model(model)

    async def set_permission_mode(self, mode: str) -> None:
        if self._client is not None:
            await self._client.set_permission_mode(mode)


# ── Smoke-test demo ────────────────────────────────────────────────────
#
# Replaces the three previous spike drivers. Runs the scenarios against
# a minimal tool registration (bash + hf_jobs) without bringing up a
# real Session / ToolRouter / ContextManager.


def _build_demo_tool_specs():
    from dataclasses import dataclass

    from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
    from agent.tools.local_tools import _bash_handler

    @dataclass
    class ToolSpec:
        name: str
        description: str
        parameters: dict
        handler: Callable

    bash = ToolSpec(
        name="bash",
        description="Execute a shell command locally.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "work_dir": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
        handler=_bash_handler,
    )
    hf_jobs = ToolSpec(
        name=HF_JOBS_TOOL_SPEC["name"],
        description=HF_JOBS_TOOL_SPEC["description"],
        parameters=HF_JOBS_TOOL_SPEC["parameters"],
        handler=hf_jobs_handler,
    )
    return [bash, hf_jobs]


SCENARIOS: dict[str, tuple[str, bool]] = {
    # Ask for something the model can only know via a tool (i.e. the
    # running process id), so it doesn't answer from context and skip
    # the bash call. `bash` refers to our MCP tool, not the builtin.
    "bash": (
        "Call the `bash` MCP tool with command `echo $$` and report the "
        "exact shell PID it prints. Do not answer from memory.",
        False,
    ),
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


async def _drain_and_print(q: asyncio.Queue, done: asyncio.Event) -> None:
    import json
    while not (done.is_set() and q.empty()):
        try:
            event = await asyncio.wait_for(q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        print(
            f"[event] {event.event_type} "
            f"{json.dumps(event.data, default=str)[:220]}",
            flush=True,
        )


async def _run_demo(scenario: str) -> None:
    import json
    prompt, deny = SCENARIOS.get(scenario, (scenario, False))
    print(f"\n>>> prompt: {prompt}\n", flush=True)

    q: asyncio.Queue = asyncio.Queue()
    async with SDKBackend(
        tool_specs=_build_demo_tool_specs(),
        event_queue=q,
        config=None,
        session=None,
        system_prompt=(
            "You are a terse assistant. Prefer single tool calls. "
            "If a tool is denied, report it and do not retry."
        ),
        max_turns=6,
        deny_all_sensitive=deny,
    ) as backend:
        done = asyncio.Event()
        drain_task = asyncio.create_task(_drain_and_print(q, done))
        try:
            summary = await backend.run_turn(prompt)
        finally:
            done.set()
            await drain_task

        usage = await backend.get_context_usage()

    print("\n--- result summary ---", flush=True)
    print(json.dumps(summary, default=str, indent=2))
    print("\n--- context usage ---", flush=True)
    print(
        f"  totalTokens={usage.get('totalTokens'):,}   "
        f"percentage={usage.get('percentage'):.2f}%   "
        f"autoCompact={usage.get('isAutoCompactEnabled')} "
        f"@ {usage.get('autoCompactThreshold')}"
    )


def demo() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "mix"
    asyncio.run(_run_demo(scenario))


if __name__ == "__main__":
    demo()
