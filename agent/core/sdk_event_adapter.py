"""
Adapter: Claude Agent SDK message stream → ml-intern Event queue.

Consumes the async iterator returned by `claude_agent_sdk.query(...)` and
pushes `agent.core.session.Event` objects onto the same asyncio.Queue the
CLI and web UI already subscribe to. Goal is drop-in parity with the
litellm-path event surface (`agent/core/agent_loop.py`).

Mapping (summary):
  SystemMessage(init)        → ready
  StreamEvent(text_delta)    → assistant_chunk
  AssistantMessage[TextBlock] → assistant_message + assistant_stream_end
  AssistantMessage[ToolUseBlock] → tool_call
  UserMessage[ToolResultBlock]   → tool_output
  ResultMessage              → turn_complete (+ error if is_error)

The permission callback (`can_use_tool`) handles `approval_required`
separately — it is NOT driven from the message stream.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from agent.core.session import Event


class SDKEventAdapter:
    """Streams SDK messages into an ml-intern-style asyncio event queue."""

    def __init__(self, event_queue: asyncio.Queue, own_mcp_server: str = "ml-intern"):
        self.q = event_queue
        # Name of the in-process MCP server we register our own tools under.
        # Used to scope the `ready` tool count to ml-intern-own tools,
        # separately from Claude Code builtins and user's other MCP plugins.
        self._own_mcp_server = own_mcp_server
        # Tool name lookup so tool_output events can include the tool name
        # (ml-intern emits `{"tool": name, "tool_call_id": id, ...}` but the
        # SDK's ToolResultBlock only carries `tool_use_id`, not the name).
        self._tool_names: dict[str, str] = {}
        # IDs of tool uses whose tool_output should be dropped (e.g. the
        # SDK's internal ToolSearch hop — see `_handle_assistant`).
        self._suppressed_tool_ids: set[str] = set()
        # Buffer partial text between AssistantMessage boundaries so we
        # emit exactly one `assistant_message` per completed text block.
        self._streaming_open = False
        # With `include_partial_messages=True` the SDK can emit the same
        # AssistantMessage multiple times during streaming. Dedupe on
        # message_id/uuid so we don't fire duplicate tool_call and
        # assistant_message events.
        self._seen_assistant_uuids: set[str] = set()
        self._seen_user_uuids: set[str] = set()
        # Set by SDKBackend.interrupt() before the user-initiated abort
        # reaches the stream. Lets `_handle_result` remap the resulting
        # is_error ResultMessage to `interrupted` instead of `error`
        # (Spike 6 finding: post-interrupt stop_reason=None, is_error=True).
        self._interrupt_pending = False
        # Captured from the first `init` SystemMessage. Resume needs this id
        # to look up the right JSONL under ~/.claude/projects/<cwd>/.
        self.sdk_session_id: str | None = None

    def mark_interrupted(self) -> None:
        """Signal the adapter that a user-initiated interrupt just fired.
        The next `is_error=True` ResultMessage will surface as an
        `interrupted` event instead of `error`."""
        self._interrupt_pending = True

    async def _emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        await self.q.put(Event(event_type=event_type, data=data))

    async def consume(self, messages: AsyncIterator) -> dict[str, Any]:
        """Consume SDK messages until the stream ends. Returns the final
        result summary (from ResultMessage) for callers that want it."""
        final: dict[str, Any] = {}

        async for msg in messages:
            if isinstance(msg, SystemMessage):
                await self._handle_system(msg)
            elif isinstance(msg, StreamEvent):
                await self._handle_stream_event(msg)
            elif isinstance(msg, AssistantMessage):
                await self._handle_assistant(msg)
            elif isinstance(msg, UserMessage):
                await self._handle_user(msg)
            elif isinstance(msg, ResultMessage):
                final = await self._handle_result(msg)

        return final

    # ── per-type handlers ───────────────────────────────────────────

    async def _handle_system(self, msg: SystemMessage) -> None:
        if msg.subtype == "init":
            sid = msg.data.get("session_id") if isinstance(msg.data, dict) else None
            if isinstance(sid, str) and sid:
                self.sdk_session_id = sid
            # SDK reports ALL tools available: Claude Code builtins
            # (Bash/Read/Edit/TodoWrite/WebFetch/…), any MCP plugins the
            # user has installed (Asana, Gmail, Notion, Sentry, …), AND
            # our ml-intern MCP server. Count only our own so the `ready`
            # event matches the user's mental model.
            tools = msg.data.get("tools") or []
            own = [
                t for t in tools
                if t.startswith(f"mcp__{self._own_mcp_server}__")
            ]
            other_mcp = [
                t for t in tools
                if t.startswith("mcp__") and t not in own
            ]
            builtin = [t for t in tools if not t.startswith("mcp__")]
            await self._emit(
                "ready",
                {
                    "message": "Agent initialized",
                    "tool_count": len(own),
                    "builtin_tool_count": len(builtin),
                    "other_mcp_tool_count": len(other_mcp),
                },
            )
        elif msg.subtype == "compact_boundary":
            # The SDK fires this when it compacts. Mirror ml-intern's shape.
            data = msg.data or {}
            await self._emit(
                "compacted",
                {
                    "old_tokens": data.get("pre_compact_tokens"),
                    "new_tokens": data.get("post_compact_tokens"),
                },
            )

    async def _handle_stream_event(self, msg: StreamEvent) -> None:
        # `event` is an Anthropic streaming event dict. Text deltas arrive
        # as {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
        ev = msg.event or {}
        ev_type = ev.get("type")
        if ev_type == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                if text:
                    if not self._streaming_open:
                        self._streaming_open = True
                    await self._emit("assistant_chunk", {"content": text})

    async def _handle_assistant(self, msg: AssistantMessage) -> None:
        # Dedupe: with include_partial_messages the SDK can emit the
        # same AssistantMessage multiple times. Fire events only on
        # the first sighting (uuid is stable; message_id is a fallback).
        key = msg.uuid or msg.message_id
        if key:
            if key in self._seen_assistant_uuids:
                return
            self._seen_assistant_uuids.add(key)

        # End any open streaming run before emitting fully-formed blocks.
        if self._streaming_open:
            await self._emit("assistant_stream_end", {})
            self._streaming_open = False

        # AssistantMessage carries the *complete* set of content blocks for
        # this turn. Text → assistant_message. Tool uses → tool_call.
        text_parts: list[str] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingBlock):
                # ml-intern has no thinking event today; skip (could add).
                continue
            elif isinstance(block, ToolUseBlock):
                # Drop the SDK's ToolSearch discovery hop — it's an
                # internal MCP-resolution step, not a user-visible tool
                # call. (Spike 3 finding.)
                if block.name == "ToolSearch":
                    self._suppressed_tool_ids.add(block.id)
                    continue
                self._tool_names[block.id] = block.name
                await self._emit(
                    "tool_call",
                    {
                        "tool": _strip_mcp_prefix(block.name),
                        "arguments": block.input,
                        "tool_call_id": block.id,
                    },
                )

        if text_parts:
            await self._emit(
                "assistant_message",
                {"content": "".join(text_parts)},
            )

        if msg.error:
            await self._emit("error", {"error": f"assistant error: {msg.error}"})

    async def _handle_user(self, msg: UserMessage) -> None:
        # Dedupe same as AssistantMessage — tool_output can otherwise
        # double-fire on partial-message streams.
        uuid = getattr(msg, "uuid", None)
        if uuid:
            if uuid in self._seen_user_uuids:
                return
            self._seen_user_uuids.add(uuid)

        # The SDK echoes tool results as UserMessages with ToolResultBlocks.
        content = msg.content
        if isinstance(content, str):
            return
        for block in content or []:
            if isinstance(block, ToolResultBlock):
                if block.tool_use_id in self._suppressed_tool_ids:
                    self._suppressed_tool_ids.discard(block.tool_use_id)
                    continue
                tool_name = self._tool_names.get(block.tool_use_id, "unknown")
                output = _flatten_tool_result(block.content)
                await self._emit(
                    "tool_output",
                    {
                        "tool": _strip_mcp_prefix(tool_name),
                        "tool_call_id": block.tool_use_id,
                        "output": output,
                        "success": not bool(block.is_error),
                    },
                )

    async def _handle_result(self, msg: ResultMessage) -> dict[str, Any]:
        # Close any lingering stream.
        if self._streaming_open:
            await self._emit("assistant_stream_end", {})
            self._streaming_open = False

        summary = {
            "stop_reason": msg.stop_reason,
            "num_turns": msg.num_turns,
            "total_cost_usd": msg.total_cost_usd,
            "duration_ms": msg.duration_ms,
            "session_id": msg.session_id,
            "is_error": msg.is_error,
            "permission_denials": msg.permission_denials,
        }

        if msg.is_error:
            if self._interrupt_pending:
                # User-initiated interrupt: surface ml-intern's `interrupted`
                # event instead of a generic error. Clear the flag so the
                # next unrelated error still reports as `error`.
                self._interrupt_pending = False
                await self._emit("interrupted", None)
            else:
                await self._emit("error", {"error": msg.result or "agent error"})
        else:
            # ml-intern emits `history_size`; we don't track that here —
            # substitute turn count so the frontend has something numeric.
            await self._emit(
                "turn_complete",
                {"history_size": msg.num_turns, **summary},
            )

        return summary


# ── Helpers ────────────────────────────────────────────────────────────

def _strip_mcp_prefix(name: str) -> str:
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return name


def _flatten_tool_result(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)
