"""Discovery, fork, and wedge-recovery for SDK-backed sessions.

The Claude Agent SDK stores per-session JSONLs at
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` — shared across
every consumer of the bundled ``claude`` binary (Claude Code TUI, ml-intern,
any other SDK app). The JSONL has no embedded marker that says "this was
ml-intern" (system prompt is passed via CLI, not stored), so we can't
content-sniff. Identification is forward-only: every ml-intern run records
the SDK session id it was given (in ``<cwd>/session_logs/*.json`` under
the ``sdk_session_id`` key) and we cross-reference against that.

This module:

* lists JSONLs whose ``sdk_session_id`` appears in the local
  ``session_logs/`` index — those are ours;
* scans each JSONL for orphan ``tool_use`` blocks (assistant-emitted tool
  calls with no matching ``tool_result``) — those are the diagnostic of a
  wedged session;
* writes a *truncated copy* under a fresh session id when the user resumes
  a wedged session, so ``resume=<new_sid>`` lands in a clean state without
  mutating the original JSONL.

Pre-feature sessions cannot be resumed via the SDK because we never
captured their session ids. They remain on disk under ``session_logs/``
as a record of the conversation, but the SDK has no way to rebuild state
from our trajectory format.
"""

from __future__ import annotations

import json
import logging
import shutil
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """One resumable SDK session and the metadata we show the user."""

    sdk_session_id: str
    jsonl_path: Path
    cwd: str
    started_at: Optional[datetime]
    last_modified: datetime
    user_message_count: int
    first_user_prompt: str
    has_orphan_tool_use: bool
    orphan_cut_uuid: Optional[str] = None
    ml_intern_session_id: Optional[str] = None
    git_branch: Optional[str] = None


@dataclass
class _JsonlScan:
    """Per-line scan results we need for filtering and forking."""

    first_user_prompt: str = ""
    user_message_count: int = 0
    started_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    git_branch: Optional[str] = None
    cwd: Optional[str] = None
    orphan_cut_uuid: Optional[str] = None
    last_completed_user_uuid: Optional[str] = None
    tool_use_ids: list[str] = field(default_factory=list)
    tool_result_ids: set[str] = field(default_factory=set)


def encoded_cwd_dir(cwd: Path) -> Path:
    """Map a cwd to the SDK's per-project JSONL dir."""
    encoded = str(cwd.resolve()).replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded


def _iter_jsonl_lines(path: Path) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.debug("Skipping malformed line in %s: %s", path, e)


def _extract_text(content) -> str:
    """Pull plain text out of a record's ``message.content``.

    SDK content can be a string, a list of blocks, or absent. We only care
    about the user's typed text — system reminders and tool results are
    irrelevant for the preview.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _content_of(rec: dict) -> list:
    msg = rec.get("message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    return content if isinstance(content, list) else []


def scan_jsonl(path: Path) -> _JsonlScan:
    """Two-pass scan: collect orphan tool-use ids, then locate the cut.

    A *turn* spans from one text-bearing user message up to (but not
    including) the next one. If an orphan tool_use lies within a turn,
    that user message is wedged and the cut point is the user message
    immediately before it. If the very first turn is wedged, the file
    has no clean cut and we report no recoverable point.
    """
    records = list(_iter_jsonl_lines(path))

    scan = _JsonlScan(last_modified=datetime.fromtimestamp(path.stat().st_mtime))

    user_turn_starts: list[tuple[int, Optional[str]]] = []
    for i, rec in enumerate(records):
        rtype = rec.get("type")
        if scan.cwd is None and isinstance(rec.get("cwd"), str):
            scan.cwd = rec["cwd"]
        if scan.git_branch is None and isinstance(rec.get("gitBranch"), str):
            scan.git_branch = rec["gitBranch"]
        if scan.started_at is None and isinstance(rec.get("timestamp"), str):
            try:
                scan.started_at = datetime.fromisoformat(
                    rec["timestamp"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

        content = _content_of(rec)

        if rtype == "user":
            text = _extract_text(content)
            if text:
                if not scan.first_user_prompt:
                    scan.first_user_prompt = text
                scan.user_message_count += 1
                user_turn_starts.append((i, rec.get("uuid")))
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    if tid:
                        scan.tool_result_ids.add(tid)
        elif rtype == "assistant":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tid = block.get("id")
                    if tid:
                        scan.tool_use_ids.append(tid)

    orphan_ids = {t for t in scan.tool_use_ids if t not in scan.tool_result_ids}

    if not orphan_ids:
        scan.last_completed_user_uuid = user_turn_starts[-1][1] if user_turn_starts else None
        return scan

    # Find the first turn that contains an orphan tool_use; cut at the
    # user message before it.
    for k, (start_idx, _uuid) in enumerate(user_turn_starts):
        end_idx = (
            user_turn_starts[k + 1][0]
            if k + 1 < len(user_turn_starts)
            else len(records)
        )
        wedged = False
        for rec in records[start_idx + 1 : end_idx]:
            if rec.get("type") != "assistant":
                continue
            for block in _content_of(rec):
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("id") in orphan_ids
                ):
                    wedged = True
                    break
            if wedged:
                break
        if wedged:
            if k == 0:
                # Very first turn is wedged — no clean cut available.
                scan.orphan_cut_uuid = None
                scan.last_completed_user_uuid = None
            else:
                # Cut by *excluding* the wedged user message and everything
                # after, so the forked JSONL ends with the previous turn's
                # complete assistant response — ready for fresh user input.
                scan.orphan_cut_uuid = _uuid
                scan.last_completed_user_uuid = user_turn_starts[k - 1][1]
            return scan

    # All orphans lay outside any turn-window — shouldn't happen in
    # practice, but if it does, treat as recoverable up to last turn.
    scan.last_completed_user_uuid = user_turn_starts[-1][1] if user_turn_starts else None
    return scan


def _ml_intern_session_logs_index(cwd: Path) -> dict[str, str]:
    """Map sdk_session_id → ml-intern session_id from saved trajectories."""
    index: dict[str, str] = {}
    log_dir = cwd / "session_logs"
    if not log_dir.is_dir():
        return index
    for path in log_dir.glob("session_*.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sdk_sid = data.get("sdk_session_id")
        if isinstance(sdk_sid, str) and sdk_sid:
            index[sdk_sid] = data.get("session_id") or ""
    return index


def discover_sessions(cwd: Path) -> list[SessionInfo]:
    """Find resumable ml-intern sessions for ``cwd``.

    Cross-references the SDK's per-cwd JSONL directory with our local
    ``session_logs/`` index — only JSONLs whose ``sdk_session_id`` we
    recorded are considered ml-intern's.
    """
    sdk_dir = encoded_cwd_dir(cwd)
    if not sdk_dir.is_dir():
        return []

    canonical = _ml_intern_session_logs_index(cwd)
    if not canonical:
        return []

    sessions: list[SessionInfo] = []
    for jsonl in sdk_dir.glob("*.jsonl"):
        sid = jsonl.stem
        if sid not in canonical:
            continue
        scan = scan_jsonl(jsonl)
        if scan.user_message_count == 0:
            continue
        sessions.append(
            SessionInfo(
                sdk_session_id=sid,
                jsonl_path=jsonl,
                cwd=scan.cwd or str(cwd),
                started_at=scan.started_at,
                last_modified=scan.last_modified or datetime.fromtimestamp(jsonl.stat().st_mtime),
                user_message_count=scan.user_message_count,
                first_user_prompt=scan.first_user_prompt,
                has_orphan_tool_use=scan.orphan_cut_uuid is not None,
                orphan_cut_uuid=scan.orphan_cut_uuid,
                ml_intern_session_id=canonical.get(sid),
                git_branch=scan.git_branch,
            )
        )

    sessions.sort(key=lambda s: s.last_modified, reverse=True)
    return sessions


# ── User-facing listing ──────────────────────────────────────────────


def _format_preview(text: str, max_chars: int = 500, width: int = 100) -> str:
    """Produce a multi-line preview of the first user prompt.

    We deliberately keep this generous — the user said sessions felt
    over-truncated. ``max_chars`` is a hard cap so a 50 KB system prompt
    doesn't blow up the menu, but within that we wrap to terminal width
    and only ellipsize at the end if we trimmed.
    """
    snippet = text.strip()
    truncated = len(snippet) > max_chars
    if truncated:
        snippet = snippet[:max_chars].rstrip()
    wrapped = textwrap.fill(
        snippet,
        width=width,
        replace_whitespace=False,
        drop_whitespace=False,
    )
    if truncated:
        wrapped = wrapped + " …"
    return wrapped


def format_sessions_menu(sessions: list[SessionInfo]) -> str:
    """Render a numbered menu for interactive selection."""
    lines: list[str] = []
    width = shutil.get_terminal_size((100, 24)).columns
    preview_width = max(60, width - 6)
    for i, s in enumerate(sessions, start=1):
        when = s.started_at.astimezone().strftime("%Y-%m-%d %H:%M") if s.started_at else "?"
        flag_str = "  [WEDGED]" if s.has_orphan_tool_use else ""
        branch = f"  branch={s.git_branch}" if s.git_branch else ""
        header = (
            f"[{i}] {when}  {s.user_message_count} msgs"
            f"{branch}{flag_str}"
            f"\n    sid={s.sdk_session_id}"
        )
        preview = _format_preview(s.first_user_prompt, width=preview_width)
        preview_indented = textwrap.indent(preview, "    ")
        lines.append(f"{header}\n{preview_indented}")
    return "\n\n".join(lines)


def select_session_interactive(sessions: list[SessionInfo]) -> Optional[SessionInfo]:
    """Auto-pick if exactly one, else prompt for a number. Returns None on cancel."""
    if not sessions:
        return None
    if len(sessions) == 1:
        s = sessions[0]
        print(f"Resuming the only session in this directory:\n  {s.sdk_session_id}")
        return s

    print(f"Found {len(sessions)} resumable ml-intern sessions in this directory:\n")
    print(format_sessions_menu(sessions))
    print()
    while True:
        try:
            raw = input(f"Select [1-{len(sessions)}] (or 'q' to cancel): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.lower() in ("q", "quit", "exit", ""):
            return None
        try:
            idx = int(raw)
        except ValueError:
            print("Not a number, try again.")
            continue
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]
        print(f"Out of range, try 1-{len(sessions)}.")


# ── Forking ───────────────────────────────────────────────────────────


@dataclass
class ForkResult:
    """The session id to feed back into ``ClaudeAgentOptions.resume``."""

    resume_session_id: str
    used_fork_session: bool  # True for clean resume (preserved original)
    truncated_messages_dropped: int


def _fork_truncated(jsonl_path: Path, cut_uuid: str) -> ForkResult:
    """Write a copy of ``jsonl_path`` keeping every record *before* ``cut_uuid``.

    ``cut_uuid`` is the wedged user message's id; we stop before it so the
    forked file ends with the previous turn's complete assistant response.
    The copy lives at ``<dir>/<new_uuid>.jsonl`` with ``sessionId`` rewritten
    on every record. The original file is untouched.
    """
    new_sid = str(uuid.uuid4())
    out_path = jsonl_path.parent / f"{new_sid}.jsonl"

    dropped = 0
    seen_cut = False
    with open(jsonl_path, "r") as src, open(out_path, "w") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not seen_cut and rec.get("uuid") == cut_uuid:
                seen_cut = True
            if seen_cut:
                dropped += 1
                continue
            rec["sessionId"] = new_sid
            dst.write(json.dumps(rec) + "\n")

    if not seen_cut:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"cut uuid {cut_uuid!r} not found in {jsonl_path.name}; aborted truncation"
        )

    return ForkResult(
        resume_session_id=new_sid,
        used_fork_session=False,
        truncated_messages_dropped=dropped,
    )


def prepare_resume(session: SessionInfo) -> ForkResult:
    """Return the session id to pass to ``ClaudeAgentOptions.resume``.

    Clean tail → resume the original id and let the SDK fork via
    ``fork_session=True`` (caller wires that). Wedged tail → write a
    truncated copy first and resume *that* id; ``fork_session`` is a no-op
    on a session that's already a fresh copy.
    """
    if session.has_orphan_tool_use and session.orphan_cut_uuid:
        try:
            return _fork_truncated(session.jsonl_path, session.orphan_cut_uuid)
        except RuntimeError as e:
            logger.warning("Wedge truncation failed (%s); resuming as-is with fork_session=True", e)

    return ForkResult(
        resume_session_id=session.sdk_session_id,
        used_fork_session=True,
        truncated_messages_dropped=0,
    )
