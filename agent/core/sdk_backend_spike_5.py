"""
Spike 5 — local training smoke test.

End-to-end test of the user's actual goal: fine-tune a tiny model on a
local dataset using local compute, driven through `SDKBackend`. We want
to confirm:

  * The agent picks the LOCAL `bash` tool — NOT `hf_jobs` — for training.
  * It loads data from a LOCAL path, not the HF Hub.
  * It writes a checkpoint to LOCAL disk.
  * It does NOT push to the HF Hub.

Prerequisites (set up ahead of the run, documented here):
  /tmp/spike5-venv/          — venv with torch/transformers/trl/datasets
  /tmp/spike5-dataset/train.jsonl  — 12 rows of Q/A text
  /tmp/spike5-output/        — expected checkpoint target

Run:
    uv run python -m agent.core.sdk_backend_spike_5

Captures:
  * Full event trace
  * Tool name histogram (did any hf_jobs calls happen?)
  * Final filesystem state of /tmp/spike5-output/
  * Cost, duration, context usage
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from agent.core.sdk_backend import SDKBackend, _build_demo_tool_specs


PROMPT = """\
Fine-tune a tiny causal language model on a local dataset.

Constraints — you MUST follow these:
  * Use the `bash` tool for everything. DO NOT use `hf_jobs` — we're
    running locally, not submitting to HF.
  * The Python environment is at `/tmp/spike5-venv/bin/python`. Use
    it via its absolute path. DO NOT run `pip install` — the venv
    already has torch, transformers, trl, datasets, accelerate.
  * The dataset is a local JSONL file at
    `/tmp/spike5-dataset/train.jsonl` (12 short Q/A rows, column
    name: `text`). Load it with
    `datasets.load_dataset("json", data_files=...)`.
  * Write all outputs under `/tmp/spike5-output/`.
  * Use the `HuggingFaceTB/SmolLM2-135M` model.
  * Train on CPU only (`device_map="cpu"`, `fp16=False`, `bf16=False`).
    CUDA is misconfigured on this machine; GPU training would crash.
  * Do NOT `push_to_hub` — this is a local experiment only.
  * Keep it small: max_steps=5, per_device_train_batch_size=1,
    max_length=128.

Succeed when you can show me the contents of /tmp/spike5-output/ after
training, including at least one file named like config.json or
pytorch_model.bin or *.safetensors.
"""


def _snapshot_output_dir() -> dict:
    """Record what /tmp/spike5-output looks like — before and after."""
    p = Path("/tmp/spike5-output")
    if not p.exists():
        return {"exists": False, "files": []}
    files = []
    for f in p.rglob("*"):
        if f.is_file():
            files.append({
                "path": str(f.relative_to(p)),
                "size": f.stat().st_size,
            })
    return {"exists": True, "files": files, "n_files": len(files)}


async def _drain(q: asyncio.Queue, done: asyncio.Event, tool_counter: Counter):
    captured = []
    while not (done.is_set() and q.empty()):
        try:
            event = await asyncio.wait_for(q.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        captured.append(event)
        etype = event.event_type
        data = event.data or {}
        if etype == "tool_call":
            tool = data.get("tool", "")
            tool_counter[tool] += 1
            args = data.get("arguments", {})
            cmd = args.get("command", "") if isinstance(args, dict) else ""
            preview = str(cmd)[:100] if cmd else json.dumps(args, default=str)[:100]
            print(f"[tool_call #{tool_counter[tool]:02d}] {tool:16s}  {preview}",
                  flush=True)
        elif etype == "approval_required":
            print(f"[APPROVAL_REQ]  {json.dumps(data, default=str)[:200]}",
                  flush=True)
        elif etype == "tool_output":
            output = str(data.get("output", ""))[:200].replace("\n", "⏎")
            print(f"[tool_output]     {data.get('tool'):16s}  success={data.get('success')}  {output}",
                  flush=True)
        elif etype in {"turn_complete", "ready", "error", "compacted"}:
            print(f"[{etype}]  {json.dumps(data, default=str)[:250]}",
                  flush=True)
    return captured


async def _run():
    print("=" * 70)
    print("SPIKE 5 — LOCAL TRAINING SMOKE TEST")
    print("=" * 70)
    print(f"venv:    /tmp/spike5-venv")
    print(f"dataset: /tmp/spike5-dataset/train.jsonl")
    print(f"output:  /tmp/spike5-output")
    print()
    print("Pre-run output dir:", _snapshot_output_dir())
    print()

    tool_counter: Counter = Counter()
    q: asyncio.Queue = asyncio.Queue()

    async with SDKBackend(
        tool_specs=_build_demo_tool_specs(),  # bash + hf_jobs
        event_queue=q,
        config=None,
        session=None,
        system_prompt=(
            "You are a careful ML engineer. Run commands locally via "
            "`bash`. Never submit `hf_jobs`. Never push to the Hugging "
            "Face Hub. Use absolute paths. If a command fails, read the "
            "error carefully and fix it."
        ),
        max_turns=80,
        deny_all_sensitive=True,   # defensively deny if agent tries hf_jobs anyway
    ) as backend:
        done = asyncio.Event()
        drain_task = asyncio.create_task(_drain(q, done, tool_counter))
        t0 = time.monotonic()
        try:
            summary = await backend.run_turn(PROMPT)
        finally:
            done.set()
            events = await drain_task
        t1 = time.monotonic()
        usage = await backend.get_context_usage()

    # ── report ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SPIKE 5 SUMMARY")
    print("=" * 70)
    print(f"wall time:    {t1 - t0:.1f}s")
    print(f"num_turns:    {summary.get('num_turns')}")
    print(f"stop_reason:  {summary.get('stop_reason')}")
    print(f"total_cost:   ${summary.get('total_cost_usd'):.4f}")
    print(f"is_error:     {summary.get('is_error')}")
    print()
    print("tool call histogram:")
    for tool, count in tool_counter.most_common():
        marker = " ❌ FORBIDDEN" if tool == "hf_jobs" else ""
        print(f"  {tool:16s}  {count:3d}{marker}")
    print()
    print(f"context_usage.totalTokens:   {usage.get('totalTokens'):,}")
    print(f"context_usage.percentage:    {usage.get('percentage'):.2f}%")
    print()
    print("Post-run output dir:")
    snap = _snapshot_output_dir()
    if snap["exists"]:
        for f in snap["files"][:40]:
            print(f"  {f['path']:50s}  {f['size']:>10,} bytes")
        if len(snap["files"]) > 40:
            print(f"  ... ({len(snap['files']) - 40} more)")
        print(f"  TOTAL: {snap['n_files']} files")
    else:
        print("  (does not exist)")
    print()

    # ── verdict ────────────────────────────────────────────────────
    verdict = {
        "used_bash":       tool_counter.get("bash", 0) > 0,
        "avoided_hf_jobs": tool_counter.get("hf_jobs", 0) == 0,
        "wrote_artifacts": snap["exists"] and snap["n_files"] > 0,
        "no_error":        not summary.get("is_error"),
    }
    print("verdict:")
    for k, v in verdict.items():
        print(f"  {k:18s}  {'PASS' if v else 'FAIL'}")
    print()
    print(
        "OVERALL:",
        "✅ pass" if all(verdict.values()) else "❌ fail",
    )


if __name__ == "__main__":
    asyncio.run(_run())
