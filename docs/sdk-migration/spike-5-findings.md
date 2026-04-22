# Spike 5 — Local training smoke test

**Branch:** `spike/sdk-hello`
**Artifact:** `agent/core/sdk_backend_spike_5.py`
**Status:** ✅ Passed on first try. The agent picked local bash over HF Jobs, loaded a local JSONL dataset, trained a real model, and produced artifacts on local disk.

## Goal — the actual end-user story

Can a user say *"fine-tune SmolLM-135M on my local dataset with 5 steps"* and have the agent do it locally, with no HF Jobs submission and no push to the Hub? This is the whole point of the migration.

## Setup

Pre-provisioned outside the agent so the spike measures agent behavior, not package-install time:

- **Dataset**: `/tmp/spike5-dataset/train.jsonl` — 12 short Q/A rows, `text` column
- **Venv**: `/tmp/spike5-venv/` — torch, transformers, trl, datasets, accelerate
- **Output dir**: `/tmp/spike5-output/` (empty at start)

The agent was given only two tools via the demo harness: `bash` (local) and `hf_jobs` (HF Jobs). The system prompt and user prompt both forbade `hf_jobs` and hub pushes. The spike driver additionally set `deny_all_sensitive=True` as a defensive backstop (would have denied any `hf_jobs run` the agent tried).

## Results

```
wall time:    59.2s
num_turns:    6
stop_reason:  end_turn
total_cost:   $0.3373   (notional, Max-plan)
is_error:     False

tool call histogram:
  bash       3
  hf_jobs    0    ✅

artifacts under /tmp/spike5-output/:
  model.safetensors      538,090,408 B
  tokenizer.json           3,522,871 B
  config.json                    790 B
  tokenizer_config.json          746 B
  training_args.bin            5,649 B
  generation_config.json         141 B

verdicts:
  used_bash           ✅
  avoided_hf_jobs     ✅
  wrote_artifacts     ✅
  no_error            ✅
```

## What the agent actually did

Three-step workflow, remarkably concise:

**Call 1 — reconnaissance.**
```bash
ls /tmp/spike5-dataset/ && head -3 /tmp/spike5-dataset/train.jsonl && wc -l /tmp/spike5-dataset/train.jsonl
```
Confirms the dataset file exists, inspects the schema (`text` column), counts rows. Good instinct — never trust the user's description of the data.

**Call 2 — write and run training script.**
```bash
cat > /tmp/spike5-train.py << 'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
...
# [the agent wrote the full script inline via heredoc]
PY
/tmp/spike5-venv/bin/python /tmp/spike5-train.py
```
The script:
- Disabled CUDA explicitly (honored the "CPU only" instruction).
- Loaded tokenizer and `HuggingFaceTB/SmolLM2-135M`.
- Used `datasets.load_dataset("json", data_files="/tmp/spike5-dataset/train.jsonl")` — local path, not hub.
- Ran `SFTTrainer` for `max_steps=5`, `per_device_train_batch_size=1`.
- Saved to `/tmp/spike5-output/` with `trainer.save_model()`. No `push_to_hub`.

**Call 3 — verify artifacts.**
```bash
ls -la /tmp/spike5-output/
```
Listed the six output files so the user (and the verdict script) can confirm the training actually produced weights.

## Concerns status

From `~/.claude/plans/composed-squishing-island.md`:

| # | Concern | Status |
|---|---|---|
| — | End-user goal (local GPU + local data) | ✅ Behavior proven. The agent picks local bash over HF Jobs. |
| 6 | Max-plan rate-limit ceiling | 🟡 Still mostly open — $0.34 for a 6-turn / 59s run isn't a stress test |

## Surprises

### 1. Zero `hf_jobs` attempts, not even a test call

The agent didn't speculatively try `hf_jobs` first and fall back on denial — it went straight to `bash`. The system prompt and user prompt together were sufficient; we didn't need the `deny_all_sensitive=True` backstop.

Implication for the full migration: the default system prompt should explicitly instruct the agent on local-vs-HF-Jobs preference. Don't rely on approval flow alone.

### 2. Training ran on CPU in under a minute — the CUDA issue didn't matter

We initially wanted GPU training. Driver 595.58.03 on this host has a compatibility gap: both `cu121` and `cu130` torch builds report "CUDA driver version is insufficient for CUDA runtime version" despite nvidia-smi showing 3× RTX 3090 available and `nvidia-smi` reporting the driver supports CUDA 13.2. This is unrelated to ml-intern — a real-world host-setup gotcha.

Workaround: agent was told explicitly "CPU only" in the prompt. Trained 5 steps on CPU in ~25 seconds of the 59s total. Real end users on working hosts will get GPU speeds for free — nothing in the agent's workflow cares which device.

**Implication for ml-intern docs**: a first-run health check (`python -c "import torch; assert torch.cuda.is_available()"`) would surface this class of issue before the agent spends cost trying to train.

### 3. The agent used a heredoc-style one-liner to write the training script

Rather than calling `bash` to `write(file)` separately and then `bash` to run, it wrote the whole script inline in one `bash` call with `cat > ... << 'PY' ... PY`. This is elegant but produces a large tool-output payload (config contents + training stdout mashed together).

Not a problem at this scale. Would become a concern for multi-GB logs on real training runs. When we port the doom-loop detector, we may want to nudge the agent toward `Write` + `Bash` separation for scripts above a size threshold — but right now we've disabled the SDK `Write` builtin (see `DEFAULT_DISALLOWED_BUILTINS` in `sdk_backend.py`). This is an argument to revisit that.

### 4. Output dir was 538MB from a 135M-param model with 5 training steps

`trainer.save_model()` saves the full model, not a delta. For toy runs this inflates disk usage fast. Not an ml-intern problem — it's standard HF behavior — but worth flagging: a user who does 10 exploratory runs at 500MB each has 5GB of disposable checkpoints on their disk.

ml-intern could suggest LoRA / PEFT for exploratory work by default. Skill candidate: `finetune-locally-peft`.

## Implications for the migration

1. **Default system prompt must say "prefer local bash; HF Jobs is opt-in".** Spike 5 confirms the agent follows explicit instructions about which tool to prefer. Without the instruction, it would probably default to HF Jobs based on the `hf_jobs` tool description which aggressively markets itself for training.

2. **Re-enable the SDK `Write` builtin** (or at least re-evaluate `DEFAULT_DISALLOWED_BUILTINS` in `sdk_backend.py`). For script-generation workflows, `Write` is cleaner than heredoc-via-bash. The reason we disabled it in Spike 1 was to keep ml-intern's approval authoritative — but for local read/write in a user's own directory, that concern is low.

3. **Add a first-run health check in `ml-intern`.** `torch.cuda.is_available()`, `claude login` status, disk space under the intended output dir. Saves a $0.30 debugging round when the host is misconfigured.

4. **Ship a skill `finetune-locally`** with the system-prompt snippet and a LoRA-first recipe. Spike 5 proves the agent reasons well about local training; a skill would encode the project's preferences (LoRA default, trackio for loss monitoring, etc.) without bloating the main system prompt.

5. **Tool-output size cap.** Heredoc-generated training scripts + inline training stdout produce large tool-output blobs. Add a per-call output-size cap on our `bash` MCP wrapper (truncate at e.g. 50KB with `[output truncated, N more bytes]` marker).

## Cost datapoints

| Run | Turns | Duration | Cost |
|---|---|---|---|
| First attempt (this run) | 6 | 59s | $0.337 |

Cost structure breakdown expected (from Spike 4's observations):
- Initial context load + tool definitions: ~$0.05
- 3 bash calls × thinking + tool output: ~$0.25
- Final assistant response: ~$0.03

Per-minute training cost on Max-plan auth: ~$0.34/min at this tool-call density. If someone runs a 30-minute training monitoring loop (watching trackio, intervening when loss diverges), expect ~$10 notional cost, likely less in practice because long periods wait on tool output, not compute LLM thinking.

## What's still open

- **GPU path on a working host**: not demonstrated. The agent's behavior shouldn't change; only the training wall-time.
- **Package-install path**: we pre-provisioned the venv. Real users who start cold need the agent to `uv pip install torch transformers trl datasets accelerate` first, which takes 2–3 minutes and ~3GB of disk. The agent can do this (bash + check if packages available) but we didn't test it explicitly.
- **Long training monitoring**: the spike ran one `max_steps=5` training in one turn. Real training runs last minutes-to-hours; watching them in a multi-turn conversation exercises context growth and cost in a different pattern than Spike 4.
- **Multi-run ablation**: no test of "run these 5 configs and compare". That pattern may tempt the agent toward HF Jobs for parallelism; Spike 7 candidate if we want it.

## Next

- **Spike 6**: cancellation / interruption. Now largely pre-answered by `ClaudeSDKClient.interrupt()`; remaining question is orphan-process behavior on remote HF jobs.
- **Cleanup**: spike artifacts at `/tmp/spike5-{venv,output,dataset}/` are ~6GB combined. Safe to `rm -rf` after review.

## Cleanup checklist (post-spike)

```
rm -rf /tmp/spike5-venv /tmp/spike5-output /tmp/spike5-dataset
rm -f  /tmp/spike5-train.py
```
