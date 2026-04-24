---
name: hf-ml-engineering
description: Guidance for ML engineering on the Hugging Face stack — fine-tuning, training, evaluation, and inference with Transformers/TRL/PEFT/Datasets. Use when the user mentions fine-tuning, LoRA, SFT/DPO/GRPO, training a model, Transformers, TRL, PEFT, Trackio, HF Jobs, or Hugging Face datasets/models. Requires the ml-intern MCP server.
---

# Hugging Face ML Engineering

Use this recipe when helping the user train, fine-tune, evaluate, or run
inference with Hugging Face libraries. It assumes the **ml-intern MCP
server** is registered in Claude Code — its tools appear as
`mcp__ml-intern__*` (e.g. `mcp__ml-intern__explore_hf_docs`). Prose below
uses short names for readability.

## Your knowledge of HF libraries is out of date

Do not write HF-ecosystem code from memory. TRL trainer class names,
Transformers APIs, PEFT argument shapes, and Trackio parameter names
drift between versions — memorised imports will produce `ImportError`,
`TypeError`, or wrong-column `KeyError` at runtime.

**Before writing any ML code:**

1. Find the landmark paper(s) for the task via `hf_papers`
2. Crawl the citation graph (`hf_papers` with `operation: citation_graph`) for recent downstream work
3. Read methodology sections of the most promising papers — prefer recent ones with strong reported results
4. Extract the concrete recipe: dataset, training method, hyperparameters
5. Validate the dataset with `hf_inspect_dataset` (columns, splits, row counts, distributions)
6. Cross-check current APIs with `explore_hf_docs` + `fetch_hf_docs`, and look at actual example scripts via `github_find_examples` / `github_read_file`

For a broad literature crawl, delegate to the **ml-research** subagent
via Claude Code's `Task` tool:

```
Task(subagent_type="ml-research", prompt="Literature crawl for <task>. Start from <paper/topic>. Crawl the citation graph for recent downstream work. Read methodology sections of the most promising papers — extract datasets, training methods, hyperparameters that produced their best results. Attribute every finding to a specific result. Also find working code examples using current TRL/Transformers APIs.")
```

`ml-research` has its own literature-first system prompt and a
read-only tool set (`hf_papers`, `explore_hf_docs`, `fetch_hf_docs`,
`github_*`, `hf_inspect_dataset`). Always pass `subagent_type="ml-research"` —
a bare `Task(...)` call gets a generic sub-agent with no research
guidance. Be specific in the prompt — name anchor papers or arxiv IDs
when known.

Skip research only for trivial non-code operations (renaming files,
reading logs, etc.).

## Mistakes to watch for

- **Hallucinated imports** — old TRL class names, deprecated Transformers APIs, wrong Trackio parameter names (`run_name` vs `name`). Fix: read a current example script before writing new code.
- **Wrong trainer arguments** — config args that don't exist in the installed version. Fix: `explore_hf_docs` + `fetch_hf_docs` for the exact class.
- **Wrong dataset format** — assumed column names. Fix: `hf_inspect_dataset` before training; confirm columns match the method (SFT: `messages`/`text`/`prompt+completion`; DPO: `prompt`/`chosen`/`rejected`; GRPO: `prompt`).
- **Silent dataset substitution** — when the requested dataset fails to load, don't quietly pick another. Tell the user and ask.
- **Missing packages** — `flash-attn` for `flash_attention_2`, etc. — install before running.
- **Scope-changing error recovery** — on OOM or other failures, do NOT switch full SFT to LoRA, reduce `max_length`, or swap datasets on your own. Those change what the user asked for. Use the minimal fix that preserves the original request; if it genuinely can't work, explain why and ask.

## Data audit — always

Before touching a dataset, run `hf_inspect_dataset` to check schema,
row counts per split, value distributions, and sample rows. Surface
anything notable: class imbalance, missing values, duplicates, outliers,
unexpected formats. Looking at data before training prevents more failed
jobs than any other single habit.

## Training logging

In `TrainingArguments` / `SFTConfig` / `DPOConfig`, always set:

```python
disable_tqdm=True,
logging_strategy="steps",
logging_first_step=True,
```

so losses are emitted as plain log lines (greppable) rather than hidden
inside tqdm redraws.

## Error recovery

- Read the full error + stack before retrying.
- Don't retry the exact same call — identify what needs to change.
- API/import errors → `explore_hf_docs` / `fetch_hf_docs`.
- OOM → (1) shrink `per_device_train_batch_size`, raise `gradient_accumulation_steps` to keep effective batch identical; (2) `gradient_checkpointing=True`; (3) larger GPU. Do NOT switch training method, reduce `max_length`, or swap models without user approval.
- Same tool failing twice for the same reason → stop and try a different approach or surface to the user.

## HF Jobs (only if `--enable-hf-infra`)

The `hf_jobs` / `hf_repo_files` / `hf_repo_git` tools are only registered
when the ml-intern MCP was installed with `--enable-hf-infra`. If not
registered, skip this section and run training locally through Claude
Code's Bash tool.

When `hf_jobs` is available, output a pre-flight check before submission:

- Reference implementation: [which example script this is based on]
- Dataset format verified: [columns confirmed via `hf_inspect_dataset`]
- `push_to_hub=True` and `hub_model_id` set (job filesystem is ephemeral — models are lost without this)
- `timeout`: [value] — default 30 min kills most training; minimum 2 h for any real training run
- Trackio monitoring configured, dashboard URL captured

For batch / ablation runs: submit **one** job first, confirm it trains
successfully from the logs, then submit the rest. Never fire the whole
sweep at once — they'll all fail for the same bug.

Hardware sizing rule of thumb:

| Model size | Flavor |
|---|---|
| 1–3B | `a10g-largex2` |
| 7–13B | `a100-large` |
| 30B+ | `l40sx4` or `a100x4` |
| 70B+ | `a100x8` |

`a10g-small` and `a10g-large` have the **same** 24 GB GPU — the flavor
difference is CPU/RAM only.

## Hyperparameter tuning

Don't tune one value at a time by hand. Write a sweep script (grid or
random over learning rate, epochs, batch size, LoRA rank, etc.) that
evaluates each run automatically. One well-designed sweep beats ten
manual experiments.

## Communication

- Concise and direct. No filler.
- Include direct Hub URLs when referencing models, datasets, Spaces, or jobs.
- On errors: state what went wrong, why, what you're doing about it.
- Act on clear intent; present options only for genuine ambiguity.
