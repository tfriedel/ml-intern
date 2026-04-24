<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# ML Intern

An ML intern that autonomously researches, writes, and ships good quality ML releated code using the Hugging Face ecosystem — with deep access to docs, papers, datasets, and cloud compute.

## Quick Start

### Installation

```bash
git clone git@github.com:huggingface/ml-intern.git
cd ml-intern
uv sync
uv tool install -e .
```

#### That's it. Now `ml-intern` works from any directory:

```bash
ml-intern
```

Create a `.env` file in the project root (or export these in your shell):

```bash
# Required for the default (litellm) backend:
ANTHROPIC_API_KEY=<your-anthropic-api-key>
# Not required when you use `--backend sdk` (Claude login auth).

HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token>
```
If no `HF_TOKEN` is set, the CLI will prompt you to paste one on first launch. To get a GITHUB_TOKEN follow the tutorial [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token).

### Backends

ml-intern has two LLM backends:

- **`litellm`** (default) — calls Claude via the Anthropic API; needs
  `ANTHROPIC_API_KEY` and charges per token. Full tool set including
  HF Jobs, HF Sandbox, and HF repo writes.
- **`sdk`** — drives Claude via the [Claude Agent SDK](https://docs.claude.com/en/api/agent-sdk/overview)
  using your `claude login` session (Pro / Max subscription, no API key
  required). Local-first: defaults to no HF Jobs / no HF Sandbox / no
  Hub writes, using Claude Code's built-in `Bash` / `Read` / `Write` /
  `Edit` tools on your own machine.

Opt into the SDK backend:

```bash
# Requires Claude Code CLI installed and `claude login` completed.
ml-intern --backend sdk "fine-tune smollm on /tmp/my-data"
```

If you want the SDK backend with HF infrastructure still enabled:

```bash
ml-intern --backend sdk --enable-hf-infra "…"
```

Conversely, you can force the litellm backend to be local-first:

```bash
ml-intern --no-hf-infra "run a 10-step experiment locally"
```

### Using from Claude Code

Instead of running the standalone `ml-intern` CLI, you can expose the
tools, skill, and research sub-agent directly inside
[Claude Code](https://docs.claude.com/en/docs/claude-code) sessions. The
installer wires up four pieces in one shot:

```bash
# Read-only (default): list/read Hub repos, search docs/papers, no writes
scripts/install-claude-code.sh

# With write ops on HF (hf_jobs, upload, delete, repo git)
scripts/install-claude-code.sh --enable-hf-infra

# Remove everything the installer added
scripts/install-claude-code.sh --uninstall
```

What it installs:

1. **`ml-intern`** stdio MCP server — ml-intern's tools (`hf_papers`,
   `explore_hf_docs`, `hf_inspect_dataset`, `github_*`, `hf_repo_files`
   read-only, …) registered via `claude mcp add`. Claude Code spawns it
   on demand through `uv --directory <repo> run ml-intern-mcp`.
2. **`hf-mcp-server`** (public HF MCP) — registered as an HTTP MCP so
   Hub search / repo details / space info are available natively. OAuth
   login triggers on first tool use. Skipped if you already have an
   entry under that name. Add `--skip-hf-mcp` to opt out.
3. **Skills** → `~/.claude/skills/` — `hf-ml-engineering` (literature-
   first ML workflow, pre-flight checks, error recovery) and
   `finetune-locally` (local LoRA recipe, no Hub writes).
4. **`ml-research` sub-agent** → `~/.claude/agents/` — read-only
   research agent invoked via
   `Task(subagent_type="ml-research", ...)`. Same literature-first prompt
   as the standalone agent, same restricted tool set.

`--enable-hf-infra` flips `ML_INTERN_MCP_HF_INFRA=1` for the MCP server:
the tool count goes from 9 (read-only) to 12 (adds `hf_jobs`,
`hf_repo_git`, and unlocks `upload`/`delete` on `hf_repo_files`).

Differences vs. the standalone CLI:

- Claude Code's built-in `Bash` / `Read` / `Write` / `Edit` / `Task`
  replace ml-intern's local/sandbox wrappers and the `research` tool.
- `HF_TOKEN` / `GITHUB_TOKEN` come from the shell environment at MCP
  launch; no `.env` is read.
- No session logging, no doom-loop detector, no auto-compaction at
  170k — Claude Code handles session transcripts and compaction
  natively.

### Usage

**Interactive mode** (start a chat session):

```bash
ml-intern
```

**Headless mode** (single prompt, auto-approve):

```bash
ml-intern "fine-tune llama on my dataset"
```

**Options:**

```bash
ml-intern --backend {litellm,sdk}   "your prompt"
ml-intern --enable-hf-infra         "your prompt"   # or --no-hf-infra
ml-intern --model anthropic/claude-opus-4-6 "your prompt"
ml-intern --max-iterations 100      "your prompt"
ml-intern --no-stream               "your prompt"
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User/CLI                            │
└────────────┬─────────────────────────────────────┬──────────┘
             │ Operations                          │ Events
             ↓ (user_input, exec_approval,         ↑
      submission_queue  interrupt, compact, ...)  event_queue
             │                                          │
             ↓                                          │
┌────────────────────────────────────────────────────┐  │
│            submission_loop (agent_loop.py)         │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │  1. Receive Operation from queue             │  │  │
│  │  2. Route to handler (run_agent/compact/...) │  │  │
│  └──────────────────────────────────────────────┘  │  │
│                      ↓                             │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │         Handlers.run_agent()                 │  ├──┤
│  │                                              │  │  │
│  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  Agentic Loop (max 300 iterations)     │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Session                          │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ContextManager             │  │  │  │  │  │
│  │  │  │  │ • Message history          │  │  │  │  │  │
│  │  │  │  │   (litellm.Message[])      │  │  │  │  │  │
│  │  │  │  │ • Auto-compaction (170k)   │  │  │  │  │  │
│  │  │  │  │ • Session upload to HF     │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  │                                  │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ToolRouter                 │  │  │  │  │  │
│  │  │  │  │  ├─ HF docs & research     │  │  │  │  │  │
│  │  │  │  │  ├─ HF repos, datasets,    │  │  │  │  │  │
│  │  │  │  │  │  jobs, papers           │  │  │  │  │  │
│  │  │  │  │  ├─ GitHub code search     │  │  │  │  │  │
│  │  │  │  │  ├─ Sandbox & local tools  │  │  │  │  │  │
│  │  │  │  │  ├─ Planning               │  │  │  │  │  │
│  │  │  │  │  └─ MCP server tools       │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Doom Loop Detector               │  │  │  │  │
│  │  │  │ • Detects repeated tool patterns │  │  │  │  │
│  │  │  │ • Injects corrective prompts     │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  Loop:                                 │  │  │  │
│  │  │    1. LLM call (litellm.acompletion)   │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    2. Parse tool_calls[]               │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    3. Approval check                   │  │  │  │
│  │  │       (jobs, sandbox, destructive ops) │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    4. Execute via ToolRouter           │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    5. Add results to ContextManager    │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    6. Repeat if tool_calls exist       │  │  │  │
│  │  └────────────────────────────────────────┘  │  │  │
│  └──────────────────────────────────────────────┘  │  │
└────────────────────────────────────────────────────┴──┘
```

### Agentic Loop Flow

```
User Message
     ↓
[Add to ContextManager]
     ↓
     ╔═══════════════════════════════════════════╗
     ║      Iteration Loop (max 300)             ║
     ║                                           ║
     ║  Get messages + tool specs                ║
     ║         ↓                                 ║
     ║  litellm.acompletion()                    ║
     ║         ↓                                 ║
     ║  Has tool_calls? ──No──> Done             ║
     ║         │                                 ║
     ║        Yes                                ║
     ║         ↓                                 ║
     ║  Add assistant msg (with tool_calls)      ║
     ║         ↓                                 ║
     ║  Doom loop check                          ║
     ║         ↓                                 ║
     ║  For each tool_call:                      ║
     ║    • Needs approval? ──Yes──> Wait for    ║
     ║    │                         user confirm ║
     ║    No                                     ║
     ║    ↓                                      ║
     ║    • ToolRouter.execute_tool()            ║
     ║    • Add result to ContextManager         ║
     ║         ↓                                 ║
     ║  Continue loop ─────────────────┐         ║
     ║         ↑                       │         ║
     ║         └───────────────────────┘         ║
     ╚═══════════════════════════════════════════╝
```

## Events

The agent emits the following events via `event_queue`:

- `processing` - Starting to process user input
- `ready` - Agent is ready for input
- `assistant_chunk` - Streaming token chunk
- `assistant_message` - Complete LLM response text
- `assistant_stream_end` - Token stream finished
- `tool_call` - Tool being called with arguments
- `tool_output` - Tool execution result
- `tool_log` - Informational tool log message
- `tool_state_change` - Tool execution state transition
- `approval_required` - Requesting user approval for sensitive operations
- `turn_complete` - Agent finished processing
- `error` - Error occurred during processing
- `interrupted` - Agent was interrupted
- `compacted` - Context was compacted
- `undo_complete` - Undo operation completed
- `shutdown` - Agent shutting down

## Development

### Adding Built-in Tools

Edit `agent/core/tools.py`:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### Adding MCP Servers

Edit `configs/main_agent_config.json`:

```json
{
  "model_name": "anthropic/claude-sonnet-4-5-20250929",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Note: Environment variables like `${YOUR_TOKEN}` are auto-substituted from `.env`.
