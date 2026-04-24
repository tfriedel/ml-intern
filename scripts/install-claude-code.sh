#!/usr/bin/env bash
# Install ml-intern's MCP server and skill into Claude Code.
#
# Usage:
#   scripts/install-claude-code.sh                       # read-only, user scope
#   scripts/install-claude-code.sh --enable-hf-infra     # also expose hf_jobs/hf_repo_*
#   scripts/install-claude-code.sh --scope project       # register in .mcp.json instead
#   scripts/install-claude-code.sh --skip-hf-mcp         # don't register huggingface.co/mcp
#   scripts/install-claude-code.sh --uninstall           # remove everything
#
# What it does:
#   1. Registers `ml-intern` as a stdio MCP server with Claude Code, launched
#      via `uv --directory <repo> run ml-intern-mcp`.
#   2. Copies agent/skills/finetune-locally.md into ~/.claude/skills/.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"

SERVER_NAME="ml-intern"
HF_MCP_NAME="hf-mcp-server"
HF_MCP_URL="https://huggingface.co/mcp?login"
CLAUDE_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
SKILLS_DST_ROOT="$CLAUDE_DIR/skills"
AGENTS_DST_ROOT="$CLAUDE_DIR/agents"

# Each entry: "<source-file>:<skill-name>". The skill file must have
# matching `name:` frontmatter. It's installed to ~/.claude/skills/<name>/SKILL.md.
SKILLS=(
    "agent/skills/finetune-locally.md:finetune-locally"
    "agent/skills/hf-ml-engineering.md:hf-ml-engineering"
)

# Each entry: "<source-file>:<agent-name>". Installed to ~/.claude/agents/<name>.md.
AGENTS=(
    ".claude/agents/ml-research.md:ml-research"
)

SCOPE="user"
ENABLE_HF_INFRA=0
SKIP_HF_MCP=0
UNINSTALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --enable-hf-infra) ENABLE_HF_INFRA=1; shift ;;
        --skip-hf-mcp)     SKIP_HF_MCP=1; shift ;;
        --scope)           SCOPE="$2"; shift 2 ;;
        --uninstall)       UNINSTALL=1; shift ;;
        -h|--help)
            sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v claude >/dev/null || { echo "claude CLI not found in PATH" >&2; exit 1; }
command -v uv     >/dev/null || { echo "uv not found in PATH"        >&2; exit 1; }

if [[ $UNINSTALL -eq 1 ]]; then
    echo "→ removing MCP server '$SERVER_NAME' (scope=$SCOPE)"
    claude mcp remove --scope "$SCOPE" "$SERVER_NAME" 2>/dev/null || echo "  (not registered — skipping)"
    # We only remove hf-mcp-server if we registered it (i.e. its URL matches ours).
    # The user may have their own entry under this name pointing elsewhere.
    if claude mcp get "$HF_MCP_NAME" 2>/dev/null | grep -qF "$HF_MCP_URL"; then
        echo "→ removing MCP server '$HF_MCP_NAME' (scope=$SCOPE)"
        claude mcp remove --scope "$SCOPE" "$HF_MCP_NAME" 2>/dev/null || true
    fi
    for entry in "${SKILLS[@]}"; do
        skill_name="${entry##*:}"
        skill_path="$SKILLS_DST_ROOT/$skill_name"
        echo "→ removing skill at $skill_path"
        rm -rf "$skill_path"
    done
    for entry in "${AGENTS[@]}"; do
        agent_name="${entry##*:}"
        agent_path="$AGENTS_DST_ROOT/$agent_name.md"
        echo "→ removing subagent at $agent_path"
        rm -f "$agent_path"
    done
    echo "✓ uninstalled"
    exit 0
fi

for entry in "${SKILLS[@]}" "${AGENTS[@]}"; do
    src="$REPO_DIR/${entry%%:*}"
    [[ -f "$src" ]] || { echo "source missing: $src" >&2; exit 1; }
done

# 1. Register the MCP server. Remove first so the command is idempotent.
claude mcp remove --scope "$SCOPE" "$SERVER_NAME" 2>/dev/null || true

ENV_ARGS=()
if [[ $ENABLE_HF_INFRA -eq 1 ]]; then
    ENV_ARGS+=(-e "ML_INTERN_MCP_HF_INFRA=1")
fi

echo "→ registering MCP server '$SERVER_NAME' (scope=$SCOPE, hf_infra=$ENABLE_HF_INFRA)"
claude mcp add \
    --scope "$SCOPE" \
    "${ENV_ARGS[@]}" \
    "$SERVER_NAME" \
    -- uv --directory "$REPO_DIR" run ml-intern-mcp

# 1b. Register the HF public MCP (Hub search, repo details, spaces, …).
# Skip if the user already has an entry under this name (don't stomp on their config).
registered_hf_mcp=0
if [[ $SKIP_HF_MCP -eq 1 ]]; then
    echo "→ skipping $HF_MCP_NAME (--skip-hf-mcp)"
elif claude mcp get "$HF_MCP_NAME" &>/dev/null; then
    echo "→ $HF_MCP_NAME already registered (scope=$SCOPE) — leaving it alone"
else
    echo "→ registering MCP server '$HF_MCP_NAME' → $HF_MCP_URL (scope=$SCOPE)"
    claude mcp add --scope "$SCOPE" --transport http "$HF_MCP_NAME" "$HF_MCP_URL"
    registered_hf_mcp=1
fi

# 2. Install skills.
installed_skills=()
for entry in "${SKILLS[@]}"; do
    skill_src="$REPO_DIR/${entry%%:*}"
    skill_name="${entry##*:}"
    skill_dir="$SKILLS_DST_ROOT/$skill_name"
    skill_dst="$skill_dir/SKILL.md"
    echo "→ installing skill '$skill_name' → $skill_dst"
    mkdir -p "$skill_dir"
    cp "$skill_src" "$skill_dst"
    installed_skills+=("$skill_dst")
done

# 3. Install subagents.
installed_agents=()
mkdir -p "$AGENTS_DST_ROOT"
for entry in "${AGENTS[@]}"; do
    agent_src="$REPO_DIR/${entry%%:*}"
    agent_name="${entry##*:}"
    agent_dst="$AGENTS_DST_ROOT/$agent_name.md"
    echo "→ installing subagent '$agent_name' → $agent_dst"
    cp "$agent_src" "$agent_dst"
    installed_agents+=("$agent_dst")
done

echo ""
echo "✓ done"
echo "  MCP server : $SERVER_NAME (scope=$SCOPE)"
if [[ $registered_hf_mcp -eq 1 ]]; then
    echo "  MCP server : $HF_MCP_NAME → $HF_MCP_URL"
    echo "               (OAuth login will trigger on first tool use)"
fi
for s in "${installed_skills[@]}"; do
    echo "  Skill      : $s"
done
for a in "${installed_agents[@]}"; do
    echo "  Subagent   : $a"
done
echo ""
echo "Verify:"
echo "  claude mcp list"
echo "  claude mcp get $SERVER_NAME"
