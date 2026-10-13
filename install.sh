#!/usr/bin/env bash
# D4FSkill installer
# Installs the server and registers the skill in Claude Code, Hermes Agent, and Codex CLI.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SRC="$SCRIPT_DIR/server"
SERVER_DST="$HOME/.local/share/d4fskill/server"

echo "=== D4FSkill installer ==="
echo ""

# ── 1. Deploy server ─────────────────────────────────────────────────────────
echo "Installing server to $SERVER_DST ..."
mkdir -p "$SERVER_DST"
cp -r "$SERVER_SRC"/. "$SERVER_DST/"

if [ ! -f "$SERVER_DST/.api_keys" ]; then
    cp "$SERVER_DST/.api_keys.example" "$SERVER_DST/.api_keys"
    echo ""
    echo "  Created $SERVER_DST/.api_keys from example."
    echo "  ACTION REQUIRED: add your DeepSeek token to that file."
    echo "  Format: <label> <token>  (one per line)"
    echo "  Get the token: deepseek.com → F12 → Application → Local Storage → userToken"
fi

# ── 2. Python venv + dependencies ────────────────────────────────────────────
echo ""
echo "Setting up Python virtual environment ..."
python3 -m venv "$SERVER_DST/.venv"
"$SERVER_DST/.venv/bin/pip" install --quiet --upgrade pip
"$SERVER_DST/.venv/bin/pip" install --quiet -r "$SERVER_DST/requirements.txt"
echo "Dependencies installed."

# ── 3. Claude Code skill ──────────────────────────────────────────────────────
CLAUDE_SKILL_DIR="$HOME/.claude/skills/deepseek4free"
if [ -d "$HOME/.claude/skills" ]; then
    echo ""
    echo "Registering skill in Claude Code → $CLAUDE_SKILL_DIR"
    mkdir -p "$CLAUDE_SKILL_DIR"
    cp "$SCRIPT_DIR/SKILL.md" "$CLAUDE_SKILL_DIR/SKILL.md"
    echo "  Done."
else
    echo ""
    echo "  Claude Code skill directory not found — skipping (~/.claude/skills/ missing)."
fi

# ── 4. Hermes Agent skill ─────────────────────────────────────────────────────
HERMES_SKILL_DIR="$HOME/.hermes/skills/software-development/deepseek4free"
if [ -d "$HOME/.hermes/skills" ]; then
    echo ""
    echo "Registering skill in Hermes Agent → $HERMES_SKILL_DIR"
    mkdir -p "$HERMES_SKILL_DIR"
    cp "$SCRIPT_DIR/SKILL.md" "$HERMES_SKILL_DIR/SKILL.md"
    echo "  Done."
else
    echo ""
    echo "  Hermes Agent skill directory not found — skipping (~/.hermes/skills/ missing)."
fi

# ── 5. Codex CLI skill ────────────────────────────────────────────────────────
CODEX_SKILL_DIR="$HOME/.codex/skills/deepseek4free"
if [ -d "$HOME/.codex/skills" ]; then
    echo ""
    echo "Registering skill in Codex CLI → $CODEX_SKILL_DIR"
    mkdir -p "$CODEX_SKILL_DIR"
    cp "$SCRIPT_DIR/SKILL.md" "$CODEX_SKILL_DIR/SKILL.md"
    echo "  Done."
else
    echo ""
    echo "  Codex CLI skill directory not found — skipping (~/.codex/skills/ missing)."
fi

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo ""
echo "Server: $SERVER_DST"
echo "Skill installed for:"
[ -d "$HOME/.claude/skills" ] && echo "  ✓ Claude Code   (/deepseek4free)"
[ -d "$HOME/.hermes/skills" ] && echo "  ✓ Hermes Agent  (/deepseek4free)"
[ -d "$HOME/.codex/skills"  ] && echo "  ✓ Codex CLI     (/deepseek4free)"
echo ""
echo "Start the server manually:"
echo "  cd $SERVER_DST && nohup .venv/bin/python main.py > /tmp/d4f.log 2>&1 &"
echo ""
echo "Or invoke /deepseek4free in any supported agent and it will start automatically."
echo ""
echo "Cursor: set 'OpenAI Base URL' to http://localhost:11434/v1 in Settings → Models."
echo "  API key can be any non-empty string (e.g. 'd4f')."
