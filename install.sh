#!/usr/bin/env bash
# D4FSkill installer — registers the deepseek4free skill in Claude Code
set -e

SKILL_DIR="$HOME/.claude/skills/deepseek4free"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing D4FSkill..."

mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/SKILL.md"

echo "Skill installed to $SKILL_DIR"
echo ""
echo "The skill will appear as /deepseek4free in Claude Code."
echo ""
echo "Prerequisites:"
echo "  1. Clone deepseek4free: https://github.com/Fanikalin/deepseek4free"
echo "  2. Set it up: cd ~/Projects/deepseek4free && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
echo "  3. Add your DeepSeek token to ~/Projects/deepseek4free/.api_keys"
echo "     Format: <label> <token>  (one per line)"
echo "     Get the token: deepseek.com → F12 → Application → Local Storage → userToken"
