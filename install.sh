#!/usr/bin/env bash
# D4FSkill installer
# Installs the server to ~/.local/share/d4fskill/ and registers the skill in Claude Code.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SRC="$SCRIPT_DIR/server"
SERVER_DST="$HOME/.local/share/d4fskill/server"
SKILL_DIR="$HOME/.claude/skills/deepseek4free"

echo "=== D4FSkill installer ==="
echo ""

# 1. Copy server files
echo "Installing server to $SERVER_DST ..."
mkdir -p "$SERVER_DST"
cp -r "$SERVER_SRC"/. "$SERVER_DST/"

# 2. Create .api_keys if missing
if [ ! -f "$SERVER_DST/.api_keys" ]; then
    cp "$SERVER_DST/.api_keys.example" "$SERVER_DST/.api_keys"
    echo ""
    echo "  Created $SERVER_DST/.api_keys from example."
    echo "  ACTION REQUIRED: add your DeepSeek token to that file."
    echo "  Format: <label> <token>  (one per line)"
    echo "  Get the token: deepseek.com → F12 → Application → Local Storage → userToken"
fi

# 3. Create Python venv and install dependencies
echo ""
echo "Setting up Python virtual environment ..."
python3 -m venv "$SERVER_DST/.venv"
"$SERVER_DST/.venv/bin/pip" install --quiet --upgrade pip
"$SERVER_DST/.venv/bin/pip" install --quiet -r "$SERVER_DST/requirements.txt"
echo "Dependencies installed."

# 4. Install Claude Code skill
echo ""
echo "Installing Claude Code skill to $SKILL_DIR ..."
mkdir -p "$SKILL_DIR"
cp "$SCRIPT_DIR/SKILL.md" "$SKILL_DIR/SKILL.md"

echo ""
echo "=== Done ==="
echo ""
echo "The skill is now available as /deepseek4free in Claude Code."
echo ""
echo "Start the server manually:"
echo "  cd $SERVER_DST && nohup .venv/bin/python main.py > /tmp/d4f.log 2>&1 &"
echo ""
echo "Or let Claude start it automatically when you invoke /deepseek4free."
