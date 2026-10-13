# D4FSkill

A [Claude Code](https://claude.ai/code) skill that routes bulk text tasks to a **free local DeepSeek proxy** ([deepseek4free](https://github.com/Fanikalin/deepseek4free)) instead of spending Claude API tokens.

## The idea

Claude is powerful but costs tokens. Many tasks — bulk rewrites, proofreading, translation, boilerplate generation — don't actually need Claude's full reasoning. DeepSeek Chat is **free** (unofficial access, no API key needed) and **significantly faster** on large volumes of text. The trade-off: the model is less capable and may produce lower-quality results on complex tasks.

This skill teaches Claude Code when and how to use DeepSeek instead of itself:
- Automatically starts the deepseek4free server if it's not running
- Sends the request to `http://localhost:11434` (Ollama-compatible port)
- Falls back gracefully and tells you if something goes wrong

## Requirements

- [Claude Code](https://claude.ai/code) CLI
- [deepseek4free](https://github.com/Fanikalin/deepseek4free) installed and configured

## Installation

```bash
git clone https://github.com/Fanikalin/D4FSkill
cd D4FSkill
bash install.sh
```

That's it. The skill is placed in `~/.claude/skills/deepseek4free/SKILL.md` and becomes available as `/deepseek4free` in Claude Code.

## Setup deepseek4free (if you haven't yet)

```bash
git clone https://github.com/Fanikalin/deepseek4free ~/Projects/deepseek4free
cd ~/Projects/deepseek4free
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then add your DeepSeek token to `~/Projects/deepseek4free/.api_keys`:

```
myaccount eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Get the token from [deepseek.com](https://chat.deepseek.com) → F12 → Application → Local Storage → `userToken`.

## Usage

In Claude Code, trigger the skill explicitly:

```
/deepseek4free rewrite this 3000-line changelog into concise bullet points
```

Or describe a token-heavy task and Claude will invoke the skill automatically when appropriate:

```
summarize all the logs in ./logs/ — there are about 500 files
```

## When Claude uses this skill

**Good fits (routes to DeepSeek):**
- Rewriting or reformatting large blocks of text
- Generating boilerplate, docstrings, repetitive structures
- Proofreading / spell-checking a big document
- Translating large texts
- Summarizing long logs, documents, or outputs
- Expanding bullet points into prose

**Not suitable (Claude handles it):**
- Security review, architectural decisions
- Complex multi-step reasoning
- Anything needing careful judgment
- Tasks where you explicitly want Claude

## How it works

The skill instructs Claude to:
1. Check if the server is running (`GET http://localhost:11434/`)
2. If not, start it: `cd ~/Projects/deepseek4free && nohup .venv/bin/python main.py > /tmp/d4f.log 2>&1 &`
3. Send requests to the OpenAI-compatible endpoint: `POST http://localhost:11434/v1/chat/completions`
4. Model name: `deepseek-chat`

The server runs on port **11434** — the same port as Ollama — so it's drop-in compatible with any Ollama client.

## License

MIT
