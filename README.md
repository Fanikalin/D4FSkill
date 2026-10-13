# D4FSkill

A skill that routes bulk text tasks to a **free local DeepSeek proxy** instead of spending tokens on Claude, Hermes Agent, or Codex CLI. Also works as a custom model backend for Cursor.

The repo is fully self-contained — it ships the full server code, the PoW WASM solver, and an install script. No external GitHub dependencies beyond this repo and Python 3.10+.

## The idea

Primary models (Claude, GPT, etc.) cost tokens. Many tasks — bulk rewrites, proofreading, translation, boilerplate generation — don't actually need full reasoning. DeepSeek Chat is **free** (unofficial access, no API key needed) and **significantly faster** on large volumes. The trade-off: the model is less capable and may produce lower-quality results on complex tasks.

This skill tells the agent when and how to use DeepSeek instead:
- Automatically starts the local proxy server if it's not running
- Sends requests to `http://localhost:11434` (Ollama-compatible port)
- Falls back gracefully and informs you if something goes wrong

## ⚠️ Privacy warning

**Any text sent through this server reaches DeepSeek Chat (chat.deepseek.com) — an external third-party service. Treat all data as publicly accessible.**

Never use this skill for passwords, API keys, personal data, proprietary source code, financial data, confidential contracts, or any information subject to NDA or compliance requirements.

## Requirements

- Python 3.10+
- At least one of: Claude Code, Hermes Agent, Codex CLI, or Cursor
- A free DeepSeek account

## Installation

```bash
git clone https://github.com/Fanikalin/D4FSkill
cd D4FSkill
bash install.sh
```

`install.sh` automatically detects which tools are installed and registers the skill for each:
- **Claude Code** → `~/.claude/skills/deepseek4free/`
- **Hermes Agent** → `~/.hermes/skills/software-development/deepseek4free/`
- **Codex CLI** → `~/.codex/skills/deepseek4free/`

The server is deployed to `~/.local/share/d4fskill/server/` with its own virtual environment.

## Add your DeepSeek token

After install, edit `~/.local/share/d4fskill/server/.api_keys`:

```
myaccount eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Get the token: [chat.deepseek.com](https://chat.deepseek.com) → F12 → Application → Local Storage → `userToken`.

You can add multiple tokens (one per line) for round-robin rotation.

## Usage

### Claude Code / Hermes Agent / Codex CLI

Invoke the skill:

```
/deepseek4free rewrite this 3000-line changelog into concise bullet points
```

Or just describe a token-heavy task — the agent decides automatically:

```
summarize all the log files in ./logs/ — there are about 500 of them
```

### Cursor

Since Cursor doesn't have a skill system, configure it to use the local server as a model backend:

1. Start the server: `cd ~/.local/share/d4fskill/server && .venv/bin/python main.py`
2. Open Cursor → Settings → Models
3. Set **OpenAI Base URL** → `http://localhost:11434/v1`
4. Set **API Key** → any non-empty string (e.g. `d4f`)
5. Set **Model** → `deepseek-chat`

The server runs on port **11434** (Ollama-compatible) so any tool that supports custom OpenAI endpoints works out of the box.

Alternatively, launch Cursor with environment variables:
```bash
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=d4f cursor
```

## When to use this

**Good fits (routed to DeepSeek, free):**
- Rewriting or reformatting large blocks of text
- Generating boilerplate, docstrings, repetitive structures
- Proofreading / spell-checking a big document
- Translating large texts
- Summarizing long logs, documents, or build output
- Expanding bullet points into prose

**Not suitable (use the primary model):**
- Complex architectural decisions
- Security review
- Multi-step reasoning across many files
- Anything sensitive or confidential

## How it works

When invoked, the agent:
1. Checks if the server is up (`GET http://localhost:11434/`)
2. If not, starts it from `~/.local/share/d4fskill/server/`
3. Sends requests via OpenAI-compatible API (`POST /v1/chat/completions`, model `deepseek-chat`)

The server retries failed generations automatically (up to 100 attempts, exponential backoff), so transient Cloudflare or PoW failures are transparent.

## Server internals

`server/` contains:
- `main.py` — entry point (Quart app on port 11434)
- `app/` — Ollama + OpenAI endpoints, tool-call handling, retry logic
- `dsk/` — DeepSeek client: HTTP, Proof-of-Work WASM solver, cookie helpers

## License

MIT
