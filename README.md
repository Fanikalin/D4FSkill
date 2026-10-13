# D4FSkill

A [Claude Code](https://claude.ai/code) skill that routes bulk text tasks to a **free local DeepSeek proxy** instead of spending Claude API tokens.

The repo is self-contained — it ships the full server code, the PoW WASM solver, and an install script. No external dependencies beyond this repo and Python 3.10+.

## The idea

Claude is powerful but costs tokens. Many tasks — bulk rewrites, proofreading, translation, boilerplate generation — don't actually need Claude's full reasoning. DeepSeek Chat is **free** (unofficial access, no API key needed) and **significantly faster** on large volumes of text. The trade-off: the model is less capable and may produce lower-quality results on complex tasks.

This skill teaches Claude Code when and how to use DeepSeek instead of itself:
- Automatically starts the local proxy server if it's not running
- Sends requests to `http://localhost:11434` (Ollama-compatible port)
- Falls back gracefully and tells you if something went wrong

## ⚠️ Privacy warning

**Any text sent through this server is transmitted to DeepSeek Chat (chat.deepseek.com) — an external third-party service. Treat all data sent through this server as publicly accessible.**

Never use this skill for passwords, API keys, personal data, proprietary source code, financial data, confidential contracts, or any information subject to NDA or compliance requirements. If there is any doubt about whether data is sensitive, use Claude directly.

## Requirements

- Python 3.10+
- [Claude Code](https://claude.ai/code) CLI
- A DeepSeek account (free) — to get an auth token

## Installation

```bash
git clone https://github.com/Fanikalin/D4FSkill
cd D4FSkill
bash install.sh
```

`install.sh` does three things:
1. Copies `server/` to `~/.local/share/d4fskill/server/`
2. Creates a Python venv there and installs dependencies
3. Copies `SKILL.md` to `~/.claude/skills/deepseek4free/`

## Add your DeepSeek token

After install, edit `~/.local/share/d4fskill/server/.api_keys`:

```
myaccount eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Get the token: [chat.deepseek.com](https://chat.deepseek.com) → F12 → Application → Local Storage → `userToken`.

You can add multiple tokens (one per line) for round-robin rotation.

## Usage

In Claude Code, invoke the skill:

```
/deepseek4free rewrite this 3000-line changelog into concise bullet points
```

Or just describe a token-heavy task — Claude will decide to invoke the skill automatically:

```
summarize all the log files in ./logs/ — there are about 500 of them
```

## When Claude uses this skill

**Good fits (routed to DeepSeek, free):**
- Rewriting or reformatting large blocks of text
- Generating boilerplate, docstrings, repetitive structures
- Proofreading / spell-checking a big document
- Translating large texts
- Summarizing long logs, documents, or build output
- Expanding bullet points into prose

**Not suitable (Claude handles it):**
- Complex architectural decisions
- Security review
- Multi-step reasoning across many files
- Anything requiring careful judgment

## How it works

The skill instructs Claude to:
1. Check if the server is up (`GET http://localhost:11434/`)
2. If not, start it from `~/.local/share/d4fskill/server/`
3. Send requests to the OpenAI-compatible endpoint (`POST /v1/chat/completions`)
4. Model name: `deepseek-chat`

The server exposes port **11434** — the same port as Ollama — so it works as a drop-in with any Ollama client (Open WebUI, Continue, etc.).

## Server internals

`server/` contains:
- `main.py` — entry point (Quart app on port 11434)
- `app/` — API server: Ollama + OpenAI endpoints, tool-call handling, retry logic
- `dsk/` — DeepSeek client: HTTP requests, Proof-of-Work solver (WASM), Cloudflare cookie helpers

The server retries failed generations automatically (up to 100 attempts with exponential backoff), so transient Cloudflare / PoW failures are handled transparently.

## License

MIT
