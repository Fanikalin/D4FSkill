---
name: deepseek4free
description: >-
  Use for token-heavy tasks that don't need Claude's full reasoning: bulk text rewriting,
  large-volume reformatting, draft generation, proofreading, translation, summarization,
  boilerplate expansion, or any task where you'd otherwise spend many Claude tokens on
  repetitive text work. Routes the request through a free local DeepSeek proxy.
  The model is less capable than Claude but completely free and significantly faster on
  large volumes. Automatically starts the server when invoked.
---

# DeepSeek4Free Skill

## What this skill does

Offloads token-expensive, bulk text tasks from Claude to a **free local DeepSeek proxy**
running at `http://localhost:11434`. The proxy emulates the Ollama and OpenAI APIs and
communicates with DeepSeek Chat in the background.

**Model trade-off:**
- DeepSeek Chat is **free** (no API tokens consumed) and **faster** on large volumes.
- It is **less capable** than Claude — it may produce lower-quality reasoning, miss nuance,
  and struggle with complex multi-step logic.
- Use it for mechanical, high-volume work; keep Claude for analysis, architecture, and
  anything that requires careful judgment.

## ⚠️ Privacy warning — read before every use

**Any text you send through this server is transmitted to DeepSeek Chat (chat.deepseek.com).**
DeepSeek is an external service operated by a third party. Treat all data sent through this
server as **publicly accessible**.

**Never use this skill for:**
- Passwords, tokens, API keys, or credentials of any kind
- Personal data: names, emails, phone numbers, addresses, IDs, medical records
- Proprietary source code, trade secrets, or internal business documents
- Financial data, contracts, or any information governed by NDA or compliance requirements
- Private messages, confidential conversations, or anything the owner hasn't consented to share

If there is **any doubt** about whether the data is sensitive, do NOT use this skill — use
Claude directly instead.

## When to invoke this skill

Trigger it when the task is:
- Rewriting or reformatting a **large block of text** (hundreds of lines, many files)
- Generating boilerplate, filler prose, docstrings, or repetitive structure
- Proofreading / spell-checking a large document
- Translating text (especially when Claude tokens are scarce)
- Summarizing long documents or logs
- Expanding bullet points into full paragraphs
- Any task where the user says "use DeepSeek" or "save tokens"

Do **not** use it for:
- Complex architectural decisions or security review
- Tasks requiring deep reasoning across many files
- Anything the user explicitly wants Claude to handle

## Server location

The server lives at `~/.local/share/d4fskill/server/` (installed by `install.sh`).

## Step 1 — Ensure the server is running

Check whether it's up:

```bash
curl -s http://localhost:11434/ 2>/dev/null | grep -q "Deepseek4Free" && echo "UP" || echo "DOWN"
```

If **DOWN**, start it:

```bash
cd ~/.local/share/d4fskill/server && nohup .venv/bin/python main.py > /tmp/d4f.log 2>&1 &
sleep 3
curl -s http://localhost:11434/ | grep -q "Deepseek4Free" && echo "Server started" || echo "Failed — check /tmp/d4f.log"
```

If startup fails, show the user the last 20 lines of `/tmp/d4f.log` and stop — do not fall back
silently to Claude without telling the user.

**One-time setup** — if the server has never been configured:
```bash
ls ~/.local/share/d4fskill/server/.api_keys 2>/dev/null || echo "MISSING"
```
If missing, the user needs to add a DeepSeek token:
```
~/.local/share/d4fskill/server/.api_keys
```
Format: `<label> <token>` (one per line).
Token location: [deepseek.com](https://chat.deepseek.com) → F12 → Application → Local Storage → `userToken`.

## Step 2 — Send the request

Use the **OpenAI-compatible endpoint**. Model name is always `deepseek-chat`.

```python
import httpx, json

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "<your system prompt>"},
        {"role": "user",   "content": "<the large text or task>"}
    ],
    "stream": False,
    "think": False
}

resp = httpx.post(
    "http://localhost:11434/v1/chat/completions",
    json=payload,
    timeout=300
)
result = resp.json()["choices"][0]["message"]["content"]
```

For **streaming** (preferred for large responses):

```python
with httpx.stream("POST", "http://localhost:11434/v1/chat/completions",
                  json={**payload, "stream": True}, timeout=300) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
```

Or via **Bash** (quick one-shot):

```bash
curl -s -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": false,
    "messages": [{"role": "user", "content": "Rewrite this: ..."}]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## Step 3 — Handle errors

If the server returns a non-200 status or the response contains `"error"`, show the error
and tell the user. Do not silently fall back to Claude — the user chose this skill to save tokens.

Common issues:
- **500 / PoW failure** — DeepSeek's anti-bot protection refreshed; the server retries
  automatically (up to 100 attempts with exponential backoff). Just wait.
- **Connection refused** — server didn't start; show `/tmp/d4f.log`.
- **AuthenticationError** — the token in `.api_keys` is expired; ask the user to refresh it.

## API reference

| Endpoint | Protocol | Notes |
|---|---|---|
| `GET /` | — | Health check → `"Deepseek4Free is running"` |
| `POST /v1/chat/completions` | OpenAI | Chat completions, tool calls, streaming |
| `GET /v1/models` | OpenAI | Lists models |
| `POST /v1/completions` | OpenAI legacy | Single-turn text completion |
| `POST /api/chat` | Ollama | Chat completions |
| `POST /api/generate` | Ollama | Single-turn generation |
| `GET /api/tags` | Ollama | Lists models |
| `POST /v1/embeddings` | OpenAI | Text embeddings via BART |

Model name for all endpoints: **`deepseek-chat`**  
Server port: **11434** (Ollama-compatible — any Ollama client works as-is)
