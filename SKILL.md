---
name: deepseek4free
description: >-
  Use for token-heavy tasks that don't need Claude's full reasoning: bulk text rewriting,
  large-volume reformatting, draft generation, proofreading, translation, summarization,
  boilerplate expansion, or any task where you'd otherwise spend many Claude tokens on
  repetitive text work. Routes the request through a free local DeepSeek proxy instead.
  The model is less capable than Claude but completely free and significantly faster on
  large volumes. Automatically starts the deepseek4free server when invoked.
---

# DeepSeek4Free Skill

## What this skill does

Offloads token-expensive, bulk text tasks from Claude to a **free local DeepSeek proxy**
running at `http://localhost:11434`. The proxy exposes both an Ollama-compatible API and
an OpenAI-compatible API backed by the unofficial DeepSeek Chat endpoint.

**Model trade-off:**
- DeepSeek Chat is **free** (no API tokens consumed) and **faster** on large volumes.
- It is **less capable** than Claude — it may produce lower-quality reasoning, miss nuance,
  and struggle with complex multi-step logic.
- Use it for mechanical, high-volume work; keep Claude for analysis, architecture, and
  anything that requires careful judgment.

## When to invoke this skill

Trigger it whenever the task is:
- Rewriting or reformatting a **large block of text** (hundreds of lines, many files)
- Generating boilerplate, filler prose, docstrings, or repetitive structure
- Proofreading / spell-checking a large document
- Translating text (especially when Claude tokens are scarce)
- Summarizing long documents or logs
- Expanding bullet points into full paragraphs
- Any task where the user explicitly says "use DeepSeek" or "save tokens"

Do **not** use it for:
- Security-sensitive code review
- Complex architectural decisions
- Tasks requiring deep reasoning across many files
- Anything the user explicitly wants Claude to handle

## Step 1 — Ensure the server is running

Before making any request, check whether the server is up:

```bash
curl -s http://localhost:11434/ 2>/dev/null | grep -q "Deepseek4Free" && echo "UP" || echo "DOWN"
```

If it is **DOWN**, start it:

```bash
cd ~/Projects/deepseek4free && nohup .venv/bin/python main.py > /tmp/d4f.log 2>&1 &
sleep 3
curl -s http://localhost:11434/ | grep -q "Deepseek4Free" && echo "Server started" || echo "Failed — check /tmp/d4f.log"
```

If startup fails, show the user the last 20 lines of `/tmp/d4f.log` and stop — do not attempt
to fall back silently to Claude without telling the user.

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

For **streaming** output (preferred for large responses):

```python
with httpx.stream("POST", "http://localhost:11434/v1/chat/completions",
                  json={**payload, "stream": True}, timeout=300) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
```

Or use the **Ollama-compatible endpoint** (also works):

```bash
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": false,
    "messages": [{"role": "user", "content": "Rewrite this: ..."}]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['message']['content'])"
```

## Step 3 — Handle errors

If the server returns a non-200 status or the response contains `"error"`, log the error
and inform the user. Do not silently fall back to Claude — the user chose this skill to save
tokens, so they should know if it failed.

Common issues:
- **500 / PoW failure** — DeepSeek's anti-bot protection refreshed; the server retries
  automatically (up to 100 attempts with exponential backoff). Wait and retry.
- **Connection refused** — server didn't start; show `/tmp/d4f.log`.
- **Empty response** — the server already retries internally; if you still get empty, retry once.

## Configuration

The server reads API tokens from `~/Projects/deepseek4free/.api_keys`.
Format: one entry per line — `<label> <token>`.
Example:
```
myaccount eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Get the token from DeepSeek Chat → F12 → Application → Local Storage → `userToken`.

## API reference (quick)

| Endpoint | Protocol | Notes |
|---|---|---|
| `GET /` | — | Health check, returns `"Deepseek4Free is running"` |
| `GET /api/tags` | Ollama | Lists available models (`deepseek-chat`) |
| `POST /api/chat` | Ollama | Chat completions, streaming supported |
| `POST /api/generate` | Ollama | Single-turn generation |
| `POST /v1/chat/completions` | OpenAI | Chat completions, tool calls supported |
| `POST /v1/completions` | OpenAI legacy | Single-turn text completion |
| `GET /v1/models` | OpenAI | Lists models |
| `POST /v1/embeddings` | OpenAI | Text embeddings via BART |

Model name for all endpoints: **`deepseek-chat`**

Server port: **11434** (same as Ollama — drop-in compatible with any Ollama client)
