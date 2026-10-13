from .ai import generate, chat, image2text, text2embeddings, embeddings2text
from .tools import is_tool_call, is_tool_call_start, is_special_token_start, tools_to_instructions, parse_block, extract_tools, clean_tools
from quart import Quart, request, jsonify, Response
import json
import asyncio
import datetime
import time
import uuid
import logging
import re

app = Quart(__name__)
# Ответ может генерироваться долго (ретраи при ошибках DeepSeek) — не обрываем его по умолчанию
# 60-секундному таймауту Quart.
app.config["RESPONSE_TIMEOUT"] = None
app.config["BODY_TIMEOUT"] = None
logging.basicConfig(level=logging.WARNING)

MODEL_NAME = "deepseek-chat"


@app.after_request
async def _add_cors_headers(response):
    """Разрешает запросы из браузерных клиентов (OpenWebUI и т.п.)."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, HEAD, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@app.route("/<path:_path>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
async def _cors_preflight(_path=""):
    return "", 204

@app.route("/api/generate", methods=["POST"])
async def generate_():
    data = await request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    prompt = data.get("prompt", "")
    system = data.get("system")
    stream = data.get("stream", False)
    think = data.get("think", False)

    extra_params = ["suffix", "images", "format", "raw", "keep_alive", "options", "logprobs", "top_logprobs"]
    for p in extra_params:
        if p in data:
            logging.warning(f"Parameter '{p}' is not supported and will be ignored")

    # Формируем сообщения для модели
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if prompt:
        messages.append({"role": "user", "content": prompt})
    # Если нет ни system, ни prompt, messages останется пустым

    # Если сообщений нет, возвращаем пустой ответ без вызова generate
    if not messages:
        if stream:
            async def empty_stream():
                final_event = {
                    "model": model,
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "prompt_eval_duration": 0,
                    "eval_count": 0,
                    "eval_duration": 0,
                }
                yield json.dumps(final_event) + "\n"
            return Response(empty_stream(), mimetype="application/x-ndjson")
        else:
            return jsonify({
                "model": model,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "response": "",
                "done": True,
                "done_reason": "stop",
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0,
            })

    # Вызов асинхронной generate
    try:
        if stream:
            # Потоковый режим – возвращаем асинхронный генератор
            async def generate_stream():
                async_gen = None
                try:
                    async_gen = await generate(messages, think=think, stream=True)
                    async for thinking_part, content_part in async_gen:
                        event = {
                            "model": model,
                            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "response": content_part,
                            "done": False,
                        }
                        if think and thinking_part:
                            event["thinking"] = thinking_part
                        yield json.dumps(event) + "\n"
                    final_event = {
                        "model": model,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "response": "",
                        "done": True,
                        "done_reason": "stop",
                        "total_duration": 0,
                        "load_duration": 0,
                        "prompt_eval_count": 0,
                        "prompt_eval_duration": 0,
                        "eval_count": 0,
                        "eval_duration": 0,
                    }
                    yield json.dumps(final_event) + "\n"
                except Exception as e:
                    logging.error(f"Error in stream: {e}")
                    error_event = {
                        "model": model,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "response": "",
                        "done": True,
                        "done_reason": "error",
                        "error": str(e)
                    }
                    yield json.dumps(error_event) + "\n"
                finally:
                    if async_gen is not None:
                        try:
                            await async_gen.aclose()
                        except Exception as close_error:
                            logging.debug(f"Failed to close async generator: {close_error}")

            return Response(generate_stream(), mimetype="application/x-ndjson")
        else:
            # Непотоковый режим – получаем полный ответ
            thinking, response = await generate(messages, think=think, stream=False)
            result = {
                "model": model,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "response": response,
                "done": True,
                "done_reason": "stop",
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0,
            }
            if think:
                result["thinking"] = thinking
            return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/show", methods=["POST"])
async def show_model():
    raw_data = await request.get_data()
    data = json.loads(raw_data) if raw_data else {}
    model = data.get("model", MODEL_NAME)
    
    response = {
        "parameters": "temperature 0.7\ntop_p 0.95\ntop_k 40\nnum_ctx 256000",
        "license": "Apache License 2.0",
        "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "deepseek",
            "families": ["deepseek"],
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M"
        },
        "capabilities": [
            "completion",
            "tools",
            "thinking"
        ],
        "model_info": {
            "general.architecture": "deepseek",
            "general.file_type": 15,
            "general.parameter_count": 7000000000,
            "general.quantization_version": 2,
            "deepseek.attention.head_count": 32,
            "deepseek.attention.head_count_kv": 4,
            "deepseek.block_count": 32,
            "deepseek.context_length": 256000,
            "deepseek.embedding_length": 4096,
            "deepseek.feed_forward_length": 14336,
            "deepseek.rope.dimension_count": 128,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.eos_token_id": 100001,
            "tokenizer.ggml.padding_token_id": 100000
        }
    }
    return jsonify(response)

@app.route("/api/pull", methods=["POST"])
async def pull_model():
    data = await request.get_json()
    if not data:
        data = {}
    model = data.get("model", MODEL_NAME)
    return jsonify({"status": "success"})

@app.route("/api/create", methods=["POST"])
@app.route("/api/push", methods=["POST"])
async def create_or_push_model():
    """Стаб управления моделями: у нас всегда ровно одна виртуальная модель,
    поэтому любой create/push считается мгновенно успешным."""
    return jsonify({"status": "success"})

@app.route("/api/copy", methods=["POST"])
async def copy_model():
    return "", 200

@app.route("/api/delete", methods=["DELETE"])
async def delete_model():
    return "", 200

@app.route("/api/blobs/<digest>", methods=["HEAD"])
async def blob_exists(digest):
    """Ollama-клиенты используют HEAD, чтобы проверить, нужно ли заливать блоб
    перед /api/create — отвечаем 'уже есть', чтобы пропустить реальную заливку."""
    return "", 200

@app.route("/api/blobs/<digest>", methods=["POST"])
async def blob_upload(digest):
    return "", 201

@app.route("/api/tags", methods=["GET"])
async def list_models():
    """Fetch a list of models and their details"""
    models = [
        {
            "name": MODEL_NAME,
            "model": MODEL_NAME,
            "modified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "size": 0,
            "digest": "dummy_digest",
            "details": {
                "format": "gguf",
                "family": "deepseek",
                "families": ["deepseek"],
                "parameter_size": "unknown",
                "quantization_level": "unknown"
            }
        }
    ]
    return jsonify({"models": models})

@app.route("/api/ps", methods=["GET"])
async def ps():
    """List running models"""
    return jsonify({"models": []})

@app.route("/api/version", methods=["GET"])
async def version():
    """Get Ollama version"""
    return jsonify({"name": "Deepseek4Free", "version": "0.12.6"})

@app.route("/", methods=["GET"])
async def main():
    return "Deepseek4Free is running"

@app.route("/api/chat", methods=["POST"])
async def ollama_chat():
    start = time.time()

    raw_data = await request.get_data()
    data = json.loads(raw_data) if raw_data else {}
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "messages is required and must be a list"}), 400
    if not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    stream = data.get("stream", True)
    think = data.get("think", True)
    if isinstance(think, str):
        think = think.lower() in ("true", "1", "yes", "high", "medium", "low", "max")

    tools = data.get("tools")
    
    extra_params = ["format", "options", "keep_alive", "logprobs", "top_logprobs"]
    for p in extra_params:
        if p in data:
            logging.warning(f"Parameter '{p}' is not supported and will be ignored")

    load_duration = time.time() - start

    try:
        if stream:
            async def generate_stream():
                async_gen = None
                try:
                    pending = ""
                    async_gen = (await chat(messages, tools, think=think, stream=True))()
                    async for chunk in async_gen:
                        thinking, content, tool_calls = chunk

                        if tool_calls: break

                        response = ""
                        for symbol in content:
                            if symbol == "<" and not pending:
                                pending += symbol
                            elif symbol == ">" and is_tool_call(pending + symbol, tools):
                                pending = ""
                            elif pending and not (is_tool_call_start(pending + symbol) or is_special_token_start(pending + symbol) or is_tool_call(pending + symbol, tools)):
                                response = pending + symbol
                                pending = ""
                            elif pending:
                                pending += symbol
                            else:
                                response += symbol

                        event = {
                            "model": model,
                            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "thinking": ""
                            },
                            "done": False
                        }

                        event["message"]["thinking"] = thinking
                        event["message"]["content"] = response

                        yield json.dumps(event) + "\n"

                    final_message = {
                        "role": "assistant",
                        "content": "",
                        "thinking": ""
                    }

                    if tools:
                        final_message["tool_calls"] = []

                        if "tool" in tool_calls:
                            tool_data = tool_calls["tool"]
                            if not isinstance(tool_data, list):
                                tool_data = [tool_data]
                            for idx, call in enumerate(tool_data):
                                name = call.get("name", "")
                                args = call.get("args", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except:
                                        args = {}
                                final_message["tool_calls"].append({
                                    "id": f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "index": idx,
                                        "name": name,
                                        "arguments": args
                                    }
                                })

                    final_event = {
                        "model": model,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "message": final_message,
                        "done": False
                    }
                    yield json.dumps(final_event) + "\n"

                    final_event = {
                        "model": model,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "done": True,
                        "done_reason": "stop",
                        "total_duration": int(time.time() - start),
                        "load_duration": int(load_duration),
                        "prompt_eval_count": 0,
                        "prompt_eval_duration": 0,
                        "eval_count": 0,
                        "eval_duration": 0
                    }
                    yield json.dumps(final_event) + "\n"
                except Exception as e:
                    logging.error(f"Error in stream: {e}")
                    error_event = {
                        "model": model,
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "thinking": ""
                        },
                        "done": True,
                        "done_reason": "error",
                        "error": str(e)
                    }
                    yield json.dumps(error_event) + "\n"
                finally:
                    if async_gen is not None:
                        try:
                            await async_gen.aclose()
                        except Exception as close_error:
                            logging.debug(f"Failed to close async generator: {close_error}")

            return Response(generate_stream(), mimetype="application/x-ndjson")
        else:
            thinking, content, tool_calls = await chat(messages, tools, think=think, stream=False)

            result = {
                "model": model,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": []
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": int(time.time() - start),
                "load_duration": int(load_duration),
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": 0,
                "eval_duration": 0
            }
            if think:
                result["message"]["thinking"] = thinking
            
            if tools:
                try:
                    if "tool" in tool_calls:
                        tool_data = tool_calls["tool"]
                        if not isinstance(tool_data, list):
                            tool_data = [tool_data]
                        for idx, call in enumerate(tool_data):
                            name = call.get("name", "")
                            args = call.get("args", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    args = {}
                            result["message"]["tool_calls"].append({
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "function": {
                                    "index": idx,
                                    "name": name,
                                    "arguments": args
                                }
                            })
                        result['done_reason'] = "tool_call"
                except Exception as e:
                    logging.error(f"Error extracting tools: {e}")

            return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/embed", methods=["POST"])
async def embed():
    data = await request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400
    input_text = data.get("input")
    if not input_text:
        return jsonify({"error": "input is required"}), 400
    truncate = data.get("truncate", True)  # ignored
    dimensions = data.get("dimensions")
    keep_alive = data.get("keep_alive")  # ignored
    options = data.get("options", {})  # ignored

    if isinstance(input_text, str):
        inputs = [input_text]
    elif isinstance(input_text, list):
        inputs = input_text
    else:
        return jsonify({"error": "input must be string or array"}), 400

    start_time = time.time()
    embeddings = []
    for text in inputs:
        try:
            embedding = await text2embeddings(text, dimensions=dimensions)
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return jsonify({"error": str(e)}), 500

    total_duration = int((time.time() - start_time) * 1e9)
    load_duration = 0
    prompt_eval_count = 0

    response = {
        "model": model,
        "embeddings": embeddings,
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count
    }
    return jsonify(response)

@app.route("/api/embeddings", methods=["POST"])
async def embed_legacy():
    """Алиас на /api/embed под старым именем — некоторые версии ollama-клиентов
    ещё используют его."""
    return await embed()

@app.route("/v1/models", methods=["GET"])
async def openai_list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "deepseek",
        }],
    })

@app.route("/v1/models/<model_id>", methods=["GET"])
async def openai_get_model(model_id):
    return jsonify({
        "id": model_id or MODEL_NAME,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "deepseek",
    })

@app.route("/v1/embeddings", methods=["POST"])
async def openai_embeddings():
    raw_data = await request.get_data()
    data = json.loads(raw_data) if raw_data else {}
    model = data.get("model", MODEL_NAME)
    input_text = data.get("input")
    if not input_text:
        return jsonify({"error": "input is required"}), 400

    inputs = [input_text] if isinstance(input_text, str) else input_text
    if not isinstance(inputs, list):
        return jsonify({"error": "input must be string or array"}), 400

    dimensions = data.get("dimensions")
    embeddings_data = []
    for idx, text in enumerate(inputs):
        try:
            embedding = await text2embeddings(text, dimensions=dimensions)
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return jsonify({"error": str(e)}), 500
        embeddings_data.append({"object": "embedding", "embedding": embedding, "index": idx})

    return jsonify({
        "object": "list",
        "data": embeddings_data,
        "model": model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    })

@app.route("/v1/completions", methods=["POST"])
async def openai_completions():
    """Legacy OpenAI text-completion API поверх той же generate()-логики,
    что и /api/generate — нужен некоторым старым/минималистичным клиентам."""
    raw_data = await request.get_data()
    data = json.loads(raw_data) if raw_data else {}
    model = data.get("model", MODEL_NAME)
    prompt = data.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    stream = data.get("stream", False)

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    try:
        if stream:
            async def generate_stream():
                completion_id = f"cmpl-{uuid.uuid4().hex}"
                async_gen = await generate([{"role": "user", "content": prompt}], think=False, stream=True)
                async for chunk in async_gen:
                    _, content = chunk
                    event = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "text": content, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(event)}\n\n"
                finish_event = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(finish_event)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(generate_stream(), mimetype="text/event-stream")
        else:
            _, content = await generate([{"role": "user", "content": prompt}], think=False, stream=False)
            return jsonify({
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": content, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/chat/completions", methods=["POST"])
async def openai_chat_completions():
    start = time.time()

    raw_data = await request.get_data()
    data = json.loads(raw_data) if raw_data else {}
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    model = data.get("model")
    if not model:
        return jsonify({"error": "model is required"}), 400

    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "messages is required and must be a list"}), 400
    if not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        elif isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            msg["content"] = "".join(text_parts)
        else:
            msg["content"] = str(content)

    stream = data.get("stream", False)
    tools = data.get("tools")
    tool_choice = data.get("tool_choice", "auto")

    think = data.get("think", True)
    if isinstance(think, str):
        think = think.lower() in ("true", "1", "yes", "high", "medium", "low", "max")

    use_tools = tools and tool_choice != "none"
    forced_function_name = None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            func = tool_choice.get("function", {})
            forced_function_name = func.get("name")
            if forced_function_name:
                use_tools = True  # ensure tools are enabled

    gen_messages = messages.copy()

    if use_tools:
        tools_message = {"role": "system", "content": tools_to_instructions(tools)}
        gen_messages.insert(-2, tools_message) if len(gen_messages) > 2 else gen_messages.insert(0, tools_message)
        if forced_function_name:
            # Add explicit instruction to force calling the specified function
            force_msg = {
                "role": "system",
                "content": f"You must call the function '{forced_function_name}' with appropriate arguments. Do not call any other functions. Provide arguments as a JSON object."
            }
            
            gen_messages.insert(-2, force_msg) if len(gen_messages) > 2 else gen_messages.insert(0, force_msg)

    extra_params = ["format", "options", "keep_alive", "logprobs", "top_logprobs", "temperature", "top_p", "n", "stop", "max_tokens", "presence_penalty", "frequency_penalty", "logit_bias", "user"]
    for p in extra_params:
        if p in data:
            logging.warning(f"Parameter '{p}' is not supported and will be ignored")

    load_duration = time.time() - start

    try:
        if stream:
            async def generate_stream():
                async_gen = None
                finished = False
                try:
                    async_gen = (await chat(gen_messages, tools, think=think, stream=True))()
                    chunks_id = f"chatcmpl-{uuid.uuid4().hex}"
                    pending = []
                    async for chunk in async_gen:
                        thinking, content, tool_calls = chunk

                        event = {
                            "id": chunks_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "system_fingerprint": "fp_ollama",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning": "",
                                    "content": ""
                                },
                                "finish_reason": None
                            }]
                        }

                        if thinking:
                            event["choices"][0]["delta"]["reasoning"] = thinking
                            
                            yield f"data: {json.dumps(event)}\n\n"
                            
                        if content:
                            if tools:
                                pending.append(content)
                            else:
                                event["choices"][0]["delta"]["content"] = content
                                
                                yield f"data: {json.dumps(event)}\n\n"

                        if tools and tool_calls:
                            tools_event = {
                                "id": event["id"],
                                "object": "chat.completion.chunk",
                                "created": event["created"],
                                "model": model,
                                "system_fingerprint": "fp_ollama",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": "",
                                            "tool_calls": []
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }

                            if "tool" in tool_calls:
                                tool_data = tool_calls["tool"]
                                if not isinstance(tool_data, list):
                                    tool_data = [tool_data]
                                for idx, call in enumerate(tool_data):
                                    name = call.get("name", "")
                                    args = call.get("args", {})
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except Exception:
                                            args = {}
                                    tools_event["choices"][0]["delta"]["tool_calls"].append({
                                        "id": f"call_{uuid.uuid4().hex[:8]}",
                                        "index": idx,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(args)
                                        }
                                    })

                            yield f"data: {json.dumps(tools_event)}\n\n"

                            finish_event = {
                                "id": event["id"],
                                "object": "chat.completion.chunk",
                                "created": event["created"],
                                "model": model,
                                "system_fingerprint": "fp_ollama",
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": []
                                    },
                                    "finish_reason": "tool_calls"
                                }]
                            }

                            yield f"data: {json.dumps(finish_event)}\n\n"
                            finished = True
                            break

                    if not finished:
                        for content in pending:
                            event = {
                                "id": chunks_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "system_fingerprint": "fp_ollama",
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "reasoning": "",
                                        "content": content
                                    },
                                    "finish_reason": None
                                }]
                            }

                            yield f"data: {json.dumps(event)}\n\n"

                        finish_event = {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(finish_event)}\n\n"

                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logging.error(f"Error in OpenAI stream: {e}")
                    error_event = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error"
                        }]
                    }
                    yield f"data: {json.dumps(error_event)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    if async_gen is not None:
                        try:
                            await async_gen.aclose()
                        except Exception as close_error:
                            logging.debug(f"Failed to close async generator: {close_error}")

            return Response(generate_stream(), mimetype="text/event-stream")
        else:
            thinking, content, tool_calls = await chat(gen_messages, tools, think=think, stream=False)
            content = content or ""
            thinking = thinking or ""

            message = {
                "role": "assistant",
                "content": content,
            }
            if think and thinking:
                message["reasoning_content"] = thinking

            tool_call_payloads = []
            finish_reason = "stop"
            if tools and tool_calls:
                if "tool" in tool_calls:
                    tool_data = tool_calls["tool"]
                    if not isinstance(tool_data, list):
                        tool_data = [tool_data]
                    for idx, call in enumerate(tool_data):
                        name = call.get("name", "")
                        args = call.get("args", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        tool_call_payloads.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "index": idx,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args)
                            }
                        })
                    if tool_call_payloads:
                        message["tool_calls"] = tool_call_payloads
                        finish_reason = "tool_calls"

            result = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
