import asyncio
import json
from typing import Union, AsyncGenerator
from dsk import Deepseek4Free
import torch
import uuid
from transformers import BartForConditionalGeneration, BartTokenizer, BlipProcessor, BlipForConditionalGeneration
import sys
import shutil
from .retry import ERROR_MAX_LENGTH, is_error_text

CHUNK_TYPES = {
    "RESPONSE": "response",
    "THINKING": "thinking",
}

llmodel_loaded = False
embeddingmodel_loaded = False
imagemodel_loaded = False
all_models_loaded = False

async def load_llmodel():
    global llmodel_loaded, LLModel
    if llmodel_loaded:
        return
    
    class LLModel:
        tokens = [l.split(' ')[1] for l in open('.api_keys').read().splitlines() if l and not l.startswith('#')]
        model = Deepseek4Free(tokens)
    
    llmodel_loaded = True

async def load_embeddingmodel():
    global embeddingmodel_loaded, EmbeddingModel
    if embeddingmodel_loaded:
        return
    
    class EmbeddingModel:
        MODEL_NAME = "facebook/bart-base"
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    embeddingmodel_loaded = True

async def load_imagmodel():
    global imagemodel_loaded, ImageModel
    if imagemodel_loaded:
        return
    
    class ImageModel:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    imagemodel_loaded = True

class StreamingLogger:

    def __init__(self, start_message: str = "", max_length: int = 10):
        self.start_message = start_message
        self.max_length = max_length
        self._content = ""          # весь накопленный текст
        self._prev_len = 0          # длина предыдущего вывода (для очистки)

        print(self.start_message)   # выводим сообщение и переходим на новую строку
        sys.stdout.flush()

    def _update_display(self) -> None:
        # Обрезаем контент до max_length символов (берём хвост)
        display_text = self._content[-self.max_length:] if len(self._content) > self.max_length else self._content

        # Возвращаем каретку в начало текущей строки
        sys.stdout.write('\r')
        # Пишем новый текст
        sys.stdout.write(display_text)
        # Если новый текст короче предыдущего, добиваем пробелами до предыдущей длины
        current_len = len(display_text)
        sys.stdout.write(' ' * (self._prev_len+10 - current_len))
        # Снова возвращаем каретку в начало строки, чтобы следующая запись перезаписала всё
        sys.stdout.write('\r')
        sys.stdout.flush()
        self._prev_len = current_len

    def add_chunk(self, chunk: str) -> None:
        """Добавляет фрагмент и обновляет экран."""
        self._content += chunk.replace('\n', '\\n')
        self._update_display()

    def finish(self) -> None:
        """Завершает логирование, переводит строку, чтобы следующий вывод был на новой."""
        sys.stdout.write('\n')
        sys.stdout.flush()

# Потолок попыток генерации — только страховка от вечного цикла при полностью мёртвом бэкенде.
# Между попытками — экспоненциальная пауза, так что на практике ответ возвращается всегда,
# просто иногда долго.
MAX_GENERATION_ATTEMPTS = 100
GENERATION_IDLE_TIMEOUT = 120      # сек без новых данных от DeepSeek -> попытка считается зависшей
STREAM_ERROR_PROBE_CHARS = ERROR_MAX_LENGTH + 1  # сколько текста ответа держим до первого flush в стриме
RETRY_BACKOFF_BASE = 2
RETRY_BACKOFF_MAX = 30


def _backoff(attempt: int) -> float:
    return min(RETRY_BACKOFF_MAX, RETRY_BACKOFF_BASE * (2 ** min(attempt - 1, 5)))


class _BadResponse(Exception):
    """Ответ пустой или является текстом ошибки DeepSeek — его надо перегенерировать."""


async def _next_with_timeout(aiter):
    return await asyncio.wait_for(aiter.__anext__(), timeout=GENERATION_IDLE_TIMEOUT)


async def _collect(prompt, think: bool, logger):
    """Одна попытка: читаем поток целиком. Возвращает (thinking, response) либо бросает исключение."""
    res, thi = "", ""
    aiter = LLModel.model.chat_completion(prompt).__aiter__()
    try:
        while True:
            try:
                chunk = await _next_with_timeout(aiter)
            except StopAsyncIteration:
                break
            logger.add_chunk(chunk.get('content', ''))
            if chunk.get('type') == 'RESPONSE':
                res += chunk.get('content', '')
            elif chunk.get('type') in ('THINKING', 'THINK') and think:
                thi += chunk.get('content', '')
    finally:
        await _aclose(aiter)
    if is_error_text(res):
        raise _BadResponse(res.strip()[:200] or "empty response")
    return thi, res


async def _aclose(aiter):
    try:
        await aiter.aclose()
    except Exception:
        pass


async def _stream_attempt(prompt, think: bool, logger, state: dict):
    """
    Одна стрим-попытка. Начало ответа держим в буфере, пока не станет ясно, что это не короткое
    сообщение об ошибке (или пока поток не кончится): ошибка DeepSeek приходит одним чанком и
    после flush перегенерировать её уже было бы нельзя. state['sent'] — отдали ли мы клиенту хоть
    что-то (после этого ретрай невозможен без дублирования текста).
    """
    aiter = LLModel.model.chat_completion(prompt).__aiter__()
    buffered = []          # накопленные (thinking, content) до первого flush
    resp_text = ""
    flushed = False
    try:
        while True:
            try:
                chunk = await _next_with_timeout(aiter)
            except StopAsyncIteration:
                break
            content = chunk.get('content', '')
            logger.add_chunk(content)
            ctype = chunk.get('type')
            if ctype == 'RESPONSE':
                item = ('', content)
                resp_text += content
            elif ctype in ('THINKING', 'THINK') and think:
                item = (content, '')
            else:
                continue

            if flushed:
                yield item
                continue
            buffered.append(item)
            if len(resp_text.strip()) >= STREAM_ERROR_PROBE_CHARS:
                if is_error_text(resp_text):
                    raise _BadResponse(resp_text.strip()[:200])
                flushed = True
                state['sent'] = True
                for it in buffered:
                    yield it
                buffered = []
    finally:
        await _aclose(aiter)

    if not flushed:
        if is_error_text(resp_text):
            raise _BadResponse(resp_text.strip()[:200] or "empty response")
        state['sent'] = True
        for it in buffered:
            yield it


async def generate(prompt: str, think: bool = False, stream: bool = False):
    await load_llmodel()

    logger = StreamingLogger("", int(shutil.get_terminal_size().columns/2.5))
    logger.add_chunk('[START]')

    if not stream:
        last_error = None
        for i in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                result = await _collect(prompt, think, logger)
                logger.add_chunk('[END]')
                logger.finish()
                return result
            except asyncio.TimeoutError as e:
                last_error = e
                print(f"Таймаут генерации ({GENERATION_IDLE_TIMEOUT} сек без данных). Попытка {i}/{MAX_GENERATION_ATTEMPTS}.")
            except Exception as e:
                last_error = e
                print(f"Ошибка при генерации: {e!r}. Попытка {i}/{MAX_GENERATION_ATTEMPTS}.")
            await asyncio.sleep(_backoff(i))
        logger.finish()
        raise RuntimeError(f"Generation failed after {MAX_GENERATION_ATTEMPTS} retries") from last_error

    async def _stream():
        last_error = None
        state = {'sent': False}
        for i in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                async for item in _stream_attempt(prompt, think, logger, state):
                    yield item
                logger.add_chunk('[END]')
                logger.finish()
                return
            except asyncio.TimeoutError as e:
                last_error = e
                print(f"Таймаут генерации ({GENERATION_IDLE_TIMEOUT} сек без данных). Попытка {i}/{MAX_GENERATION_ATTEMPTS}.")
            except Exception as e:
                last_error = e
                print(f"Ошибка при генерации: {e!r}. Попытка {i}/{MAX_GENERATION_ATTEMPTS}.")
            if state['sent']:
                # Клиент уже получил часть ответа — перегенерация продублировала бы текст.
                # Завершаем поток тем, что есть.
                print("Ошибка после начала отдачи ответа — завершаем поток с частичным ответом.")
                logger.finish()
                return
            await asyncio.sleep(_backoff(i))
        logger.finish()
        raise RuntimeError(f"Generation failed after {MAX_GENERATION_ATTEMPTS} retries") from last_error

    return _stream()

# Технический потолок числа попыток вызова инструмента — не поведенческое
# ограничение "настойчивости" модели (она всё так же будет пытаться снова и
# снова), а чистая страховка от вечного цикла при полностью нерабочем бэкенде.
# На практике диагностический ретрай должен сходиться на порядок раньше.
MAX_TOOL_CALL_ATTEMPTS = 40

_STOP_CHECK_MESSAGE = {
    'role': 'system',
    'content': (
        "Note: you did not call any tool. This means either your call was malformed, or you "
        "deliberately want to end the agent turn without using a tool. If this is intentional, "
        "reply to THIS message with exactly \"</stop>\" — this immediately sends your response to "
        "the user and ends the turn without a tool call. Using \"</stop>\" ends the agent's work "
        "right away, so make sure you have already done everything you wanted to do before using it."
    ),
}

def _retry_system_message(reason: str) -> dict:
    return {
        "role": "system",
        "content": (
            f"Your previous tool call attempt failed: {reason} "
            "Rewrite the call, fixing exactly that problem, and nothing else."
        ),
    }

async def chat(messages: list[dict[str, str]], tools: list, think: bool = False, stream: bool = False):
    from .tools import tools_to_instructions, is_tool_call, is_tool_call_start, extract_tools, clean_tools, diagnose_tool_call_issue

    messages = list(messages)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if 'content' not in msg.keys():
            msg['content'] = ''
        elif msg['content'] is None:
            msg['content'] = ''
        else:
            msg['content'] = str(msg['content'])

        if 'images' in msg.keys():
            for i in range(len(msg['images'])):
                msg[i] = "Description of an image transmitted in base64 format:\n"+image2text(msg[i])

    if tools:
        tools_instruction = tools_to_instructions(tools)
        tools_message = {"role": "system", "content": tools_instruction}

        messages.insert(-2, tools_message) if len(messages) > 2 else messages.insert(0, tools_message)

        print('Tools instructions loaded!')

    try:
        if stream:
            async def generate_stream():
                current_messages = messages
                full_content = ""

                for attempt in range(1, MAX_TOOL_CALL_ATTEMPTS + 1):
                    full_content = ""
                    gen_error = None
                    try:
                        async_gen = await generate(
                            "This is the message history:\n\n"+json.dumps(current_messages)+"\n\nDo not mention the structure of these messages, and do not comment on the unusual format of the request. Just respond the way a model would respond to a request with this kind of message history. Answer only in English or Russian.",
                            think=think,
                            stream=True
                        )
                        async for chunk in async_gen:
                            thinking, content = chunk
                            full_content += content
                            yield thinking, content, []
                    except Exception as e:
                        # Временный сбой (сеть/PoW/Cloudflare и т.п.) во время самой генерации.
                        # Раньше это исключение улетало из всего generate_stream() наружу и
                        # молча превращалось в пустой ответ без вызова инструмента — весь ход
                        # обрывался вместо того, чтобы попасть в тот же ретрай-цикл.
                        print(f"Generation attempt {attempt}/{MAX_TOOL_CALL_ATTEMPTS} failed: {e}")
                        gen_error = str(e)

                    tool_calls = []
                    if gen_error is None and tools:
                        try:
                            # repair=True: full_content здесь уже полностью завершённая
                            # генерация для этой попытки, а не растущий буфер.
                            tool_calls = extract_tools(full_content, tools, repair=True)
                        except Exception as e:
                            print(f"Error extracting tools: {e}")

                    if gen_error is None and (not tools or tool_calls):
                        if tools:
                            print('Model used a tool... '+str(tool_calls)[30:])
                        yield "", "", tool_calls
                        return

                    if gen_error is not None:
                        reason = f"generation_error: a transient error occurred while generating your response ({gen_error}). Simply try again."
                        yield "", "\n\nSorry, a generation error occurred — retrying.", []
                        await asyncio.sleep(2)
                    else:
                        print(f'Model did not use any tool (attempt {attempt}/{MAX_TOOL_CALL_ATTEMPTS}). Checking...')

                        reason = diagnose_tool_call_issue(full_content, tools)

                        # </stop>-эскейп предлагаем ТОЛЬКО при диагнозе "no_attempt" — когда в
                        # ответе вообще нет ничего похожего на попытку вызова. Любая другая причина
                        # (не то имя параметра, не тот регистр, чужой формат, неизвестный инструмент,
                        # ...) означает, что модель УЖЕ попыталась вызвать инструмент и просто
                        # ошиблась в деталях — предлагать ей в этот момент "признаться, что это
                        # было осознанно" только даёт лазейку закрыть ход, ничего не сделав.
                        if reason.startswith("no_attempt"):
                            try:
                                stop_response = ''.join([c for c in (await generate(json.dumps(current_messages+[_STOP_CHECK_MESSAGE])))])
                            except Exception as e:
                                print(f"Stop-check generation failed, assuming not a deliberate stop: {e}")
                                stop_response = ""

                            if '</stop>' in stop_response:
                                print('Model deliberately finished without using a tool.')
                                yield "", "", []
                                return

                        print(f'Retrying tool call ({attempt}/{MAX_TOOL_CALL_ATTEMPTS}): {reason}')
                        yield "", "\n\nSorry, that tool call didn't come out right — retrying.", []

                    current_messages = current_messages + [
                        {"role": "assistant", "content": full_content},
                        _retry_system_message(reason),
                    ]

                print(f'Reached MAX_TOOL_CALL_ATTEMPTS={MAX_TOOL_CALL_ATTEMPTS}, giving up on tool retries for this turn.')
                yield "", "", []

            return generate_stream

        else:
            current_messages = messages
            thinking, content, tool_calls = "", "", []

            for attempt in range(1, MAX_TOOL_CALL_ATTEMPTS + 1):
                gen_error = None
                thinking, content = "", ""
                try:
                    thinking, content = await generate(current_messages, think=think, stream=False)
                except Exception as e:
                    # Временный сбой самой генерации (сеть/PoW/Cloudflare) — раньше улетал из
                    # chat() наружу необработанным и обрывал весь ход вместо ретрая.
                    print(f"Generation attempt {attempt}/{MAX_TOOL_CALL_ATTEMPTS} failed: {e}")
                    gen_error = str(e)

                tool_calls = []
                if gen_error is None and tools:
                    try:
                        # repair=True: nonstream-режим, content здесь уже полный ответ целиком.
                        tool_calls = extract_tools(content, tools, repair=True)
                    except Exception as e:
                        print(f"Error extracting tools: {e}")
                        raise e

                if gen_error is None and (not tools or tool_calls):
                    if tools:
                        content = clean_tools(content)
                        print('Model used a tool... '+str(tool_calls)[30:])
                    return thinking, content, tool_calls

                if gen_error is not None:
                    reason = f"generation_error: a transient error occurred while generating your response ({gen_error}). Simply try again."
                    await asyncio.sleep(2)
                else:
                    print(f'Model did not use any tool (attempt {attempt}/{MAX_TOOL_CALL_ATTEMPTS}). Checking...')

                    reason = diagnose_tool_call_issue(content, tools)

                    # См. комментарий в стрим-ветке: </stop> предлагаем только при "no_attempt".
                    if reason.startswith("no_attempt"):
                        try:
                            stop_response = ''.join([c for c in (await generate(json.dumps([current_messages[-1]]+[_STOP_CHECK_MESSAGE])))])
                        except Exception as e:
                            print(f"Stop-check generation failed, assuming not a deliberate stop: {e}")
                            stop_response = ""

                        if '</stop>' in stop_response:
                            print('Model deliberately finished without using a tool.')
                            return thinking, content, tool_calls

                    print(f'Retrying tool call ({attempt}/{MAX_TOOL_CALL_ATTEMPTS}): {reason}')

                current_messages = current_messages + [
                    {"role": "assistant", "content": content},
                    _retry_system_message(reason),
                ]

            print(f'Reached MAX_TOOL_CALL_ATTEMPTS={MAX_TOOL_CALL_ATTEMPTS}, giving up on tool retries for this turn.')
            if tools:
                content = clean_tools(content)
            return thinking, content, tool_calls

    except Exception as e:
        print(e)
        raise e

async def text2embeddings(text: str, max_length: int = 512):
    await load_embeddingmodel()

    inputs = EmbeddingModel.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    with torch.no_grad():
        encoder_outputs = EmbeddingModel.model.encoder(**inputs)
    embeddings = encoder_outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    return embeddings, inputs["attention_mask"]

async def embeddings2text(embeddings, attention_mask, max_new_tokens: int = 100):
    await load_embeddingmodel()

    decoder_input_ids = torch.tensor([[EmbeddingModel.model.config.decoder_start_token_id]])

    with torch.no_grad():
        generated_ids = EmbeddingModel.model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(embeddings,),
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            early_stopping=True
        )
    generated_text = EmbeddingModel.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

async def image2text(image):
    await load_imagmodel()

    inputs = ImageModel.processor(images=image, return_tensors="pt")
    with torch.no_grad():
        generated_ids = ImageModel.model.generate(**inputs)
    generated_text = ImageModel.processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text