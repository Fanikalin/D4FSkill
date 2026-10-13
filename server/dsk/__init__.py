import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Union

from .api import DeepSeekAPI


__all__ = [
    'Deepseek4Free',
    'DeepSeekAPI',
]


class Deepseek4Free:
    """Wrapper for DeepSeek API using multiple auth tokens in round-robin order."""

    def __init__(self, tokens: Union[str, Sequence[str]]):
        if isinstance(tokens, str):
            tokens = [tokens]

        if not isinstance(tokens, (list, tuple)):
            raise ValueError('tokens must be a string or an iterable of strings')

        self._tokens: List[str] = [t for t in tokens if isinstance(t, str) and t.strip()]
        if not self._tokens:
            raise ValueError('At least one valid auth token is required')

        self._apis: List[DeepSeekAPI] = [DeepSeekAPI(token.strip()) for token in self._tokens]
        self._sessions: List[Optional[str]] = [None] * len(self._apis)
        self._lock = asyncio.Lock()
        self._next_index = 0

    async def _get_api(self) -> DeepSeekAPI:
        """Get the next API instance without creating a session."""
        async with self._lock:
            api = self._apis[self._next_index]
            self._next_index = (self._next_index + 1) % len(self._apis)
        return api

    async def _get_api_and_session(self) -> tuple[DeepSeekAPI, str]:
        """
        Get the next API instance and its associated session.
        If the session doesn't exist yet, create one.
        """
        async with self._lock:
            api = self._apis[self._next_index]
            idx = self._next_index
            self._next_index = (self._next_index + 1) % len(self._apis)

        session_id = self._sessions[idx]
        if session_id is None:
            session_id = await api.create_chat_session()
            self._sessions[idx] = session_id

        return api, session_id

    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: str = 'application/octet-stream'
    ) -> Dict[str, Any]:
        """Uploads a file using the next available token."""
        api = await self._get_api()
        return await api.upload_file(file_data, filename, content_type)

    async def chat_completion(
        self,
        messages: str,
        thinking_enabled: bool = True,
        search_enabled: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a chat completion request using the next available token.
        messages: Either a single string (user message) or a list of message dicts,
                  e.g. [{"role": "user", "content": "Hello"}]
        Returns an async generator yielding response chunks.
        """
        # Normalize messages to a list of dicts
        if isinstance(messages, str):
            if not messages.strip():
                raise ValueError("messages must be a non-empty string")
            messages_list = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            if not messages:
                raise ValueError("messages must be a non-empty list")
            # Ensure each item is a dict with at least 'content'
            for item in messages:
                if not isinstance(item, dict) or 'content' not in item:
                    raise ValueError("Each message must be a dict with 'content' field")
            messages_list = messages
        else:
            raise TypeError("messages must be a string or a list of dicts")

        # Технический потолок ретраев на "please try again later" — страховка от
        # бесконечной рекурсии/зависания при постоянно занятом бэкенде, а не
        # ограничение на количество попыток модели вызвать инструмент.
        max_busy_retries = 20
        error_msg = "A message is being generated, please try again later."

        for busy_attempt in range(1, max_busy_retries + 1):
            api, session_id = await self._get_api_and_session()
            prompt_json = json.dumps(messages_list)

            # parent_message_id is always None as requested
            stream = api.chat_completion(
                chat_session_id=session_id,
                prompt=prompt_json,
                parent_message_id=None,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            )

            hit_busy_error = False
            try:
                chunks_iter = stream.__aiter__()
                # Yield only the content chunks (text) from the stream
                async for chunk in chunks_iter:
                    if (isinstance(chunk, dict) and chunk.get('type') == 'error' and error_msg in chunk.get('content', '')) or \
                       (isinstance(chunk, str) and error_msg in chunk):
                        hit_busy_error = True
                        break
                    if chunk.get('type') in ('RESPONSE', 'THINK', 'SEARCH') and chunk.get('content'):
                        yield chunk
                    elif chunk.get('type') == 'FINISHED':
                        # Propagate finish signal
                        yield chunk
                        return
            except Exception:
                # Сессия могла протухнуть/сломаться — следующая попытка создаст новую.
                self._sessions[self._apis.index(api)] = None
                raise

            if not hit_busy_error:
                return

            await asyncio.sleep(5)

    async def close(self) -> None:
        """Closes all underlying API clients."""
        await asyncio.gather(*((await api.close()) for api in self._apis))
