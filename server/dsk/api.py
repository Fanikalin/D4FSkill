import json
import logging
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator, Literal
from dataclasses import dataclass
from pathlib import Path

import httpx
from .pow import DeepSeekPOW

logger = logging.getLogger(__name__)

ThinkingMode = Literal['detailed', 'simple', 'disabled']
SearchMode = Literal['enabled', 'disabled']


@dataclass
class ParseState:
    last_fragment_type: Optional[str] = None
    last_content_path: Optional[str] = None


class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors"""
    pass


class AuthenticationError(DeepSeekError):
    """Raised when authentication fails"""
    pass


class RateLimitError(DeepSeekError):
    """Raised when API rate limit is exceeded"""
    pass


class NetworkError(DeepSeekError):
    """Raised when network communication fails"""
    pass


class CloudflareError(DeepSeekError):
    """Raised when Cloudflare blocks the request"""
    pass


class APIError(DeepSeekError):
    """Raised when API returns an error response"""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class DeepSeekAPI:
    BASE_URL = "https://chat.deepseek.com/api/v0"

    def __init__(self, auth_token: str):
        if not auth_token or not isinstance(auth_token, str):
            raise AuthenticationError("Invalid auth token provided")

        self.auth_token = auth_token
        self.pow_solver = DeepSeekPOW()

        # Load cookies from JSON file (no automatic refresh)
        cookies_path = Path(__file__).parent / 'cookies.json'
        try:
            with open(cookies_path, 'r') as f:
                cookie_data = json.load(f)
                self.cookies = cookie_data.get('cookies', {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load cookies from {cookies_path}: {e}")
            logger.warning("Please ensure cookies.json is valid. Use a separate script to obtain fresh cookies.")
            self.cookies = {}

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None), follow_redirects=True)

    def _get_headers(self, pow_response: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'accept': '*/*',
            'accept-language': 'en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3',
            'authorization': f'Bearer {self.auth_token}',
            'content-type': 'application/json',
            'origin': 'https://chat.deepseek.com',
            'referer': 'https://chat.deepseek.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
            'x-app-version': '20241129.1',
            'x-client-locale': 'en_US',
            'x-client-platform': 'web',
            'x-client-version': '1.0.0-always',
        }
        if pow_response:
            headers['x-ds-pow-response'] = pow_response
        return headers

    async def _make_request(self, method: str, endpoint: str, json_data: Dict[str, Any], pow_required: bool = False) -> Any:
        url = f"{self.BASE_URL}{endpoint}"
        retry_count = 0
        max_retries = 2

        while retry_count < max_retries:
            try:
                headers = self._get_headers()
                if pow_required:
                    # solve_challenge is synchronous – run in thread
                    challenge = await self._get_pow_challenge()
                    pow_response = await asyncio.to_thread(self.pow_solver.solve_challenge, challenge)
                    headers = self._get_headers(pow_response)

                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    cookies=self.cookies
                )

                # Check for Cloudflare protection
                if response.status_code == 403 and "Just a moment" in response.text:
                    logger.warning("Cloudflare protection detected.")
                    raise CloudflareError("Cloudflare protection detected. Please update cookies.json manually.")

                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                elif response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                elif response.status_code >= 500:
                    raise APIError(f"Server error occurred: {response.text}", response.status_code)
                elif response.status_code != 200:
                    raise APIError(f"API request failed: {response.text}", response.status_code)

                return response.json()

            except (httpx.RequestError, httpx.TimeoutException) as e:
                raise NetworkError(f"Network error occurred: {str(e)}")
            except json.JSONDecodeError:
                raise APIError("Invalid JSON response from server")
            except CloudflareError:
                # No automatic retry – propagate
                raise

        raise APIError("Failed to bypass Cloudflare protection after multiple attempts")

    async def _get_pow_challenge(self, target_path: str = '/api/v0/chat/completion') -> Dict[str, Any]:
        response = await self._make_request(
            'POST',
            '/chat/create_pow_challenge',
            {'target_path': target_path},
            pow_required=False
        )
        try:
            return response['data']['biz_data']['challenge']
        except (KeyError, TypeError) as e:
            raise APIError(f"Invalid challenge response: {response}") from e

    async def create_chat_session(self) -> str:
        """Creates a new chat session and returns the session ID"""
        response = await self._make_request(
            'POST',
            '/chat_session/create',
            {'character_id': None}
        )
        try:
            return response['data']['biz_data']['id']
        except (KeyError, TypeError) as e:
            raise APIError(f"Invalid session creation response: {response}") from e

    async def delete_chat_session(self, chat_session_id: str) -> None:
        """Deletes a chat session to avoid resource leaks"""
        try:
            await self._make_request(
                'POST',
                '/chat_session/delete',
                {'chat_session_id': chat_session_id}
            )
            logger.debug(f"Deleted chat session {chat_session_id}")
        except Exception as e:
            logger.warning(f"Failed to delete chat session {chat_session_id}: {e}")

    async def upload_file(self, file_data: bytes, filename: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        challenge = await self._get_pow_challenge(target_path='/api/v0/file/upload_file')
        pow_response = await asyncio.to_thread(self.pow_solver.solve_challenge, challenge)

        headers = self._get_headers(pow_response)
        headers['x-file-size'] = str(len(file_data))
        headers['x-model-type'] = 'default'
        headers['x-thinking-enabled'] = '1'
        headers.pop('content-type', None)

        url = f"{self.BASE_URL}/file/upload_file"
        files = {'file': (filename, file_data, content_type)}

        try:
            response = await self._client.post(
                url,
                headers=headers,
                files=files,
                cookies=self.cookies
            )
            if response.status_code != 200:
                raise APIError(f"File upload failed: {response.text}", response.status_code)
            return response.json()
        except httpx.RequestError as e:
            raise NetworkError(f"Network error during file upload: {str(e)}")

    async def chat_completion(
            self,
            chat_session_id: str,
            prompt: str,
            parent_message_id: Optional[str] = None,
            thinking_enabled: bool = True,
            search_enabled: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message and get streaming response as async generator.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not chat_session_id or not isinstance(chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")

        state = ParseState()
        # prompt is already a JSON string (array of messages) – we'll parse it to ensure it's valid
        try:
            messages = json.loads(prompt)
        except json.JSONDecodeError:
            raise ValueError("Prompt is not a valid JSON string")

        json_data = {
            'chat_session_id': chat_session_id,
            'parent_message_id': parent_message_id,
            'prompt': prompt,
            'messages': messages,
            'ref_file_ids': [],
            'thinking_enabled': thinking_enabled,
            'search_enabled': search_enabled,
        }

        # Obtain PoW response (synchronous, but run in thread)
        challenge = await self._get_pow_challenge()
        pow_response = await asyncio.to_thread(self.pow_solver.solve_challenge, challenge)
        headers = self._get_headers(pow_response)

        url = f"{self.BASE_URL}/chat/completion"
        try:
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=json_data,
                cookies=self.cookies
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_text = error_text.decode('utf-8', 'ignore')[:200]
                    if response.status_code == 401:
                        raise AuthenticationError("Invalid or expired authentication token")
                    elif response.status_code == 429:
                        raise RateLimitError("API rate limit exceeded")
                    else:
                        raise APIError(f"API request failed: {error_text}", response.status_code)

                finished = False
                yielded_content = ""
                line_iterator = response.aiter_lines()
                try:
                    async for line in line_iterator:
                        if not line:
                            continue
                        parsed = self._parse_chunk(line.encode('utf-8'), state)
                        if parsed is None:
                            continue

                        if '｜' in parsed.get('content', ''):
                            # Похоже на спецтокены модели (напр. нативные токены DeepSeek),
                            # протёкшие в обычный текстовый фрагмент. Логируем на видимом
                            # по умолчанию уровне, чтобы при повторении было что разбирать.
                            logger.warning(f"Special-token-looking content leaked into a '{parsed.get('type')}' fragment: {line[:500]!r}")

                        if parsed.get('type') == 'FINISHED':
                            finished = True
                            yield parsed
                            continue

                        raw_content = parsed.get('content', '')
                        if raw_content.strip() == 'FINISHED':
                            logger.warning("Ignored standalone 'FINISHED' status chunk")
                            continue

                        if parsed.get('type') == 'THINK':
                            if not finished:
                                yield parsed
                        elif parsed.get('type') == 'RESPONSE' and raw_content:
                            if not finished:
                                yield parsed
                                yielded_content += raw_content
                            elif raw_content.startswith(yielded_content):
                                new_part = raw_content[len(yielded_content):]
                                if new_part:
                                    logger.debug(f"Consolidated content delta: {repr(new_part)}")
                                    yield {'content': new_part, 'type': 'RESPONSE', 'finish_reason': None}
                            else:
                                logger.debug("Consolidated content unrelated to streamed response, skipping")
                finally:
                    if hasattr(line_iterator, "aclose"):
                        await line_iterator.aclose()

        except httpx.RequestError as e:
            raise NetworkError(f"Network error occurred during streaming: {str(e)}")

    # The parse_chunk method remains essentially the same, but uses logging instead of print
    def _parse_chunk(self, chunk: bytes, state: ParseState) -> Optional[Dict[str, Any]]:
        """Parse a native Deepseek SSE chunk format"""
        if not chunk:
            return None

        try:
            if chunk.startswith(b'data: '):
                raw_data = chunk[6:]
                logger.debug(f"Raw chunk: {raw_data[:500]}")

                data = json.loads(raw_data)

                # Check for fragment append
                if data.get('o') == 'APPEND' and data.get('p') == 'response/fragments':
                    fragments = data.get('v', [])
                    if fragments and isinstance(fragments, list):
                        last_frag = fragments[-1]
                        if isinstance(last_frag, dict):
                            state.last_fragment_type = last_frag.get('type')
                            logger.debug(f"Fragment added, type: {state.last_fragment_type}")
                            # Новый фрагмент приходит уже с первым куском текста внутри
                            # (напр. {"type":"RESPONSE","content":"Hello"}) — его нельзя терять,
                            # иначе у ответа пропадает первый токен (например, '<' от '<tool>').
                            frag_content = last_frag.get('content')
                            if isinstance(frag_content, str) and frag_content:
                                return {
                                    'content': frag_content,
                                    'type': state.last_fragment_type or 'RESPONSE',
                                    'finish_reason': None
                                }
                        return None

                content_path = data.get('p', '')
                if content_path in ('response/thinking_content', 'response/content'):
                    state.last_content_path = content_path

                # Ready event
                if 'request_message_id' in data and 'response_message_id' in data:
                    logger.debug(f"Ready event: request={data['request_message_id']}, response={data['response_message_id']}")
                    return {
                        'type': 'READY',
                        'request_message_id': data['request_message_id'],
                        'response_message_id': data['response_message_id']
                    }

                # SET operations
                if data.get('o') == 'SET' and content_path in ('response/thinking_content', 'response/content'):
                    state.last_content_path = content_path
                    state.last_fragment_type = 'THINK' if 'thinking' in content_path else 'RESPONSE'

                # Handle first chunk with embedded response object
                v_field = data.get('v')
                if isinstance(v_field, dict) and 'response' in v_field:
                    response_data = v_field['response']
                    fragments = response_data.get('fragments', [])
                    if fragments and isinstance(fragments, list):
                        last_frag = fragments[-1]
                        if isinstance(last_frag, dict):
                            frag_type = last_frag.get('type')
                            if frag_type:
                                state.last_fragment_type = frag_type
                                logger.debug(f"First chunk fragment type: {state.last_fragment_type}")
                    initial_thinking = response_data.get('thinking_content', '')
                    initial_content = response_data.get('content', '')
                    # Новый формат: начальный текст лежит не в thinking_content/content,
                    # а в content последнего фрагмента.
                    if not initial_thinking and not initial_content and fragments and isinstance(fragments[-1], dict):
                        frag_content = fragments[-1].get('content')
                        if isinstance(frag_content, str) and frag_content:
                            return {
                                'content': frag_content,
                                'type': state.last_fragment_type or 'RESPONSE',
                                'finish_reason': None
                            }
                    if initial_thinking:
                        state.last_content_path = 'response/thinking_content'
                        logger.debug(f"Initial thinking from response object: {repr(initial_thinking)}")
                        return {
                            'content': initial_thinking,
                            'type': 'THINK',
                            'finish_reason': None
                        }
                    if initial_content:
                        logger.debug(f"Initial content from response object: {repr(initial_content)}")
                        return {
                            'content': initial_content,
                            'type': 'RESPONSE',
                            'finish_reason': None
                        }

                # Response finished
                if data.get('p') == 'response/status' and data.get('v') == 'FINISHED':
                    logger.debug("Response finished")
                    state.last_content_path = None
                    state.last_fragment_type = None
                    return {
                        'content': '',
                        'type': 'FINISHED',
                        'finish_reason': 'stop'
                    }

                # Consolidated content chunk (full answer after FINISHED)
                if 'content' in data and isinstance(data.get('content'), str) and data['content']:
                    chunk_type = 'RESPONSE'
                    if state.last_content_path == 'response/thinking_content':
                        chunk_type = 'THINK'
                    elif state.last_fragment_type in ('THINK', 'SEARCH'):
                        chunk_type = state.last_fragment_type
                    logger.debug(f"Got consolidated {chunk_type} content: {repr(data['content'])}")
                    return {
                        'content': data['content'],
                        'type': chunk_type,
                        'finish_reason': None
                    }

                # Direct text in {"v": "..."}
                if 'v' in data and isinstance(data['v'], str):
                    chunk_type = 'RESPONSE'
                    if state.last_content_path == 'response/thinking_content':
                        chunk_type = 'THINK'
                    elif state.last_fragment_type in ('THINK', 'SEARCH'):
                        chunk_type = state.last_fragment_type
                    return {
                        'content': data['v'],
                        'type': chunk_type,
                        'finish_reason': None
                    }

                # APPEND for .content or /content
                if data.get('o') == 'APPEND' and ('.content' in data.get('p', '') or data.get('p', '').endswith('/content')):
                    if isinstance(data.get('v'), str):
                        text = data['v']
                        path = data.get('p', '')
                        chunk_type = 'RESPONSE'
                        if 'THINK' in path:
                            chunk_type = 'THINK'
                        elif 'SEARCH' in path:
                            chunk_type = 'SEARCH'
                        elif state.last_fragment_type in ('THINK', 'SEARCH'):
                            chunk_type = state.last_fragment_type
                        elif state.last_content_path == 'response/thinking_content':
                            chunk_type = 'THINK'
                        logger.debug(f"Got {chunk_type} chunk: {repr(text)}, path: {path}")
                        return {
                            'content': text,
                            'type': chunk_type,
                            'finish_reason': None
                        }

                return None

        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {chunk[:200]}")
            return None
        except Exception as e:
            logger.error(f"Parsing error: {str(e)} for chunk: {chunk[:200]}")
            return None

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()