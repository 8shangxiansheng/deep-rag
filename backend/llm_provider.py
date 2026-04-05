from typing import AsyncIterator, List, Dict, Optional
import asyncio
import httpx
import json
import codecs
import logging
from time import monotonic
from backend.config import settings

logger = logging.getLogger(__name__)


class LLMProvider:
    _circuit_state: dict[str, dict[str, float]] = {}

    def __init__(self, provider: str = None):
        self.provider = provider or settings.api_provider
        self.config = settings.get_provider_config(self.provider)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = True
    ) -> AsyncIterator[str]:
        """Unified OpenAI-compatible chat completion"""
        base_url = self.config.get('base_url', '')
        if not base_url:
            raise ValueError(f"BASE_URL not configured for provider '{self.provider}'")
        
        if '/chat/completions' in base_url:
            url = base_url
        else:
            url = f"{base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            **self.config.get("headers", {})
        }
        
        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": settings.temperature,
            "stream": stream
        }
        
        # Newer models use max_completion_tokens, older models use max_tokens
        # gpt-5-nano and other newer models need max_completion_tokens
        if "gpt-5" in self.config["model"] or "o1" in self.config["model"]:
            payload["max_completion_tokens"] = settings.max_tokens
        else:
            payload["max_tokens"] = settings.max_tokens
        
        if tools:
            payload["tools"] = tools

        if settings.debug_logging:
            logger.debug(
                "Calling provider=%s model=%s stream=%s tools=%s",
                self.provider,
                self.config["model"],
                stream,
                bool(tools),
            )

        if self._is_circuit_open():
            yield json.dumps(
                {"type": "content", "content": "Error: upstream LLM provider is temporarily unavailable"}
            )
            return

        attempts = settings.llm_max_retries + 1
        for attempt in range(attempts):
            should_retry = False
            try:
                timeout = httpx.Timeout(settings.request_timeout_seconds)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    if stream:
                        async with client.stream("POST", url, json=payload, headers=headers) as response:
                            if response.status_code != 200:
                                error_text = await response.aread()
                                logger.error(
                                    "LLM API returned status=%s provider=%s model=%s body=%s",
                                    response.status_code,
                                    self.provider,
                                    self.config["model"],
                                    error_text.decode("utf-8", errors="ignore")[:500],
                                )
                            response.raise_for_status()
                            async for item in self._stream_response(response):
                                yield item
                    else:
                        response = await client.post(url, json=payload, headers=headers)
                        response.raise_for_status()
                        data = response.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0]["message"].get("content", "")
                            yield json.dumps({"type": "content", "content": content})

                self._record_success()
                return
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response else None
                should_retry = status_code is not None and status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                logger.exception(
                    "LLM HTTP error provider=%s model=%s status=%s retry=%s",
                    self.provider,
                    self.config.get("model", ""),
                    status_code,
                    should_retry,
                )
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
                should_retry = True
                logger.exception(
                    "LLM network timeout/connect error provider=%s model=%s",
                    self.provider,
                    self.config.get("model", ""),
                )
            except Exception:
                logger.exception(
                    "Unexpected LLM provider error provider=%s model=%s",
                    self.provider,
                    self.config.get("model", ""),
                )

            if should_retry and attempt < attempts - 1:
                backoff = settings.llm_retry_backoff_seconds * (2 ** attempt)
                await asyncio.sleep(backoff)
                continue

            self._record_failure()
            yield json.dumps({"type": "content", "content": "Error: request to LLM provider failed"})
            return

    async def _stream_response(self, response: httpx.Response) -> AsyncIterator[str]:
        buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")

        async for chunk_bytes in response.aiter_bytes():
            decoded_chunk = decoder.decode(chunk_bytes, False)
            buffer += decoded_chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        if "tool_calls" in delta:
                            yield json.dumps({"type": "tool_calls", "tool_calls": delta["tool_calls"]})
                        elif "content" in delta and delta["content"]:
                            yield json.dumps({"type": "content", "content": delta["content"]})
                except json.JSONDecodeError:
                    if settings.debug_logging:
                        logger.debug("Skipping malformed stream chunk")
                    continue

        final_chunk = decoder.decode(b"", True)
        if final_chunk and final_chunk.strip() and settings.debug_logging:
            logger.debug("Discarding trailing partial stream buffer")

    def _circuit_key(self) -> str:
        return f"{self.provider}:{self.config.get('model', '')}"

    def _is_circuit_open(self) -> bool:
        state = self._circuit_state.get(self._circuit_key(), {})
        opened_until = state.get("opened_until", 0.0)
        return opened_until > monotonic()

    def _record_success(self) -> None:
        self._circuit_state[self._circuit_key()] = {"failures": 0.0, "opened_until": 0.0}

    def _record_failure(self) -> None:
        key = self._circuit_key()
        state = self._circuit_state.get(key, {"failures": 0.0, "opened_until": 0.0})
        failures = state.get("failures", 0.0) + 1.0

        opened_until = state.get("opened_until", 0.0)
        if failures >= float(settings.circuit_breaker_failures):
            opened_until = monotonic() + float(settings.circuit_breaker_cooldown_seconds)
            failures = 0.0

        self._circuit_state[key] = {"failures": failures, "opened_until": opened_until}

llm_provider = LLMProvider()
