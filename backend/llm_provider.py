from typing import AsyncIterator, List, Dict, Optional
import httpx
import json
import codecs
import logging
from backend.config import settings

logger = logging.getLogger(__name__)


class LLMProvider:
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
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
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
                        
                        buffer = ""
                        decoder = codecs.getincrementaldecoder('utf-8')(errors='ignore')
                        
                        async for chunk_bytes in response.aiter_bytes():
                            decoded_chunk = decoder.decode(chunk_bytes, False)
                            buffer += decoded_chunk
                            
                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
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
                                            yield json.dumps({
                                                "type": "tool_calls",
                                                "tool_calls": delta["tool_calls"]
                                            })
                                        elif "content" in delta and delta["content"]:
                                            yield json.dumps({
                                                "type": "content",
                                                "content": delta["content"]
                                            })
                                except json.JSONDecodeError:
                                    if settings.debug_logging:
                                        logger.debug("Skipping malformed stream chunk")
                                    continue
                        
                        # Flush decoder to handle any remaining possible bytes
                        final_chunk = decoder.decode(b'', True)
                        if final_chunk:
                            buffer += final_chunk
                            if buffer.strip() and settings.debug_logging:
                                logger.debug("Discarding trailing partial stream buffer")
                else:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0]["message"].get("content", "")
                        yield json.dumps({"type": "content", "content": content})
        except httpx.ConnectError:
            logger.exception("Connection failed provider=%s model=%s", self.provider, self.config.get("model", ""))
            yield json.dumps({"type": "content", "content": "Connection error: unable to reach upstream LLM provider"})
        except Exception:
            logger.exception("Unexpected LLM provider error provider=%s model=%s", self.provider, self.config.get("model", ""))
            yield json.dumps({"type": "content", "content": "Error: request to LLM provider failed"})

llm_provider = LLMProvider()
