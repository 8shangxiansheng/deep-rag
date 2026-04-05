from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import logging
from typing import AsyncIterator
from dotenv import load_dotenv, find_dotenv

from backend.config import settings, Settings
from backend.models import (
    ChatRequest, FileRetrievalRequest, FileRetrievalResponse,
    KnowledgeBaseInfo, HealthResponse
)
from backend.knowledge_base import knowledge_base
from backend.llm_provider import LLMProvider
from backend.prompts import create_system_prompt, create_file_retrieval_tool, create_react_system_prompt
from backend.rate_limiter import InMemoryRateLimiter
from backend.react_handler import handle_react_mode

logger = logging.getLogger(__name__)
CONFIG_API_TOKEN_HEADER = "x-admin-token"
SENSITIVE_ENV_KEYWORDS = ("API_KEY", "TOKEN", "PASSWORD", "SECRET", "PRIVATE_KEY")
ERROR_CODE_KEY = "code"
ERROR_MESSAGE_KEY = "message"

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.DEBUG if settings.debug_logging else logging.INFO)

rate_limiter = InMemoryRateLimiter()


def _api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            ERROR_CODE_KEY: code,
            ERROR_MESSAGE_KEY: message,
        },
    )


def _enforce_rate_limit(scope: str, request: Request, max_requests: int) -> None:
    client_key = request.client.host if request.client else "unknown"
    allowed = rate_limiter.allow(
        scope=scope,
        key=client_key,
        max_requests=max_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not allowed:
        raise _api_error(
            status_code=429,
            code="RATE_LIMIT_EXCEEDED",
            message=f"Too many requests for '{scope}'. Please retry later.",
        )


def _ensure_config_api_enabled() -> None:
    if not settings.enable_config_api:
        raise _api_error(status_code=403, code="CONFIG_API_DISABLED", message="Config API is disabled")


def _is_request_from_allowed_host(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in settings.get_config_api_allowed_hosts()


def _require_config_read_access(request: Request) -> None:
    _ensure_config_api_enabled()
    if not _is_request_from_allowed_host(request):
        raise _api_error(
            status_code=403,
            code="CONFIG_API_FORBIDDEN_HOST",
            message="Config API is only available from allowed hosts",
        )


def _require_config_write_access(request: Request) -> None:
    _require_config_read_access(request)

    expected_token = settings.config_api_admin_token.strip()
    if not expected_token:
        raise _api_error(
            status_code=503,
            code="CONFIG_API_TOKEN_NOT_CONFIGURED",
            message="Config API admin token is not configured",
        )

    provided_token = request.headers.get(CONFIG_API_TOKEN_HEADER, "").strip()
    if provided_token != expected_token:
        raise _api_error(status_code=401, code="CONFIG_API_INVALID_TOKEN", message="Invalid admin token")


def _redact_env_content(content: str) -> str:
    has_trailing_newline = content.endswith("\n")
    redacted_lines = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            redacted_lines.append(line)
            continue

        key, value = line.split("=", 1)
        key_upper = key.strip().upper()

        if any(keyword in key_upper for keyword in SENSITIVE_ENV_KEYWORDS) and value.strip():
            redacted_lines.append(f"{key}=***REDACTED***")
        else:
            redacted_lines.append(line)

    redacted = "\n".join(redacted_lines)
    if has_trailing_newline:
        redacted += "\n"
    return redacted


app = FastAPI(
    title="Deep RAG",
    version="1.0.0",
    description="A Deep RAG system that teaches AI to truly understand your knowledge base"
)

cors_origins = settings.get_cors_allowed_origins()
allow_credentials = "*" not in cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check"""
    providers = settings.list_available_providers()
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "providers": providers
    }

@app.get("/config")
async def get_config():
    """Get current configuration - dynamically return the current provider's model"""
    config = settings.get_provider_config(settings.api_provider)
    
    return {
        "default_provider": settings.api_provider,
        "default_model": config.get("model", "")
    }

@app.get("/api/config")
async def get_env_config(request: Request):
    """Read redacted .env content"""
    _require_config_read_access(request)

    env_path = find_dotenv()
    if not env_path:
        raise _api_error(status_code=404, code="ENV_FILE_NOT_FOUND", message=".env file not found")
    
    with open(env_path, 'r', encoding='utf-8') as f:
        content = _redact_env_content(f.read())
    
    return {"content": content}

@app.post("/api/config")
async def update_env_config(request: Request, payload: dict):
    """Directly save .env file content"""
    _require_config_write_access(request)

    env_path = find_dotenv()
    if not env_path:
        raise _api_error(status_code=404, code="ENV_FILE_NOT_FOUND", message=".env file not found")
    
    content = payload.get("content", "")
    if not content or not content.strip():
        raise _api_error(status_code=400, code="CONFIG_CONTENT_EMPTY", message="Config content cannot be empty")
    
    try:
        # Write content directly
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Reload environment variables
        load_dotenv(dotenv_path=env_path, override=True)
        
        # Reinitialize settings object
        global settings
        settings = Settings()

        host = request.client.host if request.client else "unknown"
        logger.warning("Config updated from host=%s", host)
        
        return {"status": "success", "message": "Configuration updated successfully!"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to update configuration")
        raise _api_error(status_code=500, code="CONFIG_UPDATE_FAILED", message="Failed to update configuration")

@app.get("/knowledge-base/info", response_model=KnowledgeBaseInfo)
async def get_knowledge_base_info():
    try:
        summary = await knowledge_base.get_file_summary()
        file_tree = knowledge_base.list_files()
        return {
            "summary": summary,
            "file_tree": file_tree
        }
    except Exception:
        logger.exception("Failed to read knowledge base info")
        raise _api_error(
            status_code=500,
            code="KNOWLEDGE_BASE_INFO_LOAD_FAILED",
            message="Failed to load knowledge base info",
        )

@app.get("/system-prompt")
async def get_system_prompt():
    """Return the system prompt currently in use"""
    try:
        file_summary = await knowledge_base.get_file_summary()
        
        # Return corresponding system prompt based on configuration
        if settings.tool_calling_mode == "react":
            system_prompt = create_react_system_prompt(file_summary)
        else:
            system_prompt = create_system_prompt(file_summary)
        
        return {
            "system_prompt": system_prompt,
            "mode": settings.tool_calling_mode
        }
    except Exception:
        logger.exception("Failed to load system prompt")
        raise _api_error(status_code=500, code="SYSTEM_PROMPT_LOAD_FAILED", message="Failed to load system prompt")

@app.post("/knowledge-base/retrieve", response_model=FileRetrievalResponse)
async def retrieve_files(http_request: Request, request: FileRetrievalRequest):
    _enforce_rate_limit("knowledge-base-retrieve", http_request, settings.rate_limit_retrieve_requests)
    try:
        content = await knowledge_base.retrieve_files(request.file_paths)
        return {"content": content}
    except (ValueError, FileNotFoundError) as e:
        raise _api_error(status_code=400, code="RETRIEVE_INVALID_REQUEST", message=str(e))
    except Exception:
        logger.exception("Failed to retrieve knowledge base files")
        raise _api_error(status_code=500, code="RETRIEVE_FAILED", message="Failed to retrieve files")

@app.post("/chat")
async def chat(http_request: Request, request: ChatRequest):
    _enforce_rate_limit("chat", http_request, settings.rate_limit_chat_requests)
    try:
        provider = LLMProvider(provider=request.provider or settings.api_provider)
        
        file_summary = await knowledge_base.get_file_summary()
        
        # Check whether to use function calling or ReAct mode
        use_react = settings.tool_calling_mode == "react"
        
        if use_react:
            system_prompt = create_react_system_prompt(file_summary)
        else:
            system_prompt = create_system_prompt(file_summary)
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([msg.dict() for msg in request.messages])
        
        if use_react:
            async def generate_response() -> AsyncIterator[str]:
                async for chunk in handle_react_mode(provider, messages):
                    yield chunk
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
            return StreamingResponse(
                generate_response(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        tools = [create_file_retrieval_tool()]
        
        async def generate_response() -> AsyncIterator[str]:
            conversation_messages = messages.copy()
            max_iterations = 10
            iteration = 0
            has_content = False
            
            while iteration < max_iterations:
                iteration += 1
                accumulated_tool_call = None
                iteration_has_content = False
                
                async for chunk_str in provider.chat_completion(
                    messages=conversation_messages,
                    tools=tools,
                    stream=True
                ):
                    try:
                        chunk = json.loads(chunk_str)
                        
                        if chunk["type"] == "content":
                            has_content = True
                            iteration_has_content = True
                            yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
                        
                        elif chunk["type"] == "tool_calls":
                            tool_calls = chunk["tool_calls"]
                            
                            for tool_call in tool_calls:
                                if accumulated_tool_call is None:
                                    if tool_call.get("id") and tool_call.get("type"):
                                        accumulated_tool_call = {
                                            "index": tool_call.get("index", 0),
                                            "id": tool_call["id"],
                                            "type": tool_call["type"],
                                            "function": {
                                                "name": tool_call.get("function", {}).get("name", ""),
                                                "arguments": tool_call.get("function", {}).get("arguments", "")
                                            }
                                        }
                                else:
                                    if "function" in tool_call and "arguments" in tool_call["function"]:
                                        accumulated_tool_call["function"]["arguments"] += tool_call["function"]["arguments"]
                    
                    except json.JSONDecodeError:
                        continue
                
                if not accumulated_tool_call and iteration_has_content:
                    break
                
                if accumulated_tool_call:
                    yield f"data: {json.dumps({'type': 'tool_calls', 'tool_calls': [accumulated_tool_call]})}\n\n"
                    
                    from backend.prompts import process_tool_calls
                    tool_results = await process_tool_calls([accumulated_tool_call])
                    
                    yield f"data: {json.dumps({'type': 'tool_results', 'results': tool_results})}\n\n"
                    
                    conversation_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [accumulated_tool_call]
                    })
                    
                    conversation_messages.extend(tool_results)
                else:
                    break
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except Exception:
        logger.exception("Chat request failed")
        raise _api_error(status_code=500, code="CHAT_REQUEST_FAILED", message="Chat request failed")

@app.get("/providers")
async def list_providers():
    """List all configured providers - use dynamic scanning"""
    provider_ids = settings.list_available_providers()
    
    providers = []
    for provider_id in provider_ids:
        config = settings.get_provider_config(provider_id)
        if config.get("model"):
            providers.append({
                "id": provider_id,
                "name": provider_id.replace('_', ' ').title(),
                "models": [config["model"]]
            })
    
    return {"providers": providers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
