# src/app/endpoints/chat.py
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import CONFIG
from app.logger import logger
from app.services.gemini_client import GeminiClientNotInitializedError, get_gemini_client
from app.services.model_resolver import (
    FALLBACK_MODELS,
    model_display,
    resolve_gemini_model,
)
from app.services.telegram_notifier import TelegramNotifier
from app.services.session_manager import get_translate_session_manager
from app.utils.image_utils import (
    cleanup_temp_files,
    decode_base64_to_tempfile,
    download_to_tempfile,
    get_temp_dir,
    serialize_response_images,
)
from schemas.request import DEFAULT_MODEL, GeminiRequest, OpenAIChatRequest

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cookies(gemini_client) -> dict:
    """Extract session cookies from the underlying Gemini web client."""
    try:
        return dict(gemini_client.client.cookies)
    except Exception:
        return {}


async def _extract_multimodal_content(content) -> Tuple[str, List[Path]]:
    """
    Parse a message ``content`` field that may be:
    - a plain string
    - a list of content part dicts

    Supports both Chat Completions and Responses API part types:
    - Chat Completions: ``{"type": "text"|"image_url", ...}``
    - Responses API:    ``{"type": "input_text"|"input_image", ...}``

    For ``image_url``/``input_image``, the URL value may be:
    - Chat Completions: ``{"image_url": {"url": "data:..."}}``)
    - Responses API:    ``{"image_url": "data:..."}``  (direct string)

    Returns ``(text_prompt, temp_file_paths)``.
    Temp files are created for base64 data URIs and remote URL images; the caller
    is responsible for cleaning them up after use.
    """
    if isinstance(content, str):
        return content, []

    if not isinstance(content, list):
        return str(content) if content else "", []

    text_parts: List[str] = []
    file_paths: List[Path] = []

    for part in content:
        if not isinstance(part, dict):
            continue

        part_type = part.get("type", "")

        # Text parts — Chat Completions ("text") and Responses API ("input_text")
        if part_type in ("text", "input_text"):
            txt = part.get("text", "")
            if txt:
                text_parts.append(txt)

        # Image parts — Chat Completions ("image_url") and Responses API ("input_image")
        elif part_type in ("image_url", "input_image"):
            img_url_obj = part.get("image_url", {})
            # Chat Completions: image_url is {"url": "...", "detail": "..."}
            # Responses API:    image_url is a direct string "data:..." or "https://..."
            url: str = img_url_obj.get("url", "") if isinstance(img_url_obj, dict) else str(img_url_obj)

            if not url:
                continue

            if url.startswith("data:"):
                # base64 data URI
                try:
                    temp_path = decode_base64_to_tempfile(url)
                    file_paths.append(temp_path)
                except ValueError as exc:
                    logger.warning(f"Skipping invalid base64 image: {exc}")

            elif url.startswith("file://"):
                # Reference to a previously uploaded file — resolve file_id
                file_id = url[len("file://"):]
                # Sanitize
                if "/" not in file_id and "\\" not in file_id and ".." not in file_id:
                    candidate = get_temp_dir() / file_id
                    if candidate.exists():
                        file_paths.append(candidate)
                    else:
                        logger.warning(f"File not found for file_id: {file_id}")
                else:
                    logger.warning(f"Invalid file_id in URL: {url}")

            elif url.startswith("http://") or url.startswith("https://"):
                # Remote image URL — download to temp file
                temp_path = await download_to_tempfile(url)
                if temp_path:
                    file_paths.append(temp_path)

    return " ".join(text_parts), file_paths


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def list_models():
    """List available models from Gemini API in OpenAI-compatible format."""
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError:
        # Fallback to the canonical name list if the client isn't initialized yet
        now = int(time.time())
        models = [
            {
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "google",
            }
            for name in FALLBACK_MODELS
        ]
        return {"object": "list", "data": models}

    try:
        # Get available models from the Gemini API client (dynamic discovery).
        available_models = gemini_client.client.list_models()
        if not available_models:
            raise ValueError("empty model list")
        now = int(time.time())
        models = [
            {
                "id": model.model_name,
                "object": "model",
                "created": now,
                "owned_by": "google",
                "display_name": model.display_name,
            }
            for model in available_models
        ]
        return {"object": "list", "data": models}
    except Exception as e:
        logger.warning(f"Failed to get models from Gemini API: {e}, falling back to default list")
        now = int(time.time())
        models = [
            {
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "google",
            }
            for name in FALLBACK_MODELS
        ]
        return {"object": "list", "data": models}


# ---------------------------------------------------------------------------
# Translation endpoint
# ---------------------------------------------------------------------------

@router.post("/translate")
async def translate_chat(request: GeminiRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_manager = get_translate_session_manager()
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager is not initialized.")
    try:
        model_obj, extended = resolve_gemini_model(gemini_client.client, request.model)
        response = await session_manager.get_response(
            model_obj,
            request.message,
            request.files,
            extended_thinking=extended or request.extended_thinking,
        )
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")


# ---------------------------------------------------------------------------
# OpenAI-compatible streaming helpers
# ---------------------------------------------------------------------------

def _to_openai_format(response_text: str, model: str, images: list, stream: bool = False) -> dict:
    """Build an OpenAI-compatible chat completion response dict."""
    content = response_text
    # Append image references as markdown if present (keeps text content useful)
    if images:
        md_links = "\n".join(
            f"![{img['title']}]({img['url']})" for img in images
        )
        content = f"{response_text}\n\n{md_links}".strip()

    result = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    # Attach raw images as a top-level extension field
    if images:
        result["images"] = images
    return result


async def _stream_response(response_text: str, model: str, images: list):
    """Yield SSE chunks in OpenAI streaming format."""
    completion_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    content = response_text
    if images:
        md_links = "\n".join(f"![{img['title']}]({img['url']})" for img in images)
        content = f"{response_text}\n\n{md_links}".strip()

    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"

    content_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    if images:
        content_chunk["images"] = images
    yield f"data: {json.dumps(content_chunk)}\n\n"

    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    """
    OpenAI-compatible chat completion endpoint with multimodal support.

    Supports:
    - Plain text messages
    - ``image_url`` content parts with base64 data URIs (``data:image/...;base64,...``)
    - ``image_url`` content parts with remote HTTPS URLs (downloaded automatically)
    - ``image_url`` content parts with ``file://`` references to uploaded file IDs
    - ``thoughts`` field in response (thinking models)
    - ``images`` field in response (web/generated images)
    """
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    is_stream = bool(request.stream)

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    # Resolve model string → AvailableModel (dynamic; falls back to account default if unknown).
    model_obj, extended = resolve_gemini_model(gemini_client.client, request.model)
    model_value = model_display(model_obj) or request.model or DEFAULT_MODEL
    # request.extended_thinking (OpenAIChatRequest) overrides any name-based detection.
    if request.extended_thinking is not None:
        extended = request.extended_thinking

    # Auto-delete behavior for Gemini history:
    # - If request.store is set, honor it (store=False => temporary=True)
    # - Otherwise use config default (default: true => temporary chats)
    auto_delete_default = CONFIG.getboolean(
        "AI", "chat_completions_auto_delete", fallback=True
    )
    if request.store is None:
        temporary_mode = auto_delete_default
    else:
        temporary_mode = not bool(request.store)

    # Parse all messages — collect text parts and any image file paths
    conversation_parts: List[str] = []
    all_file_paths: List[Path] = []
    # Track which paths are temp files that should be cleaned up
    temp_file_paths: List[Path] = []

    for msg in request.messages:
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        text, file_paths = await _extract_multimodal_content(raw_content)

        # Mark newly created temp files for cleanup
        for fp in file_paths:
            if str(fp).startswith(str(get_temp_dir())):
                temp_file_paths.append(fp)
        all_file_paths.extend(file_paths)

        if not text:
            continue

        if role == "system":
            conversation_parts.append(f"System: {text}")
        elif role == "user":
            conversation_parts.append(f"User: {text}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {text}")

    if not conversation_parts:
        raise HTTPException(status_code=400, detail="No valid messages found.")

    final_prompt = "\n\n".join(conversation_parts)
    files_arg = all_file_paths if all_file_paths else None

    try:
        response = await gemini_client.generate_content(
            message=final_prompt,
            model=model_obj,
            files=files_arg,
            temporary=temporary_mode,
            extended_thinking=extended,
        )

        images = await serialize_response_images(
            response, gemini_cookies=_get_cookies(gemini_client)
        )

        if is_stream:
            return StreamingResponse(
                _stream_response(response.text, model_value, images),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return _to_openai_format(response.text, model_value, images, is_stream)

    except Exception as e:
        err_str = str(e)
        err_lower = err_str.lower()
        notifier = TelegramNotifier.get_instance()
        if "auth" in err_lower or "cookie" in err_lower:
            logger.error(f"[chat/completions] Auth error: {e}")
            await notifier.notify_error("auth", "Authentication failed", "/v1/chat/completions", err_str)
            raise HTTPException(status_code=401, detail=f"Gemini authentication failed: {err_str}")
        elif "zombie" in err_lower or "parse" in err_lower or "stalled" in err_lower:
            logger.error(f"[chat/completions] Stream error after retries (model={model_value}): {e}")
            await notifier.notify_error("503", "Stream temporarily unavailable", "/v1/chat/completions", err_str)
            raise HTTPException(status_code=503, detail="Gemini stream temporarily unavailable — please retry")
        else:
            logger.error(f"[chat/completions] Unexpected error (model={model_value}): {e}", exc_info=True)
            await notifier.notify_error("500", "Unexpected error", "/v1/chat/completions", err_str)
            raise HTTPException(status_code=500, detail=f"Error processing chat completion: {err_str}")

    finally:
        # Clean up temp files created from base64/URL image inputs
        cleanup_temp_files(temp_file_paths)
