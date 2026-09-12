# src/app/endpoints/gemini.py
from pathlib import Path
import html
import re
import time
from typing import List, Optional, Union

from fastapi import APIRouter, HTTPException

from app.config import CONFIG
from app.logger import logger
from app.services.gemini_client import GeminiClientNotInitializedError, get_gemini_client
from app.services.model_resolver import resolve_gemini_model
from app.services.telegram_notifier import TelegramNotifier
from app.services.session_manager import SessionBusyError, get_gemini_chat_manager
from app.utils.image_utils import cleanup_temp_files, serialize_response_images
from schemas.request import GeminiRequest

router = APIRouter()


def _get_cookies(gemini_client) -> dict:
    """Extract session cookies from the underlying Gemini web client."""
    try:
        return dict(gemini_client.client.cookies)
    except Exception:
        return {}


def _is_html_only_chunk(message: str) -> bool:
    """
    Return True when content has HTML/XML tags but no visible text.
    This avoids sending pure structural chunks (e.g. <img .../>) to Gemini.
    """
    if not message:
        return False
    if "<" not in message or ">" not in message:
        return False

    stripped = re.sub(r"<[^>]+>", "", message)
    stripped = html.unescape(stripped)
    return stripped.strip() == ""


@router.post("/gemini")
async def gemini_generate(request: GeminiRequest):
    """
    Stateless content generation.

    Response includes:
    - ``response``: generated text
    - ``images``: list of web/generated images (URL + base64), if any
    - ``thoughts``: chain-of-thought text (thinking models only), if any
    """
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    file_paths: List[Path] = [Path(f) for f in request.files] if request.files else []
    model_obj, extended = resolve_gemini_model(gemini_client.client, request.model)
    extended = extended or request.extended_thinking
    model_value = request.model or "gemini-flash"

    try:
        response = await gemini_client.generate_content(
            request.message,
            model=model_obj,
            files=file_paths or None,
            extended_thinking=extended,
        )

        images = await serialize_response_images(response, gemini_cookies=_get_cookies(gemini_client))

        result: dict = {"response": response.text}
        if images:
            result["images"] = images
        if response.thoughts:
            result["thoughts"] = response.thoughts
        return result

    except Exception as e:
        logger.error(f"Error in /gemini endpoint: {e}", exc_info=True)
        err_str = str(e)
        err_lower = err_str.lower()
        notifier = TelegramNotifier.get_instance()
        if "auth" in err_lower or "cookie" in err_lower:
            await notifier.notify_error("auth", "Authentication failed", "/gemini", err_str)
        else:
            await notifier.notify_error("500", "Unexpected error", "/gemini", err_str)
        raise HTTPException(status_code=500, detail=f"Error generating content: {err_str}")


@router.post("/gemini-chat")
async def gemini_chat(request: GeminiRequest):
    """
    Stateful chat with persistent session context.

    Response includes:
    - ``response``: generated text
    - ``images``: list of web/generated images (URL + base64), if any
    - ``thoughts``: chain-of-thought text (thinking models only), if any
    """
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_manager = get_gemini_chat_manager()
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager is not initialized.")

    model_obj, extended = resolve_gemini_model(gemini_client.client, request.model)
    extended = extended or request.extended_thinking
    model_value = request.model or "gemini-flash"
    image_count = len(request.files or [])
    logger.info(
        f"/gemini-chat request started (model={model_value}, has_message={bool(request.message)}, files={image_count})."
    )

    skip_html_only = CONFIG.getboolean("AI", "chat_skip_html_only_chunks", fallback=True)
    if skip_html_only and _is_html_only_chunk(request.message):
        logger.info("/gemini-chat HTML-only chunk detected; returning input without model call.")
        return {
            "response": request.message,
            "skipped_model": True,
            "skip_reason": "html_only_chunk",
        }

    try:
        started_at = time.perf_counter()
        response = await session_manager.get_response(
            model_obj,
            request.message,
            request.files,
            extended_thinking=extended,
        )
        elapsed = time.perf_counter() - started_at
        logger.info(f"/gemini-chat request completed in {elapsed:.2f}s (model={model_value}).")

        images = await serialize_response_images(response, gemini_cookies=_get_cookies(gemini_client))

        result: dict = {"response": response.text}
        if images:
            result["images"] = images
        if response.thoughts:
            result["thoughts"] = response.thoughts
        return result

    except TimeoutError as e:
        logger.error(f"/gemini-chat timeout (model={model_value}): {e}")
        notifier = TelegramNotifier.get_instance()
        await notifier.notify_error("503", "Gemini chat timeout", "/gemini-chat", str(e))

        fallback_enabled = CONFIG.getboolean(
            "AI", "chat_timeout_fallback_stateless", fallback=True
        )
        if not fallback_enabled:
            raise HTTPException(status_code=504, detail=str(e))

        logger.warning(
            f"/gemini-chat timeout fallback enabled; retrying as stateless /gemini "
            f"request (model={model_value})."
        )
        try:
            file_paths: List[Path] = [Path(f) for f in request.files] if request.files else []
            fallback_response = await gemini_client.generate_content(
                request.message,
                model=model_obj,
                files=file_paths or None,
                extended_thinking=extended,
            )
            images = await serialize_response_images(
                fallback_response,
                gemini_cookies=_get_cookies(gemini_client),
            )
            result: dict = {
                "response": fallback_response.text,
                "session_fallback": True,
            }
            if images:
                result["images"] = images
            if fallback_response.thoughts:
                result["thoughts"] = fallback_response.thoughts
            return result
        except Exception as fallback_error:
            logger.error(
                f"/gemini-chat stateless fallback failed (model={model_value}): "
                f"{fallback_error}",
                exc_info=True,
            )
            raise HTTPException(status_code=504, detail=str(e))

    except SessionBusyError as e:
        logger.warning(f"/gemini-chat busy (model={model_value}): {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.error(f"Error in /gemini-chat endpoint: {e}", exc_info=True)
        err_str = str(e)
        err_lower = err_str.lower()
        notifier = TelegramNotifier.get_instance()
        if "auth" in err_lower or "cookie" in err_lower:
            await notifier.notify_error("auth", "Authentication failed", "/gemini-chat", err_str)
        else:
            await notifier.notify_error("500", "Unexpected error", "/gemini-chat", err_str)
        raise HTTPException(status_code=500, detail=f"Error in chat: {err_str}")
