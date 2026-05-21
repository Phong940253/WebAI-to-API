# src/app/services/session_manager.py
import asyncio
import time
from app.logger import logger
from app.config import CONFIG
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError


_RETRYABLE_KEYWORDS = (
    "zombie",
    "failed to parse response body",
    "parse response",
    "stalled",
    "incomplete frame",
)


def _is_retryable_error(error: Exception) -> bool:
    err = str(error).lower()
    return any(keyword in err for keyword in _RETRYABLE_KEYWORDS)


def _read_retry_delay_seconds() -> float:
    try:
        return max(0.0, float(CONFIG["AI"].get("chat_retry_delay_seconds", "2")))
    except Exception:
        return 2.0


class SessionBusyError(Exception):
    """Raised when chat session lock could not be acquired in time."""
    pass

class SessionManager:
    def __init__(self, client):
        self.client = client
        self.session = None
        self.model = None
        self.turn_count = 0
        self.lock = asyncio.Lock()

    async def get_response(self, model, message, images):
        lock_wait_seconds = CONFIG.getint("AI", "chat_lock_wait_seconds", fallback=12)
        lock_acquired = False
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=lock_wait_seconds)
            lock_acquired = True
        except asyncio.TimeoutError:
            logger.warning(
                f"Gemini chat lock wait exceeded {lock_wait_seconds}s; "
                "request rejected as busy."
            )
            raise SessionBusyError(
                f"Chat session is busy (lock wait exceeded {lock_wait_seconds}s)"
            )

        try:
            model_value = model.value if hasattr(model, "value") else model
            timeout_seconds = CONFIG.getint("AI", "chat_timeout_seconds", fallback=45)
            retry_attempts = max(0, CONFIG.getint("AI", "chat_retry_attempts", fallback=1))
            retry_delay_seconds = _read_retry_delay_seconds()
            max_turns = max(1, CONFIG.getint("AI", "chat_session_max_turns", fallback=6))

            for attempt in range(retry_attempts + 1):
                # Start a new session if none exists or the model has changed
                if self.session is None or self.model != model or self.turn_count >= max_turns:
                    if self.session is not None:
                        logger.info(
                            f"Rotating Gemini chat session (model={model_value}, turns={self.turn_count}, max_turns={max_turns})."
                        )
                    self.session = self.client.start_chat(model=model_value)
                    self.model = model
                    self.turn_count = 0
                    logger.info(f"Started new Gemini chat session (model={model_value}).")

                try:
                    started_at = time.perf_counter()
                    response = await asyncio.wait_for(
                        self.session.send_message(prompt=message, files=images),
                        timeout=timeout_seconds,
                    )
                    elapsed = time.perf_counter() - started_at
                    logger.info(
                        f"Gemini chat session response received in {elapsed:.2f}s "
                        f"(model={model_value})."
                    )
                    self.turn_count += 1
                    return response
                except asyncio.TimeoutError:
                    # Reset stale session so retry/next request starts clean.
                    self.session = None
                    self.turn_count = 0
                    if attempt < retry_attempts:
                        logger.warning(
                            f"Gemini chat timed out after {timeout_seconds}s "
                            f"(model={model_value}, attempt={attempt + 1}/{retry_attempts + 1}). "
                            f"Retrying in {retry_delay_seconds:.1f}s with a fresh session."
                        )
                        if retry_delay_seconds:
                            await asyncio.sleep(retry_delay_seconds)
                        continue
                    logger.error(
                        f"Gemini chat session timed out after {timeout_seconds}s "
                        f"(model={model_value}). Session reset."
                    )
                    raise TimeoutError(f"Gemini chat timed out after {timeout_seconds}s")
                except Exception as e:
                    # Reset session on failures to avoid reusing a bad stream/session.
                    self.session = None
                    self.turn_count = 0
                    is_retryable = _is_retryable_error(e)
                    if is_retryable and attempt < retry_attempts:
                        logger.warning(
                            f"Gemini chat transient error "
                            f"(model={model_value}, attempt={attempt + 1}/{retry_attempts + 1}): {e!r}. "
                            f"Retrying in {retry_delay_seconds:.1f}s with a fresh session."
                        )
                        if retry_delay_seconds:
                            await asyncio.sleep(retry_delay_seconds)
                        continue
                    logger.error(f"Error in session get_response: {e}", exc_info=True)
                    raise
        finally:
            if lock_acquired:
                self.lock.release()

_translate_session_manager = None
_gemini_chat_manager = None

def init_session_managers():
    """
    Initialize session managers for translation and chat
    """
    global _translate_session_manager, _gemini_chat_manager
    try:
        client = get_gemini_client()
        _translate_session_manager = SessionManager(client)
        _gemini_chat_manager = SessionManager(client)
    except GeminiClientNotInitializedError:
        logger.warning("Session managers not initialized: Gemini client not available.")

def get_translate_session_manager():
    return _translate_session_manager

def get_gemini_chat_manager():
    return _gemini_chat_manager
