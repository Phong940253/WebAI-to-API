# src/app/services/model_resolver.py
"""
Central model resolution for gemini-webapi (>= 2.0 / master branch).

gemini-webapi now discovers models dynamically per account at init time. We map any
client-supplied model string (legacy names like ``gemini-3-flash``, Home Assistant
variants, partial ids, …) to an ``AvailableModel`` via ``client.resolve_model``, and
expose extended thinking through the ``extended_thinking`` flag — there is no separate
"thinking" model in 2.0+.

The ``Model`` enum from gemini-webapi is deprecated; we never import it. Instead we rely
on dynamic discovery so the server keeps working as Google rotates model names.
"""

from typing import Optional, Tuple

from app.logger import logger
from schemas.request import DEFAULT_MODEL

# Names shown only when the client isn't initialized yet (so /v1/models and the admin
# dropdown still render something sensible).
FALLBACK_MODELS = ["gemini-pro", "gemini-flash", "gemini-flash-lite"]


def resolve_gemini_model(client, name: Optional[str]) -> Tuple[object, bool]:
    """
    Resolve a requested model name to ``(AvailableModel | None, extended_thinking)``.

    * ``None`` model => let Google use the account's default model.
    * A name containing "thinking" enables ``extended_thinking`` and the remaining base
      name is resolved normally.
    * Unknown names fall back to the account default (``None``) instead of raising, so a
      client sending a stale name degrades gracefully rather than erroring out.

    ``client`` is the underlying ``gemini_webapi.GeminiClient`` (i.e. ``MyGeminiClient.client``).
    """
    extended = bool(name) and "thinking" in name.lower()
    base = (name or "").lower().replace("thinking", "").strip() or DEFAULT_MODEL
    try:
        return client.resolve_model(base), extended
    except (ValueError, AttributeError, TypeError):
        # Unknown name, or the client model registry isn't initialized yet.
        logger.warning(
            "Unknown Gemini model '%s' -> using account default model.", name
        )
        return None, extended


def available_model_names(client) -> list[str]:
    """Return the account's discovered model names, or the fallback list."""
    try:
        models = client.list_models()
    except Exception:
        models = None
    return [m.model_name for m in models] if models else list(FALLBACK_MODELS)


def model_display(model) -> str:
    """Best-effort display name for a resolved model (or empty string for the default)."""
    if model is None:
        return ""
    return getattr(model, "model_name", None) or str(model)
