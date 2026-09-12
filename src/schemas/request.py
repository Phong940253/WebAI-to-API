# src/schemas/request.py
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Multimodal content part schemas (OpenAI vision format)
# ---------------------------------------------------------------------------

class ImageUrlDetail(BaseModel):
    """Inner object for image_url content parts."""
    url: str
    detail: Optional[str] = "auto"


class ContentPart(BaseModel):
    """A single part of a multimodal message content array."""
    type: str  # "text" | "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrlDetail] = None


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

# gemini-webapi (>= 2.0 / master) discovers available models dynamically per
# account. These are the canonical discovered names; if the account exposes
# different ones, the server still resolves them at runtime via list_models().
DEFAULT_MODEL = "gemini-flash"


class GeminiRequest(BaseModel):
    message: str
    model: Optional[str] = Field(default=DEFAULT_MODEL, description="Model to use for Gemini (resolved dynamically).")
    files: Optional[List[str]] = []
    extended_thinking: bool = Field(
        default=False,
        description="Enable extended thinking (requires an account that supports it).",
    )


class OpenAIChatRequest(BaseModel):
    messages: List[dict]
    # Accept any model name. The endpoint resolves it dynamically (via
    # client.resolve_model) and falls back to the account default when unknown,
    # so Home Assistant and other OpenAI clients can send names like
    # "gemini-3-pro-image-preview" without breaking.
    model: Optional[str] = None
    stream: Optional[bool] = False
    # OpenAI-compatible retention control.
    # store=False -> temporary mode (not saved in Gemini history).
    store: Optional[bool] = None
    # Enable extended thinking for this request (Pro/Ultra tier accounts).
    extended_thinking: Optional[bool] = None

class Part(BaseModel):
    text: str

class Content(BaseModel):
    parts: List[Part]

class GoogleGenerativeRequest(BaseModel):
    contents: List[Content]
