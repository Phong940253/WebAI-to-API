"""Guest mode: anonymous operation limited to the guest default model."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

import app.services.providers.gemini.client as gemini_client_module
from app.config import CONFIG
from app.main import app
from app.services.providers.gemini.provider import GeminiProvider
from app.services.providers.gemini.shared import (
    GUEST_DEFAULT_MODEL,
    ensure_gemini_client_ready,
    resolve_guest_default_model,
)


def _guest_client(mocker):
    """A runtime client holding an unauthenticated guest session."""
    client = mocker.Mock()
    client.client.account_status.name = "UNAUTHENTICATED"
    client.list_models.return_value = [
        SimpleNamespace(model_name="gemini-3-flash-lite", is_available=True),
        SimpleNamespace(model_name="gemini-3-flash", is_available=False),
    ]

    def _resolve(model_name):
        for runtime_model in client.list_models.return_value:
            if runtime_model.model_name == model_name:
                return runtime_model
        raise ValueError(f"Unknown model name: '{model_name}'")

    client.resolve_model.side_effect = _resolve
    client.generate_content = mocker.AsyncMock(
        return_value=SimpleNamespace(text="guest response", images=[], videos=[], media=[])
    )
    return client


# --- Readiness gate -------------------------------------------------------


def test_ready_check_allows_guest_when_guest_mode_enabled(mocker, monkeypatch):
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")

    ensure_gemini_client_ready(_guest_client(mocker), allow_guest=True)


def test_ready_check_rejects_guest_when_guest_mode_disabled(mocker, monkeypatch):
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "false")

    with pytest.raises(HTTPException) as exc_info:
        ensure_gemini_client_ready(_guest_client(mocker), allow_guest=True)

    assert exc_info.value.status_code == 401


def test_persistent_ready_check_rejects_guest_even_when_enabled(mocker, monkeypatch):
    """Persistent paths never pass allow_guest, so sign-in stays mandatory."""
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")

    with pytest.raises(HTTPException) as exc_info:
        ensure_gemini_client_ready(_guest_client(mocker))

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"


# --- Guest model resolution ----------------------------------------------


def test_resolve_guest_default_model_prefers_available_catalog_model(mocker):
    client = mocker.Mock()
    client.list_models.return_value = [
        SimpleNamespace(model_name="gemini-3.1-flash-lite", is_available=True),
        SimpleNamespace(model_name="gemini-3-flash", is_available=False),
    ]

    assert resolve_guest_default_model(client) == "gemini-3.1-flash-lite"


def test_resolve_guest_default_model_falls_back_to_constant(mocker):
    client = mocker.Mock()
    client.list_models.return_value = None

    assert resolve_guest_default_model(client) == GUEST_DEFAULT_MODEL


# --- Stateless / temporary execution -------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_guest_session_uses_guest_default_model(
    mocker, monkeypatch, install_gemini_client
):
    """A model-less request in guest mode resolves to the guest default model."""
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")
    client = _guest_client(mocker)
    install_gemini_client(client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "guest response"
    assert client.generate_content.await_args.args[1] == "gemini-3-flash-lite"


@pytest.mark.asyncio
async def test_chat_completions_guest_session_rejects_other_models(
    mocker, monkeypatch, install_gemini_client
):
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")
    client = _guest_client(mocker)
    install_gemini_client(client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 400
    assert f"Guest sessions only support '{GUEST_DEFAULT_MODEL}'" in response.json()["detail"]
    client.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_completions_guest_session_rejects_persistent_store(
    mocker, monkeypatch, install_gemini_client
):
    """store=true (persistent) keeps requiring sign-in under guest mode."""
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")
    client = _guest_client(mocker)
    install_gemini_client(client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/v1/chat/completions",
            json={
                "store": True,
                "model": "gemini-3-flash-lite",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    client.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_translate_guest_session_allows_guest_default_model(
    mocker, monkeypatch, install_gemini_client
):
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")
    client = _guest_client(mocker)
    install_gemini_client(client)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        ok_response = await ac.post(
            "/translate",
            json={"model": "gemini-3-flash-lite", "message": "Translate this"},
        )
        rejected_response = await ac.post(
            "/translate",
            json={"model": "gemini-3-flash", "message": "Translate this"},
        )

    assert ok_response.status_code == 200
    assert ok_response.json() == {"response": "guest response"}
    assert rejected_response.status_code == 400
    assert (
        f"Guest sessions only support '{GUEST_DEFAULT_MODEL}'"
        in rejected_response.json()["detail"]
    )


# --- Model catalog --------------------------------------------------------


@pytest.mark.asyncio
async def test_guest_model_catalog_advertises_only_guest_models(
    mocker, monkeypatch, install_gemini_client
):
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")
    install_gemini_client(_guest_client(mocker))

    provider = GeminiProvider()
    models = await provider.list_models()
    model_ids = [model["id"] for model in models]

    assert "gemini-3-flash-lite" in model_ids
    assert "gemini-3-flash" not in model_ids
    assert all(not model_id.startswith("playwright/") for model_id in model_ids)


# --- Client initialization waterfall -------------------------------------


@pytest.mark.asyncio
async def test_init_gemini_client_creates_guest_client_without_login(mocker, monkeypatch):
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "true")
    mocker.patch.object(
        gemini_client_module.GeminiAuthSelector, "iter_candidates", return_value=iter([])
    )
    mocker.patch.object(gemini_client_module, "get_cookie_from_browser", return_value=None)
    mocker.patch.object(gemini_client_module, "_clear_guest_cookie_cache")

    guest_client = mocker.Mock()
    guest_client.init = mocker.AsyncMock()
    guest_client.close = mocker.AsyncMock()
    guest_client.client.account_status.name = "UNAUTHENTICATED"
    mocker.patch.object(gemini_client_module, "MyGeminiClient", return_value=guest_client)

    result = await gemini_client_module.init_gemini_client()

    assert result is True
    assert gemini_client_module._gemini_client is guest_client
    assert gemini_client_module.get_gemini_client_auth_source() == "guest mode (no login)"
    guest_client.init.assert_awaited_once_with(verbose=True, auto_refresh=False)
    gemini_client_module._clear_guest_cookie_cache.assert_called_once()


@pytest.mark.asyncio
async def test_init_gemini_client_requires_opt_in_for_guest_mode(mocker, monkeypatch):
    monkeypatch.setitem(CONFIG["Gemini"], "guest_mode", "false")
    mocker.patch.object(
        gemini_client_module.GeminiAuthSelector, "iter_candidates", return_value=iter([])
    )
    mocker.patch.object(gemini_client_module, "get_cookie_from_browser", return_value=None)
    guest_client = mocker.Mock()
    guest_client.init = mocker.AsyncMock()
    mock_client_class = mocker.patch.object(
        gemini_client_module, "MyGeminiClient", return_value=guest_client
    )

    result = await gemini_client_module.init_gemini_client()

    assert result is False
    assert gemini_client_module._gemini_client is None
    mock_client_class.assert_not_called()
    assert gemini_client_module._initialization_error
