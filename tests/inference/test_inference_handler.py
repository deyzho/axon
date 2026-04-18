"""Tests for the FastAPI inference handler (create_inference_app).

Uses httpx ASGI transport — no real server, no ports, no respx needed for
the handler layer itself. The router's outbound HTTP calls are tested separately
in test_inference_router.py.
"""

from __future__ import annotations

import json
import pytest
import httpx
import respx

from axon.inference.handler import create_inference_app

SECRET = "test-secret-key-xyz"
IONET_URL = "https://ionet-handler-test.example.com"
AKASH_URL = "https://akash-handler-test.example.com"

FAKE_COMPLETION = {
    "id": "chatcmpl-handler-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "axon-llama-3-70b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
}

SSE_RESPONSE = "\n".join([
    'data: {"id":"c","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"},"finish_reason":null,"index":0}]}',
    "data: [DONE]",
    "",
])


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> httpx.AsyncClient:
    monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
    monkeypatch.setenv("AKASH_INFERENCE_URL", AKASH_URL)
    fastapi_app = create_inference_app(secret_key=SECRET)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fastapi_app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {SECRET}"},
    )


# ─── GET /v1/models ──────────────────────────────────────────────────────────

class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_200(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.get("/v1/models")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_model_list_object(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.get("/v1/models")
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_model_list_is_non_empty(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.get("/v1/models")
        data = resp.json()
        assert len(data["data"]) > 0

    @pytest.mark.asyncio
    async def test_each_model_has_required_fields(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.get("/v1/models")
        for model in resp.json()["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"

    @pytest.mark.asyncio
    async def test_axon_llama_3_70b_in_list(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert "axon-llama-3-70b" in ids

    @pytest.mark.asyncio
    async def test_returns_401_without_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        fastapi_app = create_inference_app(secret_key=SECRET)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fastapi_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_401_with_wrong_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        fastapi_app = create_inference_app(secret_key=SECRET)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=fastapi_app),
            base_url="http://test",
            headers={"Authorization": "Bearer wrong-key"},
        ) as client:
            resp = await client.get("/v1/models")
        assert resp.status_code == 401


# ─── POST /v1/chat/completions — validation ───────────────────────────────────

class TestChatCompletionsValidation:
    @pytest.mark.asyncio
    async def test_missing_model_returns_400(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_messages_returns_400(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={"model": "axon-llama-3-70b"},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_model_returns_502(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={"model": "not-a-real-model", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_empty_messages_list_returns_400(self, app: httpx.AsyncClient) -> None:
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={"model": "axon-llama-3-70b", "messages": []},
            )
        assert resp.status_code == 400


# ─── POST /v1/chat/completions — successful routing ──────────────────────────

class TestChatCompletionsRouting:
    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_response_shape(
        self, app: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=FAKE_COMPLETION)
        )
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={
                    "model": "axon-llama-3-70b",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert isinstance(data["choices"], list)

    @pytest.mark.asyncio
    @respx.mock
    async def test_provider_error_returns_502(
        self, app: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        monkeypatch.delenv("AKASH_INFERENCE_URL", raising=False)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="upstream error")
        )
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={
                    "model": "axon-llama-3-70b",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_response_is_event_stream(
        self, app: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=SSE_RESPONSE,
                headers={"Content-Type": "text/event-stream"},
            )
        )
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={
                    "model": "axon-llama-3-70b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_response_contains_data_lines(
        self, app: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, text=SSE_RESPONSE,
                headers={"Content-Type": "text/event-stream"},
            )
        )
        async with app:
            resp = await app.post(
                "/v1/chat/completions",
                json={
                    "model": "axon-llama-3-70b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
        body = resp.text
        assert "data:" in body

    @pytest.mark.asyncio
    @respx.mock
    async def test_additional_openai_params_forwarded(
        self, app: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        captured_route = respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=FAKE_COMPLETION)
        )
        async with app:
            await app.post(
                "/v1/chat/completions",
                json={
                    "model": "axon-llama-3-70b",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.5,
                    "max_tokens": 256,
                },
            )
        sent = json.loads(captured_route.calls[0].request.content)
        assert sent.get("temperature") == 0.5
        assert sent.get("max_tokens") == 256
