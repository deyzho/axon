"""Tests for AxonInferenceRouter — model routing and HTTP forwarding."""

from __future__ import annotations

import json
import pytest
import respx
import httpx

from axon.exceptions import ProviderError
from axon.inference.router import AxonInferenceRouter, AXON_MODELS


# ─── Model registry ──────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_all_models_have_required_keys(self) -> None:
        for model_id, info in AXON_MODELS.items():
            assert "provider" in info, f"{model_id} missing 'provider'"
            assert "hardware" in info, f"{model_id} missing 'hardware'"
            assert "description" in info, f"{model_id} missing 'description'"

    def test_model_providers_are_known_values(self) -> None:
        valid = {"ionet", "akash", "acurast", "fluence", "koii"}
        for model_id, info in AXON_MODELS.items():
            assert info["provider"] in valid, (
                f"{model_id} has unknown provider '{info['provider']}'"
            )

    def test_at_least_four_models_defined(self) -> None:
        assert len(AXON_MODELS) >= 4

    def test_axon_llama_3_70b_is_present(self) -> None:
        assert "axon-llama-3-70b" in AXON_MODELS

    def test_axon_mistral_7b_is_present(self) -> None:
        assert "axon-mistral-7b" in AXON_MODELS


# ─── route() — unknown model ─────────────────────────────────────────────────

class TestRouteUnknownModel:
    @pytest.mark.asyncio
    async def test_unknown_model_raises_provider_error(self) -> None:
        router = AxonInferenceRouter({"secret_key": "test"})
        with pytest.raises(ProviderError, match="Unknown model"):
            await router.route(model="not-a-real-model", messages=[])

    @pytest.mark.asyncio
    async def test_error_message_lists_available_models(self) -> None:
        router = AxonInferenceRouter({"secret_key": "test"})
        with pytest.raises(ProviderError) as exc_info:
            await router.route(model="bad-model", messages=[])
        assert "axon-llama-3-70b" in str(exc_info.value)


# ─── route() — missing env var ───────────────────────────────────────────────

class TestRouteMissingEnvVar:
    @pytest.mark.asyncio
    async def test_missing_endpoint_env_var_raises_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("IONET_INFERENCE_URL", raising=False)
        router = AxonInferenceRouter({"secret_key": "test"})
        with pytest.raises(ProviderError, match="IONET_INFERENCE_URL"):
            await router.route(
                model="axon-llama-3-70b",
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_akash_missing_env_var_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AKASH_INFERENCE_URL", raising=False)
        router = AxonInferenceRouter({"secret_key": "test"})
        with pytest.raises(ProviderError, match="AKASH_INFERENCE_URL"):
            await router.route(
                model="axon-llama-3-8b",
                messages=[{"role": "user", "content": "hi"}],
            )


# ─── route() — non-streaming (successful) ────────────────────────────────────

FAKE_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "axon-llama-3-70b",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
}

IONET_URL = "https://ionet-test.example.com"
AKASH_URL = "https://akash-test.example.com"


class TestRouteNonStreaming:
    @pytest.mark.asyncio
    @respx.mock
    async def test_successful_completion_returns_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=FAKE_COMPLETION)
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        result = await router.route(
            model="axon-llama-3-70b",
            messages=[{"role": "user", "content": "ping"}],
        )
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "pong"

    @pytest.mark.asyncio
    @respx.mock
    async def test_request_forwarded_with_correct_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        route = respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=FAKE_COMPLETION)
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        messages = [{"role": "user", "content": "hello"}]
        await router.route(model="axon-llama-3-70b", messages=messages)

        sent = json.loads(route.calls[0].request.content)
        assert sent["model"] == "axon-llama-3-70b"
        assert sent["messages"] == messages
        assert sent["stream"] is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_provider_500_raises_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        with pytest.raises(ProviderError, match="HTTP 500"):
            await router.route(
                model="axon-llama-3-70b",
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_provider_401_raises_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        with pytest.raises(ProviderError):
            await router.route(
                model="axon-llama-3-70b",
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error_raises_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("ECONNREFUSED")
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        with pytest.raises(ProviderError, match="Could not reach"):
            await router.route(
                model="axon-llama-3-70b",
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_routes_akash_model_to_akash_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AKASH_INFERENCE_URL", AKASH_URL)
        akash_route = respx.post(f"{AKASH_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=FAKE_COMPLETION)
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        await router.route(
            model="axon-llama-3-8b",   # routes to akash
            messages=[{"role": "user", "content": "hi"}],
        )
        assert akash_route.called


# ─── route() — SSE streaming ─────────────────────────────────────────────────

SSE_CHUNKS = "\n".join([
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"He"},"finish_reason":null,"index":0}]}',
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{"content":"llo"},"finish_reason":null,"index":0}]}',
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop","index":0}]}',
    "data: [DONE]",
    "",
])


class TestRouteStreaming:
    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_yields_parsed_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text=SSE_CHUNKS,
                headers={"Content-Type": "text/event-stream"},
            )
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        gen = await router.route(
            model="axon-llama-3-70b",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        chunks = [chunk async for chunk in gen]
        assert len(chunks) == 3
        assert chunks[0]["object"] == "chat.completion.chunk"

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_stops_at_done_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=SSE_CHUNKS,
                                        headers={"Content-Type": "text/event-stream"})
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        gen = await router.route(
            model="axon-llama-3-70b",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        chunks = [chunk async for chunk in gen]
        # [DONE] should not appear as a parsed chunk
        for chunk in chunks:
            assert chunk != "[DONE]"
            assert "object" in chunk

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_skips_malformed_json_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        bad_sse = "\n".join([
            "data: {broken json here",
            'data: {"id":"c1","object":"chat.completion.chunk","choices":[]}',
            "data: [DONE]",
            "",
        ])
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text=bad_sse,
                                        headers={"Content-Type": "text/event-stream"})
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        gen = await router.route(
            model="axon-llama-3-70b",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        chunks = [chunk async for chunk in gen]
        # Malformed chunk skipped, only the valid one yields
        assert len(chunks) == 1
        assert chunks[0]["object"] == "chat.completion.chunk"

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_request_sets_stream_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        route = respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(200, text="data: [DONE]\n",
                                        headers={"Content-Type": "text/event-stream"})
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        gen = await router.route(
            model="axon-llama-3-70b",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in gen:
            pass
        sent = json.loads(route.calls[0].request.content)
        assert sent["stream"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_streaming_500_raises_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("IONET_INFERENCE_URL", IONET_URL)
        respx.post(f"{IONET_URL}/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Error")
        )
        router = AxonInferenceRouter({"secret_key": "test"})
        gen = await router.route(
            model="axon-llama-3-70b",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        with pytest.raises(ProviderError):
            async for _ in gen:
                pass
