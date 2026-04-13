"""Unit tests for FluenceProvider."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from axon.providers.fluence import FluenceProvider
from axon.exceptions import AuthError, ProviderError
from axon.types import CostEstimate, DeploymentConfig, RuntimeType


@pytest.mark.asyncio
async def test_connect_requires_key():
    """connect() with empty key and no FLUENCE_PRIVATE_KEY env var raises AuthError."""
    provider = FluenceProvider()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AuthError, match="FLUENCE_PRIVATE_KEY"):
            await provider.connect("")


@pytest.mark.asyncio
async def test_estimate_returns_cost():
    """estimate() returns a CostEstimate with provider='fluence'."""
    provider = FluenceProvider()
    config = DeploymentConfig(
        name="test-spell",
        entry_point="src/spell.js",
        runtime=RuntimeType.NODEJS,
        memory_mb=256,
        timeout_ms=30_000,
        replicas=1,
    )
    estimate = await provider.estimate(config)

    assert isinstance(estimate, CostEstimate)
    assert estimate.provider == "fluence"
    assert estimate.token == "FLT"
    assert estimate.amount >= 0


@pytest.mark.asyncio
async def test_disconnect_is_safe():
    """disconnect() when not connected does not raise."""
    provider = FluenceProvider()
    # Should not raise even though we never connected
    await provider.disconnect()
    assert not provider._connected


@pytest.mark.asyncio
async def test_on_message_returns_unsubscribe():
    """on_message() registers a handler and returns a callable that unsubscribes it."""
    provider = FluenceProvider()
    received: list = []

    def handler(msg):
        received.append(msg)

    unsubscribe = provider.on_message(handler)
    assert handler in provider._message_handlers
    assert callable(unsubscribe)

    unsubscribe()
    assert handler not in provider._message_handlers
