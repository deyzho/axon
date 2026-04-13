"""Unit tests for KoiiProvider."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from axon.providers.koii import KoiiProvider
from axon.exceptions import AuthError, ProviderError
from axon.types import CostEstimate, DeploymentConfig, RuntimeType


@pytest.mark.asyncio
async def test_connect_requires_keypair():
    """connect() with empty key and no KOII_PRIVATE_KEY env var raises AuthError."""
    provider = KoiiProvider()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AuthError, match="KOII_PRIVATE_KEY"):
            await provider.connect("")


@pytest.mark.asyncio
async def test_estimate_returns_cost():
    """estimate() returns a CostEstimate with provider='koii'."""
    provider = KoiiProvider()
    config = DeploymentConfig(
        name="test-task",
        entry_point="src/task.js",
        runtime=RuntimeType.NODEJS,
        memory_mb=256,
        timeout_ms=30_000,
        replicas=2,
    )
    estimate = await provider.estimate(config)

    assert isinstance(estimate, CostEstimate)
    assert estimate.provider == "koii"
    assert estimate.token == "KOII"
    assert estimate.amount >= 0
    # Should scale with replicas
    assert estimate.amount == 2.0


@pytest.mark.asyncio
async def test_disconnect_cleans_up():
    """disconnect() when not connected doesn't raise."""
    provider = KoiiProvider()
    # Should be safe to call without having connected
    await provider.disconnect()
    assert not provider._connected
