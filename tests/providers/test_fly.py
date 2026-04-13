"""Unit tests for FlyProvider."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from axon.providers.fly import FlyProvider
from axon.exceptions import AuthError, ProviderError
from axon.types import CostEstimate, DeploymentConfig, RuntimeType


@pytest.mark.asyncio
async def test_connect_requires_api_token():
    """connect() without FLY_API_TOKEN raises AuthError."""
    provider = FlyProvider()
    with patch.dict("os.environ", {"FLY_APP_NAME": "my-app"}, clear=True):
        with pytest.raises(AuthError, match="FLY_API_TOKEN"):
            await provider.connect("")


@pytest.mark.asyncio
async def test_connect_requires_app_name():
    """connect() without FLY_APP_NAME raises AuthError."""
    provider = FlyProvider()
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(AuthError):
            # Providing token via secret_key but no app name
            await provider.connect("fly_token_value")


@pytest.mark.asyncio
async def test_estimate_returns_cost():
    """estimate() returns a CostEstimate with provider='fly'."""
    provider = FlyProvider()
    config = DeploymentConfig(
        name="test-machine",
        entry_point="src/app.js",
        runtime=RuntimeType.DOCKER,
        memory_mb=256,
        timeout_ms=30_000,
        replicas=1,
    )
    estimate = await provider.estimate(config)

    assert isinstance(estimate, CostEstimate)
    assert estimate.provider == "fly"
    assert estimate.token == "USD"
    assert estimate.amount >= 0
    assert estimate.per_hour is True


@pytest.mark.asyncio
async def test_deploy_raises_when_not_connected():
    """deploy() without calling connect() first raises ProviderError."""
    provider = FlyProvider()
    config = DeploymentConfig(
        name="test-machine",
        entry_point="src/app.js",
        runtime=RuntimeType.DOCKER,
        memory_mb=256,
        timeout_ms=30_000,
    )
    with pytest.raises(ProviderError, match="Not connected"):
        await provider.deploy(config)
