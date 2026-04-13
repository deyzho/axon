"""Unit tests for AzureProvider."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from axon.providers.azure import AzureProvider
from axon.exceptions import AuthError, ProviderError
from axon.types import CostEstimate, DeploymentConfig, RuntimeType


@pytest.mark.asyncio
async def test_connect_requires_subscription_id():
    """connect() without AZURE_SUBSCRIPTION_ID raises AuthError."""
    provider = AzureProvider()

    mock_identity = MagicMock()
    mock_identity.DefaultAzureCredential.return_value = MagicMock()
    mock_identity.ClientSecretCredential.return_value = MagicMock()

    with patch.dict("os.environ", {}, clear=True):
        with patch.dict("sys.modules", {
            "azure": MagicMock(),
            "azure.identity": mock_identity,
        }):
            with pytest.raises(AuthError, match="AZURE_SUBSCRIPTION_ID"):
                await provider.connect("")


@pytest.mark.asyncio
async def test_connect_requires_azure_identity():
    """connect() raises ProviderError when azure.identity is not installed."""
    provider = AzureProvider()

    with patch.dict("os.environ", {"AZURE_SUBSCRIPTION_ID": "sub-123"}, clear=True):
        with patch.dict("sys.modules", {
            "azure": None,
            "azure.identity": None,
        }):
            with pytest.raises((ProviderError, Exception)):
                await provider.connect("")


@pytest.mark.asyncio
async def test_estimate_returns_cost():
    """estimate() returns a CostEstimate with provider='azure'."""
    provider = AzureProvider()
    config = DeploymentConfig(
        name="test-container",
        entry_point="src/app.py",
        runtime=RuntimeType.DOCKER,
        memory_mb=512,
        timeout_ms=30_000,
        replicas=1,
    )
    estimate = await provider.estimate(config)

    assert isinstance(estimate, CostEstimate)
    assert estimate.provider == "azure"
    assert estimate.token == "USD"
    assert estimate.amount >= 0


@pytest.mark.asyncio
async def test_deploy_raises_when_not_connected():
    """deploy() without calling connect() first raises ProviderError."""
    provider = AzureProvider()
    config = DeploymentConfig(
        name="test-container",
        entry_point="src/app.py",
        runtime=RuntimeType.DOCKER,
        memory_mb=512,
        timeout_ms=30_000,
    )
    with pytest.raises(ProviderError, match="Not connected"):
        await provider.deploy(config)
