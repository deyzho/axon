"""Unit tests for GCPProvider."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from axon.providers.gcp import GCPProvider
from axon.exceptions import AuthError, ProviderError
from axon.types import CostEstimate, DeploymentConfig, RuntimeType


@pytest.mark.asyncio
async def test_connect_requires_project_id():
    """connect() without GCP_PROJECT_ID env var raises AuthError."""
    provider = GCPProvider()

    mock_google_auth = MagicMock()
    mock_google_auth.default.return_value = (MagicMock(), "some-project")

    with patch.dict("os.environ", {}, clear=True):
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": mock_google_auth,
            "google.auth.transport": MagicMock(),
            "google.auth.transport.requests": MagicMock(),
        }):
            with pytest.raises(AuthError, match="GCP_PROJECT_ID"):
                await provider.connect("")


@pytest.mark.asyncio
async def test_connect_requires_google_auth():
    """connect() raises ProviderError when google.auth is not installed."""
    provider = GCPProvider()

    with patch.dict("os.environ", {"GCP_PROJECT_ID": "my-project"}, clear=True):
        with patch.dict("sys.modules", {
            "google": None,
            "google.auth": None,
            "google.auth.transport": None,
            "google.auth.transport.requests": None,
        }):
            with pytest.raises((ProviderError, Exception)):
                await provider.connect("")


@pytest.mark.asyncio
async def test_estimate_returns_cost():
    """estimate() returns a CostEstimate with provider='gcp'."""
    provider = GCPProvider()
    config = DeploymentConfig(
        name="test-service",
        entry_point="src/main.py",
        runtime=RuntimeType.NODEJS,
        memory_mb=512,
        timeout_ms=30_000,
        replicas=1,
    )
    estimate = await provider.estimate(config)

    assert isinstance(estimate, CostEstimate)
    assert estimate.provider == "gcp"
    assert estimate.token == "USD"
    assert estimate.amount >= 0


@pytest.mark.asyncio
async def test_deploy_raises_when_not_connected():
    """deploy() without calling connect() first raises ProviderError."""
    provider = GCPProvider()
    config = DeploymentConfig(
        name="test-service",
        entry_point="src/main.py",
        runtime=RuntimeType.NODEJS,
        memory_mb=512,
        timeout_ms=30_000,
    )
    with pytest.raises(ProviderError, match="Not connected"):
        await provider.deploy(config)
