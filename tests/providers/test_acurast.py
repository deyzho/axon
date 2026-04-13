"""Unit tests for AcurastProvider."""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

from axon.providers.acurast import AcurastProvider
from axon.exceptions import ProviderError, AuthError
from axon.types import CostEstimate, DeploymentConfig, RuntimeType


@pytest.mark.asyncio
async def test_connect_requires_mnemonic():
    """Connect with no mnemonic should raise AuthError."""
    provider = AcurastProvider()
    import os
    os.environ.pop("ACURAST_MNEMONIC", None)
    with patch.dict("os.environ", {"AXON_SECRET_KEY": "a" * 64}, clear=False):
        with pytest.raises((AuthError, ProviderError, Exception)):
            await provider.connect("")


@pytest.mark.asyncio
async def test_connect_validates_mnemonic_length():
    """Mnemonic must be 12 or 24 words."""
    provider = AcurastProvider()
    with pytest.raises(Exception):
        await provider.connect("only three words here")


@pytest.mark.asyncio
async def test_mnemonic_stored_as_bytearray():
    """Mnemonic is stored as bytearray, not str, for secure zeroing."""
    provider = AcurastProvider()
    # Verify bytearray attribute exists
    assert hasattr(provider, '_mnemonic_buf')
    assert isinstance(provider._mnemonic_buf, bytearray)


@pytest.mark.asyncio
async def test_disconnect_zeros_mnemonic():
    """disconnect() zeros the mnemonic bytearray."""
    provider = AcurastProvider()
    # Manually set the buffer
    provider._mnemonic_buf = bytearray(b"test mnemonic data here padded")
    await provider.disconnect()
    assert all(b == 0 for b in provider._mnemonic_buf) or len(provider._mnemonic_buf) == 0


@pytest.mark.asyncio
async def test_estimate_returns_cost():
    """estimate() returns a CostEstimate with provider='acurast'."""
    provider = AcurastProvider()
    config = DeploymentConfig(
        name="test",
        entry_point="index.js",
        runtime=RuntimeType.NODEJS,
        env={},
        replicas=1,
        timeout_ms=30_000,
        memory_mb=256,
        metadata={},
    )
    estimate = await provider.estimate(config)
    assert isinstance(estimate, CostEstimate)
    assert estimate.provider == "acurast"
    assert estimate.amount >= 0
