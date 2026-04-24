"""
Axon — Provider-agnostic edge compute SDK for AI workload routing.

Quickstart:
    from axon import AxonClient

    client = AxonClient(provider="ionet", secret_key="your_key")
    await client.connect()
    deployment = await client.deploy(config)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from axon.client import AxonClient
from axon.exceptions import AxonError, ConfigError, ProviderError
from axon.router import AxonRouter
from axon.types import (
    CostEstimate,
    Deployment,
    DeploymentConfig,
    HealthStatus,
    Message,
    ProviderName,
    RoutingStrategy,
)
from axon.utils.retry import with_retry

try:
    __version__ = _pkg_version("axonsdk-py")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "AxonClient",
    "AxonRouter",
    "AxonError",
    "ProviderError",
    "ConfigError",
    "DeploymentConfig",
    "Deployment",
    "CostEstimate",
    "Message",
    "ProviderName",
    "RoutingStrategy",
    "HealthStatus",
    "with_retry",
]
