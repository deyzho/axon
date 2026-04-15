"""
Axon — Provider-agnostic edge compute SDK for AI workload routing.

Quickstart:
    from axon import AxonClient

    client = AxonClient(provider="ionet", secret_key="your_key")
    await client.connect()
    deployment = await client.deploy(config)
"""

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

__version__ = "0.1.6"

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
