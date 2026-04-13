"""
Provider pricing — live where public APIs exist, documented static fallbacks elsewhere.

Sources (static values last verified 2025-01):
  AWS Lambda   : https://aws.amazon.com/lambda/pricing/
  GCP Cloud Run: https://cloud.google.com/run/pricing
  Azure ACI    : https://prices.azure.com/api/retail/prices  (live API, no auth)
  Cloudflare   : https://developers.cloudflare.com/workers/platform/pricing/
  Fly.io       : https://fly.io/docs/about/pricing/
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ProviderPricing:
    # AWS Lambda (us-east-1)
    aws_lambda_gb_sec: float = 0.0000166667   # per GB-second of compute
    aws_lambda_request: float = 0.0000002     # per invocation ($0.20/1M)

    # GCP Cloud Run
    gcp_run_vcpu_sec: float = 0.00002400      # per vCPU-second
    gcp_run_gib_sec: float = 0.00000250       # per GiB-second of memory
    gcp_run_request: float = 0.0000004        # per request ($0.40/1M)

    # Azure Container Instances (Linux, eastus)
    azure_aci_vcpu_sec: float = 0.0000135     # per vCPU-second
    azure_aci_gib_sec: float = 0.0000015      # per GiB-second of memory

    # Cloudflare Workers (paid plan)
    cf_worker_request: float = 0.0000005      # per request ($0.50/1M)
    cf_worker_free_per_day: int = 100_000     # free tier requests/day

    # Fly.io Machines
    fly_shared_cpu_1x_hour: float = 0.0101    # shared-cpu-1x, 256 MB
    fly_shared_cpu_2x_hour: float = 0.0202    # shared-cpu-2x, 512 MB

    fetched_at: float = field(default_factory=time.time)
    source: str = "static"


_STATIC_PRICING = ProviderPricing()
_cached: ProviderPricing | None = None
_cache_expires_at: float = 0.0
_CACHE_TTL = 24 * 3600  # 24 hours


async def get_pricing() -> ProviderPricing:
    """
    Return current provider pricing. Fetches live Azure pricing from the
    Azure Retail Prices API and overlays it on static constants.
    Always falls back to static pricing on any network error.
    """
    global _cached, _cache_expires_at

    if _cached is not None and time.time() < _cache_expires_at:
        return _cached

    try:
        azure_prices = await _fetch_azure_live_pricing()
        pricing = ProviderPricing(
            **{**_STATIC_PRICING.__dict__, **azure_prices,
               "fetched_at": time.time(), "source": "live"}
        )
        _cached = pricing
        _cache_expires_at = time.time() + _CACHE_TTL
        return pricing
    except Exception:
        return _STATIC_PRICING


async def _fetch_azure_live_pricing() -> dict[str, Any]:
    """
    Fetch Container Instances Linux pricing from the Azure Retail Prices API.
    This is a public endpoint — no authentication required.
    """
    params = {
        "api-version": "2023-01-01-preview",
        "$filter": (
            "serviceName eq 'Container Instances' and "
            "armRegionName eq 'eastus' and "
            "priceType eq 'Consumption' and "
            "contains(skuName, 'Linux')"
        ),
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get("https://prices.azure.com/api/retail/prices", params=params)
        resp.raise_for_status()
        data = resp.json()

    result: dict[str, Any] = {}
    for item in data.get("Items", []):
        sku: str = item.get("skuName", "").lower()
        unit: str = item.get("unitOfMeasure", "").lower()
        price: float = float(item.get("retailPrice", 0))
        if price <= 0:
            continue
        if "vcpu" in sku and "second" in unit:
            result["azure_aci_vcpu_sec"] = price
        if (" gb " in sku or sku.endswith(" gb")) and "second" in unit:
            result["azure_aci_gib_sec"] = price

    return result


def clear_pricing_cache() -> None:
    """Reset the in-memory pricing cache (useful in tests)."""
    global _cached, _cache_expires_at
    _cached = None
    _cache_expires_at = 0.0
