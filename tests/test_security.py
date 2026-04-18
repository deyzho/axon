"""Unit tests for axon.security — URL safety validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from axon.exceptions import AxonError
from axon.security import assert_safe_url, _looks_like_ip


# ---------------------------------------------------------------------------
# Valid URLs — must pass without raising
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://api.example.com",
    "https://io.net/endpoint",
    "https://inference.example.org/v1/chat",
    "https://203.0.113.5",          # TEST-NET — not in private ranges
    "https://8.8.8.8",              # Public IP
])
def test_valid_urls_pass(url: str) -> None:
    # DNS resolution may fail for test addresses — patch it out so only
    # the static regex rules are exercised here.
    with patch("axon.security.socket.gethostbyname", side_effect=OSError("no dns")):
        assert_safe_url(url, provider="test", label="URL")  # must not raise


# ---------------------------------------------------------------------------
# HTTP (non-HTTPS) — must be blocked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://api.example.com",
    "http://8.8.8.8",
])
def test_http_scheme_blocked(url: str) -> None:
    with pytest.raises(AxonError, match="HTTPS"):
        assert_safe_url(url, provider="test")


# ---------------------------------------------------------------------------
# file:// scheme — must be blocked (caught by HTTPS check)
# ---------------------------------------------------------------------------

def test_file_scheme_blocked() -> None:
    with pytest.raises(AxonError, match="HTTPS"):
        assert_safe_url("file:///etc/passwd", provider="test")


# ---------------------------------------------------------------------------
# RFC-1918 private addresses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    # Class A — 10.x.x.x
    "https://10.0.0.1",
    "https://10.255.255.255",
    # Class B — 172.16–31.x.x
    "https://172.16.0.1",
    "https://172.20.5.5",
    "https://172.31.255.255",
    # Class C — 192.168.x.x
    "https://192.168.0.1",
    "https://192.168.1.100",
])
def test_rfc1918_blocked(url: str) -> None:
    with pytest.raises(AxonError, match="private"):
        assert_safe_url(url, provider="test")


# ---------------------------------------------------------------------------
# Loopback addresses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://127.0.0.1",
    "https://127.1.2.3",
    "https://localhost",
])
def test_ipv4_loopback_blocked(url: str) -> None:
    with pytest.raises(AxonError, match="private"):
        assert_safe_url(url, provider="test")


# ---------------------------------------------------------------------------
# Link-local / cloud metadata service
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://169.254.169.254",       # AWS IMDS
    "https://169.254.0.1",
])
def test_link_local_blocked(url: str) -> None:
    with pytest.raises(AxonError, match="private"):
        assert_safe_url(url, provider="test")


# ---------------------------------------------------------------------------
# IPv6 loopback and link-local
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://[::1]",
    "https://[fe80::1]",
    "https://[fe80::dead:beef]",
])
def test_ipv6_private_blocked(url: str) -> None:
    with pytest.raises(AxonError, match="private"):
        assert_safe_url(url, provider="test")


# ---------------------------------------------------------------------------
# 172.x addresses just outside the blocked range must NOT be blocked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://172.15.255.255",   # One below 172.16 — public
    "https://172.32.0.1",       # One above 172.31 — public
])
def test_172_boundary_allowed(url: str) -> None:
    with patch("axon.security.socket.gethostbyname", side_effect=OSError("no dns")):
        assert_safe_url(url, provider="test")  # must not raise


# ---------------------------------------------------------------------------
# DNS rebinding: hostname that resolves to a private IP must be blocked
# ---------------------------------------------------------------------------

def test_dns_rebinding_blocked() -> None:
    with patch("axon.security.socket.gethostbyname", return_value="192.168.1.1"):
        with pytest.raises(AxonError, match="resolves to a private"):
            assert_safe_url("https://evil.example.com", provider="test")


def test_dns_rebinding_to_loopback_blocked() -> None:
    with patch("axon.security.socket.gethostbyname", return_value="127.0.0.1"):
        with pytest.raises(AxonError, match="resolves to a private"):
            assert_safe_url("https://evil.example.com", provider="test")


def test_dns_rebinding_to_imds_blocked() -> None:
    with patch("axon.security.socket.gethostbyname", return_value="169.254.169.254"):
        with pytest.raises(AxonError, match="resolves to a private"):
            assert_safe_url("https://metadata.internal", provider="test")


def test_dns_failure_does_not_block_valid_url() -> None:
    """A transient DNS error must not cause a legitimate URL to be rejected."""
    with patch("axon.security.socket.gethostbyname", side_effect=OSError("timeout")):
        assert_safe_url("https://api.example.com", provider="test")  # must not raise


# ---------------------------------------------------------------------------
# _looks_like_ip helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True),
    ("::1", True),
    ("[::1]", True),
    ("[fe80::1]", True),
    ("example.com", False),
    ("localhost", False),
    ("api.io.net", False),
])
def test_looks_like_ip(host: str, expected: bool) -> None:
    assert _looks_like_ip(host) == expected


# ---------------------------------------------------------------------------
# Error message includes provider and label
# ---------------------------------------------------------------------------

def test_error_message_includes_provider_and_label() -> None:
    with pytest.raises(AxonError) as exc_info:
        assert_safe_url("http://10.0.0.1", provider="ionet", label="Worker URL")
    msg = str(exc_info.value)
    assert "ionet" in msg
    assert "Worker URL" in msg
