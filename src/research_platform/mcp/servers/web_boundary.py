"""The boundary the web research server must be used within.

Section 8 requires a domain policy, a rate limit and a response-size limit. The domain
policy is also the platform's outbound-request boundary, so it refuses the addresses an
attacker would use to turn a research fetch into a request against internal
infrastructure.
"""

from __future__ import annotations

import ipaddress
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from urllib.parse import urlsplit

from research_platform.domain.models import utc_now
from research_platform.mcp.breaker import Clock

ALLOWED_SCHEMES = frozenset({"https"})

BLOCKED_HOST_SUFFIXES = (
    ".local",
    ".localdomain",
    ".internal",
    ".intranet",
    ".corp",
    ".home.arpa",
)

CLOUD_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)


class SourceNotAllowed(PermissionError):
    """Raised when a URL falls outside the approved research sources."""


@dataclass(frozen=True)
class DomainPolicy:
    """An explicit allowlist of the domains research may reach."""

    domains: frozenset[str]
    allow_subdomains: bool = True

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError("a domain policy must allow at least one domain")
        for domain in self.domains:
            if not domain or domain != domain.strip().lower() or "/" in domain:
                raise ValueError(f"invalid allowlisted domain: {domain!r}")

    def permits_host(self, host: str) -> bool:
        if host in self.domains:
            return True
        if not self.allow_subdomains:
            return False
        return any(host.endswith(f".{domain}") for domain in self.domains)


def _is_unroutable_address(host: str) -> bool:
    """Report whether a host is an IP literal that must never be fetched."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def normalize_source_url(url: str, *, policy: DomainPolicy) -> str:
    """Return the fetchable URL, or refuse it.

    Credentials, fragments and non-HTTPS schemes are refused rather than stripped, so a
    URL that was almost acceptable is never quietly turned into a different request.
    """
    parts = urlsplit(url.strip())

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise SourceNotAllowed(f"scheme {parts.scheme or '(none)'} is not permitted for research")
    if parts.username or parts.password:
        raise SourceNotAllowed("a research URL must not carry embedded credentials")

    host = (parts.hostname or "").lower()
    if not host:
        raise SourceNotAllowed("a research URL must name a host")
    if host in CLOUD_METADATA_HOSTS or _is_unroutable_address(host):
        raise SourceNotAllowed(f"host {host} is not a public research source")
    if any(host.endswith(suffix) for suffix in BLOCKED_HOST_SUFFIXES):
        raise SourceNotAllowed(f"host {host} resolves inside a private network")
    if not policy.permits_host(host):
        raise SourceNotAllowed(f"host {host} is not an approved research source")

    port = parts.port
    if port is not None and port != 443:
        raise SourceNotAllowed(f"port {port} is not permitted for research")

    path = parts.path or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"https://{host}{path}{query}"


class RateLimitExceeded(RuntimeError):
    def __init__(self, key: str, limit: int, window: timedelta) -> None:
        super().__init__(f"{key} exceeded {limit} requests per {int(window.total_seconds())}s")
        self.key = key
        self.limit = limit
        self.window = window


@dataclass
class SlidingWindowRateLimiter:
    """Cap outbound requests per tenant and host over a moving window."""

    limit: int
    window: timedelta = timedelta(seconds=60)
    clock: Clock = utc_now
    _hits: dict[str, deque[datetime]] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("a rate limit must allow at least one request")

    def remaining(self, key: str) -> int:
        with self._lock:
            return max(0, self.limit - len(self._prune(key)))

    def acquire(self, key: str) -> int:
        """Claim one request slot, refusing once the window is full."""
        with self._lock:
            hits = self._prune(key)
            if len(hits) >= self.limit:
                raise RateLimitExceeded(key, self.limit, self.window)
            hits.append(self.clock())
            return self.limit - len(hits)

    def _prune(self, key: str) -> deque[datetime]:
        hits = self._hits.setdefault(key, deque())
        cutoff = self.clock() - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        return hits


def rate_limit_key(tenant_id: str, url: str) -> str:
    """Rate limit per tenant and host, so one tenant cannot exhaust another's allowance."""
    host = (urlsplit(url).hostname or "unknown").lower()
    return f"{tenant_id}|{host}"
