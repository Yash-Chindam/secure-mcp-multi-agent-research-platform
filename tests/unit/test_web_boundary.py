from datetime import timedelta

import pytest

from research_platform.mcp.breaker import MutableClock
from research_platform.mcp.servers.web_boundary import (
    DomainPolicy,
    RateLimitExceeded,
    SlidingWindowRateLimiter,
    SourceNotAllowed,
    normalize_source_url,
    rate_limit_key,
)

POLICY = DomainPolicy(domains=frozenset({"vendor.test", "regulator.test"}))


def normalize(url: str, policy: DomainPolicy = POLICY) -> str:
    return normalize_source_url(url, policy=policy)


def test_a_policy_must_allow_at_least_one_domain() -> None:
    with pytest.raises(ValueError, match="at least one domain"):
        DomainPolicy(domains=frozenset())


@pytest.mark.parametrize("domain", ["Vendor.test", " vendor.test", "vendor.test/path", ""])
def test_a_policy_rejects_a_malformed_domain(domain: str) -> None:
    with pytest.raises(ValueError, match="invalid allowlisted domain"):
        DomainPolicy(domains=frozenset({domain}))


def test_an_allowlisted_url_is_returned_normalized() -> None:
    assert normalize("https://vendor.test/pricing?plan=team") == (
        "https://vendor.test/pricing?plan=team"
    )


def test_a_bare_host_gains_a_root_path() -> None:
    assert normalize("https://vendor.test") == "https://vendor.test/"


def test_the_host_is_lowercased() -> None:
    assert normalize("https://VENDOR.test/Pricing") == "https://vendor.test/Pricing"


def test_a_subdomain_is_allowed_by_default() -> None:
    assert normalize("https://docs.vendor.test/api") == "https://docs.vendor.test/api"


def test_subdomains_can_be_refused() -> None:
    strict = DomainPolicy(domains=frozenset({"vendor.test"}), allow_subdomains=False)

    with pytest.raises(SourceNotAllowed, match="not an approved research source"):
        normalize("https://docs.vendor.test/api", strict)


def test_a_lookalike_suffix_is_not_treated_as_a_subdomain() -> None:
    with pytest.raises(SourceNotAllowed, match="not an approved research source"):
        normalize("https://notvendor.test/pricing")


def test_a_host_outside_the_allowlist_is_refused() -> None:
    with pytest.raises(SourceNotAllowed, match="not an approved research source"):
        normalize("https://attacker.test/exfiltrate")


@pytest.mark.parametrize("url", ["http://vendor.test/", "file:///etc/passwd", "ftp://vendor.test/"])
def test_only_https_is_permitted(url: str) -> None:
    with pytest.raises(SourceNotAllowed, match="is not permitted for research"):
        normalize(url)


def test_a_url_without_a_scheme_is_refused() -> None:
    with pytest.raises(SourceNotAllowed, match="is not permitted for research"):
        normalize("vendor.test/pricing")


def test_embedded_credentials_are_refused_rather_than_stripped() -> None:
    with pytest.raises(SourceNotAllowed, match="embedded credentials"):
        normalize("https://user:secret@vendor.test/pricing")


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.1",
        "172.16.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "[::1]",
        "[fe80::1]",
    ],
)
def test_private_and_loopback_addresses_are_refused(host: str) -> None:
    permissive = DomainPolicy(domains=frozenset({"vendor.test"}))

    with pytest.raises(SourceNotAllowed, match="not a public research source"):
        normalize(f"https://{host}/pricing", permissive)


@pytest.mark.parametrize("host", ["metadata.google.internal", "instance-data"])
def test_cloud_metadata_hosts_are_refused(host: str) -> None:
    with pytest.raises(SourceNotAllowed, match="not a public research source"):
        normalize(f"https://{host}/latest/meta-data", POLICY)


@pytest.mark.parametrize("host", ["vendor.test.local", "files.corp", "db.internal"])
def test_private_network_suffixes_are_refused(host: str) -> None:
    permissive = DomainPolicy(domains=frozenset({"local", "corp", "internal"}))

    with pytest.raises(SourceNotAllowed, match="inside a private network"):
        normalize(f"https://{host}/", permissive)


def test_a_non_standard_port_is_refused() -> None:
    with pytest.raises(SourceNotAllowed, match="port 8443 is not permitted"):
        normalize("https://vendor.test:8443/pricing")


def test_the_standard_https_port_is_accepted() -> None:
    assert normalize("https://vendor.test:443/pricing") == "https://vendor.test/pricing"


def test_a_fragment_is_not_forwarded_to_the_source() -> None:
    assert normalize("https://vendor.test/pricing#section") == "https://vendor.test/pricing"


def test_the_rate_limit_must_allow_a_request() -> None:
    with pytest.raises(ValueError, match="at least one request"):
        SlidingWindowRateLimiter(limit=0)


def test_requests_within_the_limit_are_admitted() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, clock=MutableClock())

    assert limiter.acquire("acme|vendor.test") == 1
    assert limiter.acquire("acme|vendor.test") == 0


def test_exceeding_the_limit_is_refused() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, clock=MutableClock())
    limiter.acquire("acme|vendor.test")

    with pytest.raises(RateLimitExceeded) as error:
        limiter.acquire("acme|vendor.test")

    assert error.value.limit == 1


def test_the_window_slides_and_restores_the_allowance() -> None:
    clock = MutableClock()
    limiter = SlidingWindowRateLimiter(limit=1, window=timedelta(seconds=60), clock=clock)
    limiter.acquire("acme|vendor.test")

    clock.advance(timedelta(seconds=61))

    assert limiter.remaining("acme|vendor.test") == 1
    limiter.acquire("acme|vendor.test")


def test_one_tenant_cannot_exhaust_another_allowance() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, clock=MutableClock())
    limiter.acquire(rate_limit_key("acme", "https://vendor.test/a"))

    limiter.acquire(rate_limit_key("globex", "https://vendor.test/a"))


def test_the_limit_is_applied_per_host() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, clock=MutableClock())
    limiter.acquire(rate_limit_key("acme", "https://vendor.test/a"))

    limiter.acquire(rate_limit_key("acme", "https://regulator.test/a"))


def test_the_rate_limit_key_names_the_tenant_and_host() -> None:
    assert rate_limit_key("acme", "https://VENDOR.test/pricing") == "acme|vendor.test"


def test_the_rate_limit_key_tolerates_an_unparsable_url() -> None:
    assert rate_limit_key("acme", "not-a-url") == "acme|unknown"
