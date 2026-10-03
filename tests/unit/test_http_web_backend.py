"""The production web backend, against a scripted HTTP transport.

The domain policy has already approved the first URL by the time the backend runs, so
these tests are about what only the transport can enforce: where a name resolves, where
a redirect leads, how much a server sends back.
"""

from collections.abc import Callable

import httpx
import pytest

from research_platform.mcp.servers.http_web import HttpWebBackend, extract_readable
from research_platform.mcp.servers.web_boundary import SourceNotAllowed

PUBLIC = ["93.184.216.34"]

PAGE = """
<html>
  <head><title>  Pricing   update </title><style>body { color: red }</style></head>
  <body>
    <script>track("visitor")</script>
    <h1>Pricing</h1>
    <p>The list price fell to <b>4 USD</b> per million tokens.</p>
    <noscript>Enable scripts</noscript>
  </body>
</html>
"""

Handler = Callable[[httpx.Request], httpx.Response]


def backend(
    handler: Handler,
    *,
    resolves_to: dict[str, list[str]] | None = None,
    permitted: Callable[[str], bool] = lambda url: "vendor.test" in url,
    search_url: str | None = None,
    max_bytes: int = 2_000_000,
) -> HttpWebBackend:
    addresses = resolves_to or {}
    return HttpWebBackend(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        is_permitted=permitted,
        search_url=search_url,
        max_bytes=max_bytes,
        resolve=lambda host: addresses.get(host, PUBLIC),
    )


def html(body: str = PAGE) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=body)


def test_a_page_is_reduced_to_its_title_and_readable_text() -> None:
    document = backend(lambda _request: html()).fetch("https://vendor.test/pricing")

    assert document.title == "Pricing update"
    assert document.text == "Pricing\nThe list price fell to 4 USD per million tokens."


def test_scripts_styles_and_noscript_never_reach_the_agent() -> None:
    _title, text = extract_readable(PAGE)

    assert "track(" not in text
    assert "color: red" not in text
    assert "Enable scripts" not in text


def test_plain_text_is_returned_as_it_was_served() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="  4 USD  \n")

    document = backend(handler).fetch("https://vendor.test/price.txt")

    assert document.text == "4 USD"
    assert document.title == "https://vendor.test/price.txt"


def test_a_name_that_resolves_to_a_private_address_is_refused_before_any_request() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return html()

    internal = backend(handler, resolves_to={"vendor.test": ["10.0.0.5"]})

    with pytest.raises(SourceNotAllowed, match="public address"):
        internal.fetch("https://vendor.test/pricing")
    assert requested == []


def test_a_name_with_any_private_address_among_several_is_refused() -> None:
    mixed = backend(
        lambda _request: html(), resolves_to={"vendor.test": ["93.184.216.34", "127.0.0.1"]}
    )

    with pytest.raises(SourceNotAllowed):
        mixed.fetch("https://vendor.test/pricing")


def test_a_name_that_does_not_resolve_at_all_is_refused() -> None:
    unresolved = backend(lambda _request: html(), resolves_to={"vendor.test": []})

    with pytest.raises(SourceNotAllowed):
        unresolved.fetch("https://vendor.test/pricing")


def test_a_redirect_within_the_approved_sources_is_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(301, headers={"location": "/pricing"})
        return html()

    document = backend(handler).fetch("https://vendor.test/old")

    assert document.url == "https://vendor.test/pricing"


def test_a_redirect_to_an_unapproved_source_is_refused_rather_than_followed() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://elsewhere.test/collect"})

    with pytest.raises(SourceNotAllowed, match="unapproved source"):
        backend(handler).fetch("https://vendor.test/pricing")
    assert requested == ["vendor.test"]


def test_a_redirect_to_an_internal_address_is_refused_even_on_an_approved_name() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://internal.vendor.test/admin"})

    rebinding = backend(handler, resolves_to={"internal.vendor.test": ["169.254.169.254"]})

    with pytest.raises(SourceNotAllowed, match="public address"):
        rebinding.fetch("https://vendor.test/pricing")


def test_an_endless_redirect_chain_is_cut_off() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": f"{request.url.path}x"})

    with pytest.raises(SourceNotAllowed, match="redirected more than"):
        backend(handler).fetch("https://vendor.test/a")


def test_a_response_larger_than_the_limit_is_refused_not_truncated() -> None:
    oversized = backend(lambda _request: html("x" * 5_000), max_bytes=1_000)

    with pytest.raises(LookupError, match="response limit"):
        oversized.fetch("https://vendor.test/pricing")


def test_a_binary_response_is_refused() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")

    with pytest.raises(LookupError, match="not readable text"):
        backend(handler).fetch("https://vendor.test/report.pdf")


def test_a_missing_page_is_reported_as_not_found() -> None:
    with pytest.raises(LookupError, match="404"):
        backend(lambda _request: httpx.Response(404)).fetch("https://vendor.test/gone")


def test_a_page_with_no_readable_text_is_reported_as_empty() -> None:
    with pytest.raises(LookupError, match="no readable content"):
        backend(lambda _request: html("<html><script>x()</script></html>")).fetch(
            "https://vendor.test/blank"
        )


def test_a_transport_failure_is_reported_as_unfetchable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection reset", request=request)

    with pytest.raises(RuntimeError, match="could not be fetched"):
        backend(handler).fetch("https://vendor.test/pricing")


def test_search_returns_the_engines_results_up_to_the_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "token pricing"
        assert request.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": "https://vendor.test/a", "title": "A", "content": "first"},
                    {"url": "https://vendor.test/b", "title": "", "content": None},
                    {"title": "no url"},
                    {"url": "https://vendor.test/c", "title": "C"},
                ]
            },
        )

    found = backend(handler, search_url="https://search.test/search").search(
        "token pricing", limit=2
    )

    assert [(document.url, document.title, document.text) for document in found] == [
        ("https://vendor.test/a", "A", "first"),
        ("https://vendor.test/b", "https://vendor.test/b", ""),
    ]


def test_search_is_refused_outright_when_no_engine_is_configured() -> None:
    with pytest.raises(LookupError, match="no search engine"):
        backend(lambda _request: html()).search("anything", limit=5)


def test_a_failing_search_engine_is_reported_rather_than_answered_with_nothing() -> None:
    failing = backend(lambda _request: httpx.Response(502), search_url="https://search.test/s")

    with pytest.raises(RuntimeError, match="could not be queried"):
        failing.search("anything", limit=5)
