"""The production web backend: real HTTP, behind the same boundary as the stand-in.

``WebResearchService`` has already checked the requested URL against the domain policy
by the time this is called. What only the transport can check is what happens *after*
that: where the name actually resolves, and where the server redirects to. Both are how
an approved-looking URL becomes a request against internal infrastructure, so both are
re-checked here on every hop rather than trusted from the first check.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

from research_platform.mcp.servers.backends import SourceDocument
from research_platform.mcp.servers.web_boundary import SourceNotAllowed

MAX_REDIRECTS = 5
READABLE_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json")
USER_AGENT = "secure-mcp-research-platform/0.1 (+research fetch)"

Resolver = Callable[[str], Iterable[str]]
"""Return every address a host name resolves to."""


def resolve_host(host: str) -> list[str]:
    try:
        return [str(info[4][0]) for info in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)]
    except socket.gaierror as error:
        raise LookupError(f"{host} could not be resolved") from error


def _is_public(address: str) -> bool:
    parsed = ipaddress.ip_address(address)
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_reserved
        or parsed.is_multicast
        or parsed.is_unspecified
    )


class _ReadableText(HTMLParser):
    """Reduce an HTML document to its title and the text a reader would see."""

    SKIPPED = frozenset({"script", "style", "noscript", "template", "svg", "head", "iframe"})
    BLOCKS = frozenset(
        {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._parts: list[str] = []
        self._skipping = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIPPED:
            self._skipping += 1
        elif tag in self.BLOCKS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIPPED and self._skipping:
            self._skipping -= 1
        elif tag in self.BLOCKS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skipping:
            self._parts.append(data)

    @property
    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line)


def extract_readable(html: str) -> tuple[str, str]:
    """Return ``(title, text)`` for an HTML document, dropping markup and scripts."""
    parser = _ReadableText()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), parser.text


@dataclass
class HttpWebBackend:
    """Fetch approved public sources over HTTPS and search through a configured engine.

    ``is_permitted`` is the domain policy, applied again to every redirect target: a
    permitted site that redirects elsewhere does not carry its permission with it.
    ``search_url`` is a SearXNG-compatible JSON search endpoint; with none configured a
    search is refused outright rather than answered with nothing.
    """

    client: httpx.Client
    is_permitted: Callable[[str], bool]
    search_url: str | None = None
    max_bytes: int = 2_000_000
    resolve: Resolver = field(default=resolve_host)

    def fetch(self, url: str) -> SourceDocument:
        final_url, content_type, body = self._get(url)
        if content_type in ("text/html", "application/xhtml+xml"):
            title, text = extract_readable(body)
        else:
            title, text = "", body.strip()
        if not text:
            raise LookupError(f"{final_url} returned no readable content")
        return SourceDocument(url=final_url, title=title or final_url, text=text)

    def search(self, query: str, *, limit: int) -> list[SourceDocument]:
        if not self.search_url:
            raise LookupError("no search engine is configured for web research")
        try:
            response = self.client.get(
                self.search_url,
                params={"q": query, "format": "json"},
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(f"the search engine could not be queried: {error}") from error
        return [
            SourceDocument(
                url=str(result["url"]),
                title=str(result.get("title") or result["url"]),
                text=str(result.get("content") or ""),
            )
            for result in results
            if isinstance(result, dict) and result.get("url")
        ][:limit]

    def _get(self, url: str) -> tuple[str, str, str]:
        """Follow redirects by hand so every hop is checked before it is requested."""
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            self._require_public(current)
            try:
                with self.client.stream(
                    "GET",
                    current,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html, text/plain;q=0.9"},
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        current = self._redirect_target(current, response)
                        continue
                    if response.status_code >= 400:
                        raise LookupError(f"{current} answered {response.status_code}")
                    content_type = (
                        response.headers.get("content-type", "").split(";")[0].strip().lower()
                    )
                    if content_type not in READABLE_CONTENT_TYPES:
                        raise LookupError(
                            f"{current} returned {content_type or 'an undeclared type'}, "
                            "which is not readable text"
                        )
                    return current, content_type, self._read_bounded(response)
            except httpx.HTTPError as error:
                raise RuntimeError(f"{current} could not be fetched: {error}") from error
        raise SourceNotAllowed(f"{url} redirected more than {MAX_REDIRECTS} times")

    def _redirect_target(self, current: str, response: httpx.Response) -> str:
        target: str = urljoin(current, str(response.headers.get("location", "")))
        if not self.is_permitted(target):
            raise SourceNotAllowed(f"{current} redirected to an unapproved source: {target}")
        return target

    def _require_public(self, url: str) -> None:
        """Refuse a name that resolves, in whole or in part, to a non-public address.

        Checked at request time because a name on the allowlist says nothing about where
        it points today. This narrows DNS rebinding rather than eliminating it - the
        connection resolves the name again - so the network policy that keeps this
        service away from internal ranges remains the backstop.
        """
        host = urlsplit(url).hostname or ""
        addresses = list(self.resolve(host))
        if not addresses or not all(_is_public(address) for address in addresses):
            raise SourceNotAllowed(f"host {host} does not resolve to a public address")

    def _read_bounded(self, response: httpx.Response) -> str:
        received = bytearray()
        for chunk in response.iter_bytes():
            received.extend(chunk)
            if len(received) > self.max_bytes:
                raise LookupError(
                    f"{response.url} exceeded the {self.max_bytes}-byte response limit"
                )
        encoding: str = response.encoding or "utf-8"
        return received.decode(encoding, errors="replace")
