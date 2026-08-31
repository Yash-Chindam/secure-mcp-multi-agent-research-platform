"""The transports the research services read through.

Each service depends on a narrow protocol rather than a concrete client, so the boundary
checks can be tested against a substitute backend and the production transport can be
replaced without touching the enforcement logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SourceDocument:
    """A retrieved public document, before it is recorded as evidence."""

    url: str
    title: str
    text: str


class WebBackend(Protocol):
    """Retrieves approved public sources."""

    def search(self, query: str, *, limit: int) -> list[SourceDocument]: ...

    def fetch(self, url: str) -> SourceDocument: ...


class SqlBackend(Protocol):
    """Reads the tenant analytical schema through a read-only role."""

    def describe_schema(self, schema: str) -> list[str]: ...

    def run_query(self, sql: str, *, schema: str) -> list[dict[str, object]]: ...


class GitHubBackend(Protocol):
    """Reads allowlisted repositories through a scoped token."""

    def read_repository(self, repository: str, *, ref: str) -> list[str]: ...

    def read_pull_requests(self, repository: str, *, limit: int) -> list[dict[str, object]]: ...


class SandboxBackend(Protocol):
    """Runs a screened calculation inside an ephemeral, network-isolated container."""

    def run(self, code: str, *, cpu_seconds: int, memory_mib: int) -> str: ...


@dataclass
class StaticWebBackend:
    """A deterministic web backend for development and tests."""

    documents: dict[str, SourceDocument] = field(default_factory=dict)

    def search(self, query: str, *, limit: int) -> list[SourceDocument]:
        terms = [term for term in query.lower().split() if term]
        matches = [
            document
            for document in self.documents.values()
            if all(term in f"{document.title} {document.text}".lower() for term in terms)
        ]
        return sorted(matches, key=lambda document: document.url)[:limit]

    def fetch(self, url: str) -> SourceDocument:
        try:
            return self.documents[url]
        except KeyError as error:
            raise LookupError(f"{url} returned no content") from error
