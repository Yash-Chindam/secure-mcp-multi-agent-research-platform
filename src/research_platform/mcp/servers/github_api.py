"""The production GitHub backend: the REST API, read through a scoped token.

``GitHubService`` resolves every repository against the tenant allowlist before this is
called, so the only repositories requested are approved ones. The token is what bounds
the damage if that ever fails: section 8 asks for a *scoped* one, so issue it read-only
and for the allowlisted repositories alone. It is sent as a header by the HTTP client
and never appears in an argument, a result or an audit record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

GITHUB_API = "https://api.github.com"
MAX_TREE_ENTRIES = 5_000


def github_client(token: str, *, base_url: str = GITHUB_API, timeout: float = 20.0) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        timeout=timeout,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "secure-mcp-research-platform",
        },
    )


@dataclass
class GitHubApiBackend:
    client: httpx.Client

    def read_repository(self, repository: str, *, ref: str) -> list[str]:
        tree = self._get(f"/repos/{repository}/git/trees/{ref}", params={"recursive": "1"})
        paths = [
            str(entry["path"])
            for entry in tree.get("tree", [])
            if isinstance(entry, dict) and entry.get("type") == "blob"
        ]
        return sorted(paths)[:MAX_TREE_ENTRIES]

    def read_pull_requests(self, repository: str, *, limit: int) -> list[dict[str, object]]:
        pulls = self._get(
            f"/repos/{repository}/pulls",
            params={"state": "all", "per_page": str(limit), "sort": "updated", "direction": "desc"},
        )
        return [
            {
                "number": pull.get("number"),
                "title": pull.get("title"),
                "state": pull.get("state"),
                "author": (pull.get("user") or {}).get("login"),
                "created_at": pull.get("created_at"),
                "merged_at": pull.get("merged_at"),
                "url": pull.get("html_url"),
            }
            for pull in pulls
            if isinstance(pull, dict)
        ][:limit]

    def _get(self, path: str, *, params: dict[str, str]) -> Any:
        try:
            response = self.client.get(path, params=params)
        except httpx.HTTPError as error:
            raise RuntimeError(f"GitHub could not be reached: {error}") from error
        if response.status_code == 404:
            raise LookupError(f"{path} was not found, or the token cannot read it")
        if response.status_code >= 400:
            raise RuntimeError(f"GitHub answered {response.status_code} for {path}")
        return response.json()
