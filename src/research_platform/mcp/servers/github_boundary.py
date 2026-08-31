"""The boundary the GitHub server must be used within.

Section 8 requires a repository allowlist and a scoped OAuth token. The allowlist is
per tenant, so one tenant's approved repositories are never readable by another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OWNER_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
REF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")

READABLE_REF_PREFIXES = ("refs/heads/", "refs/tags/")


class RepositoryNotAllowed(PermissionError):
    """Raised when a repository or ref falls outside the tenant allowlist."""


@dataclass(frozen=True)
class RepositoryAllowlist:
    """The repositories each tenant's research may read, as ``owner/name`` pairs."""

    repositories: dict[str, frozenset[str]]

    def __post_init__(self) -> None:
        if not self.repositories:
            raise ValueError("at least one tenant repository allowlist must be configured")
        for tenant_id, entries in self.repositories.items():
            if not tenant_id:
                raise ValueError("a repository allowlist must belong to a named tenant")
            for entry in entries:
                owner, _, name = entry.partition("/")
                if not OWNER_REPO.match(owner) or not OWNER_REPO.match(name):
                    raise ValueError(f"invalid allowlisted repository: {entry!r}")

    def permits(self, tenant_id: str, repository: str) -> bool:
        return repository.lower() in {
            entry.lower() for entry in self.repositories.get(tenant_id, frozenset())
        }


def resolve_repository(
    repository: str,
    *,
    tenant_id: str,
    allowlist: RepositoryAllowlist,
) -> str:
    """Return the canonical ``owner/name`` to read, or refuse it."""
    candidate = repository.strip().removesuffix(".git")
    owner, separator, name = candidate.partition("/")
    if not separator or not OWNER_REPO.match(owner) or not OWNER_REPO.match(name):
        raise RepositoryNotAllowed(f"{repository!r} is not a valid owner/name repository")
    if not allowlist.permits(tenant_id, candidate):
        raise RepositoryNotAllowed(
            f"{candidate} is not an approved repository for tenant {tenant_id}"
        )
    return candidate


def ensure_readable_ref(ref: str) -> str:
    """Refuse a ref that is not an ordinary branch or tag name."""
    candidate = ref.strip()
    if not REF_NAME.match(candidate):
        raise RepositoryNotAllowed(f"{ref!r} is not a valid git ref")
    if ".." in candidate or candidate.endswith("/"):
        raise RepositoryNotAllowed(f"{ref!r} is not a valid git ref")
    if candidate.startswith("refs/") and not candidate.startswith(READABLE_REF_PREFIXES):
        raise RepositoryNotAllowed(f"{ref!r} is not a readable branch or tag")
    return candidate
