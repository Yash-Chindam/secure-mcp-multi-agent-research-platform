"""The boundary the filesystem server must be used within.

Section 8 requires tenant-scoped roots and a read-only default. Confinement is decided
after resolving the real path, because a symlink inside an allowed root is the standard
way to reach a file outside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

READABLE_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"})

MAX_LISTED_ENTRIES = 500


class PathNotAllowed(PermissionError):
    """Raised when a requested path falls outside the tenant workspace."""


@dataclass(frozen=True)
class WorkspaceRoots:
    """The per-tenant directories the filesystem server may read."""

    roots: dict[str, Path]

    def __post_init__(self) -> None:
        if not self.roots:
            raise ValueError("at least one tenant workspace root must be configured")
        for tenant_id, root in self.roots.items():
            if not tenant_id:
                raise ValueError("a workspace root must belong to a named tenant")
            if not root.is_absolute():
                raise ValueError(f"workspace root for {tenant_id} must be an absolute path")

    def root_for(self, tenant_id: str) -> Path:
        try:
            return self.roots[tenant_id]
        except KeyError as error:
            raise PathNotAllowed(f"tenant {tenant_id} has no configured workspace") from error


def _reject_traversal(relative_path: str) -> PurePosixPath:
    """Refuse a request that names anything other than a plain relative path."""
    candidate = relative_path.strip().replace("\\", "/")
    if not candidate:
        raise PathNotAllowed("a workspace path must not be empty")
    if candidate.startswith("/") or ":" in candidate:
        raise PathNotAllowed(f"{relative_path!r} must be relative to the tenant workspace")

    pure = PurePosixPath(candidate)
    if any(part == ".." for part in pure.parts):
        raise PathNotAllowed(f"{relative_path!r} attempts to leave the tenant workspace")
    if any(part.startswith(".") and part != "." for part in pure.parts):
        raise PathNotAllowed(f"{relative_path!r} names a hidden entry")
    return pure


def resolve_workspace_path(
    relative_path: str,
    *,
    tenant_id: str,
    roots: WorkspaceRoots,
) -> Path:
    """Return the real path to read, or refuse it.

    The resolved path is re-checked against the resolved root, so a symlink that points
    outside the workspace is refused even though its own name looked acceptable.
    """
    pure = _reject_traversal(relative_path)
    root = roots.root_for(tenant_id).resolve()
    candidate = (root / pure).resolve()

    if candidate != root and root not in candidate.parents:
        raise PathNotAllowed(f"{relative_path!r} resolves outside the tenant workspace")
    return candidate


def ensure_readable_document(path: Path) -> Path:
    """Refuse anything that is not an ordinary readable document."""
    if path.suffix.lower() not in READABLE_SUFFIXES:
        raise PathNotAllowed(f"{path.suffix or '(no suffix)'} is not a readable document type")
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is not present in the tenant workspace")
    if not path.is_file():
        raise PathNotAllowed(f"{path.name} is not an ordinary file")
    return path


def list_workspace(tenant_id: str, roots: WorkspaceRoots) -> list[str]:
    """List readable documents in the tenant workspace as workspace-relative paths."""
    root = roots.root_for(tenant_id).resolve()
    if not root.is_dir():
        raise PathNotAllowed(f"workspace for {tenant_id} is not a directory")

    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(entries) >= MAX_LISTED_ENTRIES:
            break
        if not path.is_file() or path.suffix.lower() not in READABLE_SUFFIXES:
            continue
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        entries.append(relative.as_posix())
    return entries
