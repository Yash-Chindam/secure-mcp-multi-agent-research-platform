"""Where generated artifacts live: reports, manifests and evidence bundles (section 6).

An artifact is addressed by a tenant and a name, never by a raw storage key. The store
builds the key itself, under that tenant's own prefix, so no caller can name its way
into another tenant's artifacts - the "isolation at the artifact boundary" section 11
asks for. ``InMemoryArtifactStore`` here is for development and tests;
``research_platform.persistence.object_store.MinioArtifactStore`` is the durable one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from threading import RLock
from typing import Protocol
from urllib.parse import quote
from uuid import UUID

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


class ArtifactNotFound(LookupError):
    pass


class InvalidArtifactName(ValueError):
    pass


def artifact_key(tenant_id: str, name: str) -> str:
    """The storage key for one tenant's artifact, refusing anything that could escape it.

    Each path segment of the name must be plain: no empty segment, no ``..``, no
    leading separator. The tenant identifier is percent-encoded, dots included, so any
    identifier becomes exactly one path segment and two tenants can never map onto the
    same prefix or onto each other's.
    """
    if not tenant_id:
        raise InvalidArtifactName("an artifact must belong to a tenant")
    tenant = quote(tenant_id, safe="").replace(".", "%2E")
    segments = name.split("/")
    if not all(_SEGMENT.fullmatch(segment) and ".." not in segment for segment in segments):
        raise InvalidArtifactName(f"{name!r} is not a valid artifact name")
    return f"tenants/{tenant}/{name}"


def job_artifact(job_id: UUID, filename: str) -> str:
    """The name of one of a job's artifacts, relative to its tenant."""
    return f"jobs/{job_id}/{filename}"


@dataclass(frozen=True)
class StoredArtifact:
    key: str
    size: int
    content_type: str


class ArtifactStore(Protocol):
    def put(self, tenant_id: str, name: str, data: bytes, content_type: str) -> StoredArtifact: ...

    def get(self, tenant_id: str, name: str) -> bytes: ...


class InMemoryArtifactStore:
    """Development store. Not shared between processes, like the in-memory repository."""

    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}
        self._lock = RLock()

    def put(self, tenant_id: str, name: str, data: bytes, content_type: str) -> StoredArtifact:
        key = artifact_key(tenant_id, name)
        with self._lock:
            self._objects[key] = bytes(data)
        return StoredArtifact(key=key, size=len(data), content_type=content_type)

    def get(self, tenant_id: str, name: str) -> bytes:
        key = artifact_key(tenant_id, name)
        with self._lock:
            data = self._objects.get(key)
        if data is None:
            raise ArtifactNotFound(name)
        return data
