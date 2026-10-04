"""The durable ``ArtifactStore``: MinIO, or any S3-compatible object store.

Keys are built by ``application.artifacts.artifact_key``, so every object sits under its
tenant's own prefix in one bucket. Writes replace: an artifact is named after the job it
belongs to, and a publication step Temporal redelivers must land on the same object
rather than leave a second copy.
"""

from __future__ import annotations

import io
import logging
from urllib.parse import urlsplit

from minio import Minio
from minio.error import S3Error

from research_platform.application.artifacts import (
    ArtifactNotFound,
    ArtifactStore,
    StoredArtifact,
    artifact_key,
)

logger = logging.getLogger(__name__)

_MISSING = frozenset({"NoSuchKey", "NoSuchObject", "NoSuchBucket"})


class MinioArtifactStore:
    def __init__(self, client: Minio, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def ensure_bucket(self) -> None:
        """Create the bucket if it is missing; harmless when another process won the race."""
        if self._client.bucket_exists(self._bucket):
            return
        try:
            self._client.make_bucket(self._bucket)
        except S3Error as error:
            if error.code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise

    def put(self, tenant_id: str, name: str, data: bytes, content_type: str) -> StoredArtifact:
        key = artifact_key(tenant_id, name)
        self._client.put_object(
            self._bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )
        return StoredArtifact(key=key, size=len(data), content_type=content_type)

    def get(self, tenant_id: str, name: str) -> bytes:
        key = artifact_key(tenant_id, name)
        try:
            response = self._client.get_object(self._bucket, key)
        except S3Error as error:
            if error.code in _MISSING:
                raise ArtifactNotFound(name) from error
            raise
        try:
            return bytes(response.read())
        finally:
            response.close()
            response.release_conn()


def build_artifact_store(
    endpoint: str, *, access_key: str, secret_key: str, bucket: str
) -> ArtifactStore:
    """Connect to the object store at ``endpoint`` and make sure its bucket exists.

    ``endpoint`` is a URL (``http://minio:9000``) so that whether the connection is
    encrypted is stated by the deployment rather than guessed from a port.
    """
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("the artifact endpoint must be an http(s) URL such as http://minio:9000")
    if parts.scheme == "http":
        logger.warning("artifact store %s is reached without TLS", parts.netloc)
    client = Minio(
        parts.netloc,
        access_key=access_key,
        secret_key=secret_key,
        secure=parts.scheme == "https",
    )
    store = MinioArtifactStore(client, bucket)
    store.ensure_bucket()
    return store
