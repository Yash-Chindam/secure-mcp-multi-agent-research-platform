"""FastMCP tools for tenant-confined, read-only workspace documents."""

from __future__ import annotations

import json
from dataclasses import dataclass

from fastmcp import FastMCP

from research_platform.mcp.servers.filesystem_boundary import (
    PathNotAllowed,
    WorkspaceRoots,
    ensure_readable_document,
    list_workspace,
    resolve_workspace_path,
)

DEFAULT_MAX_DOCUMENT_BYTES = 524_288


@dataclass(frozen=True)
class FilesystemService:
    """Read workspace documents only after the tenant boundary has accepted them."""

    roots: WorkspaceRoots
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES

    def __post_init__(self) -> None:
        if self.max_document_bytes < 1:
            raise ValueError("max_document_bytes must be positive")

    def list_workspace(self, tenant_id: str) -> list[str]:
        _require_tenant(tenant_id)
        return list_workspace(tenant_id, self.roots)

    def read_document(self, tenant_id: str, path: str) -> str:
        _require_tenant(tenant_id)
        resolved = ensure_readable_document(
            resolve_workspace_path(path, tenant_id=tenant_id, roots=self.roots)
        )
        content = resolved.read_bytes()
        if len(content) > self.max_document_bytes:
            raise PathNotAllowed(
                f"{resolved.name} exceeds the {self.max_document_bytes}-byte document limit"
            )
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PathNotAllowed(f"{resolved.name} is not valid UTF-8 text") from error


def _require_tenant(tenant_id: str) -> None:
    if not tenant_id or not tenant_id.strip():
        raise ValueError("a filesystem call must identify its tenant")


def build_filesystem_server(service: FilesystemService) -> FastMCP:
    """Expose tenant-confined workspace reads over MCP."""
    server: FastMCP = FastMCP(name="filesystem")

    @server.tool
    def list_workspace(tenant_id: str) -> str:
        """List readable workspace-relative document paths for this tenant."""
        try:
            entries = service.list_workspace(tenant_id)
        except (PathNotAllowed, ValueError) as error:
            raise ValueError(f"validation error: {error}") from error
        return json.dumps(entries, separators=(",", ":"))

    @server.tool
    def read_document(tenant_id: str, path: str) -> str:
        """Read one approved UTF-8 document from this tenant's workspace."""
        try:
            return service.read_document(tenant_id, path)
        except (FileNotFoundError, PathNotAllowed, ValueError) as error:
            raise ValueError(f"validation error: {error}") from error

    return server

