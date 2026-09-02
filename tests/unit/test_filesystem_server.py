from pathlib import Path

import pytest

from research_platform.mcp.servers.filesystem_boundary import PathNotAllowed, WorkspaceRoots
from research_platform.mcp.servers.filesystem_server import FilesystemService


@pytest.fixture
def service(tmp_path: Path) -> FilesystemService:
    acme = tmp_path / "acme"
    acme.mkdir()
    (acme / "brief.md").write_text("Verified evidence.", encoding="utf-8")
    (acme / "large.txt").write_text("12345", encoding="utf-8")
    (acme / "binary.txt").write_bytes(b"\xff\xfe")
    return FilesystemService(WorkspaceRoots({"acme": acme}), max_document_bytes=4)


def test_service_lists_the_tenants_documents(service: FilesystemService) -> None:
    assert service.list_workspace("acme") == ["binary.txt", "brief.md", "large.txt"]


def test_service_reads_an_approved_document(tmp_path: Path) -> None:
    workspace = tmp_path / "acme"
    workspace.mkdir()
    (workspace / "brief.md").write_text("Verified evidence.", encoding="utf-8")
    service = FilesystemService(WorkspaceRoots({"acme": workspace}))

    assert service.read_document("acme", "brief.md") == "Verified evidence."


def test_service_enforces_the_document_size_limit(service: FilesystemService) -> None:
    with pytest.raises(PathNotAllowed, match="document limit"):
        service.read_document("acme", "large.txt")


def test_service_refuses_non_utf8_documents(service: FilesystemService) -> None:
    with pytest.raises(PathNotAllowed, match="valid UTF-8"):
        service.read_document("acme", "binary.txt")


def test_service_requires_a_tenant(service: FilesystemService) -> None:
    with pytest.raises(ValueError, match="identify its tenant"):
        service.list_workspace(" ")


def test_document_limit_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        FilesystemService(WorkspaceRoots({"acme": tmp_path}), max_document_bytes=0)

