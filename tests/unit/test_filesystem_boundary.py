from pathlib import Path, PurePosixPath

import pytest

from research_platform.mcp.servers.filesystem_boundary import (
    PathNotAllowed,
    WorkspaceRoots,
    ensure_readable_document,
    list_workspace,
    resolve_workspace_path,
)


@pytest.fixture
def roots(tmp_path: Path) -> WorkspaceRoots:
    acme = tmp_path / "acme"
    globex = tmp_path / "globex"
    (acme / "reports").mkdir(parents=True)
    (acme / "reports" / "q1.md").write_text("Acme quarterly figures.", encoding="utf-8")
    (acme / "notes.txt").write_text("Acme notes.", encoding="utf-8")
    (acme / "secret.pem").write_text("private key", encoding="utf-8")
    (acme / ".env").write_text("TOKEN=live", encoding="utf-8")
    globex.mkdir()
    (globex / "plan.md").write_text("Globex plan.", encoding="utf-8")
    (tmp_path / "outside.md").write_text("Not in any workspace.", encoding="utf-8")
    return WorkspaceRoots(roots={"acme": acme, "globex": globex})


def test_roots_must_be_configured() -> None:
    with pytest.raises(ValueError, match="at least one tenant workspace root"):
        WorkspaceRoots(roots={})


def test_a_root_must_belong_to_a_named_tenant(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="named tenant"):
        WorkspaceRoots(roots={"": tmp_path})


def test_a_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute path"):
        WorkspaceRoots(roots={"acme": Path("relative")})


def test_an_unknown_tenant_has_no_workspace(roots: WorkspaceRoots) -> None:
    with pytest.raises(PathNotAllowed, match="no configured workspace"):
        roots.root_for("initech")


def test_a_document_inside_the_workspace_resolves(roots: WorkspaceRoots) -> None:
    resolved = resolve_workspace_path("reports/q1.md", tenant_id="acme", roots=roots)

    assert resolved.read_text(encoding="utf-8") == "Acme quarterly figures."


def test_windows_separators_are_accepted(roots: WorkspaceRoots) -> None:
    resolved = resolve_workspace_path("reports\\q1.md", tenant_id="acme", roots=roots)

    assert resolved.name == "q1.md"


@pytest.mark.parametrize(
    "path",
    ["../outside.md", "reports/../../outside.md", "reports/../../globex/plan.md"],
)
def test_traversal_out_of_the_workspace_is_refused(roots: WorkspaceRoots, path: str) -> None:
    with pytest.raises(PathNotAllowed, match="leave the tenant workspace"):
        resolve_workspace_path(path, tenant_id="acme", roots=roots)


def test_an_absolute_path_is_refused(roots: WorkspaceRoots) -> None:
    with pytest.raises(PathNotAllowed, match="must be relative"):
        resolve_workspace_path("/etc/passwd", tenant_id="acme", roots=roots)


def test_a_drive_qualified_path_is_refused(roots: WorkspaceRoots) -> None:
    with pytest.raises(PathNotAllowed, match="must be relative"):
        resolve_workspace_path("C:\\Windows\\win.ini", tenant_id="acme", roots=roots)


def test_an_empty_path_is_refused(roots: WorkspaceRoots) -> None:
    with pytest.raises(PathNotAllowed, match="must not be empty"):
        resolve_workspace_path("   ", tenant_id="acme", roots=roots)


def test_a_hidden_entry_is_refused(roots: WorkspaceRoots) -> None:
    with pytest.raises(PathNotAllowed, match="hidden entry"):
        resolve_workspace_path(".env", tenant_id="acme", roots=roots)


def test_one_tenant_cannot_read_another_workspace(roots: WorkspaceRoots) -> None:
    acme_view = resolve_workspace_path("plan.md", tenant_id="globex", roots=roots)

    assert acme_view.read_text(encoding="utf-8") == "Globex plan."
    with pytest.raises(FileNotFoundError):
        ensure_readable_document(resolve_workspace_path("plan.md", tenant_id="acme", roots=roots))


def test_a_symlink_pointing_outside_the_workspace_is_refused(
    roots: WorkspaceRoots, tmp_path: Path
) -> None:
    link = tmp_path / "acme" / "escape.md"
    try:
        link.symlink_to(tmp_path / "outside.md")
    except (OSError, NotImplementedError):
        pytest.skip("this platform does not permit creating symlinks in this test environment")

    with pytest.raises(PathNotAllowed, match="resolves outside the tenant workspace"):
        resolve_workspace_path("escape.md", tenant_id="acme", roots=roots)


def test_a_readable_document_type_is_accepted(roots: WorkspaceRoots) -> None:
    path = resolve_workspace_path("notes.txt", tenant_id="acme", roots=roots)

    assert ensure_readable_document(path) == path


def test_an_unreadable_document_type_is_refused(roots: WorkspaceRoots) -> None:
    path = resolve_workspace_path("secret.pem", tenant_id="acme", roots=roots)

    with pytest.raises(PathNotAllowed, match="not a readable document type"):
        ensure_readable_document(path)


def test_a_missing_document_is_reported_as_missing(roots: WorkspaceRoots) -> None:
    path = resolve_workspace_path("absent.md", tenant_id="acme", roots=roots)

    with pytest.raises(FileNotFoundError):
        ensure_readable_document(path)


def test_a_directory_is_not_an_ordinary_file(roots: WorkspaceRoots) -> None:
    path = resolve_workspace_path("reports", tenant_id="acme", roots=roots)

    with pytest.raises(PathNotAllowed, match="not a readable document type"):
        ensure_readable_document(path)


def test_listing_shows_only_the_tenant_readable_documents(roots: WorkspaceRoots) -> None:
    assert list_workspace("acme", roots) == ["notes.txt", "reports/q1.md"]
    assert list_workspace("globex", roots) == ["plan.md"]


def test_listing_an_unknown_tenant_is_refused(roots: WorkspaceRoots) -> None:
    with pytest.raises(PathNotAllowed, match="no configured workspace"):
        list_workspace("initech", roots)


def test_listing_a_missing_workspace_directory_is_refused(tmp_path: Path) -> None:
    roots = WorkspaceRoots(roots={"acme": tmp_path / "absent"})

    with pytest.raises(PathNotAllowed, match="not a directory"):
        list_workspace("acme", roots)


def test_listed_paths_are_workspace_relative(roots: WorkspaceRoots) -> None:
    for entry in list_workspace("acme", roots):
        assert not PurePosixPath(entry).is_absolute()
