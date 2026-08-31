import pytest

from research_platform.mcp.servers.github_boundary import (
    RepositoryAllowlist,
    RepositoryNotAllowed,
    ensure_readable_ref,
    resolve_repository,
)
from research_platform.mcp.servers.sandbox_boundary import (
    MAX_CODE_LENGTH,
    CalculationNotAllowed,
    SandboxLimits,
    screen_calculation,
)

ALLOWLIST = RepositoryAllowlist(
    repositories={
        "acme": frozenset({"acme/pricing-service", "acme/docs"}),
        "globex": frozenset({"globex/platform"}),
    }
)


def resolve(repository: str, tenant_id: str = "acme") -> str:
    return resolve_repository(repository, tenant_id=tenant_id, allowlist=ALLOWLIST)


def test_an_allowlist_must_be_configured() -> None:
    with pytest.raises(ValueError, match="at least one tenant repository allowlist"):
        RepositoryAllowlist(repositories={})


def test_an_allowlist_must_belong_to_a_named_tenant() -> None:
    with pytest.raises(ValueError, match="named tenant"):
        RepositoryAllowlist(repositories={"": frozenset({"acme/docs"})})


@pytest.mark.parametrize("entry", ["acme", "acme/", "/docs", "acme/do cs", "-acme/docs"])
def test_an_allowlist_rejects_a_malformed_entry(entry: str) -> None:
    with pytest.raises(ValueError, match="invalid allowlisted repository"):
        RepositoryAllowlist(repositories={"acme": frozenset({entry})})


def test_an_allowlisted_repository_resolves() -> None:
    assert resolve("acme/pricing-service") == "acme/pricing-service"


def test_a_git_suffix_is_removed() -> None:
    assert resolve("acme/docs.git") == "acme/docs"


def test_the_allowlist_match_is_case_insensitive() -> None:
    assert resolve("Acme/Docs") == "Acme/Docs"


def test_a_repository_outside_the_allowlist_is_refused() -> None:
    with pytest.raises(RepositoryNotAllowed, match="not an approved repository"):
        resolve("attacker/exfiltrate")


def test_one_tenant_cannot_read_another_approved_repository() -> None:
    assert resolve("globex/platform", tenant_id="globex") == "globex/platform"
    with pytest.raises(RepositoryNotAllowed, match="not an approved repository"):
        resolve("globex/platform", tenant_id="acme")


def test_an_unknown_tenant_has_no_approved_repositories() -> None:
    with pytest.raises(RepositoryNotAllowed, match="not an approved repository"):
        resolve("acme/docs", tenant_id="initech")


@pytest.mark.parametrize("repository", ["acme", "acme/docs/extra/deep/path", "../../etc/passwd"])
def test_a_malformed_repository_reference_is_refused(repository: str) -> None:
    with pytest.raises(RepositoryNotAllowed):
        resolve(repository)


@pytest.mark.parametrize("ref", ["main", "release/2026-08", "v1.2.3", "refs/heads/main"])
def test_an_ordinary_ref_is_accepted(ref: str) -> None:
    assert ensure_readable_ref(ref) == ref


@pytest.mark.parametrize("ref", ["", "-main", "main/", "feature/..%2f", "a b"])
def test_a_malformed_ref_is_refused(ref: str) -> None:
    with pytest.raises(RepositoryNotAllowed, match="not a valid git ref"):
        ensure_readable_ref(ref)


def test_a_non_branch_ref_namespace_is_refused() -> None:
    with pytest.raises(RepositoryNotAllowed, match="not a readable branch or tag"):
        ensure_readable_ref("refs/pull/1/merge")


def test_default_sandbox_limits_disable_the_network() -> None:
    limits = SandboxLimits()

    assert limits.network_enabled is False
    assert limits.cpu_seconds <= limits.wall_clock_seconds


def test_the_sandbox_refuses_to_be_given_network_access() -> None:
    with pytest.raises(ValueError, match="never be given network access"):
        SandboxLimits(network_enabled=True)


@pytest.mark.parametrize(
    "field",
    ["cpu_seconds", "memory_mib", "wall_clock_seconds", "max_output_bytes"],
)
def test_every_sandbox_limit_must_be_positive(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        SandboxLimits(**{field: 0})  # type: ignore[arg-type]


def test_cpu_time_cannot_exceed_the_wall_clock() -> None:
    with pytest.raises(ValueError, match="cannot exceed wall_clock_seconds"):
        SandboxLimits(cpu_seconds=60, wall_clock_seconds=30)


def test_an_ordinary_calculation_is_accepted() -> None:
    code = "import statistics\nvalues = [1, 2, 3]\nresult = statistics.mean(values)"

    assert screen_calculation(code) == code


def test_an_empty_calculation_is_refused() -> None:
    with pytest.raises(CalculationNotAllowed, match="must not be empty"):
        screen_calculation("   ")


def test_an_oversized_calculation_is_refused() -> None:
    with pytest.raises(CalculationNotAllowed, match="must not exceed"):
        screen_calculation("x = 1\n" * MAX_CODE_LENGTH)


def test_invalid_python_is_refused_before_a_container_starts() -> None:
    with pytest.raises(CalculationNotAllowed, match="must be valid Python"):
        screen_calculation("result = (1 +")


@pytest.mark.parametrize("module", ["os", "socket", "subprocess", "urllib", "requests", "ctypes"])
def test_withheld_modules_are_refused(module: str) -> None:
    with pytest.raises(CalculationNotAllowed, match=f"must not import {module}"):
        screen_calculation(f"import {module}")


def test_a_submodule_of_a_withheld_module_is_refused() -> None:
    with pytest.raises(CalculationNotAllowed, match="must not import urllib"):
        screen_calculation("import urllib.request")


def test_a_from_import_of_a_withheld_module_is_refused() -> None:
    with pytest.raises(CalculationNotAllowed, match="must not import os"):
        screen_calculation("from os import environ")


@pytest.mark.parametrize("name", ["eval", "exec", "open", "__import__", "compile"])
def test_withheld_builtins_are_refused(name: str) -> None:
    with pytest.raises(CalculationNotAllowed, match=f"must not use {name}"):
        screen_calculation(f"result = {name}('x')")


@pytest.mark.parametrize("attribute", ["__class__", "__globals__", "__subclasses__"])
def test_introspection_escapes_are_refused(attribute: str) -> None:
    with pytest.raises(CalculationNotAllowed, match=f"must not reach {attribute}"):
        screen_calculation(f"result = (1).{attribute}")
