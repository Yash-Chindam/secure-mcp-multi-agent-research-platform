from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from research_platform.domain.models import AccessClass
from research_platform.identity import Principal, Role

DEFAULT_ROLES = frozenset({Role.REQUESTER})


@dataclass(frozen=True)
class RequestIdentity:
    tenant_id: str
    requester_id: str
    roles: frozenset[Role] = DEFAULT_ROLES
    clearance: AccessClass = AccessClass.PUBLIC

    @property
    def principal(self) -> Principal:
        return Principal(
            tenant_id=self.tenant_id,
            subject_id=self.requester_id,
            roles=self.roles,
            clearance=self.clearance,
        )


def _parse_roles(raw: str | None) -> frozenset[Role]:
    if raw is None or not raw.strip():
        return DEFAULT_ROLES
    try:
        return frozenset(Role(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"unknown role: {error}") from error


def _parse_clearance(raw: str | None) -> AccessClass:
    if raw is None or not raw.strip():
        return AccessClass.PUBLIC
    try:
        return AccessClass(raw.strip())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"unknown clearance: {raw}") from error


def request_identity(
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID", min_length=1, max_length=100)],
    requester_id: Annotated[str, Header(alias="X-Requester-ID", min_length=1, max_length=200)],
    roles: Annotated[str | None, Header(alias="X-Roles", max_length=200)] = None,
    clearance: Annotated[str | None, Header(alias="X-Clearance", max_length=50)] = None,
) -> RequestIdentity:
    """Temporary trusted-proxy identity seam; Keycloak validation will replace it."""
    return RequestIdentity(
        tenant_id=tenant_id,
        requester_id=requester_id,
        roles=_parse_roles(roles),
        clearance=_parse_clearance(clearance),
    )


Identity = Annotated[RequestIdentity, Depends(request_identity)]
