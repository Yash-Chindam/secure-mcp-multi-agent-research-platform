"""How the API boundary decides who is calling.

Two identity sources exist and they are mutually exclusive. When a token issuer is
configured, the only accepted identity is a verified access token; the development
headers stop being read at all, so a deployment cannot be downgraded to them by sending
one. When no issuer is configured the headers are accepted and the platform says so on
its health endpoint, rather than leaving an operator to assume tokens are being checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from research_platform.auth import TokenRejected, TokenVerifier, bearer_token
from research_platform.domain.models import AccessClass
from research_platform.identity import Principal, Role

DEFAULT_ROLES = frozenset({Role.REQUESTER})


@dataclass(frozen=True)
class RequestIdentity:
    tenant_id: str
    requester_id: str
    roles: frozenset[Role] = DEFAULT_ROLES
    clearance: AccessClass = AccessClass.PUBLIC

    @classmethod
    def of(cls, principal: Principal) -> RequestIdentity:
        return cls(
            tenant_id=principal.tenant_id,
            requester_id=principal.subject_id,
            roles=principal.roles,
            clearance=principal.clearance,
        )

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


def _verifier_of(request: Request) -> TokenVerifier | None:
    return getattr(request.app.state, "token_verifier", None)


def request_identity(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID", max_length=100)] = None,
    requester_id: Annotated[str | None, Header(alias="X-Requester-ID", max_length=200)] = None,
    roles: Annotated[str | None, Header(alias="X-Roles", max_length=200)] = None,
    clearance: Annotated[str | None, Header(alias="X-Clearance", max_length=50)] = None,
) -> RequestIdentity:
    """Identify the caller from a verified token, or from the development headers."""
    verifier = _verifier_of(request)
    if verifier is not None:
        try:
            return RequestIdentity.of(verifier.verify(bearer_token(authorization)))
        except TokenRejected as error:
            raise HTTPException(
                status_code=401,
                detail=str(error),
                headers={"WWW-Authenticate": "Bearer"},
            ) from error

    if not tenant_id or not requester_id:
        raise HTTPException(
            status_code=401,
            detail="X-Tenant-ID and X-Requester-ID are required while no token issuer "
            "is configured",
        )
    return RequestIdentity(
        tenant_id=tenant_id,
        requester_id=requester_id,
        roles=_parse_roles(roles),
        clearance=_parse_clearance(clearance),
    )


Identity = Annotated[RequestIdentity, Depends(request_identity)]
