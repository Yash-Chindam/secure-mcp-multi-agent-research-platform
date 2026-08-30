from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header


@dataclass(frozen=True)
class RequestIdentity:
    tenant_id: str
    requester_id: str


def request_identity(
    tenant_id: Annotated[str, Header(alias="X-Tenant-ID", min_length=1, max_length=100)],
    requester_id: Annotated[str, Header(alias="X-Requester-ID", min_length=1, max_length=200)],
) -> RequestIdentity:
    """Temporary trusted-proxy identity seam; Keycloak validation will replace it."""
    return RequestIdentity(tenant_id=tenant_id, requester_id=requester_id)


Identity = Annotated[RequestIdentity, Depends(request_identity)]
