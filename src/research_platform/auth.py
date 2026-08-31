"""Verify short-lived OAuth tokens and derive the principal from their claims.

Section 11 requires short-lived OAuth tokens rather than shared long-lived keys. This
module verifies a Keycloak-issued access token against the realm's published keys and
maps its claims onto the principal every authorization decision is made against.

Nothing here trusts a claim it did not verify: the signature, issuer, audience and
lifetime are all checked, the accepted algorithms are an allowlist so a token cannot
choose a weaker one, and a role the platform does not recognize is dropped rather than
being interpreted generously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWKClient

from research_platform.domain.models import AccessClass
from research_platform.identity import Principal, Role

ACCEPTED_ALGORITHMS = ("RS256", "RS384", "RS512", "ES256", "ES384")
"""Asymmetric algorithms only, so a token cannot present a symmetric alg it also signs with."""

CLOCK_LEEWAY_SECONDS = 30


class TokenRejected(PermissionError):
    """Raised when a presented token cannot be trusted."""


@dataclass(frozen=True)
class ClaimMapping:
    """Where the platform's identity facts live in the realm's token."""

    tenant_claim: str = "tenant_id"
    clearance_claim: str = "clearance"
    roles_claim: str = "realm_access.roles"

    def read_roles(self, claims: dict[str, Any]) -> frozenset[Role]:
        """Map the realm's role names onto the roles the platform understands."""
        raw = _read_path(claims, self.roles_claim)
        if not isinstance(raw, list):
            return frozenset()
        recognized = set()
        for name in raw:
            try:
                recognized.add(Role(str(name).strip().lower()))
            except ValueError:
                continue
        return frozenset(recognized)

    def read_tenant(self, claims: dict[str, Any]) -> str:
        tenant = _read_path(claims, self.tenant_claim)
        if not isinstance(tenant, str) or not tenant.strip():
            raise TokenRejected(f"the token carries no {self.tenant_claim} claim")
        return tenant.strip()

    def read_clearance(self, claims: dict[str, Any]) -> AccessClass:
        """Default to the least permissive class when the realm states nothing."""
        raw = _read_path(claims, self.clearance_claim)
        if raw is None:
            return AccessClass.PUBLIC
        try:
            return AccessClass(str(raw).strip().lower())
        except ValueError as error:
            raise TokenRejected(f"{raw!r} is not a recognized clearance") from error


def _read_path(claims: dict[str, Any], path: str) -> Any:
    current: Any = claims
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


class KeyResolver:
    """Resolves the realm's signing key for a token, as the JWKS seam for tests."""

    def __init__(self, jwks_uri: str) -> None:
        self._client = PyJWKClient(jwks_uri, cache_keys=True)

    def key_for(self, token: str) -> Any:
        return self._client.get_signing_key_from_jwt(token).key


class TokenVerifier:
    """Verify an access token and derive the principal it authorizes."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        keys: KeyResolver,
        mapping: ClaimMapping | None = None,
    ) -> None:
        if not issuer:
            raise ValueError("a token issuer is required")
        if not audience:
            raise ValueError("a token audience is required")
        self._issuer = issuer
        self._audience = audience
        self._keys = keys
        self._mapping = mapping or ClaimMapping()

    def verify(self, token: str) -> Principal:
        """Return the principal the token authorizes, or refuse it."""
        candidate = token.strip()
        if not candidate:
            raise TokenRejected("no access token was presented")

        try:
            key = self._keys.key_for(candidate)
        except Exception as error:  # noqa: BLE001 - every key lookup failure is a refusal
            raise TokenRejected("the token signing key could not be resolved") from error

        try:
            claims = jwt.decode(
                candidate,
                key=key,
                algorithms=list(ACCEPTED_ALGORITHMS),
                issuer=self._issuer,
                audience=self._audience,
                leeway=CLOCK_LEEWAY_SECONDS,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError as error:
            raise TokenRejected("the access token has expired") from error
        except jwt.InvalidAudienceError as error:
            raise TokenRejected("the access token was issued for another audience") from error
        except jwt.InvalidIssuerError as error:
            raise TokenRejected("the access token was issued by another authority") from error
        except jwt.InvalidTokenError as error:
            raise TokenRejected(f"the access token is not valid: {error}") from error

        subject = str(claims.get("sub", "")).strip()
        if not subject:
            raise TokenRejected("the token carries no subject")

        return Principal(
            tenant_id=self._mapping.read_tenant(claims),
            subject_id=subject,
            roles=self._mapping.read_roles(claims),
            clearance=self._mapping.read_clearance(claims),
        )


def bearer_token(header_value: str | None) -> str:
    """Extract the token from an Authorization header, or refuse the request."""
    if not header_value:
        raise TokenRejected("no Authorization header was presented")
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise TokenRejected("the Authorization header must present a bearer token")
    return token.strip()
