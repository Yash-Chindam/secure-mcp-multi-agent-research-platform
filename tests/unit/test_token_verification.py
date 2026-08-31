import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from research_platform.auth import (
    ClaimMapping,
    TokenRejected,
    TokenVerifier,
    bearer_token,
)
from research_platform.domain.models import AccessClass
from research_platform.identity import Role

ISSUER = "https://keycloak.test/realms/research"
AUDIENCE = "research-platform"

SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class StaticKeyResolver:
    """Stands in for the realm's JWKS endpoint."""

    def __init__(self, key: Any = None, failure: Exception | None = None) -> None:
        self._key = key if key is not None else SIGNING_KEY.public_key()
        self._failure = failure

    def key_for(self, token: str) -> Any:
        if self._failure is not None:
            raise self._failure
        return self._key


def claims(**overrides: object) -> dict[str, Any]:
    now = datetime.now(UTC)
    base: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "b6f1-user-1",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "tenant_id": "acme",
        "realm_access": {"roles": ["requester", "offline_access"]},
    }
    base.update(overrides)
    return base


def token(
    *,
    key: Any = None,
    algorithm: str = "RS256",
    **claim_overrides: object,
) -> str:
    return jwt.encode(
        claims(**claim_overrides),
        key if key is not None else SIGNING_KEY,
        algorithm=algorithm,
    )


def verifier(
    resolver: StaticKeyResolver | None = None,
    mapping: ClaimMapping | None = None,
) -> TokenVerifier:
    return TokenVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        keys=resolver or StaticKeyResolver(),  # type: ignore[arg-type]
        mapping=mapping,
    )


def test_an_issuer_and_audience_are_both_required() -> None:
    with pytest.raises(ValueError, match="issuer is required"):
        TokenVerifier(issuer="", audience=AUDIENCE, keys=StaticKeyResolver())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="audience is required"):
        TokenVerifier(issuer=ISSUER, audience="", keys=StaticKeyResolver())  # type: ignore[arg-type]


def test_a_valid_token_yields_the_principal_it_authorizes() -> None:
    principal = verifier().verify(token())

    assert principal.tenant_id == "acme"
    assert principal.subject_id == "b6f1-user-1"
    assert principal.roles == frozenset({Role.REQUESTER})
    assert principal.clearance is AccessClass.PUBLIC
    assert principal.is_agent is False


def test_realm_roles_the_platform_does_not_know_are_dropped() -> None:
    """An unrecognized realm role must not be interpreted generously."""
    principal = verifier().verify(
        token(realm_access={"roles": ["reviewer", "uma_authorization", "superuser"]})
    )

    assert principal.roles == frozenset({Role.REVIEWER})


def test_a_token_with_no_recognized_role_authorizes_nothing() -> None:
    principal = verifier().verify(token(realm_access={"roles": ["offline_access"]}))

    assert principal.roles == frozenset()


def test_a_missing_roles_claim_authorizes_nothing() -> None:
    principal = verifier().verify(token(realm_access=None))

    assert principal.roles == frozenset()


def test_a_malformed_roles_claim_authorizes_nothing() -> None:
    principal = verifier().verify(token(realm_access={"roles": "requester"}))

    assert principal.roles == frozenset()


def test_a_stated_clearance_is_honoured() -> None:
    principal = verifier().verify(token(clearance="internal"))

    assert principal.clearance is AccessClass.INTERNAL


def test_an_unrecognized_clearance_is_refused() -> None:
    with pytest.raises(TokenRejected, match="not a recognized clearance"):
        verifier().verify(token(clearance="top-secret"))


def test_a_token_signed_by_another_key_is_refused() -> None:
    with pytest.raises(TokenRejected, match="not valid"):
        verifier().verify(token(key=OTHER_KEY))


def test_an_unsigned_token_is_refused() -> None:
    unsigned = jwt.encode(claims(), key="", algorithm="none")

    with pytest.raises(TokenRejected, match="not valid"):
        verifier().verify(unsigned)


def _forge_hs256(payload: dict[str, Any], secret: bytes) -> str:
    """Hand-build an HS256 token, which PyJWT refuses to sign with a public key."""

    def segment(document: dict[str, Any]) -> bytes:
        raw = json.dumps(document, separators=(",", ":"), default=_encode_datetime).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=")

    signing_input = segment({"alg": "HS256", "typ": "JWT"}) + b"." + segment(payload)
    signature = base64.urlsafe_b64encode(
        hmac.new(secret, signing_input, hashlib.sha256).digest()
    ).rstrip(b"=")
    return (signing_input + b"." + signature).decode()


def _encode_datetime(value: object) -> int:
    if isinstance(value, datetime):
        return int(value.timestamp())
    raise TypeError(f"cannot encode {type(value)!r}")


def test_a_symmetric_token_signed_with_the_public_key_is_refused() -> None:
    """The accepted-algorithm allowlist defeats the classic algorithm confusion attack.

    An attacker who knows the realm's public key could sign an HS256 token with it. If
    the verifier accepted HS256 it would hand that same public key to the HMAC
    verification and the forgery would pass.
    """
    public_pem = SIGNING_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    forged = _forge_hs256(claims(), public_pem)

    with pytest.raises(TokenRejected, match="not valid"):
        verifier(StaticKeyResolver(key=public_pem.decode())).verify(forged)


def test_an_expired_token_is_refused() -> None:
    past = datetime.now(UTC) - timedelta(hours=2)

    with pytest.raises(TokenRejected, match="has expired"):
        verifier().verify(token(iat=past, exp=past + timedelta(minutes=5)))


def test_a_token_for_another_audience_is_refused() -> None:
    with pytest.raises(TokenRejected, match="another audience"):
        verifier().verify(token(aud="another-service"))


def test_a_token_from_another_issuer_is_refused() -> None:
    with pytest.raises(TokenRejected, match="another authority"):
        verifier().verify(token(iss="https://attacker.test/realms/research"))


def test_a_token_not_yet_valid_is_refused() -> None:
    future = datetime.now(UTC) + timedelta(hours=1)

    with pytest.raises(TokenRejected, match="not valid"):
        verifier().verify(token(nbf=future, exp=future + timedelta(minutes=5)))


@pytest.mark.parametrize("missing", ["exp", "iat", "sub"])
def test_a_token_missing_a_required_claim_is_refused(missing: str) -> None:
    payload = claims()
    del payload[missing]
    incomplete = jwt.encode(payload, SIGNING_KEY, algorithm="RS256")

    with pytest.raises(TokenRejected, match="not valid"):
        verifier().verify(incomplete)


def test_a_token_without_a_tenant_claim_is_refused() -> None:
    with pytest.raises(TokenRejected, match="no tenant_id claim"):
        verifier().verify(token(tenant_id=None))


def test_a_blank_tenant_claim_is_refused() -> None:
    with pytest.raises(TokenRejected, match="no tenant_id claim"):
        verifier().verify(token(tenant_id="   "))


def test_a_blank_subject_is_refused() -> None:
    with pytest.raises(TokenRejected, match="no subject"):
        verifier().verify(token(sub="   "))


def test_an_empty_token_is_refused() -> None:
    with pytest.raises(TokenRejected, match="no access token"):
        verifier().verify("   ")


def test_an_unresolvable_signing_key_refuses_the_token() -> None:
    resolver = StaticKeyResolver(failure=RuntimeError("jwks unreachable"))

    with pytest.raises(TokenRejected, match="signing key could not be resolved"):
        verifier(resolver).verify(token())


def test_a_realm_can_place_the_claims_elsewhere() -> None:
    mapping = ClaimMapping(
        tenant_claim="org.tenant",
        clearance_claim="org.clearance",
        roles_claim="resource_access.research.roles",
    )
    principal = verifier(mapping=mapping).verify(
        token(
            org={"tenant": "globex", "clearance": "restricted"},
            resource_access={"research": {"roles": ["administrator"]}},
        )
    )

    assert principal.tenant_id == "globex"
    assert principal.clearance is AccessClass.RESTRICTED
    assert principal.roles == frozenset({Role.ADMINISTRATOR})


def test_a_missing_nested_claim_path_is_reported() -> None:
    mapping = ClaimMapping(tenant_claim="org.tenant")

    with pytest.raises(TokenRejected, match="no org.tenant claim"):
        verifier(mapping=mapping).verify(token())


def test_a_bearer_header_yields_the_token() -> None:
    assert bearer_token("Bearer abc.def.ghi") == "abc.def.ghi"


def test_the_bearer_scheme_is_matched_case_insensitively() -> None:
    assert bearer_token("bearer abc.def.ghi") == "abc.def.ghi"


@pytest.mark.parametrize(
    "header",
    [None, "", "abc.def.ghi", "Basic dXNlcjpwYXNz", "Bearer", "Bearer   "],
)
def test_a_header_that_is_not_a_bearer_token_is_refused(header: str | None) -> None:
    with pytest.raises(TokenRejected):
        bearer_token(header)
