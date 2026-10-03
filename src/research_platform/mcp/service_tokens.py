"""Short-lived service tokens for calling a remote MCP server (section 11).

A remote server has to know its caller is the governed gateway and not something else
that can reach it on the network. The gateway proves that with an OAuth client-credentials
token from the same issuer that signs user tokens, rather than with a shared key that
would never expire and would have to be copied into every service. The token is cached
until shortly before it expires and then replaced, so a leaked one is useful for minutes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic

import httpx

EXPIRY_MARGIN_SECONDS = 30.0
"""How long before its stated expiry a cached token stops being reused."""


class TokenUnavailable(RuntimeError):
    """The issuer did not grant a service token."""


@dataclass
class ClientCredentialsTokens:
    token_url: str
    client_id: str
    client_secret: str = field(repr=False)
    audience: str | None = None
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=10.0))
    clock: Callable[[], float] = monotonic
    _token: str | None = field(default=None, init=False, repr=False)
    _expires_at: float = field(default=0.0, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __call__(self) -> str:
        with self._lock:
            if self._token is None or self.clock() >= self._expires_at:
                self._token, lifetime = self._request()
                self._expires_at = self.clock() + max(0.0, lifetime - EXPIRY_MARGIN_SECONDS)
            return self._token

    def _request(self) -> tuple[str, float]:
        form = {"grant_type": "client_credentials"}
        if self.audience:
            form["audience"] = self.audience
        try:
            response = self.client.post(
                self.token_url, data=form, auth=(self.client_id, self.client_secret)
            )
            response.raise_for_status()
            granted = response.json()
            return str(granted["access_token"]), float(granted.get("expires_in", 60))
        except (httpx.HTTPError, KeyError, ValueError) as error:
            # The response body is deliberately left out: an issuer's error text can echo
            # the credentials it was sent.
            raise TokenUnavailable(f"{type(error).__name__} from the token endpoint") from error
