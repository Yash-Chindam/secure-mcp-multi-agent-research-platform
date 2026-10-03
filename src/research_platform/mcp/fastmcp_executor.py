"""Drive FastMCP servers from the synchronous governed gateway.

A server is reached either in process or, as section 8 requires of a remote service,
over Streamable HTTP: a target is the ``FastMCP`` instance itself or the URL it is served
at, and the two can be mixed in one deployment. A remote call carries a short-lived
service token (section 11) identifying the gateway, since a remote server can no longer
assume its only caller is the process that built it.

MCP clients are asynchronous while the gateway is synchronous, so calls are dispatched to
a dedicated event loop running on its own thread. That keeps the gateway callable from
both a request handler and a worker without either owning an event loop, and without the
gateway's enforcement order depending on the caller's concurrency model.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import TimeoutError as FutureTimeout
from types import TracebackType
from typing import Any

from fastmcp import Client, FastMCP

from research_platform.domain.invocations import ErrorClass
from research_platform.mcp.gateway import ExecutionRequest, UpstreamError

Target = FastMCP | str
"""An in-process server, or the Streamable HTTP URL a remote one is served at."""

TokenProvider = Callable[[], str]
"""Returns a currently valid access token for calling a remote MCP server."""

REFUSAL_MARKERS = ("source refused", "rate limited", "validation error", "must identify")


class _LoopThread:
    """A private event loop the executor owns for the lifetime of the process."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="mcp-executor",
            daemon=True,
        )
        self._thread.start()

    def run(self, coroutine: Any, *, timeout: float) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as error:
            future.cancel()
            raise UpstreamError("the capability did not respond in time", ErrorClass.TIMEOUT) from (
                error
            )

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


def _classify(error: Exception) -> UpstreamError:
    """Map a server-side error to the retry-aware taxonomy the audit record records."""
    message = str(error)
    if any(marker in message.lower() for marker in REFUSAL_MARKERS):
        return UpstreamError(message, ErrorClass.INVALID_ARGUMENTS)
    return UpstreamError(message, ErrorClass.UPSTREAM_UNAVAILABLE)


class FastMCPExecutor:
    """Call registered FastMCP servers on behalf of the gateway.

    The tenant is injected from the authenticated principal, never from the arguments an
    agent produced, so an agent cannot reach another tenant's data by asking for it.
    """

    def __init__(
        self, servers: Mapping[str, Target], *, token_provider: TokenProvider | None = None
    ) -> None:
        if not servers:
            raise ValueError("at least one MCP server must be registered")
        self._servers = dict(servers)
        self._token_provider = token_provider
        self._loop = _LoopThread()

    def __enter__(self) -> FastMCPExecutor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._loop.close()

    def execute(self, request: ExecutionRequest) -> str:
        capability = request.capability
        server = self._servers.get(capability.server)
        if server is None:
            raise UpstreamError(
                f"MCP server {capability.server} is not reachable",
                ErrorClass.UPSTREAM_UNAVAILABLE,
            )

        arguments: dict[str, Any] = {
            **request.arguments,
            "tenant_id": request.principal.tenant_id,
        }
        return str(
            self._loop.run(
                self._call(server, capability.name, arguments, self._token_for(server)),
                timeout=capability.timeout_seconds,
            )
        )

    def _token_for(self, server: Target) -> str | None:
        """Fetch a service token for a remote server; an in-process one needs none."""
        if not isinstance(server, str) or self._token_provider is None:
            return None
        try:
            return self._token_provider()
        except Exception as error:  # noqa: BLE001 - any failure to authenticate is the same outcome
            raise UpstreamError(
                f"a service token could not be obtained: {error}", ErrorClass.UPSTREAM_UNAVAILABLE
            ) from error

    @staticmethod
    async def _call(server: Target, tool: str, arguments: dict[str, Any], token: str | None) -> str:
        try:
            async with Client(server, auth=token) as client:
                result = await client.call_tool(tool, arguments)
        except Exception as error:  # noqa: BLE001 - the transport reports every failure this way
            raise _classify(error) from error

        blocks = [getattr(block, "text", None) for block in result.content]
        text = "\n".join(block for block in blocks if block)
        if not text:
            raise UpstreamError(
                f"{tool} returned no readable content", ErrorClass.UPSTREAM_UNAVAILABLE
            )
        return text
