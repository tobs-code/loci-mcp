"""Bearer-token protection for the HTTP transports.

The Streamable HTTP and SSE transports expose the full tool surface. Without a
check in front of them, anyone who can reach `MCP_PORT` can export, rewrite or
delete the whole memory. The middleware below requires
`Authorization: Bearer <MCP_AUTH_TOKEN>` on every HTTP request; stdio is
untouched (the client already owns the process).

Startup is fail-closed: an HTTP transport bound to a non-loopback address
without a token refuses to start unless `MCP_ALLOW_INSECURE_HTTP=true` is set
explicitly. A loopback-only bind is allowed without a token because it is not
reachable from another host.
"""

from __future__ import annotations

import hmac
from typing import Any

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
HTTP_TRANSPORTS = {"sse", "streamable-http"}

_UNAUTHORIZED_BODY = (
    b'{"ok": false, "status": "error", '
    b'"error": "unauthorized: send Authorization: Bearer <MCP_AUTH_TOKEN>", '
    b'"id": null, "warnings": []}'
)


class BearerAuthMiddleware:
    """ASGI middleware that rejects requests without the shared bearer token."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self._expected = token.encode()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        # Lifespan and websocket scopes are not tool calls; only HTTP requests
        # carry the Authorization header we check.
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.lower(): value
            for key, value in (scope.get("headers") or [])
        }
        provided = headers.get(b"authorization", b"")
        # The auth scheme is case-insensitive (RFC 9110); only the credentials
        # are compared in constant time. Splitting on whitespace also keeps a
        # header like "Bearer secret extra" invalid, because the remainder is
        # then compared as a whole.
        try:
            scheme, credentials = provided.decode("latin-1").split(None, 1)
        except ValueError:
            await _reject(send)
            return
        if scheme.lower() == "bearer" and hmac.compare_digest(
                credentials.encode("latin-1"), self._expected):
            await self.app(scope, receive, send)
            return
        await _reject(send)


async def _reject(send: Any) -> None:
    await send({
        "type": "http.response.start",
        "status": 401,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
            (b"www-authenticate", b'Bearer realm="loci"'),
        ],
    })
    await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})


def resolve_http_auth(transport: str, host: str, *, token: str,
                      allow_insecure: bool) -> str | None:
    """Return the token to enforce, or None when the transport needs none.

    Raises RuntimeError when an exposed HTTP bind would run unauthenticated and
    the caller did not opt in via `MCP_ALLOW_INSECURE_HTTP=true`.
    """
    if transport not in HTTP_TRANSPORTS:
        return None
    token = (token or "").strip()
    if token:
        return token
    if allow_insecure:
        return None
    if (host or "").strip().lower() in LOOPBACK_HOSTS:
        return None
    raise RuntimeError(
        "refusing to start an unauthenticated HTTP transport on "
        f"{host or '0.0.0.0'}: set MCP_AUTH_TOKEN (generate one with "
        "python -c \"import secrets; print(secrets.token_urlsafe(32))\"); "
        "MCP_ALLOW_INSECURE_HTTP=true only for a trusted sandbox"
    )
