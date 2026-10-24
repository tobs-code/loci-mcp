"""HTTP transport authentication.

The MCP HTTP transports expose every tool; an unauthenticated bind hands
`drop_project`/`prune_memories` to anyone who can reach the port.
"""

import asyncio
import json

import pytest

from mcp_server.http_auth import (
    BearerAuthMiddleware,
    HTTP_TRANSPORTS,
    LOOPBACK_HOSTS,
    resolve_http_auth,
)


class _Recorder:
    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


def _call(app, headers=None, scope_type="http"):
    sent = []
    scope = {
        "type": scope_type,
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in (headers or {}).items()
        ],
    }

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def _status(sent):
    return sent[0]["status"]


def _body(sent):
    return json.loads(sent[-1]["body"].decode("utf-8"))


def _header(sent, name):
    for key, value in sent[0]["headers"]:
        if key.lower() == name.encode():
            return value.decode("latin-1")
    return None


def test_stdio_needs_no_token():
    assert resolve_http_auth("stdio", "0.0.0.0", token="",
                             allow_insecure=False) is None


def test_loopback_http_bind_is_allowed_without_token():
    for host in LOOPBACK_HOSTS:
        assert resolve_http_auth("streamable-http", host, token="",
                                 allow_insecure=False) is None


def test_exposed_http_bind_without_token_is_refused():
    for transport in HTTP_TRANSPORTS:
        with pytest.raises(RuntimeError) as exc:
            resolve_http_auth(transport, "0.0.0.0", token="",
                              allow_insecure=False)
        assert "MCP_AUTH_TOKEN" in str(exc.value)


def test_insecure_opt_in_allows_an_exposed_bind():
    assert resolve_http_auth("sse", "0.0.0.0", token="",
                             allow_insecure=True) is None


def test_configured_token_is_returned_and_trims_whitespace():
    assert resolve_http_auth("streamable-http", "0.0.0.0",
                             token="  secret  ",
                             allow_insecure=False) == "secret"


def test_middleware_forwards_a_matching_request():
    inner = _Recorder()
    app = BearerAuthMiddleware(inner, "secret")

    sent = _call(app, {"Authorization": "Bearer secret"})

    assert _status(sent) == 204
    assert inner.calls == 1


def test_middleware_rejects_a_missing_header():
    inner = _Recorder()
    app = BearerAuthMiddleware(inner, "secret")

    sent = _call(app)

    assert _status(sent) == 401
    assert inner.calls == 0
    body = _body(sent)
    assert body["ok"] is False
    assert body["status"] == "error"
    assert body["id"] is None
    assert "Authorization" in body["error"]
    assert _header(sent, "www-authenticate") == 'Bearer realm="loci"'


def test_middleware_rejects_a_wrong_token():
    inner = _Recorder()
    app = BearerAuthMiddleware(inner, "secret")

    assert _status(_call(app, {"Authorization": "Bearer wrong"})) == 401
    assert inner.calls == 0


def test_middleware_rejects_a_header_with_extra_content():
    inner = _Recorder()
    app = BearerAuthMiddleware(inner, "secret")

    # `hmac.compare_digest` on the full value means only the exact header works.
    assert _status(_call(app, {"Authorization": "Bearer secret extra"})) == 401
    assert inner.calls == 0


def test_middleware_leaves_lifespan_scopes_alone():
    inner = _Recorder()
    app = BearerAuthMiddleware(inner, "secret")

    sent = _call(app, scope_type="lifespan")

    assert _status(sent) == 204
    assert inner.calls == 1
