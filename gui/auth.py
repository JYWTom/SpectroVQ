"""
Jupyter-style token auth for a GUI on a shared server.

Every user on the machine can reach any 127.0.0.1 port, so the server only answers requests that
carry its token: as ?token=... (once; it is then moved into a cookie and stripped from the URL),
as the cookie, or as an "Authorization: token ..." header for scripted clients.
"""

import html
import secrets
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, urlencode

# Gradio requests this from itself during launch() and fails to start if it is refused.
OPEN_PATHS = {"/gradio_api/startup-events"}

LOGIN_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>SpectroVQ - token required</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 32rem; margin: 15vh auto; padding: 0 1rem; }}
input {{ width: 100%; padding: .5rem; font-family: monospace; box-sizing: border-box; }}
button {{ margin-top: .75rem; padding: .5rem 1rem; }}
.error {{ color: #b00020; }}
</style></head>
<body>
<h1>SpectroVQ</h1>
<p>This server needs a token. Open the full URL printed in the terminal where you started
<code>gui.py</code>, or run <code>python gui.py --list</code> on the server and paste the token here.</p>
{error}
<form method="get" action="{action}">
<input name="token" type="password" placeholder="token" autofocus autocomplete="off">
<button type="submit">Log in</button>
</form>
</body></html>
"""


def make_token():
    return secrets.token_hex(24)


def cookie_name(port):
    # Cookies are not port-scoped: two servers tunnelled to one browser need different names.
    return f"spectrovq_token_{port}"


class TokenAuth:
    """Pure ASGI middleware (not BaseHTTPMiddleware) so Gradio's SSE streams are not buffered."""

    def __init__(self, app, token):
        self.app = app
        self.token = token

    def _ok(self, candidate):
        return bool(candidate) and secrets.compare_digest(
            candidate.encode(), self.token.encode()
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or scope["path"] in OPEN_PATHS:
            return await self.app(scope, receive, send)

        query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        query_token = next((v for k, v in query if k == "token"), None)
        headers = dict(scope.get("headers", []))
        cookie = SimpleCookie(headers.get(b"cookie", b"").decode("latin-1"))
        name = cookie_name(scope["server"][1] if scope.get("server") else "")
        cookie_token = cookie[name].value if name in cookie else None
        auth = headers.get(b"authorization", b"").decode("latin-1")
        header_token = auth[6:].strip() if auth.lower().startswith("token ") else None

        if self._ok(query_token) and scope["type"] == "http" and scope["method"] == "GET":
            # Swap the URL token for a cookie, so it stays out of history and bookmarks.
            rest = urlencode([(k, v) for k, v in query if k != "token"])
            location = scope.get("root_path", "") + scope["path"] + (f"?{rest}" if rest else "")
            return await _respond(
                send,
                302,
                b"",
                [
                    (b"location", location.encode("latin-1")),
                    (
                        b"set-cookie",
                        f"{name}={self.token}; HttpOnly; SameSite=Lax; Path=/".encode(),
                    ),
                ],
            )
        if any(self._ok(t) for t in (query_token, cookie_token, header_token)):
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        if scope["method"] in ("GET", "HEAD") and scope["path"] in ("/", ""):
            # 401 rather than 403: Gradio's own launch() check treats 401 on "/" as "server is up".
            error = '<p class="error">Wrong token.</p>' if query_token is not None else ""
            action = html.escape(scope.get("root_path", "") + "/")
            body = LOGIN_PAGE.format(error=error, action=action).encode()
            return await _respond(
                send, 401, body, [(b"content-type", b"text/html; charset=utf-8")]
            )
        return await _respond(
            send, 403, b"Forbidden: missing or wrong token\n", [(b"content-type", b"text/plain")]
        )


async def _respond(send, status, body, headers):
    headers = [*headers, (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
