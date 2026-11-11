"""Supervisor-facing endpoints (approvals, fault injection) take an API key.

The caller-facing conversation API and voice socket sit behind the telephony or
web gateway in a real deployment, which authenticates the session; here they are
open. Comparison is constant-time.
"""

import hmac

from fastapi import Header, HTTPException, Request


def require_supervisor(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    key = request.app.state.settings.supervisor_api_key
    if key is None:  # local development only; /health says so
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), key.get_secret_value().encode()):
        raise HTTPException(401, detail={"code": "unauthorized", "message": "Missing or invalid X-API-Key"})
