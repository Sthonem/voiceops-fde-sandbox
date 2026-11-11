"""HTTP client the agent uses to reach the customer's systems.

* normalised errors (`IntegrationError`) that are safe to show to the LLM
* per-attempt timeout + total deadline, retries with full-jitter backoff
  (see voiceops/retry.py)
* every POST carries an `Idempotency-Key`. Inside a tool call the key is derived
  from the conversation and the tool call id, so a retried request, or the same
  tool call executed again, cannot create a second ticket or dispute
* each call is recorded (method, path, status, attempts, latency, replayed) and
  attached to the tool span for observability
"""

import asyncio
import hashlib
import logging
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import httpx

from voiceops.config import Settings
from voiceops.retry import SAFE_METHODS, RetryPolicy, parse_retry_after

log = logging.getLogger("voiceops.integrations")

_scope: ContextVar[str | None] = ContextVar("integration_scope", default=None)
_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("integration_calls", default=None)


@contextmanager
def integration_scope(scope: str | None) -> Iterator[list[dict[str, Any]]]:
    """Within this block, POSTs get deterministic idempotency keys derived from
    `scope` and calls are collected into the yielded list."""
    calls: list[dict[str, Any]] = []
    t_scope, t_calls = _scope.set(scope), _calls.set(calls)
    try:
        yield calls
    finally:
        _scope.reset(t_scope)
        _calls.reset(t_calls)


def derive_idempotency_key(scope: str, method: str, path: str, occurrence: int = 0) -> str:
    return hashlib.sha256(f"{scope}|{method}|{path}|{occurrence}".encode()).hexdigest()[:32]


class IntegrationError(Exception):
    """Normalised failure from an external system, safe to show to the LLM."""

    def __init__(self, system: str, status: int | None, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{system}: {code} ({status}) {message}")
        self.system = system
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.attempts = 1

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status >= 500 or self.status == 429

    def to_payload(self) -> dict[str, Any]:
        payload = {"error": self.code, "status": self.status, "system": self.system, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


def _error_from_response(system: str, resp: httpx.Response) -> IntegrationError:
    try:
        detail = resp.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        code, message = detail.get("code", "error"), detail.get("message", "")
    else:
        code, message = ("validation_error" if resp.status_code == 422 else "http_error"), str(detail or resp.text)[:300]
    return IntegrationError(system, resp.status_code, code, message)


class IntegrationClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._http = http
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._rng = rng
        self._clock = clock

    @classmethod
    def from_settings(cls, settings: Settings) -> "IntegrationClient":
        policy = RetryPolicy(
            max_attempts=settings.integration_max_attempts,
            base_delay_s=settings.integration_backoff_base_s,
            max_delay_s=settings.integration_backoff_max_s,
            attempt_timeout_s=settings.integration_timeout_s,
            deadline_s=settings.integration_deadline_s,
        )
        timeout = httpx.Timeout(settings.integration_timeout_s)
        if settings.mock_api_base_url == "inprocess":
            from voiceops.mock_systems.app import mock_app

            http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app), base_url="http://mock", timeout=timeout)
        else:
            http = httpx.AsyncClient(base_url=settings.mock_api_base_url, timeout=timeout)
        return cls(http, policy)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def request(self, method: str, path: str, *, idempotency_key: str | None = None, **kwargs: Any) -> Any:
        method = method.upper()
        system = path.strip("/").split("/", 1)[0]
        headers = dict(kwargs.pop("headers", None) or {})
        calls = _calls.get()
        if method == "POST":
            if idempotency_key is None:
                scope = _scope.get()
                seen = sum(1 for c in calls or [] if c["method"] == method and c["path"] == path)
                idempotency_key = derive_idempotency_key(scope, method, path, seen) if scope else uuid.uuid4().hex
            headers["Idempotency-Key"] = idempotency_key
        retry_safe = method in SAFE_METHODS or "Idempotency-Key" in headers

        policy = self.policy
        start = self._clock()
        attempt = 0
        while True:
            attempt += 1
            remaining = policy.deadline_s - (self._clock() - start)
            retry_after: float | None = None
            try:
                async with asyncio.timeout(max(0.05, min(policy.attempt_timeout_s, remaining))):
                    resp = await self._http.request(method, path, headers=headers, **kwargs)
            except (TimeoutError, httpx.TimeoutException):
                err = IntegrationError(system, None, "timeout", f"{system} did not answer in time")
            except httpx.TransportError:
                err = IntegrationError(system, None, "unreachable", f"{system} is unreachable")
            else:
                if resp.is_success:
                    self._record(calls, method, path, resp.status_code, attempt, start, idempotency_key,
                                 replayed=resp.headers.get("idempotent-replayed") == "true")  # fmt: skip
                    return resp.json() if resp.content else None
                err = _error_from_response(system, resp)
                retry_after = parse_retry_after(resp.headers.get("retry-after"))

            transient = err.status is None or err.status in policy.retry_statuses
            delay = policy.next_delay(attempt, self._clock() - start, retry_after, self._rng) if retry_safe and transient else None
            if delay is None:
                err.attempts = attempt
                self._record(calls, method, path, err.status, attempt, start, idempotency_key, error=err.code)
                raise err
            log.warning(
                "integration_retry",
                extra={"fields": {"system": system, "method": method, "path": path, "attempt": attempt,
                                  "error": err.code, "status": err.status, "delay_ms": round(delay * 1000)}},  # fmt: skip
            )
            await self._sleep(delay)

    def _record(
        self,
        calls: list[dict[str, Any]] | None,
        method: str,
        path: str,
        status: int | None,
        attempts: int,
        start: float,
        idempotency_key: str | None,
        *,
        replayed: bool = False,
        error: str | None = None,
    ) -> None:
        if calls is None:
            return
        rec: dict[str, Any] = {
            "method": method,
            "path": path,
            "status": status,
            "attempts": attempts,
            "latency_ms": round((self._clock() - start) * 1000, 2),
        }
        if idempotency_key:
            rec["idempotency_key"] = idempotency_key[:12]
        if replayed:
            rec["replayed"] = True
        if error:
            rec["error"] = error
        calls.append(rec)

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        return await self.request("POST", path, **kwargs)
