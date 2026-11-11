"""Retry policy shared by every outbound HTTP client (integrations, speech providers).

Rules of thumb this encodes:

* retry only what is safe to repeat: reads, and writes that carry an idempotency key
* retry only transient failures: timeouts, connection errors, 429 and 5xx
* exponential backoff with full jitter, so retries from many calls don't synchronise
* a per-attempt timeout *and* a total deadline: a voice caller is waiting, so a
  slow system fails fast into a fallback ("shall we call you back?") instead of
  stacking retries
* honour `Retry-After`, but give up instead of waiting longer than the budget allows
"""

import random
from dataclasses import dataclass

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.2
    max_delay_s: float = 2.0
    attempt_timeout_s: float = 3.0
    deadline_s: float = 6.0
    retry_statuses: frozenset[int] = RETRYABLE_STATUSES

    def backoff(self, attempt: int, rng: random.Random | None = None) -> float:
        """Full-jitter delay after the `attempt`-th failure (1-based)."""
        cap = min(self.max_delay_s, self.base_delay_s * 2 ** (attempt - 1))
        return (rng or random).uniform(0, cap)

    def next_delay(self, attempt: int, elapsed_s: float, retry_after_s: float | None, rng: random.Random | None = None) -> float | None:
        """Delay before the next attempt, or None when the call should give up."""
        if attempt >= self.max_attempts:
            return None
        delay = retry_after_s if retry_after_s is not None else self.backoff(attempt, rng)
        if delay > self.max_delay_s or elapsed_s + delay >= self.deadline_s:
            return None
        return delay


def parse_retry_after(value: str | None) -> float | None:
    """Seconds form only; the HTTP-date form is treated as absent."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
