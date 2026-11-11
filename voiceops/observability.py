"""Structured logging and PII redaction.

Transcripts are the system of record and keep what the caller said. Everything
that leaves that store (trace payloads, log lines, eval exports) goes through
`redact()` first, so phone numbers, birth dates and national ids never end up in
log aggregation or dashboards.
"""

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

_SENSITIVE_KEYS = {"date_of_birth", "dob", "phone", "email", "tckn", "national_id"}
_PHONE = re.compile(r"(?<![\w-])\+?\d[\d\s\-()]{8,}(\d{4})\b")
_DATE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{4}\b")
_TCKN = re.compile(r"\b[1-9]\d{10}\b")  # Turkish national id: 11 digits


def redact_text(text: str) -> str:
    text = _TCKN.sub("[TCKN]", text)
    text = _PHONE.sub(lambda m: f"[PHONE…{m.group(1)}]", text)
    return _DATE.sub("[DATE]", text)


def redact(value: Any, key: str | None = None) -> Any:
    if key and key.lower() in _SENSITIVE_KEYS and value not in (None, ""):
        if key.lower() == "phone" and isinstance(value, str) and len(value) >= 4:
            return f"[PHONE…{value[-4:]}]"
        return "[REDACTED]"
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if isinstance(extra := getattr(record, "fields", None), dict):
            payload.update(redact(extra))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(json_logs: bool = True, level: str = "INFO") -> None:
    logger = logging.getLogger("voiceops")
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


span_logger = logging.getLogger("voiceops.trace")
