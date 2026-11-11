import json
import logging

from voiceops.observability import JsonFormatter, redact, redact_text


def test_redact_text_masks_phone_dates_and_national_id():
    out = redact_text("Numaram 0555 010 1001, doğum 12.04.1988, TCKN 12345678901")
    assert "1001" in out and "0555 010" not in out
    assert "12.04.1988" not in out and "[DATE]" in out
    assert "12345678901" not in out


def test_redact_keeps_operational_fields():
    payload = {
        "date_of_birth": "1988-04-12",
        "phone": "+905550101001",
        "transactions": [{"id": "TX-50001", "amount": "2500.00", "created_at": "2026-11-07T10:00:00+00:00"}],
    }
    out = redact(payload)
    assert out["date_of_birth"] == "[REDACTED]"
    assert out["phone"] == "[PHONE…1001]"
    assert out["transactions"][0] == payload["transactions"][0]


def test_json_log_lines_are_redacted():
    record = logging.LogRecord("voiceops.trace", logging.INFO, __file__, 1, "span", None, None)
    record.fields = {"kind": "tool", "input": {"date_of_birth": "1988-04-12"}}
    line = json.loads(JsonFormatter().format(record))
    assert line["kind"] == "tool" and line["input"]["date_of_birth"] == "[REDACTED]"


async def test_birth_date_never_reaches_stored_traces(new_chat):
    chat = await new_chat("tr")
    await chat.say("C-1001 kartım çalındı")
    await chat.say("12.04.1988")
    traces = (await chat.client.get(f"/v1/conversations/{chat.id}/traces")).json()
    dump = json.dumps(traces, ensure_ascii=False)
    assert "1988-04-12" not in dump and "12.04.1988" not in dump
    verify = next(s for t in traces for s in t["spans"] if s["name"] == "verify_identity")
    assert verify["input"]["date_of_birth"] == "[REDACTED]"


async def test_turn_response_redacts_tool_arguments(new_chat):
    chat = await new_chat()
    await chat.say("C-1003 kartımı bloke edin")
    turn = await chat.say("27.01.1996")
    verify = next(c for c in turn["tool_calls"] if c["name"] == "verify_identity")
    assert verify["arguments"] == {"date_of_birth": "[REDACTED]"}
    assert "1996" not in str(turn["tool_calls"])
