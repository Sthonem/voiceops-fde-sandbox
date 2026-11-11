# API examples

Real requests and responses, captured from a run against the mock providers
(IDs and timings differ per run; long fields trimmed with `…`). The same calls
are in [`examples/requests.http`](../examples/requests.http) (VS Code REST Client /
JetBrains HTTP client) and [`examples/curl-demo.sh`](../examples/curl-demo.sh).
Full schema: `http://localhost:8000/docs`.

Base URL `http://localhost:8000`. Supervisor endpoints take `X-API-Key` when
`SUPERVISOR_API_KEY` is set.

## Health

```http
GET /health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "llm_provider": "mock",
  "llm_model": "mock-rules-v1",
  "tools": ["lookup_customer", "verify_identity", "get_transaction_status", "search_policies", "create_ticket",
            "schedule_callback", "block_card", "dispute_transaction", "escalate_to_human"],
  "policy_passages": 36,
  "stt_provider": "mock",
  "tts_provider": "mock",
  "tts_media_type": "audio/wav",
  "mock_systems": "in-process",
  "supervisor_auth": "disabled (development)"
}
```

## A conversation turn

```http
POST /v1/conversations
{"language": "tr"}
```

```json
{"id": "f2e32da5-682d-4d9d-a65c-de53cad4f769", "channel": "text", "status": "active", "language": "tr",
 "customer_id": null, "llm_provider": "mock", "created_at": "2026-11-08T07:15:59.165337Z"}
```

```http
POST /v1/conversations/{id}/messages
{"text": "C-1002, TX-50015 işlemine itiraz ediyorum, ben yapmadım"}
```

```json
{
  "conversation_id": "f2e32da5-…",
  "turn": 1,
  "trace_id": "6c2b1556687847e7b4045da9812c32c4",
  "reply": "Teşekkürler Mehmet. Bu işlem için ek bir güvenlik doğrulaması gerekiyor. Doğum tarihinizi gün, ay ve yıl olarak söyler misiniz?",
  "intent": "dispute_transaction",
  "routing": {"intent": "dispute_transaction", "confidence": 0.76, "source": "keywords", "sensitive_category": null},
  "language": "tr",
  "status": "active",
  "customer_id": "C-1002",
  "verified": false,
  "awaiting_confirmation": null,
  "approval_ids": [],
  "citations": [],
  "tool_calls": [
    {"name": "lookup_customer", "arguments": {"customer_id": "C-1002"}, "status": "ok",
     "latency_ms": 2.96, "attempts": 1, "replayed": false}
  ],
  "latency_ms": 4.11,
  "usage": {"provider": "mock", "model": "mock-rules-v1", "input_tokens": 2550, "output_tokens": 37,
            "cache_read_tokens": 0, "estimated": true, "estimated_cost_usd": 0.01094}
}
```

The dispute is sensitive, so the registry asked for step-up verification
before doing anything. Next turn, the date of birth (redacted in every response
and span), then the confirmation gate:

```http
POST /v1/conversations/{id}/messages
{"text": "03.09.1975"}
```

```json
{
  "turn": 2,
  "reply": "06.11.2026 tarihli, Mücevher Butiği için 12.450,00 TL tutarındaki işleme itiraz kaydı açacağım. Onaylıyor musunuz? Bu tutardaki itirazlar yetkili onayından sonra açılıyor.",
  "routing": {"intent": "dispute_transaction", "confidence": 0.85, "source": "carry_over", "sensitive_category": null},
  "verified": true,
  "awaiting_confirmation": {
    "tool": "dispute_transaction",
    "summary": {"transaction_id": "TX-50015", "amount": "12450.00", "currency": "TRY",
                "counterparty": "Mücevher Butiği", "created_at": "2026-11-06T05:15:59.205635Z"}
  },
  "tool_calls": [
    {"name": "verify_identity", "arguments": {"date_of_birth": "[REDACTED]"}, "status": "ok", "attempts": 1, "replayed": false, "latency_ms": 2.79},
    {"name": "dispute_transaction", "arguments": {"transaction_id": "TX-50015", "reason": "…"}, "status": "needs_confirmation", "attempts": 1, "replayed": false, "latency_ms": 2.64}
  ]
}
```

```http
POST /v1/conversations/{id}/messages
{"text": "evet onaylıyorum"}
```

```json
{
  "turn": 3,
  "reply": "Talebinizi yetkili onayına gönderdim, referans numaranız APR-0E2DF5CE. Karar verildiğinde size bilgi vereceğiz.",
  "routing": {"intent": "dispute_transaction", "confidence": 0.95, "source": "confirmation", "sensitive_category": null},
  "approval_ids": ["APR-0E2DF5CE"],
  "tool_calls": [{"name": "dispute_transaction", "status": "pending_approval", "…": "…"}]
}
```

## Supervisor approval

```http
GET /v1/approvals
```

```json
[
  {
    "id": "APR-0E2DF5CE",
    "conversation_id": "f2e32da5-…",
    "customer_id": "C-1002",
    "tool": "dispute_transaction",
    "summary": {"transaction_id": "TX-50015", "amount": "12450.00", "currency": "TRY", "counterparty": "Mücevher Butiği", "…": "…"},
    "reason": "dispute_transaction above 10000 TRY",
    "status": "pending",
    "reviewer": null,
    "result": {}
  }
]
```

```http
POST /v1/approvals/APR-0E2DF5CE/decision
X-API-Key: <SUPERVISOR_API_KEY>
{"decision": "approve", "reviewer": "sup.demo"}
```

```json
{"id": "APR-0E2DF5CE", "status": "executed", "reviewer": "sup.demo",
 "result": {"id": "DSP-000001", "transaction_id": "TX-50015", "status": "open", "review_days": 30}, "…": "…"}
```

The caller hears the decision on their next turn, exactly once:

```http
POST /v1/conversations/{id}/messages
{"text": "Teşekkürler"}
```

```json
{"turn": 4, "intent": "goodbye",
 "reply": "Bilginiz olsun: APR-0E2DF5CE numaralı talebiniz onaylandı ve DSP-000001 numaralı kaydınız açıldı. Rica ederim. Başka bir konuda yardımcı olabilir miyim? Değilse iyi günler dilerim."}
```

A second decision on the same request returns `409 already_decided`.

## Safe client retries

A client that times out can resend the turn with the same key. The stored reply comes
back and the turn does not run again:

```http
POST /v1/conversations/{id}/messages
Idempotency-Key: retry-1
{"text": "Teşekkürler"}
```

```text
first call   → 200, turn 5, trace dda5f62e…
second call  → 200, turn 5, trace dda5f62e…, header Idempotent-Replayed: true
same key, different text → 422 idempotency_key_reused
```

## Fault injection and what the trace shows

Lose the helpdesk's response after it has created the ticket (works whether the
mock systems run in-process or as their own container):

```http
POST /v1/demo/faults
X-API-Key: <SUPERVISOR_API_KEY>
{"path_prefix": "/helpdesk/tickets", "method": "POST", "mode": "error_after_commit"}
```

```json
{"id": "FLT-66E223", "path_prefix": "/helpdesk/tickets", "method": "POST", "mode": "error_after_commit",
 "status": 503, "delay_ms": 0, "retry_after_s": null, "times": 1, "hits": 0}
```

Then ask for a ticket (after identifying as C-1003). One ticket is created, and the
tool call shows the retry and the replay:

```json
{
  "reply": "TCK-000003 numaralı kaydınızı oluşturdum. Ekibimiz en geç 48 saat içinde size dönüş yapacak.",
  "tool_calls": [
    {"name": "create_ticket", "arguments": {"category": "app", "priority": "normal", "summary": "…"},
     "status": "ok", "latency_ms": 154.1, "attempts": 2, "replayed": true}
  ]
}
```

```http
GET /v1/conversations/{id}/traces
```

```json
[{"turn": 2, "trace_id": "…", "spans": [
  {"kind": "turn", "name": "turn", "status": "ok", "latency_ms": 154.7,
   "output": {"intent": "create_ticket", "status": "active", "estimated_cost_usd": 0.012336, "model": "mock-rules-v1"}},
  {"kind": "tool", "name": "create_ticket", "status": "ok", "latency_ms": 154.1,
   "output": {"id": "TCK-000003", "sla_hours": 48, "…": "…",
              "http": [{"method": "POST", "path": "/helpdesk/tickets", "status": 201, "attempts": 2,
                        "latency_ms": 154.05, "idempotency_key": "54f56a7a8948", "replayed": true}]}}
]}]
```

Other modes: `error` (fail before the handler), `delay` (with `delay_ms`), any
status (e.g. `429` with `retry_after_s`), `times` to affect N requests.
`DELETE /v1/demo/faults` clears them.

## Voice WebSocket

`ws://localhost:8000/v1/voice`. JSON control frames plus binary audio frames;
full protocol in [`voiceops/voice/session.py`](../voiceops/voice/session.py).
Abbreviated frame sequence; the metrics are from a `python -m voiceops.voice_cli` run with mock speech.

```text
→ {"type": "session.start", "language": "tr", "audio": {"format": "wav", "sample_rate": 16000}}
← {"type": "session.started", "conversation_id": "…", "stt": "mock", "tts": "mock", "tts_media_type": "audio/wav", …}
← {"type": "tts.start", "segment": 1, "kind": "greeting", "text": "Merhaba, VoiceOps Demo Bank'a hoş geldiniz. …"}
← <binary audio> … {"type": "tts.end", "segment": 1, "bytes": …, "tts_ms": …}
← {"type": "turn.done", "kind": "greeting", …}
→ <binary WAV of the caller's utterance, in chunks>
→ {"type": "audio.end"}
← {"type": "transcript.final", "text": "0555 010 1001", "language": "tr", "confidence": 0.95, "stt_ms": 0, …}
← {"type": "agent.tool", "phase": "start", "name": "lookup_customer"}
← {"type": "tts.start", "segment": 2, "kind": "reply", "text": "Teşekkürler Ayşe."}   (while the turn is still running)
← <binary audio> … {"type": "tts.end", …}
← {"type": "agent.tool", "phase": "end", "name": "lookup_customer", "status": "ok", "latency_ms": 3}
← {"type": "agent.reply", …same body as POST /messages…}
← {"type": "tts.start", "segment": 3, …} … {"type": "tts.end", …}
← {"type": "turn.done", "kind": "agent", "metrics": {"stt_ms": 0, "agent_ms": 26, "first_audio_ms": 8,
                                                     "tts_ms": 18, "total_ms": 28, …}}
→ {"type": "interrupt"}                       (barge-in: playback stops, the turn is not cancelled)
→ {"type": "text", "text": "…"}               (typed input during a call)
→ {"type": "session.end"}
← {"type": "session.ended"}
```

With `"format": "pcm16"` the client streams raw 16 kHz PCM and the server's VAD
decides when the caller has finished (and treats speech during playback as a barge-in).

## Evals

```bash
python -m voiceops.evals --baseline evals/baselines/mock.json --no-regressions --min intent_accuracy=0.85
```

```text
running 47 scenarios × 1 with mock …
  ok   en_transfer_failed_invalid_iban  (31 ms)
  …
  lim  tr_rag_callback_hours  (9 ms)
pass 93.6% · intent 90.6% · tools 97.6% · escalation 100.0% · p95 5.7 ms · cost/conv $0.0204
report: reports/eval-mock-….md
```
