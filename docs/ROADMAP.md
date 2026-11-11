# Implementation Plan

Scenario: **VoiceOps Demo Bank** — a fictional Turkish digital bank whose contact
center handles Turkish and English calls. Every name, phone number, IBAN-like
identifier and transaction in this repository is synthetic.

## Design principles

1. **LLM proposes, orchestrator disposes.** The model may suggest tool calls, but
   identity checks, approval gates, escalation rules and step limits are enforced
   in plain Python outside the model. Swapping the model never weakens a guardrail.
2. **Integrate over the wire.** The agent talks to the CRM, core-banking,
   helpdesk, scheduling and contact-center systems through REST, exactly as it
   would at a customer site. The mock systems are a separate FastAPI app.
3. **Everything runs offline.** Mock LLM, mock STT and mock TTS providers make the
   whole system (and CI) work with zero API keys. Real providers are drop-in.
4. **Measure, then claim.** Every turn records latency, tool outcomes and token
   usage so the README's numbers come from the eval harness, not from memory.

## Milestones

| # | Scope | Status |
|---|-------|--------|
| 1 | Repo structure, architecture doc, data model, FastAPI skeleton, mock CRM / payments / helpdesk / scheduling / contact-center APIs, basic text agent flow, persistence of transcripts | **done** |
| 2 | Tool registry with guards, weighted intent router + sensitivity policy, BM25 RAG over TR/EN policy docs with citations, step-up verification, caller confirmation + supervisor approval gate, span traces + JSON logs + PII redaction, cost estimates, Claude provider (offline-tested) | **done** |
| 3 | Voice over WebSocket (push-to-talk or server VAD), STT/TTS provider abstraction (mock, Whisper local + HTTP, Azure neural TTS), streamed Claude text → sentence-level TTS, filler, barge-in, voice metrics; per-attempt timeouts + deadlines, retries with full-jitter backoff, idempotency keys (integrations and agent API), fault injection, tool-input schema validation, voice call CLI | **done** |
| 4 | Golden eval set (47 scenarios: TR + EN, text + voice, injected faults, paraphrases) in YAML with a strict schema; runner on fresh databases through the public APIs; intent accuracy, tool selection and tool-call success, escalation confusion matrix, latency (clean vs. faulted), voice time to first audio, cost per conversation, pass^k over repeats; failure-mode report; baseline diff and CI gate; retrieval coverage gate (defect found by the eval) | **done** |
| 5 | Docker Compose (API + mock systems + Postgres, separate databases), Alembic migrations for the agent schema, Postgres advisory locks across replicas, supervisor API key, web demo (simulated / browser / microphone input, supervisor queue, fault injection), deployment diagram, final README, 2-minute demo script, API examples (`docs/api-examples.md`, `examples/`), final eval report | **done** |

## Deliberate shortcuts

- Local development and tests create tables directly (`DB_AUTO_CREATE=true`);
  Docker uses Alembic. After a change to an existing table, delete the local
  `voiceops.db` or run `alembic upgrade head`.
- Per-conversation serialisation is an in-process lock on SQLite; across replicas
  it relies on the Postgres advisory lock (verified: the test suite on Postgres 16,
  and 12 concurrent turns through two API replicas with `examples/replica-race.py`).
- The mock systems write the idempotency record after the route commits, not in
  the same transaction (documented in `mock_systems/middleware.py`).
- Supervisor endpoints use a shared API key; the caller-facing API and voice socket
  are open and expect the telephony or web gateway in front to authenticate
  sessions. A production deployment needs SSO for supervisors and signed session
  tokens for the browser.
- Verified with Docker Compose (Docker 29.5, Postgres 16): fresh-volume start,
  Alembic migration, separate databases, the curl walkthrough, voice over the
  WebSocket, supervisor key enforcement and the web demo. Not covered: real speech
  providers and Claude (need keys), TLS, and a production-grade secret store.
- The Claude provider (streaming and non-streaming), Whisper and Azure TTS are
  verified against the real SDK / HTTP shapes with fake transports; they have not
  been run against live services in CI (needs keys or model downloads).
- Energy-based VAD instead of a neural VAD; utterance-level STT (Whisper) instead
  of streaming STT with partial transcripts.
- Fault rules live in process memory of the mock systems.

## Known limits

Tracked as strict xfails in `tests/` and as `known_limit` scenarios in
`evals/golden/` (both fail the build when they start passing). Current list in
the generated report: [eval-report.md](eval-report.md#known-limits).

- Voice and turn latencies from the mock providers measure the pipeline, not
  model or speech inference; costs are estimated from mock token counts.
- The Claude path has not been evaluated live (no key on the build machine);
  `python -m voiceops.evals --provider anthropic` produces the comparison.


- Retrieval: "geri arama saatleri" ranks the SLA section above the callback
  section because "saat" is frequent there; the router also reads it as a
  callback request.
- Keyword router: paraphrases without trigger words are mislabelled (3 of 5
  paraphrase scenarios fail). Not patched per sentence on purpose; an LLM or
  embedding router is the fix, and the paraphrase slice measures it.
