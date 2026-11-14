# 2-minute demo script

What the viewer should leave with: **this agent acts on real systems safely, keeps
working when those systems misbehave, and its quality is measured, not claimed.**

## Before you record

```bash
docker compose up --build        # or: uvicorn voiceops.main:app
```

- Open `http://localhost:8000/demo`. Set **Bot speaks via** to *Browser voice*
  (Chrome or Safari read the replies aloud in Turkish and English) and keep
  **Caller speaks via** on *Simulated audio*. Each line you send goes through the
  real voice path: audio → STT → agent → sentence-by-sentence TTS.
- Second tab: [`docs/eval-report.md`](eval-report.md) (or the README "Measured Results").
- Fresh database, so the dispute isn't already open (`docker compose down -v`, or
  delete `voiceops.db`).

## Script

| Time | Do | Say |
|---|---|---|
| 0:00–0:12 | README architecture diagram | "A bilingual voice agent for a bank's contact center. The model proposes, the code disposes: identity, verification, confirmations and approvals are enforced outside the model. Every system it touches is a mock behind REST, as at a customer site." |
| 0:12–0:45 | **Start call** (TR). Click the chips: *Kartım çalındı* → *0555 010 1001* → *12.04.1988* → *kredi kartım* → *Evet* | "Stolen card. It asks who I am, then for a second factor, because blocking is irreversible. Two cards match, so it asks which one. It reads back what it will do and waits for an explicit yes." Point at **Agent view**: "Each turn shows the intent, every tool call with its status, and time to first audio. The bot starts talking on the first sentence while the rest is still being written." |
| 0:45–1:15 | **End call**, **Start call**, switch language to EN. Chips: *Customer C-1005, I want to dispute TX-50016* → *19.06.1979* → *yes*. In **Supervisor queue** click **Approve**, then type *ok thanks* | "15,900 lira is above the threshold, so the agent can't open this dispute: a named supervisor has to. The core banking mock would refuse it as well, as a second line of defence. Approve, and on the caller's next turn they hear the outcome, exactly once." |
| 1:15–1:40 | **Break the backend → Helpdesk: lose response after commit**. New call (TR): type *müşteri numaram C-1003*, then *Mobil uygulama sürekli hata veriyor, şikayet kaydı açın lütfen*. Then arm **CRM: 1.5 s stall** and type *müşteri numaram C-1001* in a new call | "The helpdesk created the ticket but the response was lost. The retry carries an idempotency key, so the tool shows 2 attempts and an idempotent replay, and there is one ticket, not two. When the CRM stalls, the caller hears 'one moment' instead of silence." |
| 1:40–2:00 | Eval report headline | "55 golden conversations in Turkish and English, voice included, run on every change. The only failures are known router limits on paraphrases, flagged as such. The eval also caught a real bug: a safe-deposit-box question answered from the card-fee section. That's fixed, and the baseline diff proves nothing regressed. Next step at a customer: swap the mocks for their APIs and run the same evals against Claude." |

## If something goes wrong on camera

- **No voice in the browser**: switch **Bot speaks via** to *Muted*; the
  transcript still shows each sentence as it is spoken.
- **Dispute says it is already open**: the database isn't fresh. Use TX-50017
  (420 TL, no supervisor needed) or reset the database.
- **Supervisor queue says 401**: `SUPERVISOR_API_KEY` is set; paste it in the
  key field under the queue.

## Talking points for questions

- *Why not let the model decide confirmations?* A prompt is advice; the registry
  is enforcement. A confused or injected model still can't act for an unverified
  caller, confirm on the caller's behalf, or skip the supervisor.
- *What happens if the caller interrupts?* Playback stops; the turn is not
  cancelled, because a tool may already have changed a system of record. The
  transcript records which sentences were actually heard.
- *Data protection?* Audio is never stored; traces, logs and API responses are
  redacted (phone, date of birth, national ID); local Whisper keeps audio on-prem.
- *Numbers on a real model?* `python -m voiceops.evals --provider anthropic`
  runs the same 55 conversations against Claude Opus 5.5 and reports latency
  and real cost next to the mock baseline.
