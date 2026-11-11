"""Check that turns stay serialised per conversation across API replicas.

Fires N concurrent turns at one conversation, alternating between the given base
URLs, then checks the transcript: turn numbers 1..N with no duplicates, unique
message sequence numbers, and a final state turn of N. Exit code 1 on a race.

    docker compose up -d
    docker compose run -d --name voiceops-api-2 -p 127.0.0.1:8002:8000 api
    python examples/replica-race.py http://localhost:8000 http://localhost:8002
    docker rm -f voiceops-api-2
"""

import asyncio
import sys
from collections import Counter

import httpx

TEXTS = ["Müşteri numaram C-1001", "Ödememin durumu ne?", "EFT kesim saati kaç?", "Teşekkürler"]


async def main(bases: list[str], n: int = 12) -> int:
    async with httpx.AsyncClient(timeout=60) as client:
        conv = (await client.post(f"{bases[0]}/v1/conversations", json={"language": "tr"})).json()["id"]

        async def turn(i: int) -> tuple[int, int | None]:
            r = await client.post(f"{bases[i % len(bases)]}/v1/conversations/{conv}/messages", json={"text": TEXTS[i % len(TEXTS)]})
            return r.status_code, r.json().get("turn")

        results = await asyncio.gather(*(turn(i) for i in range(n)))
        transcript = (await client.get(f"{bases[0]}/v1/conversations/{conv}")).json()
    seqs = [m["seq"] for m in transcript["messages"]]
    turns = sorted(t for _, t in results if t is not None)
    print(f"{n} concurrent turns via {len(bases)} replica(s): HTTP {dict(Counter(s for s, _ in results))}")
    print(f"turn numbers: {turns}")
    print(f"messages: {len(seqs)}, unique seqs: {len(set(seqs))}, final state turn: {transcript['state']['turn']}")
    ok = turns == list(range(1, n + 1)) and len(seqs) == len(set(seqs)) and transcript["state"]["turn"] == n
    print("serialised" if ok else "RACE: turns were not serialised")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or ["http://localhost:8000"])))
