"""Terminal chat with the agent.

    python -m voiceops.cli                     # in-process, no server needed
    python -m voiceops.cli --url http://localhost:8000
    python -m voiceops.cli --lang en --traces
"""

import argparse
import asyncio
import contextlib
import json

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the VoiceOps agent")
    parser.add_argument("--url", help="Base URL of a running API; omit to run in-process")
    parser.add_argument("--lang", choices=["tr", "en"], default="tr")
    parser.add_argument("--traces", action="store_true", help="Print tool calls and latency per turn")
    args = parser.parse_args()

    async with contextlib.AsyncExitStack() as stack:
        if args.url:
            client = await stack.enter_async_context(httpx.AsyncClient(base_url=args.url, timeout=30))
        else:
            from voiceops.main import app

            await stack.enter_async_context(app.router.lifespan_context(app))
            client = await stack.enter_async_context(
                httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://voiceops", timeout=30)
            )

        conv = (await client.post("/v1/conversations", json={"channel": "text", "language": args.lang})).json()
        print(f"conversation {conv['id']}  (provider: {conv['llm_provider']})  — Ctrl+D to quit\n")
        while True:
            try:
                text = input("you  > ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            resp = await client.post(f"/v1/conversations/{conv['id']}/messages", json={"text": text})
            if resp.status_code != 200:
                print(f"error {resp.status_code}: {resp.text}")
                continue
            turn = resp.json()
            print(f"bot  > {turn['reply']}")
            if args.traces:
                for c in turn["tool_calls"]:
                    print(f"       ↳ {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)}) {c['status']} {c['latency_ms']}ms")
                print(f"       intent={turn['intent']} lang={turn['language']} status={turn['status']} {turn['latency_ms']}ms")
            print()


if __name__ == "__main__":
    asyncio.run(main())
