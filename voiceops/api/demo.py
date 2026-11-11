"""Web demo page and a fault-injection proxy for it.

The proxy forwards to the mock systems' /_faults endpoint through the same
integration client the agent uses, so it works whether the mock systems run
in-process or as their own container.
"""

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from voiceops.api.security import require_supervisor
from voiceops.integrations import IntegrationClient, IntegrationError
from voiceops.mock_systems.middleware import FaultRuleIn

PAGE = Path(__file__).resolve().parents[1] / "web" / "demo.html"

router = APIRouter(tags=["demo"])
faults = APIRouter(prefix="/v1/demo/faults", tags=["demo"], dependencies=[Depends(require_supervisor)])


def _integrations(request: Request) -> IntegrationClient:
    return request.app.state.orchestrator.integrations


Integrations = Annotated[IntegrationClient, Depends(_integrations)]


async def _forward(client: IntegrationClient, method: str, body: dict[str, Any] | None = None) -> Any:
    try:
        return await client.request(method, "/_faults", **({"json": body} if body is not None else {}))
    except IntegrationError as exc:
        raise HTTPException(502, detail={"code": "mock_systems_unavailable", "message": exc.message}) from exc


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/demo")


@router.get("/demo", include_in_schema=False)
async def demo_page() -> FileResponse:
    return FileResponse(PAGE, media_type="text/html")


@faults.get("")
async def list_faults(client: Integrations) -> Any:
    return await _forward(client, "GET")


@faults.post("", status_code=201)
async def add_fault(rule: FaultRuleIn, client: Integrations) -> Any:
    return await _forward(client, "POST", rule.model_dump())


@faults.delete("", status_code=204)
async def clear_faults(client: Integrations) -> None:
    await _forward(client, "DELETE")
