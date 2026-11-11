"""FastAPI entrypoint.

    uvicorn voiceops.main:app --reload

The mock enterprise systems are mounted under /mock for single-process local
runs; in Docker Compose they run as their own service.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from voiceops import __version__
from voiceops.agent.llm import build_llm_provider
from voiceops.agent.orchestrator import Orchestrator
from voiceops.agent.router import IntentRouter
from voiceops.agent.tools import build_default_registry
from voiceops.api.approvals import router as approvals_router
from voiceops.api.conversations import router as conversations_router
from voiceops.api.demo import faults as demo_faults_router
from voiceops.api.demo import router as demo_router
from voiceops.api.voice import router as voice_router
from voiceops.config import Settings, get_settings
from voiceops.db import init_db
from voiceops.integrations import IntegrationClient
from voiceops.mock_systems.app import mock_app
from voiceops.observability import configure_logging
from voiceops.rag import load_policy_store
from voiceops.speech import build_stt, build_tts
from voiceops.voice import VoiceConfig, VoiceServices


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    inprocess = settings.mock_api_base_url == "inprocess"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_json, settings.log_level)
        # The mock systems live in this process only when nothing else hosts them.
        await init_db(seed=settings.seed_on_startup, agent=settings.db_auto_create, mocks=inprocess)
        integrations = IntegrationClient.from_settings(settings)
        app.state.orchestrator = Orchestrator(
            llm=build_llm_provider(settings),
            registry=build_default_registry(),
            integrations=integrations,
            max_steps=settings.agent_max_steps,
            router=IntentRouter(settings.tz),
            policies=load_policy_store(settings.policy_dir),
            tz=settings.tz,
        )
        app.state.voice = VoiceServices(
            orchestrator=app.state.orchestrator,
            stt=build_stt(settings),
            tts=build_tts(settings),
            config=VoiceConfig.from_settings(settings),
        )
        try:
            yield
        finally:
            await app.state.voice.aclose()
            await integrations.aclose()

    app = FastAPI(
        title="VoiceOps FDE Sandbox",
        version=__version__,
        description="Bilingual (TR/EN) contact-center agent with mock enterprise integrations. Synthetic data only.",
        lifespan=lifespan,
    )
    app.include_router(conversations_router)
    app.include_router(approvals_router)
    app.include_router(voice_router)
    app.state.settings = settings
    if settings.demo_enabled:
        app.include_router(demo_router)
        app.include_router(demo_faults_router)
    if inprocess:  # single-process local runs; in Docker Compose the mock systems are their own service
        app.mount("/mock", mock_app)

    @app.get("/health", tags=["health"])
    async def health():
        orch: Orchestrator = app.state.orchestrator
        return {
            "status": "ok",
            "version": __version__,
            "llm_provider": orch.llm.name,
            "llm_model": orch.llm.model,
            "prompt_version": orch.prompt_version,
            "prompt_hash": orch.prompt_hash,
            "tools": orch.registry.names(),
            "policy_passages": len(orch.policies) if orch.policies else 0,
            "stt_provider": app.state.voice.stt.name,
            "tts_provider": app.state.voice.tts.name,
            "tts_media_type": app.state.voice.tts.media_type,
            "mock_systems": "in-process" if inprocess else settings.mock_api_base_url,
            "supervisor_auth": "api_key" if settings.supervisor_api_key else "disabled (development)",
        }

    return app


app = create_app()
