# One image, two roles: the agent API (default command) and the mock enterprise
# systems (docker-compose overrides the command).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POLICY_DIR=/app/data/policies

WORKDIR /app
RUN useradd --create-home --uid 10001 voiceops

COPY pyproject.toml README.md ./
COPY voiceops ./voiceops
RUN pip install ".[anthropic]"

COPY alembic.ini ./
COPY migrations ./migrations
COPY data ./data
COPY evals ./evals

USER voiceops
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=2)"
CMD ["uvicorn", "voiceops.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
