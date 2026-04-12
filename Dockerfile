FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps - pinned, no optional extras that can fail
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn==0.34.0 \
    pydantic==2.10.4 \
    openai==1.59.9 \
    requests==2.32.3 \
    websockets==14.1 \
    "openenv-core>=0.2.0" \
    "openenv>=0.2.0" || pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn==0.34.0 \
    pydantic==2.10.4 \
    openai==1.59.9 \
    requests==2.32.3 \
    websockets==14.1

# Copy all source files
COPY . .

# Set PYTHONPATH so all imports resolve from /app
ENV PYTHONPATH="/app"
ENV PORT=7860
ENV AP_ENV_SEED=42
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:7860/health || exit 1

# Run directly — no shell expansion issues, no optional installs
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
