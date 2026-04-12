"""client.py — HTTP/WS client for AP Workflow OpenEnv."""
from __future__ import annotations

try:
    from openenv.core.env_client import EnvClient
    from .models import APAction, APObservation

    class APEnv(EnvClient[APAction, APObservation]):
        action_type = APAction
        observation_type = APObservation

except ImportError:
    # Fallback stub if openenv-core not installed
    class APEnv:  # type: ignore
        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url
