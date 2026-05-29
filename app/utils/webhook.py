"""MeSquare webhook notifier — pushes endpoint changes to MeSquare."""

import logging
import os
import socket
from datetime import datetime, timezone

from ..config import MESQUARE_BASE_URL, SERVICE_NAME, SERVICE_PORT

try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore
    _HAS_HTTPX = False


def _get_local_ip() -> str:
    """Get LAN IP via UDP socket — avoids gethostname() DNS issues on Windows/Docker."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


SERVER_HOST = os.environ.get("SERVER_HOST", _get_local_ip())


async def notify_mesquare_api_change(event: str = "endpoints_changed"):
    """Notify MeSquare that this service's endpoints have changed."""
    if not _HAS_HTTPX:
        logging.getLogger(__name__).warning(
            "httpx not installed — cannot notify MeSquare of API changes."
        )
        return

    url = f"{MESQUARE_BASE_URL}/api/mse/api-change"
    payload = {
        "base_url": f"http://{SERVER_HOST}:{SERVICE_PORT}",
        "service_name": SERVICE_NAME,
        "event": event,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            logging.getLogger(__name__).info(
                "Notified MeSquare: %s -> %s", event, resp.status_code
            )
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to notify MeSquare at %s — is MeSquare running?", url
        )
