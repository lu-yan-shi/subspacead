"""Memory log handler for /mse/logs endpoint."""

import logging
from collections import deque
from datetime import datetime, timezone

_log_buffer: deque = deque(maxlen=2000)


class MemoryLogHandler(logging.Handler):
    """Captures log records to an in-memory buffer for MeSquare log retrieval."""

    def emit(self, record):
        try:
            msg = self.format(record)
            _log_buffer.append({
                "level": record.levelname,
                "message": msg,
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "logger": record.name,
            })
        except Exception:
            self.handleError(record)


def init_log_capture():
    handler = MemoryLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger("uvicorn").addHandler(handler)


def shutdown_log_capture():
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, MemoryLogHandler)]


def get_recent_logs(limit: int = 500):
    return list(_log_buffer)[-limit:]
