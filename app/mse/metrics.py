"""Metrics collection: MetricsCollector + EndpointMetricsTracker + CpuSpikeMonitor."""

import threading
import time
from collections import deque
from datetime import datetime, timezone

import psutil

from ..config import CPU_SPIKE_THRESHOLD


class MetricsCollector:
    """Lightweight request metrics collector for MeSquare monitoring."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_requests = 0
        self._active_connections = 0
        self._errors_24h = deque(maxlen=100000)
        self._requests_timestamps = deque(maxlen=100000)
        self._latencies = deque(maxlen=2000)

    def record_request(self, latency_s: float, status_code: int):
        now = time.time()
        is_error = status_code >= 400
        with self._lock:
            self._total_requests += 1
            self._requests_timestamps.append(now)
            if is_error:
                self._errors_24h.append(now)
            self._latencies.append(latency_s * 1000)

    def inc_active(self):
        with self._lock:
            self._active_connections += 1

    def dec_active(self):
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    def get_metrics(self) -> dict:
        now = time.time()
        cutoff_1m = now - 60
        cutoff_1h = now - 3600
        cutoff_24h = now - 86400

        with self._lock:
            total = self._total_requests
            active = self._active_connections
            latencies = list(self._latencies)

            last_min = sum(1 for t in self._requests_timestamps if t > cutoff_1m)
            last_hour = sum(1 for t in self._requests_timestamps if t > cutoff_1h)
            last_24h = sum(1 for t in self._requests_timestamps if t > cutoff_24h)
            errors_24h = sum(1 for t in self._errors_24h if t > cutoff_24h)

        error_rate = errors_24h / last_24h if last_24h > 0 else 0.0

        avg_lat = p95 = p99 = 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            avg_lat = sum(sorted_lat) / n
            p95 = sorted_lat[min(int(n * 0.95), n - 1)]
            p99 = sorted_lat[min(int(n * 0.99), n - 1)]

        return {
            "total_requests": total,
            "requests_per_minute": float(last_min),
            "requests_last_hour": last_hour,
            "requests_last_24h": last_24h,
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "error_rate_24h": round(error_rate, 4),
            "active_connections": active,
        }


class EndpointMetricsTracker:
    """Tracks per-endpoint request metrics for MeSquare."""

    def __init__(self):
        self._lock = threading.Lock()
        self._endpoints: dict[tuple[str, str], dict] = {}

    def record(self, path: str, method: str, latency_ms: float, is_error: bool):
        now = time.time()
        key = (path, method)
        with self._lock:
            if key not in self._endpoints:
                self._endpoints[key] = {
                    "count": 0, "errors": 0,
                    "latency_sum": 0.0, "timestamps": deque(),
                }
            ep = self._endpoints[key]
            ep["count"] += 1
            ep["latency_sum"] += latency_ms
            if is_error:
                ep["errors"] += 1
            ep["timestamps"].append(now)

    def get_metrics(self) -> list[dict]:
        now = time.time()
        cutoff = now - 60.0
        result = []
        with self._lock:
            for (path, method), ep in self._endpoints.items():
                ts = ep["timestamps"]
                while ts and ts[0] <= cutoff:
                    ts.popleft()
                total = ep["count"]
                rpm = len(ts)
                avg_lat = ep["latency_sum"] / total if total > 0 else 0.0
                error_rate = ep["errors"] / total if total > 0 else 0.0
                result.append({
                    "path": path,
                    "method": method,
                    "total_requests": total,
                    "requests_per_minute": round(float(rpm), 1),
                    "avg_latency_ms": round(avg_lat, 2),
                    "error_rate": round(error_rate, 4),
                })
        result.sort(key=lambda x: x["total_requests"], reverse=True)
        return result


class CpuSpikeMonitor:
    """Background CPU sampler that detects and records spike events."""

    def __init__(self, threshold: float = None, sample_interval: float = 2.0):
        self._threshold = threshold or CPU_SPIKE_THRESHOLD
        self._interval = sample_interval
        self._cpu_count = psutil.cpu_count() or 1
        self._process = psutil.Process()
        self._samples: deque = deque(maxlen=30)
        self._spikes: deque = deque(maxlen=50)
        self._in_spike = False
        self._spike_start: float = 0
        self._spike_peak: float = 0
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        self._running = True
        self._process.cpu_percent()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            time.sleep(self._interval)
            try:
                raw = self._process.cpu_percent()
                normalized = raw / self._cpu_count
                self._samples.append(normalized)
                self._check_spike(normalized)
            except Exception:
                pass

    def _check_spike(self, current: float):
        if current > self._threshold:
            if not self._in_spike:
                self._in_spike = True
                self._spike_start = time.time()
                self._spike_peak = current
            else:
                self._spike_peak = max(self._spike_peak, current)
        else:
            if self._in_spike:
                duration = time.time() - self._spike_start
                if duration >= 4.0:
                    self._spikes.append({
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "peak_percent": round(self._spike_peak, 1),
                        "duration_seconds": round(duration, 1),
                    })
                self._in_spike = False

    def get_spikes(self) -> list:
        return list(self._spikes)

    def get_current_cpu(self) -> float:
        return self._samples[-1] if self._samples else 0.0
