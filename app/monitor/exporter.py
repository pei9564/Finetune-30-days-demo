"""Prometheus metrics exporter for task and system monitoring."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client import multiprocess
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
import psutil
from redis import Redis

from app.monitor.system_metrics import SystemMetricsMonitor


def _purge_multiprocess_dir() -> None:
    """Remove any leftover metric fragments when the app boots."""

    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not directory or not os.path.isdir(directory):
        return

    for filename in os.listdir(directory):
        if filename.endswith(".db"):
            try:
                os.remove(os.path.join(directory, filename))
            except OSError:
                pass


# Use a dedicated registry and enable multiprocess mode when configured.
REGISTRY = CollectorRegistry()

if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
    if os.environ.get("PROMETHEUS_MULTIPROC_CLEANED") != "1":
        _purge_multiprocess_dir()
        os.environ["PROMETHEUS_MULTIPROC_CLEANED"] = "1"
    multiprocess.MultiProcessCollector(REGISTRY)

# Task processing metrics
TASK_DURATION_SECONDS = Histogram(
    "task_duration_seconds",
    "Distribution of task execution time in seconds.",
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, float("inf")),
    registry=REGISTRY,
)

# System resource metrics
SYSTEM_CPU_PERCENT = Gauge(
    "system_cpu_percent",
    "System-wide CPU utilisation percentage.",
    registry=REGISTRY,
)
SYSTEM_MEMORY_GB = Gauge(
    "system_memory_usage_gigabytes",
    "Resident memory usage of the API process in gigabytes.",
    registry=REGISTRY,
)

_system_monitor: Optional[SystemMetricsMonitor] = None
_redis_client: Optional[Redis] = None
router = APIRouter()

_SUCCESS_KEY = os.environ.get("TASK_SUCCESS_KEY", "metrics:task_success_total")
_FAILURE_KEY = os.environ.get("TASK_FAILURE_KEY", "metrics:task_failure_total")


def get_monitor() -> SystemMetricsMonitor:
    """Initialise or return the shared system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMetricsMonitor()
    return _system_monitor


def update_system_metrics() -> None:
    """Refresh CPU and memory gauges using the system monitor."""
    metrics = get_monitor().get_system_metrics()
    SYSTEM_CPU_PERCENT.set(metrics.get("cpu_percent", 0.0))
    SYSTEM_MEMORY_GB.set(metrics.get("memory_gb", 0.0))


def get_queue_length() -> int:
    """Fetch the current queue length directly from Redis."""

    client = _redis_client or _get_redis_client()
    if not client:
        return 0

    queue_name = os.environ.get("CELERY_TASK_QUEUE", "celery")
    try:
        return int(client.llen(queue_name))
    except Exception:
        return 0


def _increment_counter(key: str) -> None:
    client = _redis_client or _get_redis_client()
    if not client:
        return
    try:
        client.incr(key)
    except Exception:
        pass


def _get_counter_value(key: str) -> int:
    client = _redis_client or _get_redis_client()
    if not client:
        return 0
    try:
        value = client.get(key)
        return int(value) if value is not None else 0
    except Exception:
        return 0


def _get_redis_client() -> Optional[Redis]:
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    if not broker_url.startswith("redis"):
        return None

    from urllib.parse import urlparse

    parsed = urlparse(broker_url)
    _redis_client = Redis(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        password=parsed.password,
        db=int(parsed.path.lstrip("/") or 0),
        ssl=parsed.scheme == "rediss",
    )
    return _redis_client


class TaskMetricsCollector:
    """Custom collector that emits queue length and task counters."""

    @staticmethod
    def collect():
        yield GaugeMetricFamily(
            "task_queue_length",
            "Current number of pending tasks in the queue.",
            value=get_queue_length(),
        )
        yield CounterMetricFamily(
            "task_success_total",
            "Total number of successfully completed tasks.",
            value=_get_counter_value(_SUCCESS_KEY),
        )
        yield CounterMetricFamily(
            "task_failure_total",
            "Total number of tasks that ended in failure.",
            value=_get_counter_value(_FAILURE_KEY),
        )


REGISTRY.register(TaskMetricsCollector())


def cleanup_stale_multiprocess_files() -> None:
    """Remove metric fragments left behind by dead processes."""

    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not directory or not os.path.isdir(directory):
        return

    for filename in os.listdir(directory):
        if not filename.endswith(".db"):
            continue

        try:
            pid = int(filename.split(".")[0])
        except ValueError:
            continue

        if psutil.pid_exists(pid):
            continue

        try:
            multiprocess.mark_process_dead(pid)
        except Exception:
            try:
                os.remove(os.path.join(directory, filename))
            except OSError:
                pass


def record_task_success(duration_seconds: Optional[float] = None) -> None:
    """Increment success counter and optionally record the duration."""
    _increment_counter(_SUCCESS_KEY)
    if duration_seconds is not None:
        TASK_DURATION_SECONDS.observe(max(duration_seconds, 0.0))


def record_task_failure(duration_seconds: Optional[float] = None) -> None:
    """Increment failure counter and optionally record the duration."""
    _increment_counter(_FAILURE_KEY)
    if duration_seconds is not None:
        TASK_DURATION_SECONDS.observe(max(duration_seconds, 0.0))


def observe_task_duration(duration_seconds: float) -> None:
    """Manually record a task duration without affecting counters."""
    TASK_DURATION_SECONDS.observe(max(duration_seconds, 0.0))


@router.get("/metrics")
def metrics_endpoint() -> Response:
    """Expose Prometheus-formatted metrics."""
    update_system_metrics()
    cleanup_stale_multiprocess_files()
    payload = generate_latest(REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
