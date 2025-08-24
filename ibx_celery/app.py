from __future__ import annotations

from celery import Celery
from ibx_config.settings import get_settings

settings = get_settings()

app = Celery("ibx", include=["ops.tasks.contracts"])
app.conf.update(
    task_default_queue=settings.limits.queue,
    task_annotations={
        "contracts.backfill_fetch": {"rate_limit": settings.limits.fetch_rate},
    },
)

# Optional explicit routes if you split queues
app.conf.task_routes = {
    "contracts.*": {"queue": settings.limits.queue},
}

