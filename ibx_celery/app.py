from __future__ import annotations

from celery import Celery
from app_settings.settings import get_settings

settings = get_settings()

app = Celery("ibx", include=["ops.tasks.contracts"])
app.conf.update(
    task_default_queue=settings.celery.contracts.queue,
    task_annotations={
        "contracts.backfill_fetch": {"rate_limit": settings.celery.contracts.fetch_rate},
    },
)

# Optional explicit routes if you split queues
app.conf.task_routes = {"contracts.*": {"queue": settings.celery.contracts.queue}}
