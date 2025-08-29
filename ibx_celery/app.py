from __future__ import annotations

from app_settings.settings import get_settings
from celery import Celery


settings = get_settings()

app = Celery(
    "ibx",
    include=[
        "ops.tasks.contracts",
        "ops.tasks.search",
        "ops.tasks.contract_details",
        "ops.tasks.historical_data",
        "ops.tasks.option_chains",
    ],
)
app.conf.update(
    task_default_queue=settings.celery.contracts.queue,
    task_annotations={
        "contracts.backfill_fetch": {
            "rate_limit": settings.celery.contracts.fetch_rate
        },
        "historical.fetch_bars_from_ib": {
            "rate_limit": settings.celery.historical.fetch_rate
        },
    },
    worker_hijack_root_logger=False,
    worker_redirect_stdouts=False,
)
# Broker/backend from centralized settings
if settings.celery.broker_url:
    app.conf.broker_url = settings.celery.broker_url
if settings.celery.result_backend:
    app.conf.result_backend = settings.celery.result_backend

# Task routing to appropriate queues
app.conf.task_routes = {
    "contracts.*": {"queue": settings.celery.contracts.queue},
    "historical.*": {"queue": settings.celery.historical.queue},
    "search.*": {"queue": settings.celery.search.queue},
}
