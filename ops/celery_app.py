from __future__ import annotations

from celery import Celery

import os


def _make_app() -> Celery:
    broker = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    app = Celery("ops", broker=broker, backend=backend, include=["ops.tasks.contracts"])
    # Sensible JSON defaults
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        timezone=os.environ.get("TZ", "UTC"),
        enable_utc=True,
    )
    return app


app = _make_app()
