"""No-op sitecustomize.

Python auto-imports `sitecustomize` if present on sys.path. To avoid doing real
work at import time (especially in unit tests), this module intentionally does
nothing. Each entrypoint (CLI, jobs, services) is responsible for configuring
logging explicitly via `common.utils.logging_setup.setup_logging()`.
"""

# Intentionally empty to prevent side effects during interpreter startup.
