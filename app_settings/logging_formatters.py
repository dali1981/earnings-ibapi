import logging


RESET = "\x1b[0m"
DIM = "\x1b[2m"

COLORS = {
    "DEBUG": "\x1b[36m",  # cyan
    "INFO": "\x1b[32m",  # green
    "WARNING": "\x1b[33m",  # yellow
    "ERROR": "\x1b[31m",  # red
    "CRITICAL": "\x1b[1;31m",  # bold red
}


class ColorFormatter(logging.Formatter):
    """Colorize levelname and dim the logger name while keeping bracketed format."""

    def format(self, record: logging.LogRecord) -> str:
        # Copy record to avoid mutating for other handlers
        r = logging.makeLogRecord(record.__dict__.copy())
        lvl = r.levelname
        color = COLORS.get(lvl, "")
        if color:
            r.levelname = f"{color}{lvl}{RESET}"
        # Dim the logger name
        if getattr(r, "name", None):
            r.name = f"{DIM}{r.name}{RESET}"
        return super().format(r)
