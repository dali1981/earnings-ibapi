#!/usr/bin/env python3
"""
Centralized Logging Setup Utility

Provides a single function to configure logging across all scripts and modules.
Uses the configuration from config.py to ensure consistent logging behavior.

Usage:
    from common.utils.logging_setup import setup_logging

    # Basic setup
    setup_logging()

    # With custom level
    setup_logging(level="DEBUG")

    # With file logging disabled
    setup_logging(enable_file_logging=False)
"""

from pathlib import Path
from typing import Optional

import logging
import logging.config


def setup_logging(
    level: str = "INFO",
    enable_file_logging: bool = True,
    log_filename: Optional[str] = None,
) -> None:
    """
    Set up centralized logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_file_logging: Whether to enable file logging
        log_filename: Custom log filename (defaults to configured path)
    """

    try:
        # Prefer centralized app settings with colored console formatter
        from app_settings.settings import get_settings

        s = get_settings()
        cfg = s.logging.build_config(s.paths.logs)
        # Update logging level
        cfg["root"]["level"] = level.upper()
        # Handle file logging toggle/filename
        if not enable_file_logging:
            cfg["root"]["handlers"] = [
                h for h in cfg["root"]["handlers"] if h != "file"
            ]
        else:
            if log_filename:
                cfg["handlers"]["file"]["filename"] = str(s.paths.logs / log_filename)
            if "file" not in cfg["root"]["handlers"]:
                cfg["root"]["handlers"].append("file")
        logging.config.dictConfig(cfg)
        logger = logging.getLogger(__name__)
        logger.info(
            f"📝 Logging configured (app_settings): level={level}, file_logging={enable_file_logging}"
        )
    except Exception:
        try:
            # Legacy fallback: use config.py dict
            from config import LOGGING_CONFIG, LOGS_PATH

            config = LOGGING_CONFIG.copy()
            config["root"]["level"] = level.upper()
            if enable_file_logging:
                if log_filename:
                    custom_log_path = LOGS_PATH / log_filename
                    config["handlers"]["file"]["filename"] = str(custom_log_path)
                if "file" not in config["root"]["handlers"]:
                    config["root"]["handlers"].append("file")
            else:
                config["root"]["handlers"] = [
                    h for h in config["root"]["handlers"] if h != "file"
                ]
            logging.config.dictConfig(config)
            logger = logging.getLogger(__name__)
            logger.info(
                f"📝 Logging configured (legacy config.py): level={level}, file_logging={enable_file_logging}"
            )
        except Exception:
            # Minimal fallback
            logging.basicConfig(
                level=getattr(logging, level.upper(), logging.INFO),
                format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            logger = logging.getLogger(__name__)
            logger.warning("⚠️ Using basic logging configuration")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Optional level override for this specific logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logger


def setup_script_logging(
    script_name: str, level: str = "INFO", enable_file_logging: bool = True
) -> logging.Logger:
    """
    Convenience function for script-level logging setup.

    Args:
        script_name: Name of the script (usually Path(__file__).stem)
        level: Logging level
        enable_file_logging: Whether to enable file logging

    Returns:
        Configured logger for the script
    """

    # Set up centralized logging
    log_filename = f"{script_name}.log" if enable_file_logging else None
    setup_logging(
        level=level, enable_file_logging=enable_file_logging, log_filename=log_filename
    )

    # Return script-specific logger
    return get_logger(script_name)


def disable_noisy_loggers():
    """Disable or reduce verbosity of common noisy loggers."""

    # Reduce verbosity of common third-party loggers
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
        "ibapi.client",
        "ibapi.wrapper",
        "pyarrow",
        "pandas",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Auto-setup when module is imported (optional)
def auto_setup():
    """Automatically set up logging when module is imported."""
    try:
        setup_logging()
        disable_noisy_loggers()
    except Exception:
        # Silent fallback - don't break imports
        pass


# Uncomment the next line to enable auto-setup
# auto_setup()
