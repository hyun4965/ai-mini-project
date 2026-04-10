from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """프로젝트 전용 로거를 한 번만 초기화한다."""
    logger = logging.getLogger("tech_strategy")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if getattr(logger, "_tech_strategy_configured", False):
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)
        return

    handler = logging.StreamHandler()
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(numeric_level)
    logger.propagate = False
    logger._tech_strategy_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """프로젝트 네임스페이스 하위 로거를 반환한다."""
    return logging.getLogger(f"tech_strategy.{name}")

