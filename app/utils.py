"""
Shared utility functions for the survival analysis project.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the project.

    Idempotent: calling multiple times will not add duplicate handlers.
    """
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        logger.setLevel(level)


@dataclass(frozen=True)
class ProjectConfig:
    """Lightweight runtime configuration container."""

    mimic_demo_dir: str


def load_config(env: Optional[dict] = None) -> ProjectConfig:
    """Load configuration settings from environment variables.

    Prefers environment variable ``MIMIC_DEMO_DIR``; falls back to repo-relative
    ``data/raw/mimic-iv-demo``. This function never returns absolute paths with
    backslashes normalization logic; it simply relays the provided value.

    Args:
        env: Optional mapping for environment lookups (defaults to ``os.environ``).

    Returns:
        ProjectConfig: Resolved configuration.
    """
    env = env or os.environ
    mimic_demo_dir = env.get("MIMIC_DEMO_DIR", "data/raw/mimic-iv-demo")
    return ProjectConfig(mimic_demo_dir=mimic_demo_dir)


