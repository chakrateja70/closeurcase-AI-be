"""Application logging setup.

Called once from the app lifespan. Everything else in the codebase just does
`logging.getLogger(__name__)` and logs normally - no module configures
handlers of its own, so log destination and format stay a single decision made
here.

Uvicorn installs its own handlers on the root logger before our lifespan runs,
so `configure_logging` sets the format on the existing handlers rather than
adding another one - adding one would duplicate every line.
"""

from __future__ import annotations

import logging

from src.config.settings import settings

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Access logs are one line per request on top of our own; keep them, but
    # don't let them set the floor for everything else.
    logging.getLogger("uvicorn.access").setLevel(level)
