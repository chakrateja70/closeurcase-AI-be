from __future__ import annotations

import logging
import time
from collections import OrderedDict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

# Bounded so unique-IP floods can't exhaust memory; oldest entries are dropped.
MAX_TRACKED_CLIENTS = 10_000


def client_ip(request: Request) -> str:
    """The caller's address."""
    return request.client.host if request.client else "unknown"


class RequestCounter:
    """IP -> how many requests it has made since this process started."""

    def __init__(self, max_clients: int = MAX_TRACKED_CLIENTS):
        self._counts: OrderedDict[str, int] = OrderedDict()
        self._max_clients = max_clients

    def record(self, ip: str) -> int:
        count = self._counts.pop(ip, 0) + 1
        self._counts[ip] = count  # re-inserted last => most recently seen
        if len(self._counts) > self._max_clients:
            self._counts.popitem(last=False)
        return count

    def get(self, ip: str) -> int:
        return self._counts.get(ip, 0)

    def snapshot(self) -> dict[str, int]:
        """Every tracked client, busiest first."""
        return dict(
            sorted(self._counts.items(), key=lambda kv: kv[1], reverse=True)
        )


counter = RequestCounter()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Logs one line per request: who, what, the outcome, and how long.

    Runs for every request including those slowapi rejects, so a client
    hammering the API still shows up in the log with a rising count.
    """

    async def dispatch(self, request: Request, call_next):
        ip = client_ip(request)
        total = counter.record(ip)
        started = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s from %s -> %s in %.0fms (request #%d from this IP)",
            request.method,
            request.url.path,
            ip,
            response.status_code,
            elapsed_ms,
            total,
        )
        return response
