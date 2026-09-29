"""Rate-limited, retrying HTTP GET with an optional on-disk response cache.

A cached response is keyed by (url, sorted params) and never re-fetched unless `refresh=True`, which makes
crawls idempotent and resumable. Large immutable payloads (price history, trades) are fetched uncached and
persisted by the caller as parquet instead.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from pmrk.config import USER_AGENT, data_dir

RETRY_STATUS = (429, 500, 502, 503, 504)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRY_STATUS
    return isinstance(exc, httpx.TransportError)


class Client:
    """HTTP client with a minimum interval between live requests (thread-safe) and exponential backoff."""

    def __init__(self, namespace: str, min_interval_s: float = 0.2, timeout_s: float = 60.0):
        self.namespace = namespace
        self.min_interval_s = min_interval_s
        self._last = 0.0
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=timeout_s, headers={"User-Agent": USER_AGENT}, follow_redirects=True)

    def _cache_path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "?" + json.dumps(sorted((params or {}).items()), default=str)
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return data_dir() / "cache" / self.namespace / digest[:2] / f"{digest}.json"

    def get_json(
        self, url: str, params: dict[str, Any] | None = None, *, cache: bool = False, refresh: bool = False
    ) -> Any:
        if not cache:
            return json.loads(self._fetch(url, params))
        path = self._cache_path(url, params)
        if path.exists() and not refresh:
            return json.loads(path.read_text())
        text = self._fetch(url, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)
        return json.loads(text)

    def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        """Uncached, rate-limited, retried GET returning the body as text."""
        return self._fetch(url, params)

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, max=120),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    def _fetch(self, url: str, params: dict[str, Any] | None) -> str:
        with self._lock:
            wait = self.min_interval_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
        resp = self._http.get(url, params=params)
        resp.raise_for_status()
        return resp.text
