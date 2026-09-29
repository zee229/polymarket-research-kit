"""Gamma API: events (with their markets) by id, slug, series, tag and end-date range."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import httpx

from pmrk.polymarket.http import Client

GAMMA = "https://gamma-api.polymarket.com"
PAGE = 100

_client = Client("gamma", min_interval_s=0.35)


def get_event(key: str) -> dict[str, Any]:
    """One event with its markets, by numeric id or slug."""
    path = f"/events/{key}" if key.isdigit() else f"/events/slug/{key}"
    return _client.get_json(GAMMA + path)


def list_series() -> list[dict[str, Any]]:
    """All recurring series (e.g. `nyc-daily-weather`)."""
    out: list[dict[str, Any]] = []
    while page := _client.get_json(f"{GAMMA}/series", {"limit": 500, "offset": len(out)}):
        out.extend(page)
    return out


def series_events(series_id: str) -> list[dict[str, Any]]:
    """All events of a series. Full pages are cached; the trailing partial page is always re-fetched."""
    events: list[dict[str, Any]] = []
    while True:
        params = {"series_id": series_id, "limit": PAGE, "offset": len(events)}
        page = _client.get_json(f"{GAMMA}/events", params, cache=True)
        if len(page) < PAGE:
            page = _client.get_json(f"{GAMMA}/events", params, cache=True, refresh=True)
        events.extend(page)
        if len(page) < PAGE:
            return events


def _keyset_page(params: dict[str, Any], cache: bool) -> dict[str, Any]:
    """Keyset pages sometimes fail with 403 under load; back off and retry with the same cursor."""
    for attempt in range(8):
        try:
            return _client.get_json(f"{GAMMA}/events/keyset", params, cache=cache)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 403 or attempt == 7:
                raise
            time.sleep(min(300, 15 * 2**attempt))
    raise RuntimeError("unreachable")


def iter_events(
    start: date, end: date, *, tag_slug: str | None = None, closed: bool | None = True, window_days: int = 31
) -> Iterator[dict[str, Any]]:
    """Events (with markets) whose end date falls in [start, end), crawled in end-date windows.

    A single keyset cursor over a long range can hit a persistent 403, so the range is split into windows.
    Pages of fully past windows are cached, which makes a long crawl resumable.
    """
    lo = start
    while lo < end:
        hi = min(lo + timedelta(days=window_days), end)
        cache = bool(closed) and hi < date.today()
        cursor = None
        while True:
            params: dict[str, Any] = {"end_date_min": lo.isoformat(), "end_date_max": hi.isoformat(), "limit": PAGE}
            if closed is not None:
                params["closed"] = str(closed).lower()
            if tag_slug:
                params["tag_slug"] = tag_slug
            if cursor:
                params["after_cursor"] = cursor
            page = _keyset_page(params, cache)
            items = page.get("events") or []
            yield from items
            cursor = page.get("next_cursor")
            if not items or not cursor:
                break
        lo = hi
