from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict

import requests


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_MIN_INTERVAL_SECONDS = 0.11

_RATE_LIMIT_LOCK: asyncio.Lock | None = None
_NEXT_ALLOWED_REQUEST_AT = 0.0


def _get_rate_limit_lock() -> asyncio.Lock:
    global _RATE_LIMIT_LOCK
    if _RATE_LIMIT_LOCK is None:
        _RATE_LIMIT_LOCK = asyncio.Lock()
    return _RATE_LIMIT_LOCK


async def _respect_sec_rate_limit() -> None:
    global _NEXT_ALLOWED_REQUEST_AT
    async with _get_rate_limit_lock():
        now = time.monotonic()
        wait_seconds = _NEXT_ALLOWED_REQUEST_AT - now
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
            now = time.monotonic()
        _NEXT_ALLOWED_REQUEST_AT = max(now, _NEXT_ALLOWED_REQUEST_AT) + SEC_MIN_INTERVAL_SECONDS


def _validated_user_agent() -> str:
    user_agent = str(os.getenv("SEC_USER_AGENT", "")).strip()
    lowered = user_agent.lower()
    if not user_agent or "contact@example.com" in lowered or "example.com" in lowered:
        raise RuntimeError(
            "SEC_USER_AGENT must be set to a descriptive value with real contact information before SEC requests are made."
        )
    return user_agent


def _fixture_root() -> Path | None:
    raw = str(os.getenv("SEC_METRIC_FIXTURE_ROOT", "")).strip()
    if not raw:
        return None
    return Path(raw)


def _load_fixture_json(filename: str) -> Dict[str, Any]:
    root = _fixture_root()
    if root is None:
        raise FileNotFoundError(filename)
    path = root / filename
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Fixture {path} did not contain a JSON object.")
    return data


@dataclass
class SecCompanyFactsClient:
    timeout_s: float = 30.0
    retries: int = 3
    _cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    async def resolve_cik(self, ticker: str) -> str:
        raw_ticker = str(ticker).strip()
        if raw_ticker.isdigit():
            return raw_ticker.zfill(10)

        ticker_map = await self.get_company_tickers()
        target = raw_ticker.upper()
        for record in ticker_map.values():
            if str((record or {}).get("ticker", "")).upper() == target:
                return f"{int(record['cik_str']):010d}"
        raise RuntimeError(f"Ticker not found in SEC company_tickers.json: {ticker}")

    async def get_company_tickers(self) -> Dict[str, Any]:
        return await self._get_json(SEC_TICKER_MAP_URL, fixture_name="company_tickers.json")

    async def get_submissions(self, cik: str) -> Dict[str, Any]:
        cik10 = str(cik).zfill(10)
        return await self._get_json(
            SEC_SUBMISSIONS_URL.format(cik=cik10),
            fixture_name=f"submissions_CIK{cik10}.json",
        )

    async def get_companyfacts(self, cik: str) -> Dict[str, Any]:
        cik10 = str(cik).zfill(10)
        return await self._get_json(
            SEC_COMPANYFACTS_URL.format(cik=cik10),
            fixture_name=f"companyfacts_CIK{cik10}.json",
        )

    async def _get_json(self, url: str, *, fixture_name: str) -> Dict[str, Any]:
        if fixture_name in self._cache:
            return dict(self._cache[fixture_name])

        root = _fixture_root()
        if root is not None and (root / fixture_name).exists():
            payload = _load_fixture_json(fixture_name)
            self._cache[fixture_name] = dict(payload)
            return dict(payload)

        headers = {
            "User-Agent": _validated_user_agent(),
            "Accept-Encoding": "gzip, deflate",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            await _respect_sec_rate_limit()
            try:
                payload = await asyncio.to_thread(
                    self._request_json,
                    url,
                    headers,
                )
                self._cache[fixture_name] = dict(payload)
                return dict(payload)
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    await asyncio.sleep(0.5 * attempt)
        assert last_error is not None
        raise last_error

    def _request_json(self, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        response = requests.get(url, headers=headers, timeout=self.timeout_s)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"SEC response from {url} did not contain a JSON object.")
        return payload
