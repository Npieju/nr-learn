from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from racing_ml.common.progress import ProgressBar
from racing_ml.data.netkeiba_crawler import (
    RequestSettings,
    _build_session,
    _load_or_fetch_html,
    _normalize_text,
    _resolve_path,
)


RACE_ID_PARAM_PATTERN = re.compile(r"[?&]race_id=(?P<race_id>\d{12})")
RACE_HEADLINE_PATTERN = re.compile(r"^(?P<race_no>\d{1,2})R\b")
RACE_LIST_URL_TEMPLATE = "https://race.sp.netkeiba.com/?pid=race_list&kaisai_date={kaisai_date}"


def _resolve_crawl_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("crawl")
    if isinstance(nested, dict):
        return nested
    return config


def _parse_date(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value}")
    return parsed.normalize()


def _iter_target_dates(start_date: str, end_date: str, date_order: str) -> list[pd.Timestamp]:
    start_ts = _parse_date(start_date)
    end_ts = _parse_date(end_date)
    if end_ts < start_ts:
        raise ValueError(f"end_date must be >= start_date: {start_date} .. {end_date}")

    dates: list[pd.Timestamp] = []
    current = start_ts
    while current <= end_ts:
        dates.append(current)
        current = current + timedelta(days=1)

    if str(date_order).strip().lower() == "desc":
        dates.reverse()
    return dates


def _build_request_settings(crawl_cfg: dict[str, Any], *, refresh: bool, parse_only: bool) -> RequestSettings:
    return RequestSettings(
        base_url="https://race.sp.netkeiba.com",
        user_agent=str(crawl_cfg.get("user_agent", "nr-learn-netkeiba-crawler/0.1")),
        timeout_sec=float(crawl_cfg.get("timeout_sec", 20)),
        delay_sec=float(crawl_cfg.get("delay_sec", 1.5)),
        retry_count=int(crawl_cfg.get("retry_count", 3)),
        retry_backoff_sec=float(crawl_cfg.get("retry_backoff_sec", 2.0)),
        overwrite=bool(refresh),
        parse_only=bool(parse_only),
    )


def _compact_date(value: str) -> str:
    return value.replace("-", "").strip()


def _select_race_list_scopes(soup: BeautifulSoup, *, target_date: str) -> list[Any]:
    compact_target_date = _compact_date(target_date)
    matching_scopes: list[Any] = []
    for day_wrap in soup.find_all("div", class_=re.compile(r"\bRaceListDayWrap\b")):
        if day_wrap.find(attrs={"data-kaisaidate": compact_target_date}) is None:
            continue
        matching_scopes.append(day_wrap)
    return matching_scopes or [soup]


def parse_netkeiba_race_list_html(
    html: str,
    *,
    target_date: str,
    source_page_url: str,
) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []
    seen_race_ids: set[str] = set()
    scopes = _select_race_list_scopes(soup, target_date=target_date)

    for scope in scopes:
        for race_box in scope.find_all("div", class_=re.compile(r"\bRaceList_Main_Box\b")):
            anchor = race_box.find("a", href=True)
            if anchor is None:
                continue

            href = str(anchor.get("href", "")).strip()
            if not href:
                continue

            headline = _normalize_text(anchor.get_text(" ", strip=True))
            race_headline_match = RACE_HEADLINE_PATTERN.match(headline)
            if race_headline_match is None:
                continue

            race_id_match = RACE_ID_PARAM_PATTERN.search(href)
            if race_id_match is None:
                continue

            race_id = race_id_match.group("race_id")
            if race_id in seen_race_ids:
                continue
            seen_race_ids.add(race_id)

            records.append(
                {
                    "date": target_date,
                    "race_id": race_id,
                    "race_no": int(race_headline_match.group("race_no")),
                    "headline": headline,
                    "source_page_url": source_page_url,
                    "source": "netkeiba_race_list_sp",
                }
            )

    return pd.DataFrame(records, columns=["date", "race_id", "race_no", "headline", "source_page_url", "source"])


def discover_netkeiba_race_ids_from_race_list(
    crawl_config: dict[str, Any],
    *,
    base_dir: Path,
    start_date: str,
    end_date: str | None = None,
    limit: int | None = None,
    date_order: str = "asc",
    exclude_race_ids: set[str] | None = None,
    refresh: bool = False,
    parse_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    crawl_cfg = _resolve_crawl_config(crawl_config)
    normalized_end_date = end_date or start_date
    target_dates = _iter_target_dates(start_date, normalized_end_date, date_order)
    settings = _build_request_settings(crawl_cfg, refresh=refresh, parse_only=parse_only)
    raw_html_dir = _resolve_path(crawl_cfg.get("raw_html_dir", "data/external/netkeiba/raw_html"), base_dir) / "race_list"

    session = _build_session(settings)
    progress = ProgressBar(total=max(len(target_dates), 1), prefix="[prepare:race_list]", min_interval_sec=0.0)
    progress.start("starting")

    last_fetch_at: float | None = None
    pages_fetched = 0
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    seen_race_ids: set[str] = set()
    excluded_race_ids = set(exclude_race_ids or set())
    raw_discovered_count = 0

    for index, target_date_ts in enumerate(target_dates, start=1):
        compact_date = target_date_ts.strftime("%Y%m%d")
        target_date = target_date_ts.strftime("%Y-%m-%d")
        source_page_url = RACE_LIST_URL_TEMPLATE.format(kaisai_date=compact_date)
        cache_path = raw_html_dir / f"{compact_date}.html"

        try:
            html, fetched, last_fetch_at = _load_or_fetch_html(
                session=session,
                url=source_page_url,
                output_path=cache_path,
                settings=settings,
                last_fetch_at=last_fetch_at,
            )
            if fetched:
                pages_fetched += 1

            day_frame = parse_netkeiba_race_list_html(
                html,
                target_date=target_date,
                source_page_url=source_page_url,
            )
            raw_discovered_count += int(len(day_frame))

            if not day_frame.empty:
                for row in day_frame.to_dict("records"):
                    race_id = str(row.get("race_id", "")).strip()
                    if not race_id or race_id in excluded_race_ids or race_id in seen_race_ids:
                        continue
                    seen_race_ids.add(race_id)
                    records.append(row)
                    if limit is not None and limit > 0 and len(records) >= int(limit):
                        break
        except Exception as error:
            failures.append({"date": target_date, "error": str(error)})

        progress.update(current=index, message=f"dates={index}/{len(target_dates)} ids={len(records)} failures={len(failures)}")
        if limit is not None and limit > 0 and len(records) >= int(limit):
            break

    progress.complete(message=f"done ids={len(records)} failures={len(failures)}")
    race_frame = pd.DataFrame(records, columns=["date", "race_id", "race_no", "headline", "source_page_url", "source"])
    report = {
        "source": "race_list",
        "date_window": {"start": start_date, "end": normalized_end_date},
        "date_order": str(date_order),
        "requested_dates": int(len(target_dates)),
        "pages_fetched": int(pages_fetched),
        "raw_discovered_count": int(raw_discovered_count),
        "row_count": int(len(race_frame)),
        "failure_count": int(len(failures)),
        "failures": failures[:50],
        "raw_html_dir": str(raw_html_dir),
    }
    return race_frame, report