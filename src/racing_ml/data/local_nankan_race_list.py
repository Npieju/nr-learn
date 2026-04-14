from __future__ import annotations

from pathlib import Path
import re
from typing import Any, cast

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

from racing_ml.common.progress import ProgressBar
from racing_ml.data.local_nankan_collect import RequestSettings, _build_session, _load_or_fetch_html, _resolve_crawl_config, _resolve_path


CALENDAR_URL_TEMPLATE = "https://www.nankankeiba.com/calendar/{calendar_key}.do"
PROGRAM_URL_TEMPLATE = "https://www.nankankeiba.com/program/{meeting_id}.do"
PROGRAM_LINK_PATTERN = re.compile(r"/program/(?P<meeting_id>\d{14})\.do")
RESULT_LINK_PATTERN = re.compile(r"/result/(?P<race_id>\d{16})\.do")
RACE_CARD_LINK_PATTERN = re.compile(r"/syousai/(?P<race_id>\d{16})\.do")
POST_TIME_PATTERN = re.compile(r"(?P<time>\d{1,2}:\d{2})")


def _parse_date(value: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value}")
    return parsed.normalize()


def _quarter_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    month = ((int(timestamp.month) - 1) // 3) * 3 + 1
    return pd.Timestamp(year=int(timestamp.year), month=month, day=1)


def _iter_calendar_keys(start_date: str, end_date: str, date_order: str) -> list[str]:
    start_ts = _parse_date(start_date)
    end_ts = _parse_date(end_date)
    if end_ts < start_ts:
        raise ValueError(f"end_date must be >= start_date: {start_date} .. {end_date}")

    keys: list[str] = []
    current = _quarter_start(start_ts)
    final = _quarter_start(end_ts)
    while current <= final:
        keys.append(current.strftime("%Y%m"))
        current = current + pd.DateOffset(months=3)

    if str(date_order).strip().lower() == "desc":
        keys.reverse()
    return keys


def _build_request_settings(crawl_cfg: dict[str, Any]) -> RequestSettings:
    return RequestSettings(
        base_url=str(crawl_cfg.get("base_url", "https://www.nankankeiba.com")).rstrip("/"),
        user_agent=str(crawl_cfg.get("user_agent", "nr-learn-local-nankan-crawler/0.1")),
        timeout_sec=float(crawl_cfg.get("timeout_sec", 20.0)),
        delay_sec=float(crawl_cfg.get("delay_sec", 1.5)),
        retry_count=max(int(crawl_cfg.get("retry_count", 3)), 1),
        retry_backoff_sec=float(crawl_cfg.get("retry_backoff_sec", 2.0)),
        overwrite=bool(crawl_cfg.get("overwrite", False)),
    )


def _build_scheduled_post_at(date_text: str, post_time: str | None) -> str | None:
    if not post_time:
        return None
    try:
        return pd.Timestamp(f"{date_text} {post_time}", tz="Asia/Tokyo").isoformat()
    except Exception:
        return None


def _extract_program_post_time(anchor: Tag) -> str | None:
    list_item = anchor.find_parent("li", class_=re.compile(r"nk23_c-block01__list__item"))
    if not isinstance(list_item, Tag):
        return None
    texts = list_item.find(class_=re.compile(r"nk23_c-block01__texts"))
    if not isinstance(texts, Tag):
        return None
    for text_node in texts.find_all(class_=re.compile(r"nk23_c-block01__text")):
        if not isinstance(text_node, Tag):
            continue
        text = str(text_node.get_text(" ", strip=True)).strip()
        match = POST_TIME_PATTERN.search(text)
        if match is not None:
            return match.group("time")
    match = POST_TIME_PATTERN.search(str(texts.get_text(" ", strip=True)).strip())
    if match is not None:
        return match.group("time")
    return None


def _normalize_as_of_timestamp(as_of: str | None) -> pd.Timestamp | None:
    if not as_of:
        return None
    parsed = pd.Timestamp(as_of)
    if pd.isna(parsed):
        return None
    if parsed.tzinfo is None:
        return parsed.tz_localize("Asia/Tokyo")
    return parsed.tz_convert("Asia/Tokyo")


def _resolve_as_of_timestamp(as_of: str | None) -> pd.Timestamp:
    normalized = _normalize_as_of_timestamp(as_of)
    if normalized is not None:
        return normalized
    return pd.Timestamp.now(tz="Asia/Tokyo")


def _filter_upcoming_races(frame: pd.DataFrame, *, as_of: str | None) -> pd.DataFrame:
    if frame.empty or "scheduled_post_at" not in frame.columns:
        return frame
    as_of_ts = _resolve_as_of_timestamp(as_of)
    scheduled = pd.to_datetime(frame["scheduled_post_at"], errors="coerce")
    return frame.loc[scheduled.notna() & (scheduled > as_of_ts)].reset_index(drop=True)


def parse_local_nankan_calendar_html(
    html: str,
    *,
    start_date: str,
    end_date: str,
    source_page_url: str,
) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    start_ts = _parse_date(start_date)
    end_ts = _parse_date(end_date)
    records: list[dict[str, Any]] = []
    seen_meeting_ids: set[str] = set()

    for anchor in soup.find_all("a", href=PROGRAM_LINK_PATTERN):
        if not isinstance(anchor, Tag):
            continue
        href = str(anchor.get("href", "")).strip()
        match = PROGRAM_LINK_PATTERN.search(href)
        if match is None:
            continue
        meeting_id = match.group("meeting_id")
        if meeting_id == "00000000000000" or meeting_id in seen_meeting_ids:
            continue
        date_text = meeting_id[:8]
        try:
            target_date = pd.Timestamp(date_text).normalize()
        except Exception:
            continue
        if target_date < start_ts or target_date > end_ts:
            continue
        seen_meeting_ids.add(meeting_id)
        records.append(
            {
                "date": target_date.strftime("%Y-%m-%d"),
                "meeting_id": meeting_id,
                "source_page_url": source_page_url,
                "source": "local_nankan_calendar",
            }
        )

    return pd.DataFrame(records, columns=["date", "meeting_id", "source_page_url", "source"])


def parse_local_nankan_program_html(
    html: str,
    *,
    meeting_id: str,
    source_page_url: str,
    require_result_link: bool = False,
) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, Any]] = []
    seen_race_ids: set[str] = set()
    date_text = f"{meeting_id[:4]}-{meeting_id[4:6]}-{meeting_id[6:8]}"

    def _matching_anchors(pattern: re.Pattern[str]) -> list[Tag]:
        anchors: list[Tag] = []
        for anchor in soup.find_all("a", href=pattern):
            if not isinstance(anchor, Tag):
                continue
            href = str(anchor.get("href", "")).strip()
            match = pattern.search(href)
            if match is None:
                continue
            race_id = match.group("race_id")
            if not race_id.startswith(meeting_id):
                continue
            anchors.append(anchor)
        return anchors

    result_links = _matching_anchors(RESULT_LINK_PATTERN)
    if result_links:
        candidate_anchors = result_links
        candidate_pattern = RESULT_LINK_PATTERN
    elif require_result_link:
        return pd.DataFrame(columns=["date", "meeting_id", "race_id", "race_no", "post_time", "scheduled_post_at", "source_page_url", "source"])
    else:
        candidate_anchors = _matching_anchors(RACE_CARD_LINK_PATTERN)
        candidate_pattern = RACE_CARD_LINK_PATTERN

    for anchor in candidate_anchors:
        href = str(anchor.get("href", "")).strip()
        match = candidate_pattern.search(href)
        if match is None:
            continue
        race_id = match.group("race_id")
        if race_id in seen_race_ids:
            continue
        seen_race_ids.add(race_id)
        post_time = _extract_program_post_time(anchor)
        records.append(
            {
                "date": date_text,
                "meeting_id": meeting_id,
                "race_id": race_id,
                "race_no": int(race_id[-2:]),
                "post_time": post_time,
                "scheduled_post_at": _build_scheduled_post_at(date_text, post_time),
                "source_page_url": source_page_url,
                "source": "local_nankan_program",
            }
        )

    return pd.DataFrame(records, columns=["date", "meeting_id", "race_id", "race_no", "post_time", "scheduled_post_at", "source_page_url", "source"])


def discover_local_nankan_race_ids_from_calendar(
    crawl_config: dict[str, Any],
    *,
    base_dir: Path,
    start_date: str,
    end_date: str | None = None,
    limit: int | None = None,
    date_order: str = "asc",
    exclude_race_ids: set[str] | None = None,
    require_result_link: bool = False,
    upcoming_only: bool = False,
    as_of: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    crawl_cfg = _resolve_crawl_config(crawl_config)
    normalized_end_date = end_date or start_date
    calendar_keys = _iter_calendar_keys(start_date, normalized_end_date, date_order)
    settings = _build_request_settings(crawl_cfg)
    raw_html_dir = _resolve_path(crawl_cfg.get("raw_html_dir", "data/external/local_nankan/raw_html"), base_dir) / "race_list"
    calendar_dir = raw_html_dir / "calendar"
    program_dir = raw_html_dir / "program"

    session = _build_session(settings)
    calendar_progress = ProgressBar(total=max(len(calendar_keys), 1), prefix="[prepare:local-calendar]", min_interval_sec=0.0)
    calendar_progress.start("starting")

    last_fetch_at: float | None = None
    calendar_pages_fetched = 0
    program_pages_fetched = 0
    failures: list[dict[str, str]] = []
    meeting_records: list[dict[str, Any]] = []
    raw_meeting_count = 0

    for index, calendar_key in enumerate(calendar_keys, start=1):
        source_page_url = CALENDAR_URL_TEMPLATE.format(calendar_key=calendar_key)
        cache_path = calendar_dir / f"{calendar_key}.html"
        try:
            html, fetched, last_fetch_at, _ = _load_or_fetch_html(
                session=session,
                url=source_page_url,
                output_path=cache_path,
                settings=settings,
                last_fetch_at=last_fetch_at,
            )
            if fetched:
                calendar_pages_fetched += 1
            day_frame = parse_local_nankan_calendar_html(
                html,
                start_date=start_date,
                end_date=normalized_end_date,
                source_page_url=source_page_url,
            )
            raw_meeting_count += int(len(day_frame))
            if not day_frame.empty:
                for row in day_frame.to_dict("records"):
                    meeting_records.append(cast(dict[str, Any], dict(row)))
        except Exception as error:
            failures.append({"calendar_key": calendar_key, "stage": "calendar", "error": str(error)})
        calendar_progress.update(current=index, message=f"pages={index}/{len(calendar_keys)} meetings={len(meeting_records)} failures={len(failures)}")

    calendar_progress.complete(message=f"done meetings={len(meeting_records)} failures={len(failures)}")
    meeting_frame = pd.DataFrame(meeting_records, columns=["date", "meeting_id", "source_page_url", "source"])
    if not meeting_frame.empty:
        descending = str(date_order).strip().lower() == "desc"
        meeting_frame = meeting_frame.sort_values(["date", "meeting_id"], ascending=[not descending, not descending], kind="stable")
        meeting_frame = meeting_frame.drop_duplicates(subset=["meeting_id"], keep="first").reset_index(drop=True)

    program_progress = ProgressBar(total=max(len(meeting_frame), 1), prefix="[prepare:local-program]", min_interval_sec=0.0)
    program_progress.start("starting")

    excluded_race_ids = set(exclude_race_ids or set())
    records: list[dict[str, Any]] = []
    seen_race_ids: set[str] = set()
    raw_race_count = 0

    for index, row in enumerate(meeting_frame.to_dict("records"), start=1):
        meeting_id = str(row.get("meeting_id", "")).strip()
        if not meeting_id:
            continue
        source_page_url = PROGRAM_URL_TEMPLATE.format(meeting_id=meeting_id)
        cache_path = program_dir / f"{meeting_id}.html"
        try:
            html, fetched, last_fetch_at, _ = _load_or_fetch_html(
                session=session,
                url=source_page_url,
                output_path=cache_path,
                settings=settings,
                last_fetch_at=last_fetch_at,
            )
            if fetched:
                program_pages_fetched += 1
            race_frame = parse_local_nankan_program_html(
                html,
                meeting_id=meeting_id,
                source_page_url=source_page_url,
                require_result_link=require_result_link,
            )
            raw_race_count += int(len(race_frame))
            for race_row in race_frame.to_dict("records"):
                normalized_race_row = cast(dict[str, Any], dict(race_row))
                race_id = str(normalized_race_row.get("race_id", "")).strip()
                if not race_id or race_id in excluded_race_ids or race_id in seen_race_ids:
                    continue
                seen_race_ids.add(race_id)
                records.append(normalized_race_row)
                if limit is not None and limit > 0 and len(records) >= int(limit):
                    break
        except Exception as error:
            failures.append({"meeting_id": meeting_id, "stage": "program", "error": str(error)})

        program_progress.update(current=index, message=f"meetings={index}/{len(meeting_frame)} races={len(records)} failures={len(failures)}")
        if limit is not None and limit > 0 and len(records) >= int(limit):
            break

    program_progress.complete(message=f"done races={len(records)} failures={len(failures)}")
    race_frame = pd.DataFrame(records, columns=["date", "meeting_id", "race_id", "race_no", "post_time", "scheduled_post_at", "source_page_url", "source"])
    pre_filter_row_count = int(len(race_frame))
    effective_as_of = as_of
    if upcoming_only:
        effective_as_of = _resolve_as_of_timestamp(as_of).isoformat()
        race_frame = _filter_upcoming_races(race_frame, as_of=effective_as_of)
    report = {
        "source": "race_list",
        "date_window": {"start": start_date, "end": normalized_end_date},
        "date_order": str(date_order),
        "upcoming_only": bool(upcoming_only),
        "as_of": effective_as_of,
        "requested_calendar_pages": int(len(calendar_keys)),
        "calendar_pages_fetched": int(calendar_pages_fetched),
        "program_pages_fetched": int(program_pages_fetched),
        "raw_meeting_count": int(raw_meeting_count),
        "meeting_count": int(len(meeting_frame)),
        "raw_discovered_count": int(raw_race_count),
        "pre_filter_row_count": pre_filter_row_count,
        "row_count": int(len(race_frame)),
        "filtered_out_count": int(max(pre_filter_row_count - len(race_frame), 0)),
        "failure_count": int(len(failures)),
        "failures": failures[:50],
        "raw_html_dir": str(raw_html_dir),
    }
    return race_frame, report
