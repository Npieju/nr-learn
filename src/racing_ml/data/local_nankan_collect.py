from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from racing_ml.common.artifacts import display_path, utc_now_iso, write_csv_file, write_json, write_text_file
from racing_ml.common.progress import ProgressBar


HORSE_LINK_PATTERN = re.compile(r"/uma_info/(?P<id>\d+)\.do")
JOCKEY_LINK_PATTERN = re.compile(r"/kis_info/(?P<id>\d+)\.do")
TRAINER_LINK_PATTERN = re.compile(r"/cho_info/(?P<id>\d+)\.do")
TRACK_PATTERN = re.compile(r"(?P<track>浦和|船橋|大井|川崎)競馬")
DATE_PATTERN = re.compile(r"(?P<year>\d{4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日")
OPEN_DATE_PATTERN = re.compile(r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})")
DISTANCE_PATTERN = re.compile(r"(?:芝|ダート|ダ)\s*(?P<distance>[0-9,]{3,5})m")
WEATHER_PATTERN = re.compile(r"天候\s*:\s*(?P<weather>[^\s<]+)")
GROUND_PATTERN = re.compile(r"馬場\s*:\s*(?:芝|ダート|ダ)?\s*(?P<ground>[^\s<]+)")
SEX_AGE_PATTERN = re.compile(r"(?P<sex>[牡牝セ])\s*(?P<age>\d+)")
NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
SAFE_FILENAME_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")
ODDS_TAN_BLOCK_PATTERN = re.compile(r"var\s+odds_tan\s*=\s*\{(?P<body>.*?)\};", re.DOTALL)
ODDS_ENTRY_PATTERN = re.compile(
    r'"(?P<gate_no>\d+)"\s*:\s*\[\s*"(?P<odds>[^"]+)"\s*,\s*(?P<available>true|false)\s*,\s*\d+\s*,\s*(?P<popularity>\d+)',
    re.DOTALL,
)


@dataclass(frozen=True)
class RequestSettings:
    base_url: str
    user_agent: str
    timeout_sec: float
    delay_sec: float
    retry_count: int
    retry_backoff_sec: float
    overwrite: bool


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_crawl_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("crawl")
    if isinstance(nested, dict):
        return nested
    return config


def _build_target_manifest_path(base_manifest_path: Path, target_name: str) -> Path:
    return base_manifest_path.with_name(f"{base_manifest_path.stem}_{target_name}{base_manifest_path.suffix}")


def _build_lock_path(base_manifest_path: Path) -> Path:
    return base_manifest_path.with_suffix(base_manifest_path.suffix + ".lock")


def _normalize_text(value: object) -> str:
    text = str(value or "")
    text = text.replace("\xa0", " ").replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_header(value: object) -> str:
    return _normalize_text(value).replace(" ", "")


def _safe_filename(value: str) -> str:
    return SAFE_FILENAME_PATTERN.sub("_", value.strip())


def _decode_html_bytes(content: bytes) -> str:
    for encoding in ("shift_jis", "cp932", "euc-jp", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _extract_anchor_id(node: Tag | None, pattern: re.Pattern[str]) -> str | None:
    if node is None:
        return None
    anchor = node.find("a", href=pattern)
    if not isinstance(anchor, Tag):
        return None
    href = str(anchor.get("href", ""))
    match = pattern.search(href)
    if match is None:
        return None
    return match.group("id")


def _extract_anchor_text(node: Tag | None, pattern: re.Pattern[str]) -> str | None:
    if node is None:
        return None
    anchor = node.find("a", href=pattern)
    if not isinstance(anchor, Tag):
        return None
    text = _normalize_text(anchor.get_text(" ", strip=True))
    return text or None


def _extract_first_number(text: str | None) -> int | None:
    match = NUMERIC_PATTERN.search(_normalize_text(text))
    if match is None:
        return None
    try:
        return int(float(match.group(0).replace(",", "")))
    except ValueError:
        return None


def _extract_first_float_text(text: str | None) -> str | None:
    match = NUMERIC_PATTERN.search(_normalize_text(text))
    if match is None:
        return None
    return match.group(0).replace(",", "")


def _parse_sex_age(text: str | None) -> tuple[str | None, int | None]:
    match = SEX_AGE_PATTERN.search(_normalize_text(text))
    if match is None:
        return None, None
    try:
        return match.group("sex"), int(match.group("age"))
    except ValueError:
        return match.group("sex"), None


def _build_race_horse_id(race_id: str, gate_no: str | None) -> str | None:
    gate_text = _normalize_text(gate_no)
    if not gate_text:
        return None
    return f"{race_id}:{gate_text}"


def _load_target_ids(id_file: Path, id_column: str, limit: int | None) -> list[str]:
    if not id_file.exists():
        raise FileNotFoundError(f"id file not found: {id_file}")

    try:
        frame = pd.read_csv(id_file, usecols=[id_column], dtype="string", low_memory=False)
    except pd.errors.EmptyDataError:
        return []

    values: list[str] = []
    seen: set[str] = set()
    for raw_value in frame[id_column].tolist():
        value = _normalize_text(raw_value)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
        if limit is not None and limit > 0 and len(values) >= int(limit):
            break
    return values


def _extract_open_date(soup: BeautifulSoup) -> str | None:
    hidden = soup.find("input", id="gamenKirikaeMenu_openDate")
    value = _normalize_text(hidden.get("value", "") if isinstance(hidden, Tag) else "")
    match = OPEN_DATE_PATTERN.search(value)
    if match is not None:
        return f"{match.group('year')}-{match.group('month')}-{match.group('day')}"

    subtitle = soup.find(class_=re.compile(r"nk23_c-tab1__subtitle"))
    if subtitle is not None:
        match = DATE_PATTERN.search(_normalize_text(subtitle.get_text(" ", strip=True)))
        if match is not None:
            return f"{match.group('year')}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"

    legacy_header = soup.find(id="race-data01-a")
    if isinstance(legacy_header, Tag):
        match = DATE_PATTERN.search(_normalize_text(legacy_header.get_text(" ", strip=True)))
        if match is not None:
            return f"{match.group('year')}-{int(match.group('month')):02d}-{int(match.group('day')):02d}"
    return None


def _extract_track(soup: BeautifulSoup) -> str | None:
    subtitle = soup.find(class_=re.compile(r"nk23_c-tab1__subtitle|nk23_c-block01__info__title"))
    candidates: list[str] = []
    if subtitle is not None:
        candidates.append(_normalize_text(subtitle.get_text(" ", strip=True)))

    legacy_header = soup.find(id="race-data01-a")
    if isinstance(legacy_header, Tag):
        candidates.append(_normalize_text(legacy_header.get_text(" ", strip=True)))

    for text in candidates:
        match = TRACK_PATTERN.search(text)
        if match is not None:
            return match.group("track")
    return None


def _extract_distance(soup: BeautifulSoup) -> int | None:
    candidates: list[str] = []
    subtitle = soup.find(class_=re.compile(r"nk23_c-tab1__subtitle"))
    if subtitle is not None:
        candidates.append(_normalize_text(subtitle.get_text(" ", strip=True)))

    legacy_header = soup.find(id="race-data01-a")
    if isinstance(legacy_header, Tag):
        candidates.append(_normalize_text(legacy_header.get_text(" ", strip=True)))

    for text in candidates:
        match = DISTANCE_PATTERN.search(text)
        if match is None:
            continue
        try:
            return int(match.group("distance").replace(",", ""))
        except ValueError:
            continue
    return None


def _extract_weather_and_ground(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    candidates: list[str] = []
    for node in soup.find_all(class_=re.compile(r"nk23_c-table01__txt|nk23_c-block01__info__texts")):
        candidates.append(_normalize_text(node.get_text(" ", strip=True)))

    legacy_table = soup.find("table", attrs={"summary": "天候・馬場"})
    if isinstance(legacy_table, Tag):
        rows = legacy_table.find_all("tr")
        if len(rows) >= 2:
            legacy_row = rows[1]
            if isinstance(legacy_row, Tag):
                cells = legacy_row.find_all(["td", "th"])
                if len(cells) >= 2:
                    weather = _normalize_text(cells[0].get_text(" ", strip=True)) or None
                    ground_text = _normalize_text(cells[1].get_text(" ", strip=True))
                    ground_match = GROUND_PATTERN.search(f"馬場:{ground_text}")
                    ground = ground_match.group("ground") if ground_match is not None else ground_text or None
                    return weather, ground

    for text in candidates:
        weather_match = WEATHER_PATTERN.search(text)
        ground_match = GROUND_PATTERN.search(text)
        weather = weather_match.group("weather") if weather_match is not None else None
        ground = ground_match.group("ground") if ground_match is not None else None
        if weather == "−":
            weather = None
        if ground == "−":
            ground = None
        if weather is not None or ground is not None:
            return weather, ground
    return None, None


def _extract_race_name(soup: BeautifulSoup) -> str | None:
    heading = soup.find(class_=re.compile(r"nk23_c-tab1__title__text|nk23_c-block01__list__title"))
    if heading is None:
        heading = soup.find(class_="race-name")
    if heading is None:
        return None
    text = _normalize_text(heading.get_text(" ", strip=True))
    return text or None


def _extract_race_metadata(soup: BeautifulSoup, race_id: str) -> dict[str, Any]:
    weather, ground = _extract_weather_and_ground(soup)
    return {
        "date": _extract_open_date(soup),
        "race_id": race_id,
        "track": _extract_track(soup),
        "distance": _extract_distance(soup),
        "weather": weather,
        "ground_condition": ground,
        "race_name": _extract_race_name(soup),
    }


def _empty_race_result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "race_id",
            "track",
            "distance",
            "weather",
            "ground_condition",
            "race_name",
            "horse_id",
            "horse_key",
            "horse_name",
            "rank",
            "frame_no",
            "gate_no",
            "sex",
            "age",
            "weight",
            "jockey_id",
            "jockey_key",
            "trainer_id",
            "trainer_key",
            "finish_time",
            "margin",
            "closing_time_3f",
            "passing_order",
            "popularity",
        ]
    )


def _is_cancelled_race_result_page(soup: BeautifulSoup) -> bool:
    scratch_node = soup.find(id="tl-scratch")
    scratch_text = _normalize_text(scratch_node.get_text(" ", strip=True) if isinstance(scratch_node, Tag) else "")
    if scratch_text and any(keyword in scratch_text for keyword in ("取り止め", "取止め", "不成立")):
        return True

    body_text = _normalize_text(soup.get_text(" ", strip=True))
    return any(keyword in body_text for keyword in ("競走は取り止めとなりました", "競走は取止めとなりました", "不成立"))


def parse_local_nankan_race_result_html(html: str, race_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", attrs={"summary": "レース結果"})
    table: Tag | None = None
    if isinstance(container, Tag):
        candidate = container.find("table", class_=re.compile(r"nk23_c-table01__table"))
        if isinstance(candidate, Tag):
            table = candidate

    if table is None:
        legacy_table = soup.find("table", attrs={"summary": "レース結果"})
        if isinstance(legacy_table, Tag):
            table = legacy_table

    if not isinstance(table, Tag):
        if _is_cancelled_race_result_page(soup):
            return _empty_race_result_frame()
        raise ValueError(f"race result table not found for race_id={race_id}")

    header_row = table.find("tr")
    if not isinstance(header_row, Tag):
        raise ValueError(f"race result header row not found for race_id={race_id}")

    headers = [
        _normalize_header(th.get_text(" ", strip=True))
        for th in header_row.find_all("th")
        if isinstance(th, Tag)
    ]
    header_map = {
        "着": "rank",
        "枠": "frame_no",
        "馬番": "gate_no",
        "馬名": "horse_name",
        "性齢": "sex_age",
        "負担": "carried_weight",
        "馬体重": "weight",
        "増減": "weight_change",
        "騎手": "jockey_id",
        "調教師": "trainer_id",
        "タイム": "finish_time",
        "着差": "margin",
        "上がり3F": "closing_time_3f",
        "コーナー通過順": "passing_order",
        "人気": "popularity",
    }

    metadata = _extract_race_metadata(soup, race_id)
    rows: list[dict[str, Any]] = []
    for row_tag in table.find_all("tr")[1:]:
        if not isinstance(row_tag, Tag):
            continue
        cells = row_tag.find_all("td", recursive=False)
        if len(cells) < 8:
            continue

        raw_row = {
            header_map[header]: _normalize_text(cells[index].get_text(" ", strip=True))
            for index, header in enumerate(headers[: len(cells)])
            if header in header_map
        }
        sex, age = _parse_sex_age(raw_row.get("sex_age"))
        horse_key = _extract_anchor_id(row_tag, HORSE_LINK_PATTERN)
        horse_name = _extract_anchor_text(row_tag, HORSE_LINK_PATTERN) or raw_row.get("horse_name")
        jockey_name = _extract_anchor_text(row_tag, JOCKEY_LINK_PATTERN) or raw_row.get("jockey_id")
        trainer_name = _extract_anchor_text(row_tag, TRAINER_LINK_PATTERN) or raw_row.get("trainer_id")
        gate_no = raw_row.get("gate_no")

        rows.append(
            {
                **metadata,
                "horse_id": _build_race_horse_id(race_id, gate_no),
                "horse_key": horse_key,
                "horse_name": horse_name,
                "rank": raw_row.get("rank"),
                "frame_no": raw_row.get("frame_no"),
                "gate_no": gate_no,
                "sex": sex,
                "age": age,
                "weight": _extract_first_number(raw_row.get("weight")),
                "jockey_id": jockey_name,
                "jockey_key": _extract_anchor_id(row_tag, JOCKEY_LINK_PATTERN),
                "trainer_id": trainer_name,
                "trainer_key": _extract_anchor_id(row_tag, TRAINER_LINK_PATTERN),
                "finish_time": raw_row.get("finish_time"),
                "margin": raw_row.get("margin"),
                "closing_time_3f": _extract_first_float_text(raw_row.get("closing_time_3f")),
                "passing_order": raw_row.get("passing_order"),
                "popularity": raw_row.get("popularity"),
            }
        )

    if not rows:
        raise ValueError(f"no race result rows parsed for race_id={race_id}")
    return pd.DataFrame(rows)


def _extract_multiline_cell_values(cell: Tag | None) -> list[str]:
    if cell is None:
        return []
    values = [_normalize_text(piece) for piece in cell.stripped_strings]
    return [value for value in values if value]


def _parse_local_nankan_odds_js(script_text: str | None) -> dict[str, dict[str, str]]:
    if not script_text:
        return {}
    block_match = ODDS_TAN_BLOCK_PATTERN.search(script_text)
    if block_match is None:
        return {}

    odds_map: dict[str, dict[str, str]] = {}
    for match in ODDS_ENTRY_PATTERN.finditer(block_match.group("body")):
        if str(match.group("available")).strip().lower() != "true":
            continue
        gate_no = _normalize_text(match.group("gate_no"))
        if not gate_no:
            continue
        odds_map[gate_no] = {
            "odds": _normalize_text(match.group("odds")),
            "popularity": _normalize_text(match.group("popularity")),
        }
    return odds_map


def parse_local_nankan_race_card_html(html: str, race_id: str, odds_js: str | None = None) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    metadata = _extract_race_metadata(soup, race_id)
    odds_map = _parse_local_nankan_odds_js(odds_js)
    table = soup.select_one(".nk23_c-table23__inner table.nk23_c-table23__table")
    if not isinstance(table, Tag):
        raise ValueError(f"race card table not found for race_id={race_id}")

    rows: list[dict[str, Any]] = []
    current_frame_no: str | None = None
    for row_tag in table.select("tbody tr"):
        if not isinstance(row_tag, Tag):
            continue
        cells = [cell for cell in row_tag.find_all("td", recursive=False) if isinstance(cell, Tag)]
        if len(cells) == 12:
            frame_no = _normalize_text(cells[0].get_text(" ", strip=True))
            current_frame_no = frame_no or current_frame_no
            gate_cell_index = 1
            horse_cell_index = 2
        elif len(cells) == 11 and current_frame_no:
            frame_no = current_frame_no
            gate_cell_index = 0
            horse_cell_index = 1
        else:
            continue

        horse_key = _extract_anchor_id(cells[horse_cell_index], HORSE_LINK_PATTERN)
        horse_name = _extract_anchor_text(cells[horse_cell_index], HORSE_LINK_PATTERN)
        sex, age = _parse_sex_age(cells[horse_cell_index + 1].get_text(" ", strip=True))
        weight_values = _extract_multiline_cell_values(cells[horse_cell_index + 3])
        parent_values = _extract_multiline_cell_values(cells[horse_cell_index + 6])
        owner_values = _extract_multiline_cell_values(cells[horse_cell_index + 8])
        gate_no = _normalize_text(cells[gate_cell_index].get_text(" ", strip=True))
        odds_entry = odds_map.get(gate_no, {})

        rows.append(
            {
                **metadata,
                "horse_id": _build_race_horse_id(race_id, gate_no),
                "horse_key": horse_key,
                "horse_name": horse_name,
                "frame_no": frame_no,
                "gate_no": gate_no,
                "sex": sex,
                "age": age,
                "weight": _extract_first_number(weight_values[0] if weight_values else None),
                "weight_change": _extract_first_number(weight_values[1] if len(weight_values) > 1 else None),
                "odds": _extract_first_float_text(odds_entry.get("odds")),
                "popularity": _extract_first_number(odds_entry.get("popularity")),
                "carried_weight": _extract_first_float_text(cells[horse_cell_index + 4].get_text(" ", strip=True)),
                "jockey_id": _extract_anchor_text(cells[horse_cell_index + 5], JOCKEY_LINK_PATTERN) or _normalize_text(cells[horse_cell_index + 5].get_text(" ", strip=True)),
                "jockey_key": _extract_anchor_id(cells[horse_cell_index + 5], JOCKEY_LINK_PATTERN),
                "trainer_id": _extract_anchor_text(cells[horse_cell_index + 7], TRAINER_LINK_PATTERN) or _normalize_text(cells[horse_cell_index + 7].get_text(" ", strip=True)),
                "trainer_key": _extract_anchor_id(cells[horse_cell_index + 7], TRAINER_LINK_PATTERN),
                "sire_name": parent_values[0] if len(parent_values) > 0 else None,
                "dam_name": parent_values[1] if len(parent_values) > 1 else None,
                "owner_name": owner_values[0] if len(owner_values) > 0 else None,
                "breeder_name": owner_values[1] if len(owner_values) > 1 else None,
            }
        )

    if not rows:
        raise ValueError(f"no race card rows parsed for race_id={race_id}")
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=["race_id", "horse_id"], keep="first").reset_index(drop=True)


def _extract_horse_name(profile_soup: BeautifulSoup, horse_key: str) -> str | None:
    heading = profile_soup.find(id="tl-prof")
    if isinstance(heading, Tag):
        text = _normalize_text(heading.get_text(" ", strip=True))
        if text:
            return text
    title = profile_soup.find("title")
    title_text = _normalize_text(title.get_text(" ", strip=True) if title is not None else "")
    if title_text and title_text != "競走馬検索":
        return re.split(r"\s*[|｜]\s*", title_text, maxsplit=1)[0] or horse_key
    heading = profile_soup.find(["h1", "h2"], class_=re.compile(r"title|horse", re.IGNORECASE))
    if heading is None:
        return None
    return _normalize_text(heading.get_text(" ", strip=True)) or None


def parse_local_nankan_pedigree_html(html: str, horse_key: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    blood_table = soup.find("table", attrs={"summary": "血統"})
    if not isinstance(blood_table, Tag):
        raise ValueError(f"blood table not found for horse_key={horse_key}")

    pedigree_map: dict[str, str] = {}
    for row in blood_table.find_all("tr"):
        if not isinstance(row, Tag):
            continue
        cells = row.find_all("td", recursive=False)
        if len(cells) < 3:
            continue
        label = _normalize_text(cells[0].get_text(" ", strip=True))
        pedigree_map[label] = _normalize_text(cells[2].get_text(" ", strip=True))

    info_table = soup.select_one("#disp-chg-pc table#horse-info") or soup.find("table", id="horse-info")
    info_map: dict[str, str] = {}
    if isinstance(info_table, Tag):
        cells = info_table.find_all("td")
        for index in range(0, len(cells) - 1, 2):
            label = _normalize_text(cells[index].get_text(" ", strip=True))
            value = _normalize_text(cells[index + 1].get_text(" ", strip=True))
            if label:
                info_map[label] = value

    row = {
        "horse_key": horse_key,
        "horse_name": _extract_horse_name(soup, horse_key),
        "sire_name": pedigree_map.get("父"),
        "dam_name": pedigree_map.get("母"),
        "damsire_name": pedigree_map.get("母父"),
        "owner_name": info_map.get("馬主名"),
        "breeder_name": info_map.get("生産牧場"),
    }
    return pd.DataFrame([row])


def _build_session(settings: RequestSettings) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": settings.user_agent,
            "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        }
    )
    return session


def _fetch_html(session: requests.Session, url: str, settings: RequestSettings) -> str:
    last_error: Exception | None = None
    for attempt in range(max(settings.retry_count, 1)):
        try:
            response = session.get(url, timeout=settings.timeout_sec)
            response.raise_for_status()
            return _decode_html_bytes(response.content)
        except Exception as error:
            last_error = error
            if attempt + 1 >= max(settings.retry_count, 1):
                break
            time.sleep(settings.retry_backoff_sec * (attempt + 1))
    assert last_error is not None
    raise last_error


def _load_or_fetch_html(
    *,
    session: requests.Session,
    url: str,
    output_path: Path,
    settings: RequestSettings,
    last_fetch_at: float | None,
) -> tuple[str, bool, float | None]:
    if output_path.exists() and not settings.overwrite:
        return output_path.read_text(encoding="utf-8"), False, last_fetch_at

    if last_fetch_at is not None and settings.delay_sec > 0:
        elapsed = time.perf_counter() - last_fetch_at
        sleep_sec = settings.delay_sec - elapsed
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    html = _fetch_html(session, url, settings)
    write_text_file(output_path, html, label="raw html output")
    return html, True, time.perf_counter()


def _dedupe_frame(frame: pd.DataFrame, dedupe_on: list[str]) -> pd.DataFrame:
    columns = [column for column in dedupe_on if column in frame.columns]
    if not columns:
        return frame.reset_index(drop=True)
    return frame.drop_duplicates(subset=columns, keep="last").reset_index(drop=True)


def _load_existing_output_frame(output_file: Path) -> pd.DataFrame:
    if not output_file.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_file, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_cumulative_output(output_file: Path, batch_frame: pd.DataFrame, dedupe_on: list[str]) -> tuple[pd.DataFrame, int]:
    existing = _load_existing_output_frame(output_file)
    existing_rows = int(len(existing))
    if existing.empty:
        combined = batch_frame.copy()
    elif batch_frame.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, batch_frame], ignore_index=True, sort=False)
        combined = _dedupe_frame(combined, dedupe_on)
    write_csv_file(output_file, combined, index=False, label="crawl output")
    return combined, existing_rows


def _target_status(*, requested_ids: int, parsed_ids: int, failure_count: int) -> str:
    if requested_ids == 0:
        return "completed"
    if parsed_ids == requested_ids and failure_count == 0:
        return "completed"
    if parsed_ids > 0:
        return "partial"
    return "failed"


def _build_target_report(
    *,
    target_name: str,
    requested_ids: int,
    processed_ids: int,
    parsed_ids: int,
    fetched_count: int,
    failures: list[dict[str, str]],
    output_file: Path,
    manifest_path: Path,
    raw_html_dir: Path,
    base_dir: Path,
    rows_written: int,
    dry_run: bool,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    status = "planned" if dry_run else _target_status(
        requested_ids=requested_ids,
        parsed_ids=parsed_ids,
        failure_count=len(failures),
    )
    return {
        "target": target_name,
        "status": status,
        "requested_ids": int(requested_ids),
        "processed_ids": int(processed_ids),
        "parsed_ids": int(parsed_ids),
        "fetched_count": int(fetched_count),
        "rows_written": int(rows_written),
        "failure_count": len(failures),
        "output_file": display_path(output_file, workspace_root=base_dir),
        "manifest_file": display_path(manifest_path, workspace_root=base_dir),
        "raw_html_path": display_path(raw_html_dir / target_name, workspace_root=base_dir),
        "sample_ids": list(sample_ids or []),
        "failures": failures[:50],
    }


def _build_summary(
    *,
    crawl_cfg: dict[str, Any],
    target_reports: list[dict[str, Any]],
    base_manifest_path: Path,
    lock_path: Path,
    base_dir: Path,
    dry_run: bool,
    started_at: str,
) -> dict[str, Any]:
    requested_total = sum(int(report.get("requested_ids") or 0) for report in target_reports)
    failure_total = sum(int(report.get("failure_count") or 0) for report in target_reports)
    selected_targets = [str(report.get("target")) for report in target_reports]
    all_completed = all(str(report.get("status") or "") in {"completed", "planned"} for report in target_reports)

    if dry_run:
        status = "planned"
        current_phase = "planned"
        recommended_action = "review_collect_plan"
        error_code = None
        error_message = None
    elif all_completed and failure_total == 0:
        status = "completed"
        current_phase = "crawl_completed"
        recommended_action = "run_local_materialize"
        error_code = None
        error_message = None
    else:
        status = "partial"
        current_phase = "crawl_completed_with_failures"
        recommended_action = "inspect_local_crawl_failures"
        error_code = "crawl_partial_failures"
        error_message = f"local_nankan crawl finished with {failure_total} failures"

    highlights = [
        f"selected_targets={len(selected_targets)}",
        f"requested_ids_total={requested_total}",
        f"failure_count_total={failure_total}",
    ]
    if dry_run:
        highlights.append("dry run only; provider fetch is not executed")
    else:
        highlights.append("provider fixed to nankankeiba.com result/syousai/uma_info")

    return {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "status": status,
        "provider": str(crawl_cfg.get("provider", "nankankeiba")),
        "manifest_file": display_path(base_manifest_path, workspace_root=base_dir),
        "lock_file": display_path(lock_path, workspace_root=base_dir),
        "selected_targets": selected_targets,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "highlights": highlights,
        "error_code": error_code,
        "error_message": error_message,
        "targets": target_reports,
    }


@contextmanager
def local_nankan_crawl_lock(crawl_cfg: dict[str, Any], *, base_dir: Path):
    manifest_base_path = _resolve_path(
        str(crawl_cfg.get("manifest_file", "artifacts/reports/local_nankan_crawl_manifest.json")),
        base_dir,
    )
    lock_path = _build_lock_path(manifest_base_path)
    payload = {
        "pid": os.getpid(),
        "started_at": utc_now_iso(),
        "lock_file": str(lock_path),
    }
    while True:
        try:
            file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as error:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            pid = int(existing.get("pid", 0) or 0) if isinstance(existing, dict) else 0
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            except PermissionError:
                pass
            raise RuntimeError(f"local_nankan crawl is already running: {lock_path}") from error

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _dry_run_collect(
    *,
    crawl_cfg: dict[str, Any],
    targets: dict[str, Any],
    base_dir: Path,
    target_filter: str | None,
    override_limit: int | None,
) -> dict[str, Any]:
    raw_html_dir = _resolve_path(str(crawl_cfg.get("raw_html_dir", "data/external/local_nankan/raw_html")), base_dir)
    base_manifest_path = _resolve_path(str(crawl_cfg.get("manifest_file", "artifacts/reports/local_nankan_crawl_manifest.json")), base_dir)
    lock_path = _build_lock_path(base_manifest_path)
    started_at = utc_now_iso()
    target_reports: list[dict[str, Any]] = []

    for target_name, target_cfg in targets.items():
        if not isinstance(target_cfg, dict):
            continue
        if target_filter not in {None, "all", target_name}:
            continue
        if not bool(target_cfg.get("enabled", True)):
            continue
        id_file = _resolve_path(str(target_cfg.get("id_file")), base_dir)
        id_column = str(target_cfg.get("id_column", "id"))
        ids = _load_target_ids(id_file, id_column, override_limit)
        manifest_path = _build_target_manifest_path(base_manifest_path, target_name)
        output_file = _resolve_path(str(target_cfg.get("output_file")), base_dir)
        report = _build_target_report(
            target_name=target_name,
            requested_ids=len(ids),
            processed_ids=0,
            parsed_ids=0,
            fetched_count=0,
            failures=[],
            output_file=output_file,
            manifest_path=manifest_path,
            raw_html_dir=raw_html_dir,
            base_dir=base_dir,
            rows_written=0,
            dry_run=True,
            sample_ids=ids[:5],
        )
        write_json(manifest_path, report)
        target_reports.append(report)

    if not target_reports:
        raise ValueError(f"No enabled targets selected for filter: {target_filter or 'all'}")

    summary = _build_summary(
        crawl_cfg=crawl_cfg,
        target_reports=target_reports,
        base_manifest_path=base_manifest_path,
        lock_path=lock_path,
        base_dir=base_dir,
        dry_run=True,
        started_at=started_at,
    )
    write_json(base_manifest_path, summary)
    return summary


def collect_local_nankan_from_config(
    config: dict[str, Any],
    *,
    base_dir: Path,
    target_filter: str | None = None,
    override_limit: int | None = None,
    dry_run: bool = False,
    use_lock: bool = True,
) -> dict[str, Any]:
    crawl_cfg = _resolve_crawl_config(config)
    targets = crawl_cfg.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("crawl.targets must contain at least one target")

    if dry_run:
        return _dry_run_collect(
            crawl_cfg=crawl_cfg,
            targets=targets,
            base_dir=base_dir,
            target_filter=target_filter,
            override_limit=override_limit,
        )

    if use_lock:
        with local_nankan_crawl_lock(crawl_cfg, base_dir=base_dir):
            return collect_local_nankan_from_config(
                config,
                base_dir=base_dir,
                target_filter=target_filter,
                override_limit=override_limit,
                dry_run=False,
                use_lock=False,
            )

    settings = RequestSettings(
        base_url=str(crawl_cfg.get("base_url", "https://www.nankankeiba.com")).rstrip("/"),
        user_agent=str(crawl_cfg.get("user_agent", "nr-learn-local-nankan-crawler/0.1")),
        timeout_sec=float(crawl_cfg.get("timeout_sec", 20.0)),
        delay_sec=float(crawl_cfg.get("delay_sec", 0.05)),
        retry_count=max(int(crawl_cfg.get("retry_count", 3)), 1),
        retry_backoff_sec=float(crawl_cfg.get("retry_backoff_sec", 2.0)),
        overwrite=bool(crawl_cfg.get("overwrite", False)),
    )

    raw_html_dir = _resolve_path(str(crawl_cfg.get("raw_html_dir", "data/external/local_nankan/raw_html")), base_dir)
    base_manifest_path = _resolve_path(str(crawl_cfg.get("manifest_file", "artifacts/reports/local_nankan_crawl_manifest.json")), base_dir)
    lock_path = _build_lock_path(base_manifest_path)
    started_at = utc_now_iso()
    target_reports: list[dict[str, Any]] = []
    write_json(
        base_manifest_path,
        {
            "started_at": started_at,
            "finished_at": None,
            "status": "running",
            "provider": str(crawl_cfg.get("provider", "nankankeiba")),
            "manifest_file": display_path(base_manifest_path, workspace_root=base_dir),
            "lock_file": display_path(lock_path, workspace_root=base_dir),
            "selected_targets": [],
            "current_phase": "crawl_running",
            "recommended_action": None,
            "highlights": [],
            "error_code": None,
            "error_message": None,
            "targets": target_reports,
        },
    )

    parser_by_target = {
        "race_result": ("result", parse_local_nankan_race_result_html, ["race_id", "horse_id"]),
        "race_card": ("syousai", parse_local_nankan_race_card_html, ["race_id", "horse_id"]),
        "pedigree": ("uma_info", parse_local_nankan_pedigree_html, ["horse_key"]),
    }

    session = _build_session(settings)
    try:
        for target_name, target_cfg in targets.items():
            if not isinstance(target_cfg, dict):
                continue
            if target_filter not in {None, "all", target_name}:
                continue
            if not bool(target_cfg.get("enabled", True)):
                continue
            if target_name not in parser_by_target:
                raise ValueError(f"Unsupported crawl target: {target_name}")

            id_file = _resolve_path(str(target_cfg.get("id_file")), base_dir)
            id_column = str(target_cfg.get("id_column", "id"))
            output_file = _resolve_path(str(target_cfg.get("output_file")), base_dir)
            limit = override_limit if override_limit is not None else target_cfg.get("limit")
            limit_value = None
            if limit not in {None, "", 0}:
                limit_value = int(str(limit))
            ids = _load_target_ids(id_file, id_column, limit_value)
            manifest_path = _build_target_manifest_path(base_manifest_path, target_name)
            endpoint, parser, dedupe_on = parser_by_target[target_name]
            progress = ProgressBar(total=max(len(ids), 1), prefix=f"[crawl:{target_name}]", min_interval_sec=0.0)
            progress.start("starting")

            frames: list[pd.DataFrame] = []
            failures: list[dict[str, str]] = []
            fetched_count = 0
            last_fetch_at: float | None = None

            for index, entity_id in enumerate(ids, start=1):
                raw_html_path = raw_html_dir / target_name / f"{_safe_filename(entity_id)}.html"
                url = urljoin(settings.base_url + "/", f"{endpoint}/{entity_id}.do")
                try:
                    html, fetched, last_fetch_at = _load_or_fetch_html(
                        session=session,
                        url=url,
                        output_path=raw_html_path,
                        settings=settings,
                        last_fetch_at=last_fetch_at,
                    )
                    if fetched:
                        fetched_count += 1
                    if target_name == "race_card":
                        odds_js: str | None = None
                        odds_path = raw_html_dir / "race_card_odds" / f"{_safe_filename(entity_id)}.js"
                        odds_url = urljoin(settings.base_url + "/", f"oddsJS/{entity_id}.do")
                        try:
                            odds_js, odds_fetched, last_fetch_at = _load_or_fetch_html(
                                session=session,
                                url=odds_url,
                                output_path=odds_path,
                                settings=settings,
                                last_fetch_at=last_fetch_at,
                            )
                            if odds_fetched:
                                fetched_count += 1
                        except Exception as error:
                            failures.append({"id": entity_id, "stage": "race_card_odds", "error": str(error)})
                        frames.append(parse_local_nankan_race_card_html(html, entity_id, odds_js=odds_js))
                    else:
                        frames.append(parser(html, entity_id))
                except Exception as error:
                    failures.append({"id": entity_id, "stage": target_name, "error": str(error)})

                progress.update(current=index, message=f"parsed={len(frames)} failed={len(failures)}")
                running_rows = int(sum(len(frame) for frame in frames))
                running_report = _build_target_report(
                    target_name=target_name,
                    requested_ids=len(ids),
                    processed_ids=index,
                    parsed_ids=len(frames),
                    fetched_count=fetched_count,
                    failures=failures,
                    output_file=output_file,
                    manifest_path=manifest_path,
                    raw_html_dir=raw_html_dir,
                    base_dir=base_dir,
                    rows_written=running_rows,
                    dry_run=False,
                    sample_ids=ids[:5],
                )
                write_json(manifest_path, running_report)

            batch_frame = _dedupe_frame(pd.concat(frames, ignore_index=True, sort=False), dedupe_on) if frames else pd.DataFrame()
            combined, _ = _write_cumulative_output(output_file, batch_frame, dedupe_on)
            report = _build_target_report(
                target_name=target_name,
                requested_ids=len(ids),
                processed_ids=len(ids),
                parsed_ids=len(frames),
                fetched_count=fetched_count,
                failures=failures,
                output_file=output_file,
                manifest_path=manifest_path,
                raw_html_dir=raw_html_dir,
                base_dir=base_dir,
                rows_written=int(len(combined)),
                dry_run=False,
                sample_ids=ids[:5],
            )
            target_reports.append(report)
            write_json(manifest_path, report)
            write_json(
                base_manifest_path,
                {
                    "started_at": started_at,
                    "finished_at": None,
                    "status": "running",
                    "provider": str(crawl_cfg.get("provider", "nankankeiba")),
                    "manifest_file": display_path(base_manifest_path, workspace_root=base_dir),
                    "lock_file": display_path(lock_path, workspace_root=base_dir),
                    "selected_targets": [str(item.get("target")) for item in target_reports],
                    "current_phase": "crawl_running",
                    "recommended_action": None,
                    "highlights": [],
                    "error_code": None,
                    "error_message": None,
                    "targets": target_reports,
                },
            )
    finally:
        session.close()

    summary = _build_summary(
        crawl_cfg=crawl_cfg,
        target_reports=target_reports,
        base_manifest_path=base_manifest_path,
        lock_path=lock_path,
        base_dir=base_dir,
        dry_run=False,
        started_at=started_at,
    )
    write_json(base_manifest_path, summary)
    return summary