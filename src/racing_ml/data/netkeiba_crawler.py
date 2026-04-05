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

from racing_ml.common.artifacts import write_csv_file, write_json, write_text_file
from racing_ml.common.progress import ProgressBar


HORSE_LINK_PATTERN = re.compile(r"/horse/([0-9A-Za-z]+)/?")
JOCKEY_LINK_PATTERN = re.compile(r"/jockey(?:/result/recent)?/([0-9A-Za-z]+)/?")
TRAINER_LINK_PATTERN = re.compile(r"/trainer(?:/result/recent)?/([0-9A-Za-z]+)/?")
OWNER_LINK_PATTERN = re.compile(r"/owner(?:/result/recent)?/([0-9A-Za-z]+)/?")
RACECARD_DATE_PARAM_PATTERN = re.compile(r"[?&]kaisai_date=(?P<date>\d{8})")
DATE_PATTERN = re.compile(r"(?P<year>\d{4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日")
DISTANCE_PATTERN = re.compile(r"(?P<distance>\d{3,4})m")
WEATHER_PATTERN = re.compile(r"天候\s*:\s*(?P<weather>[^\s/]+)")
GROUND_PATTERN = re.compile(r"(?:芝|ダート|障害)\s*:\s*(?P<ground>[^\s/]+)")
RACECARD_GROUND_PATTERN = re.compile(r"馬場\s*:\s*(?P<ground>[^\s/|]+)")
BRACKET_PREFIX_PATTERN = re.compile(r"^\[[^\]]+\]\s*")
BRACKET_PREFIX_CAPTURE_PATTERN = re.compile(r"^\[(?P<prefix>[^\]]+)\]\s*")
SAFE_FILENAME_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")
MOBILE_DATE_PATTERN = re.compile(r"(?P<month>\d{1,2})/(?P<day>\d{1,2})\([^)]+\)")
MOBILE_TRACK_PATTERN = re.compile(r"今週へ\s+(?P<track>札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)")
MOBILE_RACE_INFO_PATTERN = re.compile(
    r"(?P<race_no>\d{1,2})R\s+(?P<condition>.*?)\s+\d{1,2}:\d{2}\s+"
    r"(?P<surface>芝|ダート|ダ|障害)\s*(?P<distance>\d{3,4})m\s*(?:\((?P<direction>右|左|直線)\))?\s+"
    r"\d+頭\s+(?P<weather>晴|曇|雨|雪)\s+(?P<ground>良|稍重|重|不良)"
)
MOBILE_RESULT_WEIGHT_PATTERN = re.compile(r"(?P<weight>\d+kg\s*\([^)]+\))")
MOBILE_RESULT_CARRY_PATTERN = re.compile(
    r"(?P<sex>[牡牝セ])(?P<age>\d+)(?:\S+)?\s+(?P<weight>\d+kg\s*\([^)]+\))\s+"
    r"(?P<jockey>.+?)\s+(?P<carried>\d+(?:\.\d+)?)\s+(?P<trainer_region>栗東|美浦|地方|海外)･(?P<trainer>.+)$"
)
MOBILE_CARD_JOCKEY_PATTERN = re.compile(
    r"(?P<sex>[牡牝セ])(?P<age>\d+)(?:\S+)?\s+(?P<jockey>.+?)\s+(?P<carried>\d+(?:\.\d+)?)$"
)
MOBILE_ODDS_PATTERN = re.compile(r"(?P<odds>\d+(?:\.\d+)?)倍\s+(?P<popularity>\d+)人気")
MOBILE_CARD_ODDS_PATTERN = re.compile(r"(?P<odds>\d+(?:\.\d+)?|---\.--)\s*\(\s*(?P<popularity>\d+|\*\*)人気\s*\)")
MOBILE_TIME_PATTERN = re.compile(r"(?P<time>\d+:\d+\.\d+)")
MOBILE_CLOSING_PATTERN = re.compile(r"\((?P<closing>\d+\.\d+)\)")
FULLWIDTH_DIGIT_TRANSLATION = str.maketrans("０１２３４５６７８９", "0123456789")
CLASS_TOKEN_REGEX = (
    r"未出走|新馬|未勝利|300万下|400万下|500万下|600万下|700万下|"
    r"800万下|900万下|1000万下|1400万下|1500万下|1600万下|"
    r"1勝クラス|2勝クラス|3勝クラス|オープン|OPEN|OP"
)
CONDITION_WITH_AGE_PATTERN = re.compile(
    fr"(?P<age_group>(?:障害)?\d歳(?:以上|上)?)\s*(?P<class_token>{CLASS_TOKEN_REGEX})"
)
CLASS_TOKEN_PATTERN = re.compile(fr"(?P<class_token>{CLASS_TOKEN_REGEX})")
RACE_GRADE_PATTERN = re.compile(r"\((?P<grade>(?:J\.?)?G(?:III|II|I|3|2|1)|L)\)")
SURFACE_PATTERN = re.compile(r"(?P<surface>芝|ダート|ダ|障害)(?:\s*(?:右|左|直線|\d))")
JRA_VENUES = (
    "札幌",
    "函館",
    "福島",
    "新潟",
    "東京",
    "中山",
    "中京",
    "京都",
    "阪神",
    "小倉",
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
    parse_only: bool
    pedigree_prefer_mobile: bool = False


def _normalize_text(value: object) -> str:
    text = str(value or "")
    text = text.translate(FULLWIDTH_DIGIT_TRANSLATION)
    text = text.replace("\xa0", " ").replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_header(value: object) -> str:
    text = _normalize_text(value)
    return text.replace(" ", "")


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _safe_filename(entity_id: str) -> str:
    return SAFE_FILENAME_PATTERN.sub("_", entity_id.strip())


def _decode_html_bytes(content: bytes) -> str:
    for encoding in ("euc-jp", "cp932", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    for encoding in ("euc-jp", "cp932", "utf-8"):
        try:
            return content.decode(encoding, errors="ignore")
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _extract_anchor_id(node: Tag | None, pattern: re.Pattern[str]) -> str | None:
    if node is None:
        return None
    anchor = node.find("a", href=pattern)
    if anchor is None:
        return None
    href = str(anchor.get("href", ""))
    match = pattern.search(href)
    if match is None:
        return None
    return match.group(1)


def _extract_anchor_text(node: Tag | None, pattern: re.Pattern[str]) -> str | None:
    if node is None:
        return None
    anchor = node.find("a", href=pattern)
    if anchor is None:
        return None
    text = _normalize_text(anchor.get_text(" ", strip=True))
    return text or None


def _strip_bracket_prefix(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = BRACKET_PREFIX_PATTERN.sub("", _normalize_text(text))
    return stripped or None


def _extract_bracket_prefix(text: str | None) -> str | None:
    if text is None:
        return None
    match = BRACKET_PREFIX_CAPTURE_PATTERN.match(_normalize_text(text))
    if match is None:
        return None
    return match.group("prefix") or None


def _extract_racecard_date(soup: BeautifulSoup) -> str | None:
    date_container = soup.find("div", class_="RaceList_Date")
    if date_container is None:
        return None

    active_day = date_container.find("dd", class_=re.compile(r"\bActive\b"))
    active_link = active_day.find("a", href=True) if isinstance(active_day, Tag) else None
    if active_link is None:
        active_link = date_container.find("a", href=RACECARD_DATE_PARAM_PATTERN)
    if active_link is None:
        return None

    href = str(active_link.get("href", ""))
    match = RACECARD_DATE_PARAM_PATTERN.search(href)
    if match is None:
        return None

    value = match.group("date")
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def _extract_race_title(soup: BeautifulSoup) -> str | None:
    title_tag = soup.find("title")
    title_text = _normalize_text(title_tag.get_text(" ", strip=True) if title_tag is not None else "")
    if not title_text:
        return None

    title_text = re.split(r"\s*[｜|]\s*", title_text, maxsplit=1)[0]

    title_text = re.sub(r"\s*出馬表\s*$", "", title_text)
    title_text = re.sub(r"\s*-\s*netkeiba\s*$", "", title_text)
    return title_text or None


def _extract_class_condition(*texts: str | None) -> str | None:
    for raw_text in texts:
        text = _normalize_text(raw_text).replace("サラ系", "")
        if not text:
            continue
        match = CONDITION_WITH_AGE_PATTERN.search(text)
        if match is not None:
            return f"{match.group('age_group')}{match.group('class_token')}"

    for raw_text in texts:
        text = _normalize_text(raw_text)
        if not text:
            continue
        match = CLASS_TOKEN_PATTERN.search(text)
        if match is not None:
            return match.group("class_token")

    return None


def _normalize_stakes_grade(value: str | None) -> str | None:
    if value is None:
        return None
    text = _normalize_text(value).upper()
    if not text:
        return None

    if text.startswith("JG"):
        text = text.replace("JG", "J.G", 1)

    replacements = {
        "J.GIII": "J.G3",
        "J.GII": "J.G2",
        "J.GI": "J.G1",
        "GIII": "G3",
        "GII": "G2",
        "GI": "G1",
    }
    return replacements.get(text, text)


def _extract_stakes_grade(*texts: str | None) -> str | None:
    for raw_text in texts:
        text = _normalize_text(raw_text)
        if not text:
            continue
        match = RACE_GRADE_PATTERN.search(text)
        if match is not None:
            return _normalize_stakes_grade(match.group("grade"))
    return None


def _extract_surface(text: str | None) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    match = SURFACE_PATTERN.search(normalized)
    if match is None:
        return None
    surface = match.group("surface")
    if surface == "ダ":
        return "ダート"
    return surface


def _extract_course_direction(text: str | None) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for token in ("直線", "右", "左"):
        if token in normalized:
            return token
    return None


def _extract_course_side(text: str | None) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    for token in ("外内", "内外", "外", "内", "襷"):
        if re.search(fr"(?:^|[\s(/]){token}(?:$|[\s)A-D])", normalized):
            return token
    return None


def _extract_race_metadata(soup: BeautifulSoup) -> dict[str, Any]:
    intro = soup.find("div", class_="data_intro")
    intro_text = _normalize_text(intro.get_text(" ", strip=True) if intro is not None else "")
    smalltxt = soup.find("p", class_="smalltxt")
    smalltxt_text = _normalize_text(smalltxt.get_text(" ", strip=True) if smalltxt is not None else "")
    race_data_01 = soup.find("div", class_="RaceData01")
    race_data_01_text = _normalize_text(race_data_01.get_text(" ", strip=True) if race_data_01 is not None else "")
    race_data_02 = soup.find("div", class_="RaceData02")
    race_data_02_text = _normalize_text(race_data_02.get_text(" ", strip=True) if race_data_02 is not None else "")
    race_name_node = soup.find("h1", class_="RaceName")
    if race_name_node is None and intro is not None:
        race_name_node = intro.find("h1")
    race_name_text = _normalize_text(race_name_node.get_text(" ", strip=True) if race_name_node is not None else "")
    race_title = _extract_race_title(soup)
    page_text = " ".join(
        text
        for text in [intro_text, smalltxt_text, race_data_01_text, race_data_02_text, race_name_text, race_title]
        if text
    ).strip()

    date_match = DATE_PATTERN.search(page_text)
    date_value = None
    if date_match is not None:
        date_value = (
            f"{int(date_match.group('year')):04d}-"
            f"{int(date_match.group('month')):02d}-"
            f"{int(date_match.group('day')):02d}"
        )
    if date_value is None:
        date_value = _extract_racecard_date(soup)

    distance_match = DISTANCE_PATTERN.search(intro_text) or DISTANCE_PATTERN.search(race_data_01_text)
    weather_match = WEATHER_PATTERN.search(intro_text) or WEATHER_PATTERN.search(race_data_01_text)
    ground_match = (
        GROUND_PATTERN.search(intro_text)
        or GROUND_PATTERN.search(race_data_01_text)
        or RACECARD_GROUND_PATTERN.search(race_data_01_text)
    )
    venue_text = smalltxt_text or race_data_02_text or page_text
    venue = next((venue_name for venue_name in JRA_VENUES if venue_name in venue_text), None)
    course_text = intro_text or race_data_01_text
    surface = _extract_surface(course_text)

    return {
        "date": date_value,
        "track": venue,
        "distance": distance_match.group("distance") if distance_match is not None else None,
        "weather": weather_match.group("weather") if weather_match is not None else None,
        "ground_condition": ground_match.group("ground") if ground_match is not None else None,
        "競争条件": _extract_class_condition(smalltxt_text, race_data_02_text, race_title),
        "リステッド・重賞競走": _extract_stakes_grade(race_name_text, race_title),
        "芝・ダート区分": surface,
        "芝・ダート区分2": None,
        "右左回り・直線区分": _extract_course_direction(course_text),
        "内・外・襷区分": _extract_course_side(course_text),
    }


def _find_table_with_classes(soup: BeautifulSoup, required_classes: list[str]) -> Tag | None:
    required = set(required_classes)
    for table in soup.find_all("table"):
        classes = set(table.get("class") or [])
        if required.issubset(classes):
            return table
    return None


def _extract_mobile_race_metadata(soup: BeautifulSoup, race_id: str) -> dict[str, Any]:
    page_text = _normalize_text(soup.get_text(" ", strip=True))
    title_text = _normalize_text(soup.title.get_text(" ", strip=True) if soup.title is not None else "")
    year = race_id[:4]

    date_value = None
    track = None
    date_match = MOBILE_DATE_PATTERN.search(page_text)
    if date_match is not None:
        month = int(date_match.group("month"))
        day = int(date_match.group("day"))
        date_value = f"{year}-{month:02d}-{day:02d}"
    if date_value is None:
        canonical_date_match = DATE_PATTERN.search(title_text) or DATE_PATTERN.search(page_text)
        if canonical_date_match is not None:
            date_value = (
                f"{int(canonical_date_match.group('year')):04d}-"
                f"{int(canonical_date_match.group('month')):02d}-"
                f"{int(canonical_date_match.group('day')):02d}"
            )
    track_match = MOBILE_TRACK_PATTERN.search(page_text)
    if track_match is not None:
        track = track_match.group("track")
    if track is None:
        track_source_text = title_text or page_text
        track = next((venue_name for venue_name in JRA_VENUES if venue_name in track_source_text), None)

    info_match = MOBILE_RACE_INFO_PATTERN.search(page_text)
    condition = None
    surface = None
    distance = None
    weather = None
    ground = None
    direction = None
    if info_match is not None:
        surface = info_match.group("surface")
        if surface == "ダ":
            surface = "ダート"
        distance = info_match.group("distance")
        weather = info_match.group("weather")
        ground = info_match.group("ground")
        direction = info_match.group("direction")

    race_data = soup.find("div", class_="Race_Data")
    race_data_text = _normalize_text(race_data.get_text(" ", strip=True) if race_data is not None else "")
    if surface is None and race_data is not None:
        if race_data.find("span", class_="Turf") is not None:
            surface = "芝"
        elif race_data.find("span", class_="Dirt") is not None:
            surface = "ダート"
    if distance is None:
        distance_match = DISTANCE_PATTERN.search(race_data_text)
        if distance_match is not None:
            distance = distance_match.group("distance")
    if weather is None and race_data is not None:
        weather_tag = race_data.find("span", class_="WeatherData")
        weather_text = _normalize_text(weather_tag.get_text(" ", strip=True) if weather_tag is not None else "")
        weather_match = re.search(r"(晴|曇|雨|雪)", weather_text)
        if weather_match is not None:
            weather = weather_match.group(1)
    if ground is None and race_data is not None:
        ground_tag = race_data.find("span", class_="Item03")
        ground_text = _normalize_text(ground_tag.get_text(" ", strip=True) if ground_tag is not None else "")
        ground_match = re.search(r"(良|稍重|重|不良)", ground_text)
        if ground_match is not None:
            ground = ground_match.group(1)
    if direction is None:
        direction_match = re.search(r"\((?P<direction>右|左|直線)", race_data_text)
        if direction_match is not None:
            direction = direction_match.group("direction")

    for heading in soup.find_all("h1"):
        heading_text = _normalize_text(heading.get_text(" ", strip=True))
        if heading_text and heading_text != "netkeiba":
            condition = heading_text
            break

    return {
        "date": date_value,
        "track": track,
        "distance": distance,
        "weather": weather,
        "ground_condition": ground,
        "競争条件": condition,
        "リステッド・重賞競走": _extract_stakes_grade(page_text),
        "芝・ダート区分": surface,
        "芝・ダート区分2": None,
        "右左回り・直線区分": direction,
        "内・外・襷区分": None,
    }


def _parse_mobile_result_runner_text(text: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    match = MOBILE_RESULT_CARRY_PATTERN.search(normalized)
    if match is None:
        return {}
    return {
        "sex": match.group("sex"),
        "age": int(match.group("age")),
        "weight": match.group("weight").replace(" ", ""),
        "jockey_id": _normalize_text(match.group("jockey")),
        "斤量": match.group("carried"),
        "東西・外国・地方区分": match.group("trainer_region"),
        "trainer_id": _normalize_text(match.group("trainer")),
    }


def _parse_mobile_result_time_cell(text: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    output: dict[str, Any] = {}
    time_match = MOBILE_TIME_PATTERN.search(normalized)
    closing_match = MOBILE_CLOSING_PATTERN.search(normalized)
    if time_match is not None:
        output["finish_time"] = time_match.group("time")
    if closing_match is not None:
        output["closing_time_3f"] = closing_match.group("closing")
    return output


def _parse_mobile_odds_cell(text: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    match = MOBILE_ODDS_PATTERN.search(normalized)
    if match is None:
        return {}
    return {
        "odds": match.group("odds"),
        "popularity": match.group("popularity"),
    }


def _parse_mobile_odds_tag(cell: Tag | None) -> dict[str, Any]:
    if cell is None:
        return {}

    odds_tag = cell.find("dt", class_="Odds_Ninki")
    if odds_tag is None:
        odds_tag = cell.find("dt")
    popularity_tag = cell.find("dd")
    odds = _normalize_text(odds_tag.get_text(" ", strip=True) if odds_tag is not None else "")
    popularity_text = _normalize_text(popularity_tag.get_text(" ", strip=True) if popularity_tag is not None else "")

    output: dict[str, Any] = {}
    if odds:
        odds_match = re.search(r"\d+(?:\.\d+)?", odds)
        if odds_match is not None:
            output["odds"] = odds_match.group(0)
    if popularity_text:
        popularity_match = re.search(r"(\d+)", popularity_text)
        if popularity_match is not None:
            output["popularity"] = popularity_match.group(1)

    if output:
        return output
    return _parse_mobile_odds_cell(cell.get_text(" ", strip=True))


def parse_netkeiba_race_result_mobile_html(html: str, race_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = _find_table_with_classes(soup, ["RaceCommon_Table", "ResultRefund"])
    if table is None:
        raise ValueError(f"Mobile race result table not found for race_id={race_id}")

    metadata = _extract_mobile_race_metadata(soup, race_id)
    rows: list[dict[str, Any]] = []
    for row_tag in table.find_all("tr")[1:]:
        cells = row_tag.find_all("td", recursive=False)
        if len(cells) < 5:
            continue

        rank_text = _normalize_text(cells[0].get_text(" ", strip=True)).replace(" 着", "")
        frame_no = _normalize_text(cells[1].get_text(" ", strip=True))
        gate_no = _normalize_text(cells[2].get_text(" ", strip=True))
        horse_text = _normalize_text(cells[3].get_text(" ", strip=True))
        horse_name = _extract_anchor_text(cells[3], HORSE_LINK_PATTERN)
        horse_key = _extract_anchor_id(cells[3], HORSE_LINK_PATTERN)

        row: dict[str, Any] = {
            **metadata,
            "race_id": race_id,
            "horse_id": _build_race_horse_id(race_id, gate_no),
            "horse_key": horse_key,
            "horse_name": horse_name,
            "rank": rank_text,
            "frame_no": frame_no,
            "gate_no": gate_no,
        }
        row.update(_parse_mobile_result_runner_text(horse_text))
        row.update(_parse_mobile_result_time_cell(cells[4].get_text(" ", strip=True)))
        if len(cells) >= 6:
            row.update(_parse_mobile_odds_tag(cells[5]))
        rows.append(row)

    if not rows:
        raise ValueError(f"No mobile race result rows parsed for race_id={race_id}")
    return pd.DataFrame(rows)


def parse_netkeiba_race_card_mobile_past_html(html: str, race_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = _find_table_with_classes(soup, ["Shutuba_Table", "Shutuba_Past5_Table"])
    if table is None:
        raise ValueError(f"Mobile past race card table not found for race_id={race_id}")

    metadata = _extract_mobile_race_metadata(soup, race_id)
    rows: list[dict[str, Any]] = []
    for row_tag in table.find_all("tr")[1:]:
        cells = row_tag.find_all("td", recursive=False)
        if len(cells) < 4:
            continue

        gate_no = _normalize_text(cells[0].get_text(" ", strip=True))
        info_text = _normalize_text(cells[2].get_text(" ", strip=True))
        jockey_text = _normalize_text(cells[3].get_text(" ", strip=True))
        horse_name = _extract_anchor_text(cells[2], HORSE_LINK_PATTERN)
        horse_key = _extract_anchor_id(cells[2], HORSE_LINK_PATTERN)
        trainer_text = _extract_anchor_text(cells[2], TRAINER_LINK_PATTERN)
        trainer_region = None
        trainer_name = None
        if trainer_text and "・" in trainer_text:
            trainer_region, trainer_name = trainer_text.split("・", 1)

        jockey_name = _extract_anchor_text(cells[3], JOCKEY_LINK_PATTERN)
        sex_age_match = MOBILE_CARD_JOCKEY_PATTERN.search(jockey_text)
        odds_match = MOBILE_CARD_ODDS_PATTERN.search(info_text)
        weight_match = MOBILE_RESULT_WEIGHT_PATTERN.search(info_text)

        row: dict[str, Any] = {
            **metadata,
            "race_id": race_id,
            "horse_id": _build_race_horse_id(race_id, gate_no),
            "horse_key": horse_key,
            "horse_name": horse_name,
            "frame_no": None,
            "gate_no": gate_no,
            "weight": weight_match.group("weight").replace(" ", "") if weight_match is not None else None,
            "jockey_id": jockey_name,
            "jockey_key": _extract_anchor_id(cells[3], JOCKEY_LINK_PATTERN),
            "東西・外国・地方区分": trainer_region,
            "trainer_id": trainer_name,
            "trainer_key": _extract_anchor_id(cells[2], TRAINER_LINK_PATTERN),
        }
        if sex_age_match is not None:
            row["sex"] = sex_age_match.group("sex")
            row["age"] = int(sex_age_match.group("age"))
            row["斤量"] = sex_age_match.group("carried")
            if not row.get("jockey_id"):
                row["jockey_id"] = _normalize_text(sex_age_match.group("jockey"))
        if odds_match is not None and odds_match.group("odds") != "---.--":
            row["odds"] = odds_match.group("odds")
        if odds_match is not None and odds_match.group("popularity") != "**":
            row["popularity"] = odds_match.group("popularity")
        rows.append(row)

    if not rows:
        raise ValueError(f"No mobile past race card rows parsed for race_id={race_id}")
    return pd.DataFrame(rows)


def _build_race_horse_id(race_id: str, gate_no: str | None) -> str | None:
    if gate_no is None:
        return None
    match = re.search(r"(\d+)", gate_no)
    if match is None:
        return None
    return f"{race_id}{int(match.group(1)):02d}"


def _parse_sex_age(value: str | None) -> tuple[str | None, int | None]:
    if value is None:
        return None, None
    text = _normalize_text(value)
    match = re.match(r"(?P<sex>.)(?P<age>\d+)", text)
    if match is None:
        return None, None
    return match.group("sex"), int(match.group("age"))


def _expand_passing_order(value: str | None) -> dict[str, Any]:
    positions = re.findall(r"\d+", str(value or ""))
    output: dict[str, Any] = {"passing_order_raw": _normalize_text(value)} if value else {}
    for index, token in enumerate(positions[:4], start=1):
        output[f"corner_{index}_position"] = token
    return output


def _find_required_table(soup: BeautifulSoup, *, summary_pattern: str | None = None, class_name: str | None = None) -> Tag:
    table: Tag | None = None
    if summary_pattern is not None:
        table = soup.find("table", attrs={"summary": re.compile(summary_pattern)})
    if table is None and class_name is not None:
        table = soup.find("table", class_=re.compile(class_name))
    if table is None:
        raise ValueError(f"Required table not found: summary_pattern={summary_pattern}, class_name={class_name}")
    return table


def parse_netkeiba_race_result_html(html: str, race_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = _find_required_table(soup, summary_pattern="レース結果", class_name="race_table_01")
    header_row = table.find("tr")
    if header_row is None:
        raise ValueError("Race result header row not found")

    headers = [_normalize_header(th.get_text(" ", strip=True)) for th in header_row.find_all("th")]
    metadata = _extract_race_metadata(soup)

    header_map = {
        "着順": "rank",
        "枠番": "frame_no",
        "馬番": "gate_no",
        "馬名": "horse_name",
        "性齢": "sex_age",
        "斤量": "斤量",
        "騎手": "jockey_id",
        "タイム": "finish_time",
        "通過": "passing_order",
        "上り": "closing_time_3f",
        "単勝": "odds",
        "人気": "popularity",
        "馬体重": "weight",
        "調教師": "trainer_id",
        "馬主": "owner_name",
    }

    rows: list[dict[str, Any]] = []
    for row_tag in table.find_all("tr")[1:]:
        cells = row_tag.find_all("td")
        if not cells:
            continue

        cell_values = [_normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
        if len(cell_values) < 5:
            continue

        raw_row = {
            header_map[header]: cell_values[index]
            for index, header in enumerate(headers[:len(cell_values)])
            if header in header_map
        }

        sex, age = _parse_sex_age(raw_row.get("sex_age"))
        horse_key = _extract_anchor_id(row_tag, HORSE_LINK_PATTERN)
        horse_name = _extract_anchor_text(row_tag, HORSE_LINK_PATTERN) or raw_row.get("horse_name")
        jockey_name = _extract_anchor_text(row_tag, JOCKEY_LINK_PATTERN) or raw_row.get("jockey_id")
        trainer_raw_text = raw_row.get("trainer_id")
        trainer_name = _extract_anchor_text(row_tag, TRAINER_LINK_PATTERN) or trainer_raw_text
        owner_name = _extract_anchor_text(row_tag, OWNER_LINK_PATTERN) or raw_row.get("owner_name")
        trainer_region = _extract_bracket_prefix(trainer_raw_text or trainer_name)

        row: dict[str, Any] = {
            **metadata,
            "race_id": race_id,
            "horse_id": _build_race_horse_id(race_id, raw_row.get("gate_no")),
            "horse_key": horse_key,
            "horse_name": horse_name,
            "rank": raw_row.get("rank"),
            "frame_no": raw_row.get("frame_no"),
            "gate_no": raw_row.get("gate_no"),
            "sex": sex,
            "age": age,
            "斤量": raw_row.get("斤量"),
            "weight": raw_row.get("weight"),
            "jockey_id": jockey_name,
            "jockey_key": _extract_anchor_id(row_tag, JOCKEY_LINK_PATTERN),
            "東西・外国・地方区分": trainer_region,
            "trainer_id": _strip_bracket_prefix(trainer_raw_text or trainer_name),
            "trainer_key": _extract_anchor_id(row_tag, TRAINER_LINK_PATTERN),
            "owner_name": owner_name,
            "owner_key": _extract_anchor_id(row_tag, OWNER_LINK_PATTERN),
            "finish_time": raw_row.get("finish_time"),
            "closing_time_3f": raw_row.get("closing_time_3f"),
            "odds": raw_row.get("odds"),
            "popularity": raw_row.get("popularity"),
        }
        row.update(_expand_passing_order(raw_row.get("passing_order")))
        rows.append(row)

    if not rows:
        raise ValueError(f"No race result rows parsed for race_id={race_id}")
    return pd.DataFrame(rows)


def parse_netkeiba_race_card_html(html: str, race_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = _find_required_table(soup, class_name="ShutubaTable|Shutuba_Table")
    header_row = table.find("tr")
    if header_row is None:
        raise ValueError("Race card header row not found")

    headers = [_normalize_header(th.get_text(" ", strip=True)) for th in header_row.find_all("th")]
    metadata = _extract_race_metadata(soup)

    header_map = {
        "枠": "frame_no",
        "馬番": "gate_no",
        "馬名": "horse_name",
        "性齢": "sex_age",
        "斤量": "斤量",
        "騎手": "jockey_id",
        "厩舎": "trainer_id",
        "馬体重(増減)": "weight",
        "馬体重": "weight",
    }

    rows: list[dict[str, Any]] = []
    row_tags = table.find_all("tr", class_=re.compile(r"\bHorseList\b"))
    if not row_tags:
        row_tags = table.find_all("tr")[1:]

    for row_tag in row_tags:
        cells = row_tag.find_all("td", recursive=False)
        if not cells:
            continue

        cell_values = [_normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
        if len(cell_values) < 7:
            continue

        raw_row = {
            header_map[header]: cell_values[index]
            for index, header in enumerate(headers[:len(cell_values)])
            if header in header_map
        }

        sex, age = _parse_sex_age(raw_row.get("sex_age"))
        horse_key = _extract_anchor_id(row_tag, HORSE_LINK_PATTERN)
        horse_name = _extract_anchor_text(row_tag, HORSE_LINK_PATTERN) or raw_row.get("horse_name")
        jockey_name = _extract_anchor_text(row_tag, JOCKEY_LINK_PATTERN) or raw_row.get("jockey_id")
        trainer_raw_text = raw_row.get("trainer_id")
        trainer_name = _extract_anchor_text(row_tag, TRAINER_LINK_PATTERN) or trainer_raw_text
        trainer_region = _extract_bracket_prefix(trainer_raw_text or trainer_name)

        rows.append(
            {
                **metadata,
                "race_id": race_id,
                "horse_id": _build_race_horse_id(race_id, raw_row.get("gate_no")),
                "horse_key": horse_key,
                "horse_name": horse_name,
                "frame_no": raw_row.get("frame_no"),
                "gate_no": raw_row.get("gate_no"),
                "sex": sex,
                "age": age,
                "斤量": raw_row.get("斤量"),
                "weight": raw_row.get("weight"),
                "jockey_id": jockey_name,
                "jockey_key": _extract_anchor_id(row_tag, JOCKEY_LINK_PATTERN),
                "東西・外国・地方区分": trainer_region,
                "trainer_id": _strip_bracket_prefix(trainer_raw_text or trainer_name),
                "trainer_key": _extract_anchor_id(row_tag, TRAINER_LINK_PATTERN),
            }
        )

    if not rows:
        raise ValueError(f"No race card rows parsed for race_id={race_id}")

    frame = pd.DataFrame(rows)
    dedupe_columns = [column for column in ["race_id", "horse_id"] if column in frame.columns]
    if dedupe_columns:
        frame = frame.drop_duplicates(subset=dedupe_columns, keep="first").reset_index(drop=True)
    return frame


def _extract_horse_name(profile_soup: BeautifulSoup, horse_key: str) -> str | None:
    title = profile_soup.find("title")
    title_text = _normalize_text(title.get_text(" ", strip=True) if title is not None else "")
    if title_text:
        match = re.match(r"(?P<name>[^|(]+?)(?:\s*\(|の競走馬データ)", title_text)
        if match is not None:
            return _normalize_text(match.group("name"))

    header_link = profile_soup.find("a", href=re.compile(rf"/horse/{re.escape(horse_key)}/"))
    if header_link is None:
        heading = profile_soup.find("h1")
        heading_text = _normalize_text(heading.get_text(" ", strip=True) if heading is not None else "")
        return heading_text or None
    return _normalize_text(header_link.get_text(" ", strip=True))


def _extract_profile_label_map(profile_soup: BeautifulSoup) -> dict[str, str]:
    for table in profile_soup.find_all("table"):
        label_to_value: dict[str, str] = {}
        for row in table.find_all("tr"):
            label = row.find("th")
            value = row.find("td")
            if label is None or value is None:
                continue
            label_to_value[_normalize_header(label.get_text(" ", strip=True))] = _normalize_text(value.get_text(" ", strip=True))
        if {"馬主", "生産者(産地)", "父", "母"} & set(label_to_value):
            return label_to_value
    return {}


def _parse_profile_pedigree_fields(profile_html: str) -> dict[str, Any]:
    soup = BeautifulSoup(profile_html, "html.parser")
    label_to_value = _extract_profile_label_map(soup)
    sire_name = label_to_value.get("父")
    dam_raw = label_to_value.get("母")
    if sire_name is None or dam_raw is None:
        raise ValueError("Profile page does not contain fallback pedigree fields")

    damsire_name = None
    dam_name = dam_raw
    if "母父:" in dam_raw:
        dam_name, damsire_name = dam_raw.split("母父:", maxsplit=1)

    return {
        "sire_name": _normalize_text(sire_name) or None,
        "dam_name": _normalize_text(dam_name) or None,
        "damsire_name": _normalize_text(damsire_name) or None,
    }


def _parse_profile_fields(profile_html: str, horse_key: str) -> dict[str, Any]:
    soup = BeautifulSoup(profile_html, "html.parser")
    try:
        table = _find_required_table(soup, class_name="db_prof_table")
        label_to_value: dict[str, str] = {}
        for row in table.find_all("tr"):
            label = row.find("th")
            value = row.find("td")
            if label is None or value is None:
                continue
            label_to_value[_normalize_header(label.get_text(" ", strip=True))] = _normalize_text(value.get_text(" ", strip=True))
    except ValueError:
        label_to_value = _extract_profile_label_map(soup)

    breeder_name = label_to_value.get("生産者") or label_to_value.get("生産者(産地)")
    if breeder_name is not None:
        breeder_name = breeder_name.split("(", maxsplit=1)[0].strip()

    return {
        "horse_key": horse_key,
        "horse_name": _extract_horse_name(soup, horse_key),
        "owner_name": label_to_value.get("馬主"),
        "breeder_name": breeder_name,
    }


def _parse_pedigree_fields(pedigree_html: str) -> dict[str, Any]:
    soup = BeautifulSoup(pedigree_html, "html.parser")
    table = _find_required_table(soup, class_name="blood_table")
    rows = table.find_all("tr", recursive=False)
    if len(rows) < 3:
        raise ValueError("Pedigree table does not have enough rows")

    sire_cells = rows[0].find_all("td", recursive=False)
    dam_cells = rows[2].find_all("td", recursive=False)
    if len(sire_cells) < 1 or len(dam_cells) < 2:
        raise ValueError("Pedigree table structure is incomplete")

    return {
        "sire_name": _normalize_text(sire_cells[0].get_text(" ", strip=True)) or None,
        "dam_name": _normalize_text(dam_cells[0].get_text(" ", strip=True)) or None,
        "damsire_name": _normalize_text(dam_cells[1].get_text(" ", strip=True)) or None,
    }


def parse_netkeiba_pedigree_html(profile_html: str, pedigree_html: str, horse_key: str) -> pd.DataFrame:
    row = _parse_profile_fields(profile_html, horse_key)
    try:
        row.update(_parse_pedigree_fields(pedigree_html))
    except ValueError:
        row.update(_parse_profile_pedigree_fields(profile_html))
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

    if settings.parse_only:
        raise FileNotFoundError(f"Cached HTML not found for parse-only mode: {output_path}")

    if last_fetch_at is not None and settings.delay_sec > 0:
        elapsed = time.perf_counter() - last_fetch_at
        sleep_sec = settings.delay_sec - elapsed
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    html = _fetch_html(session, url, settings)
    write_text_file(output_path, html, label="raw html output")
    return html, True, time.perf_counter()


def _load_target_ids(id_file: Path, id_column: str, limit: int | None) -> list[str]:
    if not id_file.exists():
        raise FileNotFoundError(f"ID file not found: {id_file}")

    if id_file.suffix.lower() == ".csv":
        frame = pd.read_csv(id_file, dtype={id_column: "string"}, low_memory=False)
        if id_column not in frame.columns:
            raise ValueError(f"ID column '{id_column}' not found in {id_file}")
        values = frame[id_column].astype(str).tolist()
    else:
        values = id_file.read_text(encoding="utf-8").splitlines()

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


def _dedupe_frame(frame: pd.DataFrame, dedupe_on: list[str]) -> pd.DataFrame:
    columns = [column for column in dedupe_on if column in frame.columns]
    if not columns:
        return frame.reset_index(drop=True)
    work = frame.copy()
    dedupe_keys: list[str] = []
    for column in columns:
        key_column = f"__dedupe_key__{column}"
        normalized = work[column].where(work[column].notna(), "").astype(str).str.strip()
        normalized = normalized.replace({"nan": "", "None": "", "NaT": ""})
        work[key_column] = normalized
        dedupe_keys.append(key_column)
    work = work.drop_duplicates(subset=dedupe_keys, keep="first")
    return work.drop(columns=dedupe_keys).reset_index(drop=True)


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
        columns = [column for column in dedupe_on if column in combined.columns]
        if columns:
            dedupe_keys: list[str] = []
            for column in columns:
                key_column = f"__dedupe_key__{column}"
                normalized = combined[column].where(combined[column].notna(), "").astype(str).str.strip()
                normalized = normalized.replace({"nan": "", "None": "", "NaT": ""})
                combined[key_column] = normalized
                dedupe_keys.append(key_column)
            combined = combined.drop_duplicates(subset=dedupe_keys, keep="last")
            combined = combined.drop(columns=dedupe_keys)
        combined = combined.reset_index(drop=True)

    write_csv_file(output_file, combined, index=False, label="crawl output")
    return combined, existing_rows


def _build_checkpoint_batch_frame(frames: list[pd.DataFrame], dedupe_on: list[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    batch_frame = pd.concat(frames, ignore_index=True, sort=False)
    return _dedupe_frame(batch_frame, dedupe_on)


def _build_target_manifest_path(base_path: Path, target_name: str) -> Path:
    if not base_path.suffix:
        return base_path.parent / f"{base_path.name}_{target_name}.json"
    return base_path.with_name(f"{base_path.stem}_{target_name}{base_path.suffix}")


def _build_lock_path(base_path: Path) -> Path:
    if not base_path.suffix:
        return base_path.parent / f"{base_path.name}.lock"
    return base_path.with_name(f"{base_path.stem}{base_path.suffix}.lock")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    write_json(path, data)


def _build_target_report(
    *,
    target_name: str,
    started_at: str,
    status: str,
    requested_ids: int,
    processed_ids: int,
    parsed_ids: int,
    fetched_count: int,
    failures: list[dict[str, str]],
    output_file: Path,
    raw_html_path: Path,
    batch_rows_written: int | None = None,
    rows_written: int | None = None,
    existing_rows_merged: int | None = None,
    finished_at: str | None = None,
    process_id: int | None = None,
    lock_file: Path | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "started_at": started_at,
        "status": status,
        "target": target_name,
        "requested_ids": int(requested_ids),
        "processed_ids": int(processed_ids),
        "parsed_ids": int(parsed_ids),
        "fetched_count": int(fetched_count),
        "failure_count": len(failures),
        "failures": failures[:50],
        "output_file": str(output_file),
        "raw_html_dir": str(raw_html_path),
    }
    if process_id is not None:
        report["pid"] = int(process_id)
    if lock_file is not None:
        report["lock_file"] = str(lock_file)
    if batch_rows_written is not None:
        report["batch_rows_written"] = int(batch_rows_written)
    if rows_written is not None:
        report["rows_written"] = int(rows_written)
    if existing_rows_merged is not None:
        report["existing_rows_merged"] = int(existing_rows_merged)
    if finished_at is not None:
        report["finished_at"] = finished_at
    return report


def _build_crawl_summary(
    *,
    started_at: str,
    target_reports: list[dict[str, Any]],
    finished_at: str | None = None,
    status: str,
    process_id: int | None = None,
    lock_file: Path | None = None,
) -> dict[str, Any]:
    summary = {
        "source": "netkeiba",
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "targets": target_reports,
    }
    if process_id is not None:
        summary["pid"] = int(process_id)
    if lock_file is not None:
        summary["lock_file"] = str(lock_file)
    return summary


def _read_lock_payload(lock_path: Path) -> dict[str, Any]:
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def netkeiba_crawl_lock(config: dict[str, Any], *, base_dir: Path):
    crawl_cfg = config.get("crawl", config)
    manifest_base_path = _resolve_path(
        crawl_cfg.get("manifest_file", "artifacts/reports/netkeiba_crawl_manifest.json"),
        base_dir,
    )
    lock_path = _build_lock_path(manifest_base_path)
    lock_payload = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lock_file": str(lock_path),
    }

    while True:
        try:
            file_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as error:
            payload = _read_lock_payload(lock_path)
            pid = int(payload.get("pid", 0) or 0)
            if payload and not _pid_is_running(pid):
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass
            details = json.dumps(payload, ensure_ascii=False, indent=2) if payload else ""
            suffix = f" Existing lock details: {details}" if details else ""
            raise RuntimeError(f"netkeiba crawl is already running: {lock_path}.{suffix}") from error

    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(lock_payload, file, ensure_ascii=False, indent=2)
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _crawl_race_result_target(
    *,
    session: requests.Session,
    ids: list[str],
    output_file: Path,
    raw_html_dir: Path,
    dedupe_on: list[str],
    settings: RequestSettings,
    base_url: str,
    manifest_path: Path | None = None,
    checkpoint_interval: int = 0,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    progress = ProgressBar(total=max(len(ids), 1), prefix="[crawl:race_result]", min_interval_sec=0.0)
    progress.start("starting")

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    fetched_count = 0
    last_fetch_at: float | None = None
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    initial_existing_rows = int(len(_load_existing_output_frame(output_file)))
    current_rows_written = initial_existing_rows
    current_batch_rows = 0

    for index, race_id in enumerate(ids, start=1):
        raw_html_path = raw_html_dir / "race_result" / f"{_safe_filename(race_id)}.html"
        url = urljoin(base_url, f"/race/{race_id}/")
        try:
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
                frames.append(parse_netkeiba_race_result_html(html, race_id))
            except Exception:
                mobile_path = raw_html_dir / "race_result_mobile" / f"{_safe_filename(race_id)}.html"
                mobile_url = f"https://race.sp.netkeiba.com/?pid=race_result&race_id={race_id}"
                mobile_html, mobile_fetched, last_fetch_at = _load_or_fetch_html(
                    session=session,
                    url=mobile_url,
                    output_path=mobile_path,
                    settings=settings,
                    last_fetch_at=last_fetch_at,
                )
                if mobile_fetched:
                    fetched_count += 1
                frames.append(parse_netkeiba_race_result_mobile_html(mobile_html, race_id))
        except Exception as error:
            failures.append({"id": race_id, "stage": "race_result", "error": str(error)})

        if checkpoint_interval > 0 and (index % checkpoint_interval) == 0:
            checkpoint_frame = _build_checkpoint_batch_frame(frames, dedupe_on)
            combined, _ = _write_cumulative_output(output_file, checkpoint_frame, dedupe_on)
            current_batch_rows = int(len(checkpoint_frame))
            current_rows_written = int(len(combined))

        progress.update(current=index, message=f"parsed={len(frames)} failed={len(failures)}")
        if manifest_path is not None:
            _write_json(
                manifest_path,
                _build_target_report(
                    target_name="race_result",
                    started_at=started_at,
                    status="running",
                    requested_ids=len(ids),
                    processed_ids=index,
                    parsed_ids=len(frames),
                    fetched_count=fetched_count,
                    failures=failures,
                    output_file=output_file,
                    raw_html_path=raw_html_dir / "race_result",
                    batch_rows_written=current_batch_rows,
                    rows_written=current_rows_written,
                    existing_rows_merged=initial_existing_rows,
                    process_id=os.getpid(),
                    lock_file=lock_path,
                ),
            )

    batch_frame = _build_checkpoint_batch_frame(frames, dedupe_on)
    combined, _ = _write_cumulative_output(output_file, batch_frame, dedupe_on)

    report = _build_target_report(
        target_name="race_result",
        started_at=started_at,
        status="completed",
        requested_ids=len(ids),
        processed_ids=len(ids),
        parsed_ids=len(frames),
        fetched_count=fetched_count,
        failures=failures,
        output_file=output_file,
        raw_html_path=raw_html_dir / "race_result",
        batch_rows_written=len(batch_frame),
        rows_written=len(combined),
        existing_rows_merged=initial_existing_rows,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        process_id=os.getpid(),
        lock_file=lock_path,
    )
    if manifest_path is not None:
        _write_json(manifest_path, report)
    return report


def _crawl_race_card_target(
    *,
    session: requests.Session,
    ids: list[str],
    output_file: Path,
    raw_html_dir: Path,
    dedupe_on: list[str],
    settings: RequestSettings,
    base_url: str,
    manifest_path: Path | None = None,
    checkpoint_interval: int = 0,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    progress = ProgressBar(total=max(len(ids), 1), prefix="[crawl:race_card]", min_interval_sec=0.0)
    progress.start("starting")

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    fetched_count = 0
    last_fetch_at: float | None = None
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    initial_existing_rows = int(len(_load_existing_output_frame(output_file)))
    current_rows_written = initial_existing_rows
    current_batch_rows = 0

    for index, race_id in enumerate(ids, start=1):
        raw_html_path = raw_html_dir / "race_card" / f"{_safe_filename(race_id)}.html"
        url = urljoin(base_url, f"/race/shutuba.html?race_id={race_id}")
        try:
            mobile_past_path = raw_html_dir / "race_card_mobile_past" / f"{_safe_filename(race_id)}.html"
            mobile_past_url = f"https://race.sp.netkeiba.com/?pid=shutuba_past&race_id={race_id}"
            html, fetched, last_fetch_at = _load_or_fetch_html(
                session=session,
                url=mobile_past_url,
                output_path=mobile_past_path,
                settings=settings,
                last_fetch_at=last_fetch_at,
            )
            if fetched:
                fetched_count += 1
            try:
                frames.append(parse_netkeiba_race_card_mobile_past_html(html, race_id))
            except Exception:
                html, fetched, last_fetch_at = _load_or_fetch_html(
                    session=session,
                    url=url,
                    output_path=raw_html_path,
                    settings=settings,
                    last_fetch_at=last_fetch_at,
                )
                if fetched:
                    fetched_count += 1
                try:
                    frames.append(parse_netkeiba_race_card_html(html, race_id))
                except Exception:
                    mobile_path = raw_html_dir / "race_card_mobile" / f"{_safe_filename(race_id)}.html"
                    mobile_url = f"https://race.sp.netkeiba.com/race/shutuba.html?race_id={race_id}"
                    mobile_html, mobile_fetched, last_fetch_at = _load_or_fetch_html(
                        session=session,
                        url=mobile_url,
                        output_path=mobile_path,
                        settings=settings,
                        last_fetch_at=last_fetch_at,
                    )
                    if mobile_fetched:
                        fetched_count += 1
                    frames.append(parse_netkeiba_race_card_mobile_past_html(mobile_html, race_id))
        except Exception as error:
            failures.append({"id": race_id, "stage": "race_card", "error": str(error)})

        if checkpoint_interval > 0 and (index % checkpoint_interval) == 0:
            checkpoint_frame = _build_checkpoint_batch_frame(frames, dedupe_on)
            combined, _ = _write_cumulative_output(output_file, checkpoint_frame, dedupe_on)
            current_batch_rows = int(len(checkpoint_frame))
            current_rows_written = int(len(combined))

        progress.update(current=index, message=f"parsed={len(frames)} failed={len(failures)}")
        if manifest_path is not None:
            _write_json(
                manifest_path,
                _build_target_report(
                    target_name="race_card",
                    started_at=started_at,
                    status="running",
                    requested_ids=len(ids),
                    processed_ids=index,
                    parsed_ids=len(frames),
                    fetched_count=fetched_count,
                    failures=failures,
                    output_file=output_file,
                    raw_html_path=raw_html_dir / "race_card",
                    batch_rows_written=current_batch_rows,
                    rows_written=current_rows_written,
                    existing_rows_merged=initial_existing_rows,
                    process_id=os.getpid(),
                    lock_file=lock_path,
                ),
            )

    batch_frame = _build_checkpoint_batch_frame(frames, dedupe_on)
    combined, _ = _write_cumulative_output(output_file, batch_frame, dedupe_on)

    report = _build_target_report(
        target_name="race_card",
        started_at=started_at,
        status="completed",
        requested_ids=len(ids),
        processed_ids=len(ids),
        parsed_ids=len(frames),
        fetched_count=fetched_count,
        failures=failures,
        output_file=output_file,
        raw_html_path=raw_html_dir / "race_card",
        batch_rows_written=len(batch_frame),
        rows_written=len(combined),
        existing_rows_merged=initial_existing_rows,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        process_id=os.getpid(),
        lock_file=lock_path,
    )
    if manifest_path is not None:
        _write_json(manifest_path, report)
    return report


def _crawl_pedigree_target(
    *,
    session: requests.Session,
    ids: list[str],
    output_file: Path,
    raw_html_dir: Path,
    dedupe_on: list[str],
    settings: RequestSettings,
    base_url: str,
    manifest_path: Path | None = None,
    checkpoint_interval: int = 0,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    progress = ProgressBar(total=max(len(ids), 1), prefix="[crawl:pedigree]", min_interval_sec=0.0)
    progress.start("starting")

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    fetched_count = 0
    last_fetch_at: float | None = None
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    initial_existing_rows = int(len(_load_existing_output_frame(output_file)))
    current_rows_written = initial_existing_rows
    current_batch_rows = 0

    for index, horse_key in enumerate(ids, start=1):
        # Pedigree runs can span many hours; use a fresh session per horse to avoid
        # carrying cookies or connection state across thousands of profile fetches.
        horse_session = _build_session(settings)
        profile_path = raw_html_dir / "pedigree" / f"{_safe_filename(horse_key)}.profile.html"
        pedigree_path = raw_html_dir / "pedigree" / f"{_safe_filename(horse_key)}.pedigree.html"
        mobile_profile_path = raw_html_dir / "pedigree" / f"{_safe_filename(horse_key)}.profile.mobile.html"
        mobile_pedigree_path = raw_html_dir / "pedigree" / f"{_safe_filename(horse_key)}.pedigree.mobile.html"
        profile_url = urljoin(base_url, f"/horse/{horse_key}/")
        pedigree_url = urljoin(base_url, f"/horse/ajax_horse_pedigree.html?id={horse_key}")
        mobile_profile_url = f"https://db.sp.netkeiba.com/horse/{horse_key}/"
        mobile_pedigree_url = f"https://db.sp.netkeiba.com/horse/ped/{horse_key}/"
        try:
            def _fetch_and_parse_desktop() -> None:
                nonlocal last_fetch_at, fetched_count
                profile_html, fetched_profile, last_fetch_at = _load_or_fetch_html(
                    session=horse_session,
                    url=profile_url,
                    output_path=profile_path,
                    settings=settings,
                    last_fetch_at=last_fetch_at,
                )
                if fetched_profile:
                    fetched_count += 1
                pedigree_html, fetched_pedigree, last_fetch_at = _load_or_fetch_html(
                    session=horse_session,
                    url=pedigree_url,
                    output_path=pedigree_path,
                    settings=settings,
                    last_fetch_at=last_fetch_at,
                )
                if fetched_pedigree:
                    fetched_count += 1
                frames.append(parse_netkeiba_pedigree_html(profile_html, pedigree_html, horse_key))

            def _fetch_and_parse_mobile() -> None:
                nonlocal last_fetch_at, fetched_count
                profile_html, fetched_profile, last_fetch_at = _load_or_fetch_html(
                    session=horse_session,
                    url=mobile_profile_url,
                    output_path=mobile_profile_path,
                    settings=settings,
                    last_fetch_at=last_fetch_at,
                )
                if fetched_profile:
                    fetched_count += 1
                pedigree_html, fetched_pedigree, last_fetch_at = _load_or_fetch_html(
                    session=horse_session,
                    url=mobile_pedigree_url,
                    output_path=mobile_pedigree_path,
                    settings=settings,
                    last_fetch_at=last_fetch_at,
                )
                if fetched_pedigree:
                    fetched_count += 1
                frames.append(parse_netkeiba_pedigree_html(profile_html, pedigree_html, horse_key))

            if settings.pedigree_prefer_mobile:
                try:
                    _fetch_and_parse_mobile()
                except Exception as mobile_error:
                    try:
                        _fetch_and_parse_desktop()
                    except Exception as desktop_error:
                        raise RuntimeError(f"mobile={mobile_error}; desktop={desktop_error}") from desktop_error
            else:
                try:
                    _fetch_and_parse_desktop()
                except Exception as desktop_error:
                    try:
                        _fetch_and_parse_mobile()
                    except Exception as mobile_error:
                        raise RuntimeError(f"desktop={desktop_error}; mobile={mobile_error}") from mobile_error
        except Exception as error:
            failures.append({"id": horse_key, "stage": "pedigree", "error": str(error)})
        finally:
            horse_session.close()

        if checkpoint_interval > 0 and (index % checkpoint_interval) == 0:
            checkpoint_frame = _build_checkpoint_batch_frame(frames, dedupe_on)
            combined, _ = _write_cumulative_output(output_file, checkpoint_frame, dedupe_on)
            current_batch_rows = int(len(checkpoint_frame))
            current_rows_written = int(len(combined))

        progress.update(current=index, message=f"parsed={len(frames)} failed={len(failures)}")
        if manifest_path is not None:
            _write_json(
                manifest_path,
                _build_target_report(
                    target_name="pedigree",
                    started_at=started_at,
                    status="running",
                    requested_ids=len(ids),
                    processed_ids=index,
                    parsed_ids=len(frames),
                    fetched_count=fetched_count,
                    failures=failures,
                    output_file=output_file,
                    raw_html_path=raw_html_dir / "pedigree",
                    batch_rows_written=current_batch_rows,
                    rows_written=current_rows_written,
                    existing_rows_merged=initial_existing_rows,
                    process_id=os.getpid(),
                    lock_file=lock_path,
                ),
            )

    batch_frame = _build_checkpoint_batch_frame(frames, dedupe_on)
    combined, _ = _write_cumulative_output(output_file, batch_frame, dedupe_on)

    report = _build_target_report(
        target_name="pedigree",
        started_at=started_at,
        status="completed",
        requested_ids=len(ids),
        processed_ids=len(ids),
        parsed_ids=len(frames),
        fetched_count=fetched_count,
        failures=failures,
        output_file=output_file,
        raw_html_path=raw_html_dir / "pedigree",
        batch_rows_written=len(batch_frame),
        rows_written=len(combined),
        existing_rows_merged=initial_existing_rows,
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        process_id=os.getpid(),
        lock_file=lock_path,
    )
    if manifest_path is not None:
        _write_json(manifest_path, report)
    return report


def crawl_netkeiba_from_config(
    config: dict[str, Any],
    *,
    base_dir: Path,
    target_filter: str | None = None,
    override_limit: int | None = None,
    refresh: bool = False,
    parse_only: bool = False,
    use_lock: bool = True,
) -> dict[str, Any]:
    if use_lock:
        with netkeiba_crawl_lock(config, base_dir=base_dir):
            return crawl_netkeiba_from_config(
                config,
                base_dir=base_dir,
                target_filter=target_filter,
                override_limit=override_limit,
                refresh=refresh,
                parse_only=parse_only,
                use_lock=False,
            )

    crawl_cfg = config.get("crawl", config)
    targets = crawl_cfg.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("crawl.targets must contain at least one target")

    settings = RequestSettings(
        base_url=str(crawl_cfg.get("base_url", "https://db.netkeiba.com")).rstrip("/"),
        user_agent=str(crawl_cfg.get("user_agent", "nr-learn-netkeiba-crawler/0.1")),
        timeout_sec=float(crawl_cfg.get("timeout_sec", 20.0)),
        delay_sec=float(crawl_cfg.get("delay_sec", 1.5)),
        retry_count=max(int(crawl_cfg.get("retry_count", 3)), 1),
        retry_backoff_sec=float(crawl_cfg.get("retry_backoff_sec", 2.0)),
        overwrite=bool(crawl_cfg.get("overwrite", False) or refresh),
        parse_only=bool(parse_only),
        pedigree_prefer_mobile=bool(crawl_cfg.get("pedigree_prefer_mobile", False)),
    )

    raw_html_dir = _resolve_path(crawl_cfg.get("raw_html_dir", "data/external/netkeiba/raw_html"), base_dir)
    manifest_base_path = _resolve_path(crawl_cfg.get("manifest_file", "artifacts/reports/netkeiba_crawl_manifest.json"), base_dir)
    lock_path = _build_lock_path(manifest_base_path)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    target_reports: list[dict[str, Any]] = []
    _write_json(
        manifest_base_path,
        _build_crawl_summary(
            started_at=started_at,
            target_reports=target_reports,
            finished_at=None,
            status="running",
            process_id=os.getpid(),
            lock_file=lock_path,
        ),
    )
    session = _build_session(settings)
    try:
        for target_name, target_cfg in targets.items():
            if not isinstance(target_cfg, dict):
                continue
            if target_filter is not None and target_name != target_filter:
                continue
            if not bool(target_cfg.get("enabled", True)):
                continue

            id_file = _resolve_path(target_cfg.get("id_file"), base_dir)
            id_column = str(target_cfg.get("id_column", "id"))
            output_file = _resolve_path(target_cfg.get("output_file"), base_dir)
            limit = override_limit if override_limit is not None else target_cfg.get("limit")
            limit_value = int(limit) if limit not in {None, "", 0} else None
            dedupe_on = [str(column) for column in target_cfg.get("dedupe_on", [])]
            checkpoint_interval_raw = target_cfg.get("checkpoint_interval", crawl_cfg.get("checkpoint_interval", 25))
            checkpoint_interval = max(int(checkpoint_interval_raw or 0), 0)
            ids = _load_target_ids(id_file, id_column, limit_value)
            target_base_url = str(target_cfg.get("base_url", settings.base_url)).rstrip("/")
            manifest_path = _build_target_manifest_path(manifest_base_path, target_name)

            if target_name == "race_result":
                report = _crawl_race_result_target(
                    session=session,
                    ids=ids,
                    output_file=output_file,
                    raw_html_dir=raw_html_dir,
                    dedupe_on=dedupe_on or ["race_id", "horse_id"],
                    settings=settings,
                    base_url=target_base_url,
                    manifest_path=manifest_path,
                    checkpoint_interval=checkpoint_interval,
                    lock_path=lock_path,
                )
            elif target_name == "race_card":
                report = _crawl_race_card_target(
                    session=session,
                    ids=ids,
                    output_file=output_file,
                    raw_html_dir=raw_html_dir,
                    dedupe_on=dedupe_on or ["race_id", "horse_id"],
                    settings=settings,
                    base_url=target_base_url,
                    manifest_path=manifest_path,
                    checkpoint_interval=checkpoint_interval,
                    lock_path=lock_path,
                )
            elif target_name == "pedigree":
                report = _crawl_pedigree_target(
                    session=session,
                    ids=ids,
                    output_file=output_file,
                    raw_html_dir=raw_html_dir,
                    dedupe_on=dedupe_on or ["horse_key"],
                    settings=settings,
                    base_url=target_base_url,
                    manifest_path=manifest_path,
                    checkpoint_interval=checkpoint_interval,
                    lock_path=lock_path,
                )
            else:
                raise ValueError(f"Unsupported crawl target: {target_name}")

            report["manifest_file"] = str(manifest_path)
            target_reports.append(report)
            _write_json(
                manifest_base_path,
                _build_crawl_summary(
                    started_at=started_at,
                    target_reports=target_reports,
                    finished_at=None,
                    status="running",
                    process_id=os.getpid(),
                    lock_file=lock_path,
                ),
            )
    finally:
        session.close()

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary = _build_crawl_summary(
        started_at=started_at,
        target_reports=target_reports,
        finished_at=finished_at,
        status="completed",
        process_id=os.getpid(),
        lock_file=lock_path,
    )
    _write_json(manifest_base_path, summary)
    return summary