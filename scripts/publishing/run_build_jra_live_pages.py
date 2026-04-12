from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import write_json, write_text_file
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.version import get_source_version


JRA_VENUE_CODE_MAP = {
  "01": "札幌",
  "02": "函館",
  "03": "福島",
  "04": "新潟",
  "05": "東京",
  "06": "中山",
  "07": "中京",
  "08": "京都",
  "09": "阪神",
  "10": "小倉",
}
HARVILLE_MARKET_OPTIONS = [
    {"key": "quinella", "label": "馬連", "odds_key": "馬連", "combo_size": 2, "ordered": False, "payout_rate": 0.775},
    {"key": "exacta", "label": "馬単", "odds_key": "馬単", "combo_size": 2, "ordered": True, "payout_rate": 0.75},
    {"key": "wide", "label": "ワイド", "odds_key": "ワイド", "combo_size": 2, "ordered": False, "payout_rate": 0.775},
    {"key": "trio", "label": "三連複", "odds_key": "三連複", "combo_size": 3, "ordered": False, "payout_rate": 0.75},
    {"key": "trifecta", "label": "三連単", "odds_key": "三連単", "combo_size": 3, "ordered": True, "payout_rate": 0.75},
]
HARVILLE_MODEL_PROB_WEIGHT = 0.35
HARVILLE_MARKET_PROB_WEIGHT = 0.65
HARVILLE_SUMMARY_LIMIT = 16
HARVILLE_SUMMARY_PER_MARKET_LIMIT = 4
HARVILLE_DETAIL_LIMIT = 120
HARVILLE_SENTINEL_ODDS_THRESHOLD = 999000.0
ODDS_ANALYZE_API_URL = "https://nk-calculator-api.onrender.com/v1/analyze"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[jra-live-pages {now}] {message}", flush=True)


def latest_file(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {path}/{pattern}")
    return files[-1]


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return None
    return payload


def derive_target_date(prediction_path: Path, summary_payload: dict[str, Any] | None) -> str:
    summary_date = summary_payload.get("target_date") if isinstance(summary_payload, dict) else None
    if isinstance(summary_date, str) and summary_date.strip():
        return summary_date.strip()

    stem = prediction_path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 8:
        raw = parts[1]
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    raise ValueError(f"Unable to derive target date from {prediction_path}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _coerce_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_cell(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return _coerce_cell(value.item())
        except Exception:
            pass
    return str(value)


def _top_row(frame: pd.DataFrame, primary: str, secondary: str) -> pd.Series | None:
    if primary not in frame.columns:
        return None
    ranked = frame.copy()
    if primary in ranked.columns:
        ranked[primary] = pd.to_numeric(ranked[primary], errors="coerce")
    if secondary in ranked.columns:
        ranked[secondary] = pd.to_numeric(ranked[secondary], errors="coerce")
    ranked = ranked.sort_values([primary, secondary], ascending=False, na_position="last")
    if ranked.empty:
        return None
    return ranked.iloc[0]


def _race_headline(frame: pd.DataFrame, race_id: str) -> str:
    if "headline" not in frame.columns:
        return race_id
    for value in frame["headline"]:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return race_id


def _derive_venue_code(race_id: str) -> str | None:
    text = str(race_id)
    if len(text) < 6:
        return None
    return text[4:6]


def _derive_venue_name(race_id: str) -> str | None:
    venue_code = _derive_venue_code(race_id)
    if venue_code is None:
        return None
    return JRA_VENUE_CODE_MAP.get(venue_code)


def _parse_odds_value(value: Any) -> float | None:
  if value is None:
    return None
  text = str(value).replace(",", "").strip()
  if not text:
    return None
  if "-" in text:
    parts = [float(part.strip()) for part in text.split("-") if part.strip()]
    if not parts:
      return None
    return sum(parts) / len(parts)
  try:
    return float(text)
  except ValueError:
    return None


def _normalize_horse_no(value: Any) -> str:
  text = str(value or "").strip()
  if not text:
    return ""
  if text.isdigit():
    return str(int(text))
  try:
    numeric = float(text)
  except ValueError:
    return text
  if numeric.is_integer():
    return str(int(numeric))
  return text


def _horse_sort_key(horse_no: str) -> tuple[int, str]:
  return (int(horse_no), "") if horse_no.isdigit() else (9999, horse_no)


def _pair_key(left: str, right: str) -> tuple[str, str]:
  return tuple(sorted([left, right], key=_horse_sort_key))


def _trio_key(a: str, b: str, c: str) -> tuple[str, str, str]:
  return tuple(sorted([a, b, c], key=_horse_sort_key))


def _combo_numbers(value: Any) -> list[str]:
  return [_normalize_horse_no(item) for item in str(value or "").split("-") if _normalize_horse_no(item)]


def _build_race_url(race_id: str) -> str:
  return f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}&rf=race_list"


def _fetch_race_analyze_payload(race_id: str) -> dict[str, Any]:
  last_error: Exception | None = None
  for attempt in range(3):
    try:
      response = requests.post(
        ODDS_ANALYZE_API_URL,
        json={"race_url": _build_race_url(race_id), "excluded_horses": [], "force_refresh": False},
        timeout=30,
      )
      response.raise_for_status()
      payload = response.json()
      if not isinstance(payload, dict):
        raise ValueError(f"Invalid analyze payload for race_id={race_id}")
      return payload
    except (requests.RequestException, ValueError) as error:
      last_error = error
      if attempt >= 2:
        break
      time.sleep(2 * (attempt + 1))
  assert last_error is not None
  raise last_error


def _build_race_master(race_frame: pd.DataFrame, analyze_payload: dict[str, Any]) -> list[dict[str, str]]:
  entries = analyze_payload.get("entries") if isinstance(analyze_payload, dict) else None
  master: list[dict[str, str]] = []
  seen: set[str] = set()
  if isinstance(entries, list):
    for row in entries:
      if not isinstance(row, dict):
        continue
      horse_no = _normalize_horse_no(row.get("馬番") or row.get("col_2") or row.get("col_1"))
      horse_name = str(row.get("馬名") or row.get("col_4") or "").strip()
      if not horse_no or horse_no in seen:
        continue
      seen.add(horse_no)
      master.append({"horse_no": horse_no, "horse_name": horse_name})
  if master:
    return sorted(master, key=lambda item: _horse_sort_key(item["horse_no"]))

  for _, row in race_frame.iterrows():
    horse_no = _normalize_horse_no(row.get("gate_no"))
    horse_name = str(row.get("horse_name") or "").strip()
    if not horse_no or horse_no in seen:
      continue
    seen.add(horse_no)
    master.append({"horse_no": horse_no, "horse_name": horse_name})
  return sorted(master, key=lambda item: _horse_sort_key(item["horse_no"]))


def _build_model_win_probability_map(race_frame: pd.DataFrame, master: list[dict[str, str]]) -> dict[str, float]:
  working = race_frame.copy()
  if "gate_no" in working.columns:
    working["gate_no"] = working["gate_no"].map(_normalize_horse_no)
  if "score" in working.columns:
    working["score"] = pd.to_numeric(working["score"], errors="coerce")

  rows_by_horse_no = {
    _normalize_horse_no(row.get("gate_no")): row
    for row in working.to_dict(orient="records")
    if _normalize_horse_no(row.get("gate_no"))
  }
  rows_by_horse_name = {
    str(row.get("horse_name") or "").strip(): row
    for row in working.to_dict(orient="records")
    if str(row.get("horse_name") or "").strip()
  }

  raw: dict[str, float] = {}
  total = 0.0
  for item in master:
    row = rows_by_horse_no.get(item["horse_no"]) or rows_by_horse_name.get(item["horse_name"])
    if not isinstance(row, dict):
      continue
    score = _safe_float(row.get("score"))
    if score is None or score <= 0:
      continue
    raw[item["horse_no"]] = score
    total += score
  if not total > 0:
    return {}
  return {horse_no: value / total for horse_no, value in raw.items()}


def _build_market_win_probability_map(
  win_odds_map: dict[str, float],
  horse_numbers: list[str],
) -> dict[str, float]:
  raw: dict[str, float] = {}
  total = 0.0
  for horse_no in horse_numbers:
    odds = _safe_float(win_odds_map.get(horse_no))
    if odds is None or odds <= 0:
      continue
    implied = 1.0 / odds
    raw[horse_no] = implied
    total += implied
  if not total > 0:
    return {}
  return {horse_no: value / total for horse_no, value in raw.items()}


def _blend_win_probability_maps(
  model_probability_map: dict[str, float],
  market_probability_map: dict[str, float],
  horse_numbers: list[str],
) -> dict[str, float]:
  blended_raw: dict[str, float] = {}
  total = 0.0
  for horse_no in horse_numbers:
    model_probability = _safe_float(model_probability_map.get(horse_no))
    market_probability = _safe_float(market_probability_map.get(horse_no))
    if model_probability is None and market_probability is None:
      continue
    if model_probability is None:
      blended = float(market_probability)
    elif market_probability is None:
      blended = float(model_probability)
    else:
      blended = (
        HARVILLE_MODEL_PROB_WEIGHT * float(model_probability) +
        HARVILLE_MARKET_PROB_WEIGHT * float(market_probability)
      )
    if blended <= 0:
      continue
    blended_raw[horse_no] = blended
    total += blended
  if not total > 0:
    return {}
  return {horse_no: value / total for horse_no, value in blended_raw.items()}


def _build_win_odds_map(odds_payload: dict[str, Any]) -> dict[str, float]:
  rows = odds_payload.get("単勝") if isinstance(odds_payload, dict) else None
  result: dict[str, float] = {}
  if not isinstance(rows, list):
    return result
  for row in rows:
    if not isinstance(row, dict):
      continue
    horse_no = _normalize_horse_no(row.get("馬番"))
    odds_value = _parse_odds_value(row.get("オッズ"))
    if not horse_no or odds_value is None or odds_value <= 0:
      continue
    result[horse_no] = odds_value
  return result


def _harville_ordered_probability(win_probability_map: dict[str, float], combo: list[str]) -> float | None:
  seen: set[str] = set()
  remaining = 1.0
  probability = 1.0
  for horse_no in combo:
    if not horse_no or horse_no in seen:
      return None
    win_probability = win_probability_map.get(horse_no)
    if win_probability is None or win_probability <= 0 or remaining <= 0 or win_probability >= remaining + 1e-12:
      return None
    probability *= win_probability / remaining
    remaining -= win_probability
    seen.add(horse_no)
  return probability if probability > 0 else None


def _harville_unordered_probability(win_probability_map: dict[str, float], combo: list[str]) -> float | None:
  total = 0.0
  for permutation in permutations(combo, len(combo)):
    value = _harville_ordered_probability(win_probability_map, list(permutation))
    if value is not None:
      total += value
  return total if total > 0 else None


def _harville_wide_probability(win_probability_map: dict[str, float], pair: list[str], horses: list[str]) -> float | None:
  if len(pair) != 2:
    return None
  total = 0.0
  left, right = pair
  for horse_no in horses:
    if horse_no in {left, right}:
      continue
    value = _harville_unordered_probability(win_probability_map, [left, right, horse_no])
    if value is not None:
      total += value
  return total if total > 0 else None


def _build_harville_rows_for_market(
  *,
  config: dict[str, Any],
  actual_rows: list[dict[str, Any]],
  horse_name_map: dict[str, str],
  horse_numbers: list[str],
  win_odds_map: dict[str, float],
  win_probability_map: dict[str, float],
) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  seen: set[tuple[str, ...] | tuple[str, str] | tuple[str, str, str]] = set()
  for row in actual_rows:
    if not isinstance(row, dict):
      continue
    combo_raw = _combo_numbers(row.get("組み合わせ"))
    if len(combo_raw) != int(config["combo_size"]):
      continue
    combo = combo_raw if bool(config["ordered"]) else sorted(combo_raw, key=_horse_sort_key)
    if bool(config["ordered"]):
      dedupe_key: tuple[str, ...] | tuple[str, str] | tuple[str, str, str] = tuple(combo)
    elif int(config["combo_size"]) == 2:
      dedupe_key = _pair_key(combo[0], combo[1])
    else:
      dedupe_key = _trio_key(combo[0], combo[1], combo[2])
    if dedupe_key in seen:
      continue
    seen.add(dedupe_key)

    actual_odds = _parse_odds_value(row.get("オッズ"))
    if actual_odds is None or actual_odds <= 0 or actual_odds >= HARVILLE_SENTINEL_ODDS_THRESHOLD:
      continue
    if config["key"] == "wide":
      harville_probability = _harville_wide_probability(win_probability_map, combo, horse_numbers)
    elif bool(config["ordered"]):
      harville_probability = _harville_ordered_probability(win_probability_map, combo)
    else:
      harville_probability = _harville_unordered_probability(win_probability_map, combo)
    if harville_probability is None or harville_probability <= 0:
      continue
    harville_odds = float(config["payout_rate"]) / harville_probability
    spread = (actual_odds / harville_odds) * 100 if harville_odds > 0 else None
    payload: dict[str, Any] = {
      "marketKey": config["key"],
      "marketLabel": config["label"],
      "market_odds": actual_odds,
      "harville_odds": harville_odds,
      "spread": spread,
      "edge": spread - 100 if spread is not None else None,
      "ev_ratio": actual_odds / harville_odds if harville_odds > 0 else None,
    }
    for index, horse_no in enumerate(combo):
      suffix = ["a", "b", "c"][index]
      payload[f"horse_no_{suffix}"] = horse_no
      payload[f"horse_name_{suffix}"] = horse_name_map.get(horse_no, "")
      payload[f"win_odds_{suffix}"] = win_odds_map.get(horse_no)
    rows.append(payload)
  return sorted(
    rows,
    key=lambda item: (
      -9999 if _safe_float(item.get("edge")) is None else -float(item.get("edge")),
      -9999 if _safe_float(item.get("market_odds")) is None else -float(item.get("market_odds")),
    ),
  )


def _build_harville_payload_for_race(race_frame: pd.DataFrame, race_id: str) -> dict[str, Any]:
  try:
    analyze_payload = _fetch_race_analyze_payload(race_id)
  except Exception as error:
    return {
      "available": False,
      "message": f"odds api unavailable: {error}",
      "meta": {},
      "marketOptions": [],
      "summaryRows": [],
      "rowsByMarket": {},
    }

  master = _build_race_master(race_frame, analyze_payload)
  if not master:
    return {
      "available": False,
      "message": "odds api returned no entries",
      "meta": {},
      "marketOptions": [],
      "summaryRows": [],
      "rowsByMarket": {},
    }

  odds_payload = analyze_payload.get("odds") if isinstance(analyze_payload, dict) else {}
  if not isinstance(odds_payload, dict):
    odds_payload = {}
  horse_numbers = [item["horse_no"] for item in master]
  horse_name_map = {item["horse_no"]: item["horse_name"] for item in master}
  win_odds_map = _build_win_odds_map(odds_payload)
  model_probability_map = _build_model_win_probability_map(race_frame, master)
  market_probability_map = _build_market_win_probability_map(win_odds_map, horse_numbers)
  win_probability_map = _blend_win_probability_maps(model_probability_map, market_probability_map, horse_numbers)

  rows_by_market: dict[str, list[dict[str, Any]]] = {}
  for config in HARVILLE_MARKET_OPTIONS:
    rows_by_market[config["key"]] = _build_harville_rows_for_market(
      config=config,
      actual_rows=odds_payload.get(config["odds_key"], []) if isinstance(odds_payload.get(config["odds_key"]), list) else [],
      horse_name_map=horse_name_map,
      horse_numbers=horse_numbers,
      win_odds_map=win_odds_map,
      win_probability_map=win_probability_map,
    )[:HARVILLE_DETAIL_LIMIT]

  summary_candidates: list[dict[str, Any]] = []
  for config in HARVILLE_MARKET_OPTIONS:
    key = config["key"]
    positive_rows = [row for row in rows_by_market.get(key, []) if (_safe_float(row.get("edge")) or 0) > 0]
    summary_candidates.extend(positive_rows[:HARVILLE_SUMMARY_PER_MARKET_LIMIT])
  summary_rows = sorted(
    summary_candidates,
    key=lambda item: (
      -9999 if _safe_float(item.get("edge")) is None else -float(item.get("edge")),
      -9999 if _safe_float(item.get("market_odds")) is None else -float(item.get("market_odds")),
    ),
  )[:HARVILLE_SUMMARY_LIMIT]
  market_options = [
    {"key": config["key"], "label": config["label"], "rows": len(rows_by_market.get(config["key"], []))}
    for config in HARVILLE_MARKET_OPTIONS
    if rows_by_market.get(config["key"])
  ]
  race_meta = analyze_payload.get("race") if isinstance(analyze_payload.get("race"), dict) else {}
  return {
    "available": bool(market_options),
    "message": "model score 正規化勝率と単勝市場の暗黙確率をブレンドして Harville 理論オッズを計算し、build 時点の市場オッズ snapshot と比較します。",
    "meta": {
      "oddsUpdatedAt": race_meta.get("odds_updated_at"),
      "analyzedAt": race_meta.get("analyzed_at"),
      "horseCount": len(master),
      "positiveRows": len(summary_rows),
      "probabilitySource": "blend:model35_market65",
    },
    "marketOptions": market_options,
    "summaryRows": summary_rows,
    "rowsByMarket": rows_by_market,
  }


def build_payload(
    *,
    prediction_path: Path,
    summary_path: Path,
    live_summary_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_payload = load_optional_json(summary_path)
    if summary_payload is None:
        raise FileNotFoundError(f"Prediction summary JSON not found or invalid: {summary_path}")

    live_summary_payload = load_optional_json(live_summary_path) if live_summary_path else None
    target_date = derive_target_date(prediction_path, summary_payload)
    frame = pd.read_csv(prediction_path)
    all_columns = list(frame.columns)

    numeric_columns = [
        "score",
        "pred_rank",
        "odds",
        "popularity",
        "policy_prob",
        "policy_market_prob",
        "policy_expected_value",
        "policy_edge",
        "policy_weight",
        "policy_min_prob",
        "policy_odds_min",
        "policy_odds_max",
        "policy_min_edge",
        "policy_fractional_kelly",
        "policy_max_fraction",
        "policy_top_k",
        "policy_min_expected_value",
        "expected_value",
        "ev_rank",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    focus_columns = [
        column
        for column in [
            "race_no",
            "horse_name",
            "rank",
            "odds",
            "popularity",
            "score",
            "pred_rank",
            "policy_prob",
            "policy_market_prob",
            "policy_expected_value",
            "policy_edge",
            "expected_value",
            "ev_rank",
            "policy_selected",
            "policy_reject_reason_primary",
        ]
        if column in frame.columns
    ]
    race_diagnostics = summary_payload.get("policy_diagnostics", {}).get("race_diagnostics", [])
    race_diag_map = {
        str(item.get("race_id")): item
        for item in race_diagnostics
        if isinstance(item, dict) and item.get("race_id")
    }

    races: list[dict[str, Any]] = []
    day_overview: list[dict[str, Any]] = []
    venue_map: dict[str, dict[str, Any]] = {}
    harville_available_races = 0
    sorted_frame = frame.sort_values(["race_id", "pred_rank", "score"], ascending=[True, True, False], na_position="last")
    for race_id, race_frame in sorted_frame.groupby("race_id", sort=True):
        race_id_str = str(race_id)
        race_headline = _race_headline(race_frame, race_id_str)
        venue_code = _derive_venue_code(race_id_str)
        venue_name = _derive_venue_name(race_id_str)
        harville_payload = _build_harville_payload_for_race(race_frame, race_id_str)
        if harville_payload.get("available"):
            harville_available_races += 1
        top_score_row = _top_row(race_frame, "score", "expected_value")
        top_raw_ev_row = _top_row(race_frame, "expected_value", "score")
        top_policy_ev_row = _top_row(race_frame, "policy_expected_value", "expected_value")
        selected_rows = int(race_frame["policy_selected"].fillna(False).astype(bool).sum()) if "policy_selected" in race_frame.columns else 0
        positive_edge_rows = int((race_frame.get("policy_edge", pd.Series(dtype=float)) > 0).fillna(False).sum()) if "policy_edge" in race_frame.columns else 0
        race_diag = race_diag_map.get(race_id_str, {})
        blocker = None
        top_reasons = race_diag.get("top_primary_reject_reasons") if isinstance(race_diag, dict) else None
        if isinstance(top_reasons, list) and top_reasons:
            blocker = str(top_reasons[0].get("reason") or "")

        race_summary = {
            "raceId": race_id_str,
            "headline": race_headline,
            "venueCode": venue_code,
            "venue": venue_name,
            "raceNo": _safe_int(race_frame["race_no"].iloc[0]) if "race_no" in race_frame.columns else None,
            "track": str(race_frame["track"].iloc[0]) if "track" in race_frame.columns else None,
            "distance": _safe_int(race_frame["distance"].iloc[0]) if "distance" in race_frame.columns else None,
            "rows": int(len(race_frame)),
            "selectedRows": selected_rows,
            "positiveEdgeRows": positive_edge_rows,
            "blocker": blocker,
            "topScoreHorse": str(top_score_row.get("horse_name")) if top_score_row is not None else None,
            "topScore": _safe_float(top_score_row.get("score")) if top_score_row is not None else None,
            "topRawEvHorse": str(top_raw_ev_row.get("horse_name")) if top_raw_ev_row is not None else None,
            "topRawEv": _safe_float(top_raw_ev_row.get("expected_value")) if top_raw_ev_row is not None else None,
            "topPolicyEvHorse": str(top_policy_ev_row.get("horse_name")) if top_policy_ev_row is not None else None,
            "topPolicyEv": _safe_float(top_policy_ev_row.get("policy_expected_value")) if top_policy_ev_row is not None else None,
            "topReasons": top_reasons if isinstance(top_reasons, list) else [],
        }
        race_rows = [{column: _coerce_cell(value) for column, value in row.items()} for row in race_frame.to_dict(orient="records")]
        races.append({**race_summary, "rowsData": race_rows, "harville": _coerce_cell(harville_payload)})
        day_overview.append(race_summary)
        venue_key = venue_code or "unknown"
        if venue_key not in venue_map:
            venue_map[venue_key] = {
                "venueCode": venue_code,
                "venue": venue_name or venue_key,
                "raceCount": 0,
                "rowCount": 0,
                "selectedRows": 0,
                "raceIds": [],
            }
        venue_map[venue_key]["raceCount"] += 1
        venue_map[venue_key]["rowCount"] += int(len(race_frame))
        venue_map[venue_key]["selectedRows"] += selected_rows
        venue_map[venue_key]["raceIds"].append(race_id_str)

    venues = sorted(
        venue_map.values(),
        key=lambda item: (
            999 if item.get("venueCode") is None else int(str(item.get("venueCode"))),
            str(item.get("venue") or ""),
        ),
    )

    payload = {
        "metadata": {
            "sourceVersion": (
                live_summary_payload.get("source_version")
                if isinstance(live_summary_payload, dict) and live_summary_payload.get("source_version")
                else get_source_version()
            ),
            "title": f"JRA Live Predictions {target_date}",
            "targetDate": target_date,
            "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "predictionFile": artifact_display_path(prediction_path, workspace_root=ROOT),
            "summaryFile": artifact_display_path(summary_path, workspace_root=ROOT),
            "liveSummaryFile": artifact_display_path(live_summary_path, workspace_root=ROOT) if live_summary_path else None,
            "profile": summary_payload.get("profile"),
            "scoreSource": summary_payload.get("score_source"),
            "scoreSourceModelConfig": summary_payload.get("score_source_model_config"),
            "policyName": summary_payload.get("policy_name"),
            "policyStrategyKind": summary_payload.get("policy_strategy_kind"),
            "oddsOfficialDatetimeMax": live_summary_payload.get("odds_official_datetime_max") if isinstance(live_summary_payload, dict) else None,
            "raceCount": int(frame["race_id"].nunique()) if "race_id" in frame.columns else len(races),
            "rowCount": int(len(frame)),
            "policySelectedRows": _safe_int(summary_payload.get("policy_selected_rows")),
            "harvilleAvailableRaces": harville_available_races,
            "likelyBlockerReason": summary_payload.get("policy_diagnostics", {}).get("likely_blocker_reason"),
            "oddsLabel": "前日オッズ snapshot",
        },
        "policyDiagnostics": summary_payload.get("policy_diagnostics", {}),
        "venues": venues,
        "dayOverview": day_overview,
        "focusColumns": focus_columns,
        "allColumns": all_columns,
        "races": races,
    }
    site_manifest = {
      "source_version": payload["metadata"]["sourceVersion"],
        "target_date": target_date,
        "title": payload["metadata"]["title"],
        "relative_path": f"jra-live/{target_date}/",
        "race_count": payload["metadata"]["raceCount"],
        "row_count": payload["metadata"]["rowCount"],
        "policy_selected_rows": payload["metadata"]["policySelectedRows"],
        "odds_official_datetime_max": payload["metadata"]["oddsOfficialDatetimeMax"],
        "profile": payload["metadata"]["profile"],
        "built_at": payload["metadata"]["generatedAt"],
    }
    return payload, site_manifest


def render_live_page(*, page_title: str) -> str:
    template = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__PAGE_TITLE__</title>
  <meta name="description" content="JRA live prediction viewer">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #e7edf3;
      --bg-strong: #dbe4ec;
      --surface: rgba(247, 250, 252, 0.94);
      --surface-strong: #f8fbfd;
      --ink: #1e2b36;
      --muted: #667786;
      --line: rgba(30, 43, 54, 0.11);
      --accent: #6f8799;
      --accent-soft: rgba(111, 135, 153, 0.10);
      --navy: #466276;
      --navy-soft: rgba(70, 98, 118, 0.09);
      --gold: #8d9cab;
      --selected: #4d6f82;
      --selected-soft: rgba(77, 111, 130, 0.11);
      --danger-soft: rgba(111, 135, 153, 0.10);
      --shadow: 0 10px 28px rgba(30, 43, 54, 0.06);
      --radius: 20px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans JP", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(111, 135, 153, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(70, 98, 118, 0.08), transparent 32%),
        linear-gradient(180deg, #edf3f8 0%, #e7eef4 52%, #dde6ee 100%);
    }
    a { color: inherit; }
    .shell {
      width: min(1480px, calc(100vw - 14px));
      margin: 8px auto 20px;
    }
    .hero {
      display: grid;
      gap: 5px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(249, 252, 254, 0.96), rgba(238, 245, 250, 0.92));
      box-shadow: 0 8px 20px rgba(30, 43, 54, 0.05);
    }
    .eyebrow {
      margin: 0 0 4px;
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-weight: 700;
    }
    h1 {
      margin: 0;
      line-height: 1.08;
      font-size: clamp(24px, 2.5vw, 34px);
    }
    .hero-meta,
    .hero-note {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.7;
    }
    .hero-note {
      font-size: 12px;
    }
    .section {
      margin-top: 8px;
      padding: 8px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: var(--surface);
      box-shadow: 0 12px 28px rgba(30, 43, 54, 0.05);
    }
    .sticky-panel {
      position: sticky;
      top: 8px;
      z-index: 9;
      overflow: visible;
      backdrop-filter: blur(14px);
      background: rgba(247, 250, 252, 0.94);
    }
    .sticky-panel.collapsed {
      padding: 0;
      min-height: 0;
      border-color: transparent;
      background: transparent;
      box-shadow: none;
    }
    .sticky-panel.collapsed .venue-row {
      display: block;
      height: 0;
      min-height: 0;
      overflow: visible;
      gap: 0;
    }
    .sticky-panel.collapsed #venue-tabs,
    .sticky-panel.collapsed #race-tabs-row {
      display: none;
    }
    .sticky-panel.collapsed .venue-row .tab-panel-label {
      display: none;
    }
    .tab-panel-head-inline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
    }
    .sticky-panel.collapsed .tab-panel-head-inline {
      display: inline-flex;
      position: absolute;
      top: 2px;
      left: -10px;
      z-index: 2;
    }
    .tabs-toggle {
      border: 1px solid var(--line);
      background: rgba(251, 253, 255, 0.84);
      color: var(--navy);
      border-radius: 8px;
      padding: 3px 8px;
      font: inherit;
      font-size: 12px;
      line-height: 1.2;
      cursor: pointer;
      box-shadow: 0 6px 16px rgba(30, 43, 54, 0.10);
      flex: 0 0 auto;
    }
    .tabs-toggle:hover {
      background: rgba(255, 255, 255, 0.96);
    }
    .sticky-panel.collapsed .tabs-toggle {
      min-width: 28px;
      padding: 4px 8px;
    }
    .tab-stack {
      display: grid;
      gap: 3px;
    }
    .tab-panel-row {
      display: grid;
      gap: 2px;
    }
    .tab-panel-row[hidden],
    .overview-section[hidden],
    .race-section[hidden] {
      display: none;
    }
    .tab-panel-label {
      margin: 0;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .section-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }
    .section-title {
      margin: 0;
      font-size: 16px;
    }
    .section-note {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }
    .overview-table-wrap,
    .table-wrap {
      overflow: auto;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: var(--surface-strong);
      max-width: 100%;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 560px;
    }
    table.compact-table {
      table-layout: fixed;
      min-width: 520px;
    }
    table.raw-table {
      table-layout: fixed;
      min-width: 980px;
    }
    th,
    td {
      padding: 7px 7px;
      text-align: left;
      border-bottom: 1px solid rgba(23, 32, 51, 0.08);
      vertical-align: top;
      font-size: 12px;
    }
    th {
      position: sticky;
      top: 0;
      background: #eef4f8;
      color: var(--navy);
      font-weight: 700;
      z-index: 2;
      white-space: nowrap;
    }
    th button {
      all: unset;
      cursor: pointer;
      color: inherit;
    }
    th.mark-col, td.mark-col {
      width: 44px;
      max-width: 44px;
    }
    th.name-col, td.name-col {
      width: 104px;
      max-width: 104px;
    }
    th.num-col, td.num-col,
    th.tiny-col, td.tiny-col {
      width: 56px;
      max-width: 56px;
    }
    th.note-col, td.note-col {
      width: 116px;
      max-width: 116px;
    }
    .cell-text {
      display: block;
      white-space: normal;
      word-break: break-word;
      line-height: 1.35;
    }
    .cell-text.wrap {
      line-height: 1.35;
    }
    tbody tr:hover {
      background: rgba(33, 57, 91, 0.04);
    }
    tbody tr.active-overview {
      background: rgba(33, 57, 91, 0.08);
    }
    tbody tr.row-selected {
      background: var(--selected-soft);
    }
    tbody tr.row-positive-edge {
      box-shadow: inset 4px 0 0 var(--gold);
    }
    .chip-row,
    .tabs {
      display: flex;
      gap: 4px;
      flex-wrap: wrap;
    }
    .tabs {
      overflow-x: auto;
      flex-wrap: nowrap;
      padding-bottom: 1px;
    }
    #race-tabs {
      gap: 2px;
    }
    .tab {
      border: 1px solid var(--line);
      background: rgba(251, 253, 255, 0.82);
      color: var(--ink);
      border-radius: 9px;
      padding: 4px 7px;
      font: inherit;
      white-space: nowrap;
      cursor: pointer;
      transition: 140ms ease;
      font-size: 10px;
      line-height: 1.1;
      position: relative;
    }
    #race-tabs .tab {
      min-width: 32px;
      padding: 4px 6px;
      text-align: center;
    }
    #race-tabs .tab.tab-overview {
      min-width: 54px;
    }
    .tab.has-selection {
      border-color: rgba(77, 111, 130, 0.26);
      color: #375668;
      background: rgba(77, 111, 130, 0.09);
    }
    .tab.has-selection::after {
      content: "";
      position: absolute;
      left: 10px;
      right: 10px;
      bottom: 3px;
      height: 2px;
      border-radius: 999px;
      background: currentColor;
      opacity: 0.85;
    }
    .tab.active {
      border-color: rgba(33, 57, 91, 0.4);
      background: var(--navy);
      color: white;
      box-shadow: 0 4px 12px rgba(82, 99, 111, 0.14);
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 10px;
      font-size: 11px;
      border: 1px solid var(--line);
      background: rgba(250, 253, 255, 0.8);
      color: var(--muted);
    }
    .chip.strong {
      color: var(--navy);
      background: var(--navy-soft);
      border-color: rgba(33, 57, 91, 0.18);
    }
    .chip.good {
      color: var(--selected);
      background: var(--selected-soft);
      border-color: rgba(27, 127, 93, 0.2);
    }
    .chip.warn {
      color: var(--accent);
      background: var(--danger-soft);
      border-color: rgba(183, 71, 42, 0.18);
    }
    .race-layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .race-subtabs {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin: 2px 0 10px;
    }
    .race-subtabs .tab {
      font-size: 12px;
      padding: 7px 11px;
      border-radius: 12px;
    }
    .race-panel[hidden] {
      display: none !important;
    }
    .race-header {
      display: grid;
      gap: 5px;
      padding: 8px 10px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(251, 253, 255, 0.9), rgba(238, 245, 250, 0.92));
    }
    .race-commentary {
      margin: 0;
      padding: 7px 8px;
      border-radius: 10px;
      background: rgba(70, 98, 118, 0.06);
      color: var(--ink);
      line-height: 1.55;
      font-size: 12px;
    }
    .race-title {
      margin: 0;
      font-size: clamp(18px, 1.8vw, 24px);
      line-height: 1.2;
    }
    .race-meta {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }
    .race-kickers {
      display: flex;
      gap: 6px 12px;
      flex-wrap: wrap;
      margin: 0;
      color: var(--ink);
      font-size: 11px;
      line-height: 1.6;
    }
    .race-kickers span {
      color: var(--muted);
    }
    .info-summary {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 52px;
      padding: 6px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(251, 253, 255, 0.86);
      color: var(--navy);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      list-style: none;
    }
    .info-summary::-webkit-details-marker {
      display: none;
    }
    .info-panel {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(248, 251, 253, 0.78);
      padding: 8px 10px 10px;
    }
    .info-panel[open] {
      background: rgba(250, 253, 255, 0.9);
    }
    .guide-grid {
      display: grid;
      gap: 6px;
      margin-top: 8px;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .guide-card {
      padding: 8px 9px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: rgba(241, 247, 251, 0.92);
    }
    .guide-card-head {
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin: 0 0 6px;
    }
    .guide-label {
      font-weight: 700;
      color: var(--navy);
      font-size: 13px;
    }
    .guide-name {
      color: var(--muted);
      font-size: 12px;
    }
    .guide-copy {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 12px;
    }
    .mark-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 28px;
      height: 24px;
      border-radius: 999px;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(251, 253, 255, 0.84);
    }
    .mark-badge.mark-top {
      background: rgba(141, 156, 171, 0.18);
      color: #4e6272;
    }
    .mark-badge.mark-mid {
      background: rgba(70, 98, 118, 0.10);
      color: var(--navy);
    }
    .mark-badge.mark-low {
      background: rgba(111, 135, 153, 0.12);
      color: var(--accent);
    }
    .controls {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin: 0 0 4px;
    }
    .search {
      min-width: 200px;
      flex: 1;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px 10px;
      background: rgba(250, 253, 255, 0.88);
      color: var(--ink);
      font: inherit;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .market-card {
      margin: 14px 0 18px;
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(248, 251, 253, 0.78);
      box-shadow: 0 8px 24px rgba(30, 43, 54, 0.05);
    }
    .market-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .market-tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .ghost-button {
      appearance: none;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.8);
      color: var(--ink);
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }
    .ghost-button:hover {
      border-color: rgba(70, 98, 118, 0.28);
      background: rgba(255, 255, 255, 0.96);
    }
    .market-meta {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }
    .market-empty {
      padding: 18px;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.45);
    }
    .edge-positive {
      color: var(--navy);
      font-weight: 700;
    }
    .edge-negative {
      color: var(--muted);
    }
    .mono {
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
    }
    .footer {
      margin-top: 22px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.8;
    }
    .details-panel {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(246, 250, 253, 0.72);
      padding: 8px 10px 10px;
    }
    .details-panel[open] {
      background: rgba(249, 252, 255, 0.84);
    }
    .details-summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--navy);
      list-style: none;
    }
    .details-summary::-webkit-details-marker {
      display: none;
    }
    .loading,
    .empty {
      padding: 28px;
      text-align: center;
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 1080px) {
      .hero {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 720px) {
      .shell {
        width: min(100vw - 8px, 1480px);
        margin-top: 4px;
      }
      .hero,
      .section {
        padding: 7px;
        border-radius: 12px;
      }
      th,
      td {
        padding: 6px 6px;
      }
      table {
        min-width: 620px;
      }
      table.compact-table {
        min-width: 560px;
      }
      table.raw-table {
        min-width: 900px;
      }
      th.name-col, td.name-col {
        width: 112px;
        max-width: 112px;
      }
      th.num-col, td.num-col,
      th.tiny-col, td.tiny-col {
        width: 60px;
        max-width: 60px;
      }
      th.note-col, td.note-col {
        width: 132px;
        max-width: 132px;
      }
      .tab,
      .chip,
      .info-summary,
      .tabs-toggle {
        font-size: 9px;
      }
      #race-tabs {
        gap: 1px;
      }
      #race-tabs .tab {
        min-width: 29px;
        padding: 3px 4px;
      }
      #race-tabs .tab.tab-overview {
        min-width: 46px;
      }
      .race-title {
        font-size: 17px;
      }
      .search {
        min-width: 150px;
      }
      .guide-grid {
        grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <p class="eyebrow">GitHub Pages / JRA Live Board</p>
      <h1 id="page-title">__PAGE_TITLE__</h1>
      <p class="hero-meta mono" id="page-meta">loading metadata...</p>
      <p class="hero-note" id="page-note">overview で全体を見てから、開催場とレースへ降りて市場オッズ・推論値・policy 判定を確認できます。</p>
    </section>

    <section class="section sticky-panel" id="sticky-panel">
      <div class="tab-stack" id="tabs-stack">
        <div class="tab-panel-row venue-row">
          <div class="tab-panel-head-inline">
            <p class="tab-panel-label">Venue</p>
            <button class="tabs-toggle" id="tabs-toggle" type="button" aria-expanded="true" aria-label="タブを折りたたむ">◀</button>
          </div>
          <div class="tabs" id="venue-tabs"></div>
        </div>
        <div class="tab-panel-row" id="race-tabs-row">
          <p class="tab-panel-label">Overview / Race</p>
          <div class="tabs" id="race-tabs"></div>
        </div>
      </div>
    </section>

    <section class="section overview-section" id="overview-section">
      <div class="section-head">
        <div>
          <h2 class="section-title" id="overview-title">Overview</h2>
          <p class="section-note">開催別の命中寄り上位、妙味上位、policy 採否、主 blocker を 1 表で見ます。行クリックで該当レースへ移動します。</p>
        </div>
      </div>
      <div class="overview-table-wrap" id="overview-wrap">
        <div class="loading">loading overview...</div>
      </div>
    </section>

    <section class="section race-section" id="race-section" hidden>
      <div class="race-header">
        <div>
          <h2 class="race-title" id="race-title">loading...</h2>
          <p class="race-meta" id="race-meta"></p>
        </div>
        <p class="race-commentary" id="race-commentary">loading commentary...</p>
        <div class="race-kickers" id="race-kickers"></div>
      </div>
    </section>

    <section class="section race-layout race-section" id="race-data-section" hidden>
      <div>
        <div class="race-subtabs" id="race-subtabs">
          <button class="tab active" type="button" data-race-panel="focused">Focused Metrics</button>
          <button class="tab" type="button" data-race-panel="harville">Multi-Market EV</button>
        </div>
        <section class="race-panel" id="focused-panel">
          <div class="controls">
            <input id="horse-filter" class="search" type="search" placeholder="馬名または reject reason で絞り込み">
            <label class="toggle"><input id="selected-only" type="checkbox"> selected only</label>
            <label class="toggle"><input id="positive-edge-only" type="checkbox"> positive edge only</label>
          </div>
          <div class="section-head">
            <div>
              <h3 class="section-title">Focused Metrics</h3>
              <p class="section-note">市場オッズ、推論値、policy 指標を中心に並べた主表です。header を押すと列ソートできます。</p>
            </div>
            <details class="info-panel">
              <summary class="info-summary">info</summary>
              <div class="guide-grid" id="column-guide"></div>
            </details>
          </div>
          <div class="table-wrap" id="focused-table-wrap"></div>
        </section>
        <section class="market-card race-panel" id="harville-panel" hidden>
          <div class="section-head">
            <div>
              <h3 class="section-title">Multi-Market EV</h3>
              <p class="section-note" id="harville-note">score を race 内で正規化した勝率から Harville 理論オッズを計算し、build 時点の市場オッズ snapshot と比較します。</p>
            </div>
          </div>
          <div class="market-toolbar">
            <div class="market-tabs" id="harville-market-tabs"></div>
            <span class="mono">build snapshot</span>
          </div>
          <p class="market-meta mono" id="harville-meta">building snapshot...</p>
          <div class="table-wrap" id="harville-summary-wrap"></div>
          <details class="details-panel" open>
            <summary class="details-summary">Harville Detail</summary>
            <div class="table-wrap" id="harville-detail-wrap"></div>
          </details>
        </section>
      </div>

      <div id="raw-panel-wrap">
        <details class="details-panel">
          <summary class="details-summary">All Columns</summary>
          <p class="section-note">予想 CSV の全列をそのまま出しています。必要なときだけ開いて、生データ確認や downstream 加工前の inspection に使ってください。</p>
          <div class="table-wrap" id="raw-table-wrap"></div>
        </details>
      </div>
    </section>

    <p class="footer mono" id="footer-meta">loading metadata...</p>
  </div>

  <script>
    const numericSix = new Set([]);
    const numericThree = new Set(["score", "policy_prob", "policy_market_prob", "policy_weight", "policy_expected_value", "policy_edge", "expected_value", "policy_min_prob", "policy_min_edge", "policy_fractional_kelly", "policy_min_expected_value", "policy_blend_weight", "policy_max_fraction"]);
    const numericOne = new Set(["odds", "policy_odds_min", "policy_odds_max"]);
    const numericZero = new Set(["popularity", "pred_rank", "ev_rank", "rank", "race_no", "distance", "policy_selection_rank", "policy_stage_index", "policy_top_k"]);
    const boolCols = new Set(["policy_selected"]);
    const focusColumnMeta = {
      recommendation_mark: { label: "印", description: "総合期待値ベースの推奨印。◎◯▲★が上位、消は見送り寄り。", className: "mark-col" },
      horse_name: { label: "馬", description: "出走馬名。", className: "name-col" },
      pred_rank: { label: "勝率順", description: "モデル勝率ベースの順位。", className: "tiny-col" },
      ev_rank: { label: "期待値順", description: "期待値ベースの順位。", className: "tiny-col" },
      odds: { label: "オッズ", description: "この viewer が見ている前日オッズ snapshot。", className: "num-col" },
      popularity: { label: "人気", description: "市場人気順。", className: "tiny-col" },
      score: { label: "勝率", description: "モデルが見ている勝率 proxy。高いほど勝ち切る評価が高い。", className: "num-col" },
      expected_value: { label: "期待値", description: "勝率 × オッズで計算した生の期待値。", className: "num-col" },
      policy_prob: { label: "採用確率", description: "runtime policy が使う blended probability。", className: "num-col" },
      policy_expected_value: { label: "採用期待値", description: "policy 判定に使う期待値。1.0 超でプラス期待値読み。", className: "num-col" },
      policy_edge: { label: "Edge", description: "PEV - 1.0。市場に対する上振れ幅。", className: "num-col" },
      policy_selected: { label: "選", description: "runtime policy が実際に選択したか。", className: "tiny-col" },
      recommendation_reason: { label: "寸評", description: "印や見送りの短い理由。", className: "note-col" },
      policy_reject_reason_primary: { label: "主拒否", description: "未選択時の主な reject reason。", className: "note-col" },
    };
    const preferredFocusColumns = [
      "recommendation_mark",
      "horse_name",
      "pred_rank",
      "ev_rank",
      "odds",
      "popularity",
      "score",
      "expected_value",
      "policy_prob",
      "policy_expected_value",
      "policy_edge",
      "policy_selected",
      "recommendation_reason",
      "policy_reject_reason_primary",
    ];
    const markPriority = { "◎": 0, "◯": 1, "▲": 2, "★": 3, "": 4, "-": 4, "消": 5 };
    const ODDS_API_BASE_URL = "https://nk-calculator-api.onrender.com";
    const HARVILLE_MARKET_OPTIONS = [
      { key: "quinella", label: "馬連", oddsKey: "馬連", comboSize: 2, ordered: false, payoutRate: 0.775 },
      { key: "exacta", label: "馬単", oddsKey: "馬単", comboSize: 2, ordered: true, payoutRate: 0.75 },
      { key: "wide", label: "ワイド", oddsKey: "ワイド", comboSize: 2, ordered: false, payoutRate: 0.775 },
      { key: "trio", label: "三連複", oddsKey: "三連複", comboSize: 3, ordered: false, payoutRate: 0.75 },
      { key: "trifecta", label: "三連単", oddsKey: "三連単", comboSize: 3, ordered: true, payoutRate: 0.75 },
    ];
    const HARVILLE_DETAIL_ROW_LIMIT = 120;
    const oddsAnalyzeCache = new Map();
    const oddsAnalyzePending = new Map();
    const state = {
      data: null,
      viewMode: "overview",
      tabsCollapsed: false,
      selectedVenueCode: null,
      selectedRaceId: null,
      racePanel: "focused",
      harvilleMarket: "quinella",
      activeOddsRaceId: null,
      filter: "",
      selectedOnly: false,
      positiveEdgeOnly: false,
      focusedSort: { key: "recommendation_mark", dir: 1 },
      rawSort: { key: "pred_rank", dir: 1 },
    };

    function formatValue(key, value) {
      if (value === null || value === undefined || value === "") {
        return "-";
      }
      if (boolCols.has(key)) {
        return value ? "yes" : "no";
      }
      if (Array.isArray(value)) {
        return value.length ? value.map((item) => formatValue(key, item)).join(" | ") : "-";
      }
      if (typeof value === "object") {
        return JSON.stringify(value);
      }
      if (typeof value === "number") {
        if (numericSix.has(key)) {
          return value.toFixed(6);
        }
        if (numericThree.has(key)) {
          return value.toFixed(3);
        }
        if (numericOne.has(key)) {
          return value.toFixed(1);
        }
        if (numericZero.has(key)) {
          return String(Math.trunc(value));
        }
        return Number.isInteger(value) ? String(value) : value.toFixed(3);
      }
      return String(value);
    }

    function formatFocusedValue(key, value) {
      const number = numericValue(value);
      if (number === null) {
        return formatValue(key, value);
      }
      if (Math.abs(number) >= 1 || number === 0) {
        return formatValue(key, value);
      }
      return String(Number(number.toPrecision(3)));
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function numericValue(value) {
      if (value === null || value === undefined || value === "") {
        return null;
      }
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    function parseOddsValue(value) {
      if (value === null || value === undefined || value === "") {
        return null;
      }
      if (typeof value === "number") {
        return Number.isFinite(value) ? value : null;
      }
      const text = String(value).replaceAll(",", "").trim();
      if (!text) {
        return null;
      }
      if (text.includes("-")) {
        const parts = text.split("-").map((item) => Number(item.trim())).filter((item) => Number.isFinite(item));
        if (!parts.length) {
          return null;
        }
        return parts.reduce((acc, item) => acc + item, 0) / parts.length;
      }
      const number = Number(text);
      return Number.isFinite(number) ? number : null;
    }

    function normalizeHorseNo(value) {
      const text = String(value ?? "").trim();
      if (!text) {
        return "";
      }
      return /^\d+$/.test(text) ? String(Number(text)) : text;
    }

    function normalizeHorseName(value) {
      return String(value ?? "").trim();
    }

    function horseSortKey(horseNo) {
      return /^\d+$/.test(String(horseNo)) ? Number(horseNo) : 9999;
    }

    function horsePairSort(left, right) {
      const leftKey = horseSortKey(left);
      const rightKey = horseSortKey(right);
      if (leftKey !== rightKey) {
        return leftKey - rightKey;
      }
      return String(left).localeCompare(String(right), "ja");
    }

    function pairKey(a, b) {
      return [a, b].sort(horsePairSort).join("|");
    }

    function trioKey(a, b, c) {
      return [a, b, c].sort(horsePairSort).join("|");
    }

    function comboNumbers(combo) {
      return String(combo ?? "")
        .split("-")
        .map((item) => normalizeHorseNo(item))
        .filter(Boolean);
    }

    function formatOddsText(value) {
      const number = parseOddsValue(value);
      if (number === null) {
        return "-";
      }
      return `${number >= 100 ? number.toFixed(1) : number.toFixed(2)}倍`;
    }

    function formatEdgePercent(value) {
      const number = numericValue(value);
      if (number === null) {
        return "-";
      }
      return `${number >= 0 ? "+" : ""}${number.toFixed(1)}%`;
    }

    function buildRaceUrl(race) {
      return `https://race.netkeiba.com/race/shutuba.html?race_id=${encodeURIComponent(race.raceId)}&rf=race_list`;
    }

    function buildRaceHorseMaster(race, analyzeData) {
      const master = [];
      const seen = new Set();
      const analyzeEntries = Array.isArray(analyzeData?.entries) ? analyzeData.entries : [];
      for (const row of analyzeEntries) {
        const horseNo = normalizeHorseNo(row?.["馬番"]);
        const horseName = normalizeHorseName(row?.["馬名"]);
        if (!horseNo || seen.has(horseNo)) {
          continue;
        }
        seen.add(horseNo);
        master.push({ horse_no: horseNo, horse_name: horseName });
      }
      if (master.length) {
        return master.sort((left, right) => horsePairSort(left.horse_no, right.horse_no));
      }
      for (const row of race.rowsData || []) {
        const horseNo = normalizeHorseNo(row?.gate_no);
        const horseName = normalizeHorseName(row?.horse_name);
        if (!horseNo || seen.has(horseNo)) {
          continue;
        }
        seen.add(horseNo);
        master.push({ horse_no: horseNo, horse_name: horseName });
      }
      return master.sort((left, right) => horsePairSort(left.horse_no, right.horse_no));
    }

    function buildPredictionMaps(race) {
      const byHorseNo = new Map();
      const byHorseName = new Map();
      for (const row of race.rowsData || []) {
        const horseNo = normalizeHorseNo(row?.gate_no);
        const horseName = normalizeHorseName(row?.horse_name);
        if (horseNo && !byHorseNo.has(horseNo)) {
          byHorseNo.set(horseNo, row);
        }
        if (horseName && !byHorseName.has(horseName)) {
          byHorseName.set(horseName, row);
        }
      }
      return { byHorseNo, byHorseName };
    }

    function buildModelWinProbabilityMap(race, master) {
      const { byHorseNo, byHorseName } = buildPredictionMaps(race);
      const raw = {};
      let total = 0;
      for (const item of master) {
        const row = byHorseNo.get(item.horse_no) || byHorseName.get(item.horse_name);
        const score = numericValue(row?.score);
        if (score === null || score <= 0) {
          continue;
        }
        raw[item.horse_no] = score;
        total += score;
      }
      if (!(total > 0)) {
        return {};
      }
      const normalized = {};
      for (const [horseNo, value] of Object.entries(raw)) {
        normalized[horseNo] = value / total;
      }
      return normalized;
    }

    function buildWinOddsMap(odds) {
      const rows = Array.isArray(odds?.["単勝"]) ? odds["単勝"] : [];
      const out = {};
      for (const row of rows) {
        const horseNo = normalizeHorseNo(row?.["馬番"]);
        const odd = parseOddsValue(row?.["オッズ"]);
        if (!horseNo || odd === null || odd <= 0) {
          continue;
        }
        out[horseNo] = odd;
      }
      return out;
    }

    function getPermutations(items) {
      if (!Array.isArray(items) || items.length === 0) {
        return [];
      }
      if (items.length === 1) {
        return [items];
      }
      const out = [];
      for (let index = 0; index < items.length; index += 1) {
        const head = items[index];
        const rest = items.slice(0, index).concat(items.slice(index + 1));
        for (const tail of getPermutations(rest)) {
          out.push([head, ...tail]);
        }
      }
      return out;
    }

    function getHarvilleOrderedProbability(winProbabilityMap, combo) {
      const seen = new Set();
      let remaining = 1;
      let probability = 1;
      for (const horse of combo || []) {
        if (!horse || seen.has(horse)) {
          return null;
        }
        const winProbability = numericValue(winProbabilityMap?.[horse]);
        if (winProbability === null || winProbability <= 0 || remaining <= 0 || winProbability >= remaining + 1e-12) {
          return null;
        }
        probability *= winProbability / remaining;
        remaining -= winProbability;
        seen.add(horse);
      }
      return probability > 0 ? probability : null;
    }

    function getHarvilleUnorderedProbability(winProbabilityMap, combo) {
      let total = 0;
      for (const permutation of getPermutations(combo || [])) {
        const value = getHarvilleOrderedProbability(winProbabilityMap, permutation);
        if (value !== null) {
          total += value;
        }
      }
      return total > 0 ? total : null;
    }

    function getHarvilleWideProbability(winProbabilityMap, pair, horses) {
      const [a, b] = pair || [];
      if (!a || !b) {
        return null;
      }
      let total = 0;
      for (const c of horses || []) {
        if (c === a || c === b) {
          continue;
        }
        const value = getHarvilleUnorderedProbability(winProbabilityMap, [a, b, c]);
        if (value !== null) {
          total += value;
        }
      }
      return total > 0 ? total : null;
    }

    function buildHarvilleRowsForMarket(config, actualRows, horseNameMap, horseNumbers, winOddsMap, winProbabilityMap) {
      const rows = [];
      const seen = new Set();
      for (const row of actualRows || []) {
        const comboRaw = comboNumbers(row?.["組み合わせ"]);
        if (comboRaw.length !== config.comboSize) {
          continue;
        }
        const combo = config.ordered ? comboRaw : [...comboRaw].sort(horsePairSort);
        const dedupeKey = config.ordered
          ? combo.join("|")
          : config.comboSize === 2
            ? pairKey(combo[0], combo[1])
            : trioKey(combo[0], combo[1], combo[2]);
        if (seen.has(dedupeKey)) {
          continue;
        }
        seen.add(dedupeKey);
        const actualOdds = parseOddsValue(row?.["オッズ"]);
        if (actualOdds === null || actualOdds <= 0) {
          continue;
        }
        let harvilleProbability = null;
        if (config.key === "wide") {
          harvilleProbability = getHarvilleWideProbability(winProbabilityMap, combo, horseNumbers);
        } else if (config.ordered) {
          harvilleProbability = getHarvilleOrderedProbability(winProbabilityMap, combo);
        } else {
          harvilleProbability = getHarvilleUnorderedProbability(winProbabilityMap, combo);
        }
        if (harvilleProbability === null || harvilleProbability <= 0) {
          continue;
        }
        const harvilleOdds = config.payoutRate / harvilleProbability;
        const spread = harvilleOdds > 0 ? (actualOdds / harvilleOdds) * 100 : null;
        const payload = {
          marketKey: config.key,
          marketLabel: config.label,
          market_odds: actualOdds,
          harville_odds: harvilleOdds,
          spread,
          edge: spread === null ? null : spread - 100,
          ev_ratio: harvilleOdds > 0 ? actualOdds / harvilleOdds : null,
        };
        combo.forEach((horseNo, index) => {
          const suffix = ["a", "b", "c"][index];
          if (!suffix) {
            return;
          }
          payload[`horse_no_${suffix}`] = horseNo;
          payload[`horse_name_${suffix}`] = horseNameMap[horseNo] || "";
          payload[`win_odds_${suffix}`] = winOddsMap[horseNo] ?? null;
        });
        rows.push(payload);
      }
      return rows.sort((left, right) => {
        const byEdge = (numericValue(right.edge) ?? -9999) - (numericValue(left.edge) ?? -9999);
        if (byEdge !== 0) {
          return byEdge;
        }
        return (numericValue(right.market_odds) ?? -9999) - (numericValue(left.market_odds) ?? -9999);
      });
    }

    function buildHarvilleComparisons(race, analyzeData) {
      const odds = analyzeData?.odds || {};
      const master = buildRaceHorseMaster(race, analyzeData);
      if (!master.length) {
        return { rowsByMarket: {}, screenerRows: [], masterCount: 0 };
      }
      const horseNumbers = master.map((item) => item.horse_no);
      const horseNameMap = Object.fromEntries(master.map((item) => [item.horse_no, item.horse_name]));
      const winOddsMap = buildWinOddsMap(odds);
      const winProbabilityMap = buildModelWinProbabilityMap(race, master);
      const rowsByMarket = {};
      for (const config of HARVILLE_MARKET_OPTIONS) {
        rowsByMarket[config.key] = buildHarvilleRowsForMarket(
          config,
          Array.isArray(odds?.[config.oddsKey]) ? odds[config.oddsKey] : [],
          horseNameMap,
          horseNumbers,
          winOddsMap,
          winProbabilityMap,
        );
      }
      const screenerRows = Object.values(rowsByMarket)
        .flatMap((rows) => rows)
        .filter((row) => (numericValue(row.edge) ?? 0) > 0)
        .sort((left, right) => {
          const byEdge = (numericValue(right.edge) ?? -9999) - (numericValue(left.edge) ?? -9999);
          if (byEdge !== 0) {
            return byEdge;
          }
          return (numericValue(right.market_odds) ?? -9999) - (numericValue(left.market_odds) ?? -9999);
        });
      return { rowsByMarket, screenerRows, masterCount: master.length };
    }

    function availableHarvilleMarkets(rowsByMarket) {
      return HARVILLE_MARKET_OPTIONS.filter((item) => Array.isArray(rowsByMarket?.[item.key]) && rowsByMarket[item.key].length > 0);
    }

    function buildHarvilleOutcomeLabel(row) {
      return [row.horse_no_a, row.horse_no_b, row.horse_no_c].filter(Boolean).join("-") || "-";
    }

    function harvilleSummaryTableHtml(rows) {
      if (!rows.length) {
        return '<div class="market-empty">Harville 理論値を上回る行がまだ見つかっていません。</div>';
      }
      const body = rows.slice(0, 16).map((row) => `
        <tr>
          <td>${escapeHtml(row.marketLabel || "-")}</td>
          <td>${escapeHtml(buildHarvilleOutcomeLabel(row))}</td>
          <td>${escapeHtml(formatOddsText(row.market_odds))}</td>
          <td>${escapeHtml(formatOddsText(row.harville_odds))}</td>
          <td>${escapeHtml((numericValue(row.ev_ratio) ?? 0).toFixed(3))}</td>
          <td class="${(numericValue(row.edge) ?? 0) >= 0 ? "edge-positive" : "edge-negative"}">${escapeHtml(formatEdgePercent(row.edge))}</td>
        </tr>
      `).join("");
      return `<table class="compact-table"><thead><tr><th>券種</th><th>対象</th><th>実オッズ</th><th>Harville</th><th>EV倍率</th><th>上振れ</th></tr></thead><tbody>${body}</tbody></table>`;
    }

    function harvilleDetailTableHtml(marketKey, rows) {
      if (!rows.length) {
        return '<div class="market-empty">この券種のオッズはまだ取得できていません。</div>';
      }
      const isTriple = ["trio", "trifecta"].includes(String(marketKey || ""));
      const head = isTriple
        ? "<tr><th>馬A</th><th>馬B</th><th>馬C</th><th>単勝A</th><th>単勝B</th><th>単勝C</th><th>実オッズ</th><th>Harville</th><th>EV倍率</th><th>上振れ</th></tr>"
        : "<tr><th>馬A</th><th>馬B</th><th>単勝A</th><th>単勝B</th><th>実オッズ</th><th>Harville</th><th>EV倍率</th><th>上振れ</th></tr>";
      const body = rows.slice(0, HARVILLE_DETAIL_ROW_LIMIT).map((row) => {
        const nameA = `${row.horse_no_a || "-"} ${row.horse_name_a || ""}`.trim();
        const nameB = `${row.horse_no_b || "-"} ${row.horse_name_b || ""}`.trim();
        const nameC = `${row.horse_no_c || "-"} ${row.horse_name_c || ""}`.trim();
        const leadingCells = isTriple
          ? `<td>${escapeHtml(nameA)}</td><td>${escapeHtml(nameB)}</td><td>${escapeHtml(nameC)}</td>`
          : `<td>${escapeHtml(nameA)}</td><td>${escapeHtml(nameB)}</td>`;
        const oddsCells = isTriple
          ? `<td>${escapeHtml(formatOddsText(row.win_odds_a))}</td><td>${escapeHtml(formatOddsText(row.win_odds_b))}</td><td>${escapeHtml(formatOddsText(row.win_odds_c))}</td>`
          : `<td>${escapeHtml(formatOddsText(row.win_odds_a))}</td><td>${escapeHtml(formatOddsText(row.win_odds_b))}</td>`;
        return `<tr>${leadingCells}${oddsCells}<td>${escapeHtml(formatOddsText(row.market_odds))}</td><td>${escapeHtml(formatOddsText(row.harville_odds))}</td><td>${escapeHtml((numericValue(row.ev_ratio) ?? 0).toFixed(3))}</td><td class="${(numericValue(row.edge) ?? 0) >= 0 ? "edge-positive" : "edge-negative"}">${escapeHtml(formatEdgePercent(row.edge))}</td></tr>`;
      }).join("");
      return `<table class="compact-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
    }

    function renderHarvilleLoading(message) {
      document.getElementById("harville-meta").textContent = message;
      document.getElementById("harville-market-tabs").innerHTML = "";
      document.getElementById("harville-summary-wrap").innerHTML = '<div class="market-empty">building snapshot...</div>';
      document.getElementById("harville-detail-wrap").innerHTML = '<div class="market-empty">loading detail...</div>';
    }

    function renderHarvilleError(error) {
      const message = `odds api unavailable: ${error}`;
      document.getElementById("harville-meta").textContent = message;
      document.getElementById("harville-summary-wrap").innerHTML = `<div class="market-empty">${escapeHtml(message)}</div>`;
      document.getElementById("harville-detail-wrap").innerHTML = '<div class="market-empty">detail unavailable</div>';
    }

    function renderHarvilleSnapshot(race) {
      const harville = race?.harville;
      if (!harville || !harville.available) {
        renderHarvilleError(harville?.message || "snapshot unavailable");
        return;
      }
      const availableMarkets = Array.isArray(harville.marketOptions) && harville.marketOptions.length
        ? harville.marketOptions
        : availableHarvilleMarkets(harville.rowsByMarket);
      const activeMarket = availableMarkets.some((item) => item.key === state.harvilleMarket)
        ? state.harvilleMarket
        : availableMarkets[0]?.key || null;
      state.harvilleMarket = activeMarket;
      document.getElementById("harville-note").textContent = harville.message || "model score を race 内で正規化した勝率から Harville 理論オッズを計算し、市場オッズがどれだけ上振れているかを見ます。";
      document.getElementById("harville-meta").textContent = [
        harville?.meta?.oddsUpdatedAt ? `odds ${harville.meta.oddsUpdatedAt}` : null,
        harville?.meta?.analyzedAt ? `analyzed ${harville.meta.analyzedAt}` : null,
        harville?.meta?.horseCount ? `horses ${harville.meta.horseCount}` : null,
        harville?.meta?.positiveRows !== undefined ? `positive rows ${harville.meta.positiveRows}` : null,
      ].filter(Boolean).join(" / ");
      document.getElementById("harville-market-tabs").innerHTML = availableMarkets.length
        ? availableMarkets.map((item) => `<button class="tab ${activeMarket === item.key ? "active" : ""}" type="button" data-harville-market="${item.key}">${escapeHtml(item.label)}</button>`).join("")
        : "";
      document.getElementById("harville-summary-wrap").innerHTML = harvilleSummaryTableHtml(harville.summaryRows || []);
      document.getElementById("harville-detail-wrap").innerHTML = activeMarket
        ? harvilleDetailTableHtml(activeMarket, harville.rowsByMarket?.[activeMarket] || [])
        : '<div class="market-empty">この race では multi-market odds をまだ取得できていません。</div>';
    }

    function scoreStrengthMap(rows, key) {
      const ranked = rows
        .map((row, index) => ({ index, value: numericValue(row[key]) }))
        .filter((item) => item.value !== null)
        .sort((left, right) => left.value - right.value);
      const result = new Map();
      ranked.forEach((item, index) => {
        result.set(item.index, (index + 1) / ranked.length);
      });
      return result;
    }

    function orderedMarkedRows(rows) {
      return [...rows]
        .filter((row) => ["◎", "◯", "▲", "★"].includes(row.recommendation_mark))
        .sort((left, right) => {
          const byMark = (markPriority[left.recommendation_mark] ?? 9) - (markPriority[right.recommendation_mark] ?? 9);
          if (byMark !== 0) return byMark;
          return (numericValue(right.recommendation_score) ?? -1) - (numericValue(left.recommendation_score) ?? -1);
        });
    }

    function buildRecommendationReason(row) {
      const mark = row.recommendation_mark || "";
      const policyEv = numericValue(row.policy_expected_value) ?? 0;
      const edge = numericValue(row.policy_edge) ?? 0;
      const rejectReason = row.policy_reject_reason_primary ? String(row.policy_reject_reason_primary) : "";
      if (mark === "◎") {
        if (row.policy_selected) return "policy も選んだ本線候補";
        if (policyEv >= 1.0) return "期待値と能力の両立";
        return "総合評価で最上位";
      }
      if (mark === "◯") {
        return policyEv >= 1.0 ? "対抗評価。妙味も足りる" : "総合評価の対抗";
      }
      if (mark === "▲") {
        return edge > 0 ? "単穴。妙味寄りで拾う" : "単穴。能力寄りで残す";
      }
      if (mark === "★") {
        return edge > 0 ? "穴で一考" : "押さえ候補";
      }
      if (mark === "消") {
        return rejectReason ? `見送り寄り: ${rejectReason}` : "総合評価が低位";
      }
      if (row.policy_selected) {
        return "policy 選択";
      }
      if (edge > 0) {
        return "プラス域の妙味";
      }
      return rejectReason || "";
    }

    function decorateRace(race) {
      const rows = race.rowsData.map((row) => ({ ...row, recommendation_mark: "", recommendation_reason: "" }));
      const scoreStrength = scoreStrengthMap(rows, "score");
      const rawEvStrength = scoreStrengthMap(rows, "expected_value");
      const policyEvStrength = scoreStrengthMap(rows, "policy_expected_value");
      rows.forEach((row, index) => {
        const selectedBoost = row.policy_selected ? 0.1 : 0.0;
        row.recommendation_score = (
          0.50 * (policyEvStrength.get(index) || 0) +
          0.25 * (scoreStrength.get(index) || 0) +
          0.15 * (rawEvStrength.get(index) || 0) +
          selectedBoost
        );
      });
      const ranked = [...rows].sort((left, right) => {
        const byScore = (numericValue(right.recommendation_score) ?? -1) - (numericValue(left.recommendation_score) ?? -1);
        if (byScore !== 0) return byScore;
        const byPolicyEv = (numericValue(right.policy_expected_value) ?? -1) - (numericValue(left.policy_expected_value) ?? -1);
        if (byPolicyEv !== 0) return byPolicyEv;
        return (numericValue(right.score) ?? -1) - (numericValue(left.score) ?? -1);
      });
      ["◎", "◯", "▲", "★"].forEach((mark, index) => {
        if (ranked[index]) {
          ranked[index].recommendation_mark = mark;
        }
      });
      let lowAssigned = 0;
      const lowTarget = rows.length >= 12 ? 2 : 1;
      [...ranked].reverse().forEach((row) => {
        if (lowAssigned >= lowTarget || row.recommendation_mark) {
          return;
        }
        const recommendationScore = numericValue(row.recommendation_score) ?? 0;
        const policyEv = numericValue(row.policy_expected_value) ?? 0;
        if (recommendationScore <= 0.24 || policyEv < 0.72) {
          row.recommendation_mark = "消";
          lowAssigned += 1;
        }
      });
      rows.forEach((row) => {
        row.recommendation_reason = buildRecommendationReason(row);
      });
      const marked = orderedMarkedRows(rows);
      const dismissed = rows.filter((row) => row.recommendation_mark === "消");
      const topPolicyEv = numericValue(race.topPolicyEv) ?? 0;
      const topScore = numericValue(race.topScore) ?? 0;
      const positiveEdgeRows = rows.filter((row) => (numericValue(row.policy_edge) ?? 0) > 0).length;
      const selectedRows = rows.filter((row) => row.policy_selected).length;
      const topPair = marked.slice(0, 2);
      const topPairGap = topPair.length >= 2
        ? (numericValue(topPair[0].recommendation_score) ?? 0) - (numericValue(topPair[1].recommendation_score) ?? 0)
        : null;
      const longshotCount = rows.filter((row) => (numericValue(row.odds) ?? 0) >= 20).length;
      const commentary = [];
      if (selectedRows > 0) {
        commentary.push(`見送りではなく ${selectedRows} 頭まで絞れているレース。`);
      } else {
        commentary.push("runtime policy は見送り寄り。");
      }
      if (topPair.length >= 1 && topPairGap !== null && topPairGap >= 0.12) {
        commentary.push(`${topPair[0].horse_name} が一段抜けた形。`);
      } else if (topPair.length >= 2 && topPairGap !== null && topPairGap <= 0.04) {
        commentary.push("上位評価が拮抗していて軸はやや定めづらい。");
      }
      if (topPolicyEv >= 1.05) {
        commentary.push("妙味は一応立っている。");
      } else if (topPolicyEv < 0.9) {
        commentary.push("全体的にしょっぱめで、強く買いたい水準ではない。");
      }
      if (positiveEdgeRows >= 3) {
        commentary.push("プラス域の候補が複数いて、広めに見たいレース。");
      } else if (positiveEdgeRows === 0) {
        commentary.push("市場に対して強く上振れを見る馬がほぼいない。");
      }
      if (topScore >= 0.3) {
        commentary.push("能力評価の先頭は比較的はっきりしている。");
      }
      if (longshotCount >= Math.ceil(rows.length / 3)) {
        commentary.push("人気薄が多く、妙味はあるが散りやすい組み合わせ。");
      }
      race.rowsData = rows;
      race.recommendationSummary = marked.length
        ? marked.map((row) => `${row.recommendation_mark}${row.horse_name}`).join(" / ")
        : "印なし";
      race.dismissSummary = dismissed.length
        ? dismissed.map((row) => `${row.recommendation_mark}${row.horse_name}`).join(" / ")
        : "消印なし";
      race.raceCommentary = commentary.join(" ");
      race.topMarkedHorse = marked[0]?.horse_name || null;
      race.topMarkedLabel = marked[0]?.recommendation_mark || null;
      return race;
    }

    function decoratePayload(payload) {
      payload.races = payload.races.map((race) => decorateRace(race));
      payload.focusColumns = preferredFocusColumns.filter((column) => payload.races.some((race) => race.rowsData.some((row) => Object.prototype.hasOwnProperty.call(row, column))));
      return payload;
    }

    function focusedColumnDefs() {
      return state.data.focusColumns.map((column) => ({
        key: column,
        label: focusColumnMeta[column]?.label || column,
        description: focusColumnMeta[column]?.description || column,
        className: focusColumnMeta[column]?.className || "note-col",
      }));
    }

    function rawColumnDefs() {
      return state.data.allColumns.map((column) => ({
        key: column,
        label: column,
        description: column,
        className: String(column).includes("reason") ? "note-col" : "num-col",
      }));
    }

    function renderColumnGuide() {
      document.getElementById("column-guide").innerHTML = focusedColumnDefs().map((column) => `
        <article class="guide-card">
          <div class="guide-card-head">
            <span class="guide-label">${column.label}</span>
            <span class="guide-name mono">${column.key}</span>
          </div>
          <p class="guide-copy">${column.description}</p>
        </article>
      `).join("");
    }

    function sortRows(rows, sortState) {
      const copied = [...rows];
      copied.sort((left, right) => {
        const leftValue = left[sortState.key];
        const rightValue = right[sortState.key];
        if (sortState.key === "recommendation_mark") {
          return ((markPriority[leftValue] ?? 9) - (markPriority[rightValue] ?? 9)) * sortState.dir;
        }
        if (leftValue === null || leftValue === undefined || leftValue === "") return 1;
        if (rightValue === null || rightValue === undefined || rightValue === "") return -1;
        if (typeof leftValue === "number" && typeof rightValue === "number") {
          return (leftValue - rightValue) * sortState.dir;
        }
        return String(leftValue).localeCompare(String(rightValue), "ja") * sortState.dir;
      });
      return copied;
    }

    function currentRace() {
      const pool = venueRaces();
      return pool.find((race) => race.raceId === state.selectedRaceId) || pool[0] || state.data.races[0];
    }

    function currentVenue() {
      return state.data.venues.find((venue) => venue.venueCode === state.selectedVenueCode) || state.data.venues[0];
    }

    function venueRaces() {
      return state.data.races.filter((race) => race.venueCode === state.selectedVenueCode);
    }

    function syncSelection() {
      const activeVenue = currentVenue();
      if (!activeVenue) {
        state.selectedVenueCode = state.data.venues[0]?.venueCode || null;
      }
      const races = venueRaces();
      if (!state.selectedRaceId || !races.some((race) => race.raceId === state.selectedRaceId)) {
        state.selectedRaceId = races[0]?.raceId || state.data.races[0]?.raceId || null;
      }
    }

    function filteredRows(race) {
      const keyword = state.filter.trim().toLowerCase();
      return race.rowsData.filter((row) => {
        if (state.selectedOnly && !row.policy_selected) {
          return false;
        }
        if (state.positiveEdgeOnly && !(typeof row.policy_edge === "number" && row.policy_edge > 0)) {
          return false;
        }
        if (!keyword) {
          return true;
        }
        const haystack = [row.horse_name, row.policy_reject_reason_primary, row.policy_reject_reasons]
          .filter((value) => value !== null && value !== undefined)
          .map((value) => Array.isArray(value) ? value.join(" ") : String(value))
          .join(" ")
          .toLowerCase();
        return haystack.includes(keyword);
      });
    }

    function renderPageMeta() {
      const meta = state.data.metadata;
      const diag = state.data.policyDiagnostics || {};
      const metaBits = [
        meta.targetDate,
        `${meta.raceCount ?? "-"} races`,
        `${meta.rowCount ?? "-"} rows`,
        `policy ${meta.policySelectedRows ?? "-"}`,
        meta.oddsOfficialDatetimeMax ? `odds ${meta.oddsOfficialDatetimeMax}` : null,
      ].filter(Boolean);
      document.getElementById("page-meta").textContent = metaBits.join(" / ");
      document.getElementById("page-note").textContent = diag.likely_blocker_reason
        ? `主 blocker は ${diag.likely_blocker_reason}。開催別 overview からレースへ降りて市場オッズ・推論値・policy 判定を確認できます。`
        : "開催別 overview からレースへ降りて市場オッズ・推論値・policy 判定を確認できます。";
      document.getElementById("footer-meta").textContent = `version=${meta.sourceVersion || "-"} | source=${meta.predictionFile} | summary=${meta.summaryFile} | built=${meta.generatedAt}`;
    }

    function renderOverview() {
      const activeVenue = currentVenue();
      const rows = state.data.dayOverview
        .filter((row) => row.venueCode === state.selectedVenueCode)
        .sort((left, right) => (left.raceNo || 0) - (right.raceNo || 0));
      document.getElementById("overview-title").textContent = `${activeVenue?.venue || "-"} Overview`;
      const html = [`<table><thead><tr>
        <th>R</th>
        <th>命中寄り</th>
        <th>妙味寄り</th>
        <th>policy</th>
        <th>阻害要因</th>
      </tr></thead><tbody>`];
      rows.forEach((row) => {
        const activeClass = row.raceId === state.selectedRaceId ? "active-overview" : "";
        html.push(`
          <tr class="${activeClass}" data-race-id="${row.raceId}">
            <td><button class="tab-link" data-race-id="${row.raceId}">${formatValue("raceNo", row.raceNo)}R</button></td>
            <td>${row.topScoreHorse || "-"} <span class="mono">${formatValue("score", row.topScore)}</span></td>
            <td>${row.topRawEvHorse || "-"} <span class="mono">${formatValue("expected_value", row.topRawEv)}</span></td>
            <td>${row.topPolicyEvHorse || "-"} <span class="mono">${formatValue("selectedRows", row.selectedRows)} / ${formatValue("policy_expected_value", row.topPolicyEv)}</span></td>
            <td>${row.blocker || "-"}</td>
          </tr>
        `);
      });
      html.push("</tbody></table>");
      document.getElementById("overview-wrap").innerHTML = html.join("");
    }

    function renderVenueTabs() {
      const buttons = state.data.venues.map((venue) => {
        const activeClass = venue.venueCode === state.selectedVenueCode ? "active" : "";
        const selectedClass = (venue.selectedRows || 0) > 0 ? "has-selection" : "";
        return `<button class="tab ${activeClass} ${selectedClass}" data-venue-code="${venue.venueCode || ""}">${venue.venue || "Unknown"}</button>`;
      });
      const html = buttons.join("");
      document.getElementById("venue-tabs").innerHTML = html;
    }

    function renderTabsToggle() {
      const button = document.getElementById("tabs-toggle");
      const panel = document.getElementById("sticky-panel");
      if (state.tabsCollapsed) {
        panel.classList.add("collapsed");
        button.textContent = "▶";
        button.setAttribute("aria-expanded", "false");
        button.setAttribute("aria-label", "タブを展開する");
      } else {
        panel.classList.remove("collapsed");
        button.textContent = "◀";
        button.setAttribute("aria-expanded", "true");
        button.setAttribute("aria-label", "タブを折りたたむ");
      }
    }

    function renderTabs() {
      const html = [
        `<button class="tab tab-overview ${state.viewMode === "overview" ? "active" : ""}" data-view-mode="overview">Overview</button>`,
        ...venueRaces().map((race) => {
          const activeClass = state.viewMode !== "overview" && race.raceId === state.selectedRaceId ? "active" : "";
          const selectedClass = (race.selectedRows || 0) > 0 ? "has-selection" : "";
          return `<button class="tab ${activeClass} ${selectedClass}" data-race-id="${race.raceId}">${race.raceNo || "-"}R</button>`;
        }),
      ].join("");
      document.getElementById("race-tabs-row").hidden = false;
      document.getElementById("race-tabs").innerHTML = html;
    }

    function renderRaceHeader(race) {
      document.getElementById("race-title").textContent = race.headline;
      document.getElementById("race-meta").textContent = `${race.venue || "-"} ${formatValue("raceNo", race.raceNo)}R / race_id=${race.raceId} / ${race.track || "-"} ${formatValue("distance", race.distance)}m / odds ${state.data.metadata.oddsOfficialDatetimeMax || "-"}`;
      const oddsTimestamp = state.data.metadata.oddsOfficialDatetimeMax || "-";
      document.getElementById("race-commentary").textContent = race.raceCommentary || "race commentary unavailable";
      const kickers = [
        `印 ${race.recommendationSummary || "印なし"}`,
        `top score ${race.topScoreHorse || "-"} ${formatValue("score", race.topScore)}`,
        `top PEV ${race.topPolicyEvHorse || "-"} ${formatValue("policy_expected_value", race.topPolicyEv)}`,
        `selected ${formatValue("selectedRows", race.selectedRows)}`,
        `blocker ${race.blocker || "-"}`,
        `odds ${oddsTimestamp}`,
      ];
      document.getElementById("race-kickers").innerHTML = kickers.map((line) => `<span>${escapeHtml(line)}</span>`).join("");
    }

    function markClass(mark) {
      if (mark === "◎") return "mark-top";
      if (["◯", "▲", "★"].includes(mark)) return "mark-mid";
      if (mark === "消") return "mark-low";
      return "";
    }

    function cellHtml(column, row, tableName) {
      const value = row[column.key];
      const formatted = tableName === "focused"
        ? formatFocusedValue(column.key, value)
        : formatValue(column.key, value);
      if (column.key === "recommendation_mark") {
        const safeMark = escapeHtml(formatted);
        return `<span class="mark-badge ${markClass(formatted)}" title="${escapeHtml(column.description)}">${safeMark || "-"}</span>`;
      }
      const wrapClass = column.className === "note-col" ? "cell-text wrap" : "cell-text";
      return `<span class="${wrapClass}">${escapeHtml(formatted)}</span>`;
    }

    function tableHtml(columns, rows, sortState, tableName) {
      if (!rows.length) {
        return '<div class="empty">該当行がありません。</div>';
      }
      const sortedRows = sortRows(rows, sortState);
      const header = columns.map((column) => {
        const arrow = sortState.key === column.key ? (sortState.dir === 1 ? " ↑" : " ↓") : "";
        return `<th class="${column.className || ""}"><button data-table="${tableName}" data-column="${column.key}" title="${escapeHtml(column.description)}">${escapeHtml(column.label)}${arrow}</button></th>`;
      }).join("");
      const body = sortedRows.map((row) => {
        const classes = [
          row.policy_selected ? "row-selected" : "",
          typeof row.policy_edge === "number" && row.policy_edge > 0 ? "row-positive-edge" : "",
        ].filter(Boolean).join(" ");
        return `<tr class="${classes}">${columns.map((column) => `<td class="${column.className || ""}">${cellHtml(column, row, tableName)}</td>`).join("")}</tr>`;
      }).join("");
      const tableClass = tableName === "focused" ? "compact-table" : "raw-table";
      return `<table class="${tableClass}"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
    }

    function renderTables(race) {
      const rows = filteredRows(race);
      document.getElementById("focused-table-wrap").innerHTML = tableHtml(focusedColumnDefs(), rows, state.focusedSort, "focused");
      document.getElementById("raw-table-wrap").innerHTML = tableHtml(rawColumnDefs(), rows, state.rawSort, "raw");
    }

    function renderRacePanels() {
      const focusedActive = state.racePanel !== "harville";
      document.getElementById("focused-panel").hidden = !focusedActive;
      document.getElementById("harville-panel").hidden = focusedActive;
      document.getElementById("raw-panel-wrap").hidden = !focusedActive;
      document.getElementById("race-subtabs").innerHTML = [
        { key: "focused", label: "Focused Metrics" },
        { key: "harville", label: "Harville EV" },
      ].map((item) => `<button class="tab ${state.racePanel === item.key ? "active" : ""}" type="button" data-race-panel="${item.key}">${escapeHtml(item.label)}</button>`).join("");
    }

    function renderRace() {
      syncSelection();
      renderTabsToggle();
      renderVenueTabs();
      renderTabs();
      renderOverview();
      const overviewSection = document.getElementById("overview-section");
      const raceSection = document.getElementById("race-section");
      const raceDataSection = document.getElementById("race-data-section");
      if (state.viewMode === "overview") {
        overviewSection.hidden = false;
        raceSection.hidden = true;
        raceDataSection.hidden = true;
        return;
      }
      const race = currentRace();
      state.selectedRaceId = race.raceId;
      overviewSection.hidden = true;
      raceSection.hidden = false;
      raceDataSection.hidden = false;
      renderRaceHeader(race);
      renderRacePanels();
      renderTables(race);
      renderHarvilleSnapshot(race);
    }

    function bindEvents() {
      document.body.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
          return;
        }
        const viewMode = target.getAttribute("data-view-mode");
        if (viewMode === "overview") {
          state.viewMode = "overview";
          renderRace();
          return;
        }
        if (target.id === "tabs-toggle") {
          state.tabsCollapsed = !state.tabsCollapsed;
          renderTabsToggle();
          return;
        }
        const venueCode = target.getAttribute("data-venue-code");
        if (venueCode !== null) {
          state.viewMode = "overview";
          state.selectedVenueCode = venueCode || null;
          syncSelection();
          renderRace();
          return;
        }
        const raceId = target.getAttribute("data-race-id");
        if (raceId) {
          const race = state.data.races.find((item) => item.raceId === raceId);
          if (race && race.venueCode) {
            state.viewMode = "venue";
            state.selectedVenueCode = race.venueCode;
          }
          state.selectedRaceId = raceId;
          renderRace();
          return;
        }
        const harvilleMarket = target.getAttribute("data-harville-market");
        if (harvilleMarket) {
          state.harvilleMarket = harvilleMarket;
          renderHarvilleSnapshot(currentRace());
          return;
        }
        const racePanel = target.getAttribute("data-race-panel");
        if (racePanel) {
          state.racePanel = racePanel;
          renderRacePanels();
          return;
        }
        const column = target.getAttribute("data-column");
        const tableName = target.getAttribute("data-table");
        if (column && tableName) {
          const sortState = tableName === "focused" ? state.focusedSort : state.rawSort;
          if (sortState.key === column) {
            sortState.dir = sortState.dir * -1;
          } else {
            sortState.key = column;
            sortState.dir = 1;
          }
          renderRace();
        }
      });
      document.getElementById("horse-filter").addEventListener("input", (event) => {
        state.filter = event.target.value || "";
        renderTables(currentRace());
      });
      document.getElementById("selected-only").addEventListener("change", (event) => {
        state.selectedOnly = Boolean(event.target.checked);
        renderTables(currentRace());
      });
      document.getElementById("positive-edge-only").addEventListener("change", (event) => {
        state.positiveEdgeOnly = Boolean(event.target.checked);
        renderTables(currentRace());
      });
    }

    async function main() {
      const response = await fetch('./data.json');
      const payload = decoratePayload(await response.json());
      state.data = payload;
      state.selectedVenueCode = payload.venues[0]?.venueCode || payload.races[0]?.venueCode || null;
      state.selectedRaceId = payload.races[0]?.raceId || null;
      renderPageMeta();
      renderColumnGuide();
      renderRace();
      bindEvents();
    }

    main().catch((error) => {
      document.getElementById('overview-wrap').innerHTML = `<div class="empty">viewer load failed: ${error}</div>`;
      document.getElementById('page-note').textContent = `viewer load failed: ${error}`;
    });
  </script>
</body>
</html>
"""
    return template.replace("__PAGE_TITLE__", page_title)


def render_root_page(*, manifests: list[dict[str, Any]]) -> str:
    cards = []
    for manifest in manifests:
        cards.append(
            """
            <a class="card" href="{relative_path}">
              <p class="card-eyebrow">{target_date}</p>
              <h2>{title}</h2>
              <p class="card-copy mono">version={source_version}</p>
              <p class="card-copy">races={race_count} / rows={row_count} / policy_selected={policy_selected_rows}</p>
              <p class="card-copy mono">{odds_official_datetime_max}</p>
            </a>
            """.format(
                relative_path=manifest.get("relative_path", "#"),
                target_date=manifest.get("target_date", "-"),
                title=manifest.get("title", "JRA Live Viewer"),
                source_version=manifest.get("source_version", "-"),
                race_count=manifest.get("race_count", "-"),
                row_count=manifest.get("row_count", "-"),
                policy_selected_rows=manifest.get("policy_selected_rows", "-"),
                odds_official_datetime_max=manifest.get("odds_official_datetime_max", "odds timestamp unavailable"),
            )
        )
    return """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nr-learn JRA Live Pages</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+JP:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f4efe6;
      --surface: rgba(255, 250, 244, 0.9);
      --ink: #172033;
      --muted: #667085;
      --line: rgba(23, 32, 51, 0.12);
      --navy: #21395b;
      --accent: #b7472a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans JP", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f2ea 0%, #eee4d5 100%);
    }
    .shell {
      width: min(1200px, calc(100vw - 28px));
      margin: 24px auto 40px;
    }
    .hero {
      padding: 28px;
      border-radius: 28px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(247, 239, 229, 0.94));
      box-shadow: 0 18px 48px rgba(23, 32, 51, 0.08);
    }
    .eyebrow {
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      font-size: clamp(30px, 4vw, 52px);
      line-height: 1.08;
    }
    .lead {
      margin: 16px 0 0;
      color: var(--muted);
      line-height: 1.9;
      max-width: 70ch;
    }
    .grid {
      margin-top: 20px;
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }
    .card {
      display: block;
      text-decoration: none;
      color: inherit;
      padding: 20px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: var(--surface);
      box-shadow: 0 14px 30px rgba(23, 32, 51, 0.05);
      transition: transform 140ms ease, box-shadow 140ms ease;
    }
    .card:hover {
      transform: translateY(-2px);
      box-shadow: 0 18px 36px rgba(23, 32, 51, 0.10);
    }
    .card-eyebrow {
      margin: 0 0 10px;
      color: var(--navy);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .card h2 {
      margin: 0 0 12px;
      font-size: 24px;
      line-height: 1.3;
    }
    .card-copy {
      margin: 0;
      color: var(--muted);
      line-height: 1.8;
      font-size: 14px;
    }
    .mono { font-family: "IBM Plex Mono", monospace; font-size: 12px; }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <p class="eyebrow">nr-learn / GitHub Pages</p>
      <h1>JRA Live Prediction Viewer</h1>
      <p class="lead">repo で生成した live 予想を、レース別 tab と全列テーブルで GitHub Pages 公開できる形にした入口です。下の card から各 target date の静的 viewer に入れます。</p>
    </section>
    <section class="grid">
      __CARDS__
    </section>
  </div>
</body>
</html>
""".replace("__CARDS__", "\n".join(cards))


def render_redirect_page(*, relative_target: str) -> str:
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="0; url={relative_target}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Redirecting...</title>
</head>
<body>
  <p>Redirecting to <a href="{relative_target}">{relative_target}</a></p>
</body>
</html>
"""


def collect_site_manifests(site_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted((site_dir / "jra-live").glob("*/site_manifest.json"), reverse=True):
        if path.parent.name == "latest":
            continue
        payload = load_optional_json(path)
        if payload is None:
            continue
        manifests.append(payload)
    manifests.sort(key=lambda item: str(item.get("target_date", "")), reverse=True)
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--live-summary-file", default=None)
    parser.add_argument("--output-dir", default="pages")
    args = parser.parse_args()
    progress = ProgressBar(total=4, prefix="[jra-live-pages]", logger=log_progress, min_interval_sec=0.0)

    try:
        progress.start("resolving live page inputs")
        predictions_dir = ROOT / "artifacts" / "predictions"
        prediction_path = Path(args.predictions_file) if args.predictions_file else latest_file(predictions_dir, "predictions_*_jra_live.csv")
        summary_path = Path(args.summary_file) if args.summary_file else prediction_path.with_suffix(".summary.json")
        live_summary_path = Path(args.live_summary_file) if args.live_summary_file else prediction_path.with_suffix(".live.json")
        output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
        progress.update(message=f"inputs resolved prediction={prediction_path.name} output_dir={artifact_display_path(output_dir, workspace_root=ROOT)}")

        with Heartbeat("[jra-live-pages]", "loading prediction payloads", logger=log_progress):
            payload, site_manifest = build_payload(
                prediction_path=prediction_path,
                summary_path=summary_path,
                live_summary_path=live_summary_path,
            )
        target_date = str(payload["metadata"]["targetDate"])
        date_dir = output_dir / "jra-live" / target_date
        latest_dir = output_dir / "jra-live" / "latest"
        progress.update(message=f"payload built target_date={target_date} races={payload['metadata']['raceCount']} rows={payload['metadata']['rowCount']}")

        with Heartbeat("[jra-live-pages]", "writing static page artifacts", logger=log_progress):
            write_json(date_dir / "data.json", payload)
            write_json(date_dir / "site_manifest.json", site_manifest)
            write_text_file(date_dir / "index.html", render_live_page(page_title=payload["metadata"]["title"]))
            write_text_file(latest_dir / "index.html", render_redirect_page(relative_target=f"../{target_date}/"))
            write_text_file(output_dir / ".nojekyll", "")
            manifests = collect_site_manifests(output_dir)
            write_text_file(output_dir / "index.html", render_root_page(manifests=manifests))
            write_text_file(output_dir / "jra-live" / "index.html", render_root_page(manifests=manifests))
        progress.update(message=f"static pages written path={artifact_display_path(date_dir, workspace_root=ROOT)}")

        progress.complete(message="jra live pages build finished")
        print(f"[jra-live-pages] page root: {artifact_display_path(output_dir / 'index.html', workspace_root=ROOT)}")
        print(f"[jra-live-pages] target page: {artifact_display_path(date_dir / 'index.html', workspace_root=ROOT)}")
        print(f"[jra-live-pages] data file: {artifact_display_path(date_dir / 'data.json', workspace_root=ROOT)}")
        return 0
    except KeyboardInterrupt:
        print("[jra-live-pages] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[jra-live-pages] failed: {error}")
        return 1
    except Exception as error:
        print(f"[jra-live-pages] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())