from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any
import zlib

import pandas as pd
import requests

from racing_ml.common.artifacts import display_path, write_csv_file, write_json, write_text_file
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.data.netkeiba_crawler import crawl_netkeiba_from_config
from racing_ml.data.netkeiba_race_list import discover_netkeiba_race_ids_from_race_list
from racing_ml.features.builder import build_features
from racing_ml.serving.predict_batch import run_predict_from_frame


def log_live_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[jra-live {now}] {message}", flush=True)


def _resolve_workspace_path(path_value: str | Path, workspace_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return workspace_root / path


def _normalize_race_id_list(race_ids: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for race_id in race_ids or []:
        token = str(race_id).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _build_runtime_dir(workspace_root: Path, race_date: str) -> Path:
    date_tag = pd.Timestamp(race_date).strftime("%Y%m%d")
    return workspace_root / "artifacts" / "tmp" / "jra_live" / date_tag


def _build_runtime_crawl_config(template_config: dict[str, Any], runtime_dir: Path) -> dict[str, Any]:
    config = deepcopy(template_config)
    crawl_cfg = config.setdefault("crawl", {})
    targets = crawl_cfg.setdefault("targets", {})
    race_card_cfg = deepcopy(targets.get("race_card", {}))
    pedigree_cfg = deepcopy(targets.get("pedigree", {}))
    crawl_cfg["targets"] = {
        "race_card": race_card_cfg,
        "pedigree": pedigree_cfg,
    }
    crawl_cfg["raw_html_dir"] = (runtime_dir / "raw_html").as_posix()
    crawl_cfg["manifest_file"] = (runtime_dir / "reports" / "netkeiba_crawl_manifest.json").as_posix()

    race_card_cfg["id_file"] = (runtime_dir / "ids" / "race_ids.csv").as_posix()
    race_card_cfg["id_column"] = "race_id"
    race_card_cfg["output_file"] = (runtime_dir / "racecard" / "live_racecard.csv").as_posix()

    pedigree_cfg["id_file"] = (runtime_dir / "ids" / "horse_keys.csv").as_posix()
    pedigree_cfg["id_column"] = "horse_key"
    pedigree_cfg["output_file"] = (runtime_dir / "pedigree" / "live_pedigree.csv").as_posix()
    return config


def _filter_race_list_frame(
    race_list_frame: pd.DataFrame,
    *,
    selected_race_ids: list[str],
    headline_contains: str | None,
    limit: int | None,
) -> pd.DataFrame:
    filtered = race_list_frame.copy()
    if headline_contains and "headline" in filtered.columns:
        filtered = filtered[
            filtered["headline"].astype("string").str.contains(str(headline_contains), case=False, na=False)
        ].copy()
    if selected_race_ids:
        filtered = filtered[filtered["race_id"].astype(str).isin(selected_race_ids)].copy()
    if limit is not None and limit > 0:
        sort_columns = [column for column in ["date", "race_no", "race_id"] if column in filtered.columns]
        filtered = filtered.sort_values(sort_columns).head(int(limit)).copy() if sort_columns else filtered.head(int(limit)).copy()
    return filtered.reset_index(drop=True)


def _merge_live_racecard_with_pedigree(
    racecard_frame: pd.DataFrame,
    pedigree_frame: pd.DataFrame,
    race_list_frame: pd.DataFrame,
    *,
    target_date: str,
) -> pd.DataFrame:
    live = racecard_frame.copy()
    if live.empty:
        raise ValueError("Live race card rows are empty")

    live["date"] = pd.to_datetime(live.get("date", target_date), errors="coerce").fillna(pd.Timestamp(target_date))
    live["date"] = live["date"].dt.strftime("%Y-%m-%d")

    pedigree_columns = [
        column
        for column in ["horse_key", "owner_name", "breeder_name", "sire_name", "dam_name", "damsire_name"]
        if column in pedigree_frame.columns
    ]
    if pedigree_columns:
        pedigree_view = pedigree_frame[pedigree_columns].drop_duplicates(subset=["horse_key"], keep="first")
        live = live.merge(pedigree_view, on="horse_key", how="left")

    race_meta_columns = [
        column for column in ["race_id", "race_no", "headline", "source_page_url", "source"] if column in race_list_frame.columns
    ]
    if race_meta_columns:
        race_meta = race_list_frame[race_meta_columns].drop_duplicates(subset=["race_id"], keep="first")
        live = live.merge(race_meta, on="race_id", how="left")

    if "rank" not in live.columns:
        live["rank"] = pd.NA
    live["is_win"] = 0

    for numeric_col in ["frame_no", "gate_no", "age", "weight", "斤量", "distance", "odds", "popularity"]:
        if numeric_col in live.columns:
            live[numeric_col] = pd.to_numeric(live[numeric_col], errors="coerce")

    live = live.drop_duplicates(subset=["race_id", "horse_id"], keep="first").reset_index(drop=True)
    sort_columns = [column for column in ["date", "race_id", "gate_no", "horse_id"] if column in live.columns]
    if sort_columns:
        live = live.sort_values(sort_columns).reset_index(drop=True)
    return live


def _parse_live_win_odds_payload(payload: dict[str, Any], *, race_id: str) -> pd.DataFrame:
    odds_block = payload.get("odds")
    if not isinstance(odds_block, dict):
        return pd.DataFrame(columns=["race_id", "gate_no", "odds", "popularity", "odds_official_datetime"])
    win_odds = odds_block.get("1")
    if not isinstance(win_odds, dict):
        return pd.DataFrame(columns=["race_id", "gate_no", "odds", "popularity", "odds_official_datetime"])

    official_datetime = payload.get("official_datetime")
    records: list[dict[str, Any]] = []
    for gate_no, values in win_odds.items():
        if not isinstance(values, list) or not values:
            continue
        odds_value = values[0] if len(values) >= 1 else None
        popularity = values[2] if len(values) >= 3 else None
        records.append(
            {
                "race_id": race_id,
                "gate_no": pd.to_numeric(gate_no, errors="coerce"),
                "odds": pd.to_numeric(odds_value, errors="coerce"),
                "popularity": pd.to_numeric(popularity, errors="coerce"),
                "odds_official_datetime": official_datetime,
            }
        )
    return pd.DataFrame(records)


def _fetch_live_win_odds(
    race_ids: list[str],
    *,
    runtime_dir: Path,
    user_agent: str,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    odds_dir = runtime_dir / "raw_html" / "odds_view"
    odds_dir.mkdir(parents=True, exist_ok=True)

    for race_id in race_ids:
        response = requests.get(
            "https://race.sp.netkeiba.com/",
            params={
                "pid": "api_get_jra_odds",
                "input": "UTF-8",
                "output": "jsonp",
                "race_id": race_id,
                "type": "1",
                "action": "init",
                "sort": "odds",
                "compress": "1",
            },
            timeout=20,
            headers={
                "User-Agent": user_agent,
                "Referer": f"https://race.sp.netkeiba.com/?pid=shutuba_past&race_id={race_id}",
            },
        )
        response.raise_for_status()
        payload_text = response.text.strip()
        write_text_file(odds_dir / f"{race_id}.jsonp", payload_text, label="live odds payload")
        if payload_text.startswith("(") and payload_text.endswith(")"):
            payload_text = payload_text[1:-1]
        payload = json.loads(payload_text)
        compressed = str(payload.get("data", ""))
        if not compressed:
            raise ValueError(f"Live odds payload missing data for race_id={race_id}: {payload.get('reason', 'unknown')}")
        body = zlib.decompress(base64.b64decode(compressed), zlib.MAX_WBITS).decode("utf-8")
        write_text_file(odds_dir / f"{race_id}.json", body, label="live odds json")
        records.append(_parse_live_win_odds_payload(json.loads(body), race_id=race_id))

    if not records:
        return pd.DataFrame(columns=["race_id", "gate_no", "odds", "popularity", "odds_official_datetime"])
    return pd.concat(records, ignore_index=True, sort=False)


def _format_metric(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def summarize_live_prediction_tradeoff(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            "top_score_horse": None,
            "top_expected_value_horse": None,
            "largest_divergence_horse": None,
            "quadrant_counts": {},
            "policy_selected_rows": 0,
        }

    enriched = predictions.copy()
    enriched["score"] = pd.to_numeric(enriched.get("score"), errors="coerce")
    enriched["expected_value"] = pd.to_numeric(enriched.get("expected_value"), errors="coerce")
    enriched["pred_rank"] = pd.to_numeric(enriched.get("pred_rank"), errors="coerce")
    enriched["ev_rank"] = pd.to_numeric(enriched.get("ev_rank"), errors="coerce")
    enriched["rank_gap"] = enriched["pred_rank"] - enriched["ev_rank"]

    score_median = enriched["score"].median(skipna=True)
    ev_median = enriched["expected_value"].median(skipna=True)

    def classify_quadrant(row: pd.Series) -> str:
        score_value = row.get("score")
        ev_value = row.get("expected_value")
        if pd.isna(score_value) or pd.isna(ev_value):
            return "unknown"
        high_score = float(score_value) >= float(score_median)
        high_ev = float(ev_value) >= float(ev_median)
        if high_score and high_ev:
            return "high_acc_high_ev"
        if high_score and not high_ev:
            return "high_acc_low_ev"
        if not high_score and high_ev:
            return "low_acc_high_ev"
        return "low_acc_low_ev"

    enriched["tradeoff_quadrant"] = enriched.apply(classify_quadrant, axis=1)
    top_score_row = enriched.sort_values(["score", "expected_value"], ascending=False).iloc[0]
    top_ev_row = enriched.sort_values(["expected_value", "score"], ascending=False).iloc[0]
    largest_gap_row = enriched.assign(abs_rank_gap=enriched["rank_gap"].abs()).sort_values(
        ["abs_rank_gap", "expected_value"],
        ascending=False,
    ).iloc[0]

    return {
        "top_score_horse": str(top_score_row.get("horse_name", "")) or None,
        "top_score": None if pd.isna(top_score_row.get("score")) else float(top_score_row.get("score")),
        "top_expected_value_horse": str(top_ev_row.get("horse_name", "")) or None,
        "top_expected_value": None if pd.isna(top_ev_row.get("expected_value")) else float(top_ev_row.get("expected_value")),
        "largest_divergence_horse": str(largest_gap_row.get("horse_name", "")) or None,
        "largest_rank_gap": None if pd.isna(largest_gap_row.get("rank_gap")) else float(largest_gap_row.get("rank_gap")),
        "quadrant_counts": {
            str(key): int(value)
            for key, value in enriched["tradeoff_quadrant"].value_counts(dropna=False).to_dict().items()
        },
        "policy_selected_rows": int(enriched["policy_selected"].fillna(False).sum()) if "policy_selected" in enriched.columns else 0,
    }


def build_live_prediction_report(predictions: pd.DataFrame, *, race_date: str, odds_available_rows: int) -> str:
    lines = [
        f"# JRA live prediction report {race_date}",
        "",
        f"- records: {len(predictions)}",
        f"- races: {predictions['race_id'].nunique()}",
        f"- odds_available_rows: {odds_available_rows}",
        "",
    ]

    for race_id, race_frame in predictions.groupby("race_id", sort=True):
        race_frame = race_frame.copy()
        race_frame["score"] = pd.to_numeric(race_frame.get("score"), errors="coerce")
        race_frame["expected_value"] = pd.to_numeric(race_frame.get("expected_value"), errors="coerce")
        race_frame["pred_rank"] = pd.to_numeric(race_frame.get("pred_rank"), errors="coerce")
        race_frame["ev_rank"] = pd.to_numeric(race_frame.get("ev_rank"), errors="coerce")
        race_frame["rank_gap"] = race_frame["pred_rank"] - race_frame["ev_rank"]
        race_frame = race_frame.sort_values(["pred_rank", "score"], ascending=[True, False]).reset_index(drop=True)
        headline = next(
            (
                str(value)
                for value in race_frame.get("headline", pd.Series(dtype="string"))
                if pd.notna(value) and str(value).strip()
            ),
            str(race_id),
        )
        score_median = race_frame["score"].median(skipna=True)
        ev_median = race_frame["expected_value"].median(skipna=True)

        def classify_quadrant(row: pd.Series) -> str:
            score_value = row.get("score")
            ev_value = row.get("expected_value")
            if pd.isna(score_value) or pd.isna(ev_value):
                return "unknown"
            high_score = float(score_value) >= float(score_median)
            high_ev = float(ev_value) >= float(ev_median)
            if high_score and high_ev:
                return "high_acc_high_ev"
            if high_score and not high_ev:
                return "high_acc_low_ev"
            if not high_score and high_ev:
                return "low_acc_high_ev"
            return "low_acc_low_ev"

        race_frame["tradeoff_quadrant"] = race_frame.apply(classify_quadrant, axis=1)
        top_score_row = race_frame.sort_values(["score", "expected_value"], ascending=False).iloc[0]
        top_ev_row = race_frame.sort_values(["expected_value", "score"], ascending=False).iloc[0]
        largest_gap_row = race_frame.assign(abs_rank_gap=race_frame["rank_gap"].abs()).sort_values(
            ["abs_rank_gap", "expected_value"],
            ascending=False,
        ).iloc[0]
        lines.append(f"## {headline}")
        lines.append("")
        lines.append("score は予想 win probability の proxy、expected_value は `score * current_odds`。")
        lines.append("単純な EV/score 比はほぼ odds に一致するため、この report では `score順位` と `EV順位` の乖離で命中寄り / 妙味寄りを読む。")
        lines.append("")
        lines.append("### Summary")
        lines.append("")
        lines.append(f"- top score: {top_score_row.get('horse_name', '')} score={_format_metric(top_score_row.get('score'), 6)} odds={_format_metric(top_score_row.get('odds'), 1)} EV={_format_metric(top_score_row.get('expected_value'), 3)}")
        lines.append(f"- top EV: {top_ev_row.get('horse_name', '')} score={_format_metric(top_ev_row.get('score'), 6)} odds={_format_metric(top_ev_row.get('odds'), 1)} EV={_format_metric(top_ev_row.get('expected_value'), 3)}")
        lines.append(f"- largest rank divergence: {largest_gap_row.get('horse_name', '')} pred_rank={'' if pd.isna(largest_gap_row.get('pred_rank')) else int(largest_gap_row.get('pred_rank'))} ev_rank={'' if pd.isna(largest_gap_row.get('ev_rank')) else int(largest_gap_row.get('ev_rank'))} gap={_format_metric(largest_gap_row.get('rank_gap'), 0)} quadrant={largest_gap_row.get('tradeoff_quadrant', '')}")
        lines.append(f"- quadrant counts: {', '.join(f'{key}={value}' for key, value in race_frame['tradeoff_quadrant'].value_counts().to_dict().items())}")
        lines.append("")
        lines.append("### Accuracy-First")
        lines.append("")
        lines.append("| pred_rank | horse_name | score | odds | popularity | expected_value | policy_selected |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in race_frame.head(5).iterrows():
            pred_rank = row.get("pred_rank")
            score = row.get("score")
            odds = row.get("odds")
            popularity = row.get("popularity")
            expected_value = row.get("expected_value")
            policy_selected = row.get("policy_selected", False)
            lines.append(
                "| {pred_rank} | {horse_name} | {score} | {odds} | {popularity} | {expected_value} | {policy_selected} |".format(
                    pred_rank="" if pd.isna(pred_rank) else int(pred_rank),
                    horse_name=str(row.get("horse_name", "")),
                    score="" if pd.isna(score) else f"{float(score):.6f}",
                    odds="" if pd.isna(odds) else odds,
                    popularity="" if pd.isna(popularity) else popularity,
                    expected_value="" if pd.isna(expected_value) else f"{float(expected_value):.3f}",
                    policy_selected="yes" if bool(policy_selected) else "no",
                )
            )
        lines.append("")
        lines.append("### Value-First")
        lines.append("")
        lines.append("| ev_rank | horse_name | expected_value | score | odds | popularity | pred_rank | quadrant |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, row in race_frame.sort_values(["expected_value", "score"], ascending=False).head(5).iterrows():
            lines.append(
                "| {ev_rank} | {horse_name} | {expected_value} | {score} | {odds} | {popularity} | {pred_rank} | {quadrant} |".format(
                    ev_rank="" if pd.isna(row.get("ev_rank")) else int(row.get("ev_rank")),
                    horse_name=str(row.get("horse_name", "")),
                    expected_value=_format_metric(row.get("expected_value"), 3),
                    score=_format_metric(row.get("score"), 6),
                    odds=_format_metric(row.get("odds"), 1),
                    popularity=_format_metric(row.get("popularity"), 0),
                    pred_rank="" if pd.isna(row.get("pred_rank")) else int(row.get("pred_rank")),
                    quadrant=str(row.get("tradeoff_quadrant", "")),
                )
            )
        lines.append("")
        lines.append("### Divergence")
        lines.append("")
        lines.append("| horse_name | pred_rank | ev_rank | rank_gap | score | expected_value | odds | quadrant |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        divergence_frame = race_frame.assign(abs_rank_gap=race_frame["rank_gap"].abs()).sort_values(
            ["abs_rank_gap", "expected_value"],
            ascending=False,
        )
        for _, row in divergence_frame.head(5).iterrows():
            lines.append(
                "| {horse_name} | {pred_rank} | {ev_rank} | {rank_gap} | {score} | {expected_value} | {odds} | {quadrant} |".format(
                    horse_name=str(row.get("horse_name", "")),
                    pred_rank="" if pd.isna(row.get("pred_rank")) else int(row.get("pred_rank")),
                    ev_rank="" if pd.isna(row.get("ev_rank")) else int(row.get("ev_rank")),
                    rank_gap=_format_metric(row.get("rank_gap"), 0),
                    score=_format_metric(row.get("score"), 6),
                    expected_value=_format_metric(row.get("expected_value"), 3),
                    odds=_format_metric(row.get("odds"), 1),
                    quadrant=str(row.get("tradeoff_quadrant", "")),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_jra_live_predict(
    *,
    model_config_path: str | Path,
    data_config_path: str | Path,
    feature_config_path: str | Path,
    crawl_config_path: str | Path,
    race_date: str,
    profile_name: str | None = None,
    race_ids: list[str] | None = None,
    headline_contains: str | None = None,
    limit: int | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    workspace_root = Path.cwd()
    resolved_data_config_path = _resolve_workspace_path(data_config_path, workspace_root)
    resolved_model_config_path = _resolve_workspace_path(model_config_path, workspace_root)
    resolved_feature_config_path = _resolve_workspace_path(feature_config_path, workspace_root)
    resolved_crawl_config_path = _resolve_workspace_path(crawl_config_path, workspace_root)

    progress = ProgressBar(total=6, prefix="[jra-live]", logger=log_live_progress, min_interval_sec=0.0)
    progress.start(message=f"starting race_date={race_date} profile={profile_name or 'custom'}")

    crawl_config = load_yaml(resolved_crawl_config_path)
    crawl_root = crawl_config.get("crawl", crawl_config)
    user_agent = str(crawl_root.get("user_agent", "nr-learn-netkeiba-crawler/0.1"))
    race_list_frame, race_list_report = discover_netkeiba_race_ids_from_race_list(
        crawl_config,
        base_dir=workspace_root,
        start_date=race_date,
        end_date=race_date,
        date_order="asc",
        refresh=refresh,
    )
    selected_race_ids = _normalize_race_id_list(race_ids)
    filtered_race_list = _filter_race_list_frame(
        race_list_frame,
        selected_race_ids=selected_race_ids,
        headline_contains=headline_contains,
        limit=limit,
    )
    if filtered_race_list.empty and selected_race_ids:
        filtered_race_list = pd.DataFrame({"date": race_date, "race_id": selected_race_ids})
    if filtered_race_list.empty:
        raise ValueError(f"No live JRA races found for {race_date}")

    selected_race_ids = filtered_race_list["race_id"].astype(str).drop_duplicates().tolist()
    runtime_dir = _build_runtime_dir(workspace_root, race_date)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for child_dir in [
        runtime_dir / "ids",
        runtime_dir / "reports",
        runtime_dir / "racecard",
        runtime_dir / "pedigree",
        runtime_dir / "raw_html",
    ]:
        child_dir.mkdir(parents=True, exist_ok=True)
    runtime_crawl_config = _build_runtime_crawl_config(crawl_config, runtime_dir)
    write_csv_file(runtime_dir / "ids" / "race_ids.csv", pd.DataFrame({"race_id": selected_race_ids}), index=False)
    progress.update(message=f"race discovery completed races={len(selected_race_ids)}")

    with Heartbeat("[jra-live]", "crawling live race card", logger=log_live_progress):
        crawl_netkeiba_from_config(
            runtime_crawl_config,
            base_dir=workspace_root,
            target_filter="race_card",
            refresh=refresh,
        )
    live_racecard_path = runtime_dir / "racecard" / "live_racecard.csv"
    if not live_racecard_path.exists():
        raise FileNotFoundError(f"Live race card output not found: {live_racecard_path}")
    live_racecard = pd.read_csv(live_racecard_path, low_memory=False)
    live_racecard = live_racecard[live_racecard["race_id"].astype(str).isin(selected_race_ids)].copy()
    if live_racecard.empty:
        raise ValueError("Live race card crawl returned zero rows for selected races")
    live_racecard["race_id"] = live_racecard["race_id"].astype(str)
    live_racecard["gate_no"] = pd.to_numeric(live_racecard.get("gate_no"), errors="coerce")
    live_win_odds = _fetch_live_win_odds(selected_race_ids, runtime_dir=runtime_dir, user_agent=user_agent)
    if not live_win_odds.empty:
        live_win_odds["race_id"] = live_win_odds["race_id"].astype(str)
        live_racecard = live_racecard.drop(columns=["odds", "popularity"], errors="ignore")
        live_racecard = live_racecard.merge(live_win_odds, on=["race_id", "gate_no"], how="left")
    odds_available_rows = int(pd.to_numeric(live_racecard.get("odds"), errors="coerce").notna().sum()) if "odds" in live_racecard.columns else 0
    if odds_available_rows == 0:
        raise ValueError("Live race card did not include current odds")
    progress.update(message=f"race card ready rows={len(live_racecard)} odds_rows={odds_available_rows}")

    horse_keys_series = live_racecard.get("horse_key", pd.Series(dtype="string")).astype("string").str.strip()
    horse_keys = horse_keys_series[horse_keys_series.str.fullmatch(r"\d+", na=False)].drop_duplicates().tolist()
    write_csv_file(runtime_dir / "ids" / "horse_keys.csv", pd.DataFrame({"horse_key": horse_keys}), index=False)
    with Heartbeat("[jra-live]", "crawling pedigree", logger=log_live_progress):
        crawl_netkeiba_from_config(
            runtime_crawl_config,
            base_dir=workspace_root,
            target_filter="pedigree",
            refresh=refresh,
        )
    live_pedigree_path = runtime_dir / "pedigree" / "live_pedigree.csv"
    live_pedigree = pd.read_csv(live_pedigree_path, low_memory=False) if live_pedigree_path.exists() else pd.DataFrame()
    progress.update(message=f"pedigree ready rows={len(live_pedigree)} horse_keys={len(horse_keys)}")

    data_config = load_yaml(resolved_data_config_path)
    dataset_cfg = data_config.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    with Heartbeat("[jra-live]", "loading historical training table", logger=log_live_progress):
        historical_frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=workspace_root)

    live_frame = _merge_live_racecard_with_pedigree(
        live_racecard,
        live_pedigree,
        filtered_race_list,
        target_date=race_date,
    )
    if "race_id" in historical_frame.columns:
        historical_frame = historical_frame[~historical_frame["race_id"].astype(str).isin(selected_race_ids)].copy()
    combined_frame = pd.concat([historical_frame, live_frame], ignore_index=True, sort=False)
    sort_columns = [column for column in ["date", "race_id", "gate_no", "horse_id"] if column in combined_frame.columns]
    if sort_columns:
        combined_frame = combined_frame.sort_values(sort_columns).reset_index(drop=True)

    with Heartbeat("[jra-live]", "building features on combined frame", logger=log_live_progress):
        featured_frame = build_features(combined_frame)
    if "date" in featured_frame.columns:
        featured_frame["date"] = pd.to_datetime(featured_frame["date"], errors="coerce").dt.normalize()
    progress.update(message=f"features built rows={len(featured_frame)}")

    prediction_summary = run_predict_from_frame(
        model_config_path=resolved_model_config_path,
        feature_config_path=resolved_feature_config_path,
        frame=featured_frame,
        race_date=race_date,
        profile_name=profile_name,
        output_file_suffix="jra_live",
    )
    prediction_path = workspace_root / str(prediction_summary["prediction_file"])
    predictions = pd.read_csv(prediction_path, low_memory=False)
    tradeoff_summary = summarize_live_prediction_tradeoff(predictions)
    report_text = build_live_prediction_report(predictions, race_date=race_date, odds_available_rows=odds_available_rows)
    report_path = prediction_path.with_suffix(".report.md")
    live_summary_path = prediction_path.with_suffix(".live.json")
    write_text_file(report_path, report_text, label="live prediction report")

    live_summary = {
        "target_date": race_date,
        "profile": profile_name,
        "prediction_file": prediction_summary["prediction_file"],
        "prediction_summary_file": prediction_summary["summary_file"],
        "live_report_file": display_path(report_path, workspace_root),
        "runtime_dir": display_path(runtime_dir, workspace_root),
        "race_count": int(predictions["race_id"].nunique()),
        "record_count": int(len(predictions)),
        "selected_race_ids": selected_race_ids,
        "headline_filter": headline_contains,
        "odds_available_rows": int(odds_available_rows),
        "odds_missing_rows": int(len(live_racecard) - odds_available_rows),
        "odds_official_datetime_max": str(live_racecard["odds_official_datetime"].dropna().max()) if "odds_official_datetime" in live_racecard.columns and live_racecard["odds_official_datetime"].notna().any() else None,
        "tradeoff_summary": tradeoff_summary,
        "race_list_report": race_list_report,
    }
    write_json(live_summary_path, live_summary)
    progress.complete(message=f"completed predictions={prediction_summary['prediction_file']}")
    print(f"[jra-live] report saved: {report_path}")
    print(f"[jra-live] live summary saved: {live_summary_path}")
    return live_summary