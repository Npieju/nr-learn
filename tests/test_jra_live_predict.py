from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.serving.jra_live_predict import (
    _merge_live_racecard_with_pedigree,
    _parse_live_win_odds_payload,
    _resolve_live_pre_feature_max_rows,
    build_live_prediction_report,
    run_jra_live_predict,
    summarize_runner_history_coverage,
    summarize_live_prediction_tradeoff,
)
from racing_ml.serving.runtime_policy import summarize_policy_diagnostics
from racing_ml.version import get_source_version


class JraLivePredictTest(unittest.TestCase):
    def test_run_jra_live_predict_writes_live_summary_with_new_diagnostics_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runtime_dir = root / "artifacts" / "tmp" / "jra_live" / "20260405"
            racecard_dir = runtime_dir / "racecard"
            pedigree_dir = runtime_dir / "pedigree"
            prediction_dir = root / "artifacts" / "predictions"
            racecard_dir.mkdir(parents=True, exist_ok=True)
            pedigree_dir.mkdir(parents=True, exist_ok=True)
            prediction_dir.mkdir(parents=True, exist_ok=True)

            (racecard_dir / "live_racecard.csv").write_text(
                "race_id,horse_id,horse_key,horse_name,gate_no,date\n"
                "202604050811,h1,2019001,サンプルA,1,2026-04-05\n",
                encoding="utf-8",
            )
            (pedigree_dir / "live_pedigree.csv").write_text(
                "horse_key,owner_name,breeder_name,sire_name,dam_name\n"
                "2019001,owner-a,breeder-a,sire-a,dam-a\n",
                encoding="utf-8",
            )
            (prediction_dir / "predictions_20260405_jra_live.csv").write_text(
                "race_id,horse_id,horse_name,pred_rank,score,odds,popularity,expected_value,policy_selected,policy_reject_reason_primary\n"
                "202604050811,h1,サンプルA,1,0.33,3.4,1,1.10,True,\n",
                encoding="utf-8",
            )

            written_json: list[tuple[Path, dict[str, object]]] = []
            written_text: list[tuple[Path, str]] = []
            runner_history_coverage = {
                "available": True,
                "runner_count": 1,
                "group_columns": {"horse_history": ["horse_last_3_avg_rank"], "pedigree_history": ["owner_last_50_win_rate"]},
                "profile_counts": {"history_rich": 1},
                "race_diagnostics": [{"race_id": "202604050811", "runner_count": 1, "profile_counts": {"history_rich": 1}}],
                "runners": [{"race_id": "202604050811", "horse_id": "h1", "horse_name": "サンプルA", "history_profile": "history_rich"}],
            }
            policy_diagnostics = {
                "available": True,
                "policy_selected_rows": 1,
                "policy_selected_races": 1,
                "rejected_rows": 0,
                "likely_blocker_reason": None,
                "primary_reject_reason_counts": {},
                "primary_reject_reason_race_counts": {},
                "primary_reject_reason_examples": {},
                "stage_fallback_reason_counts": {},
                "race_diagnostics": [{"race_id": "202604050811", "selected_rows": 1}],
            }

            with patch.object(Path, "cwd", return_value=root), patch(
                "racing_ml.serving.jra_live_predict.load_yaml",
                side_effect=[
                    {"crawl": {"user_agent": "test-agent"}},
                    {"dataset": {"raw_dir": "data/raw", "primary_tail_cache_manifest_file": "artifacts/reports/primary_tail_cache.json"}},
                    {},
                ],
            ), patch(
                "racing_ml.serving.jra_live_predict.discover_netkeiba_race_ids_from_race_list",
                return_value=(
                    pd.DataFrame([
                        {"date": "2026-04-05", "race_id": "202604050811", "race_no": 8, "headline": "8R テスト 14:00 芝1600m 12頭"}
                    ]),
                    {"source": "race_list", "records": 1},
                ),
            ), patch(
                "racing_ml.serving.jra_live_predict.crawl_netkeiba_from_config",
                return_value=None,
            ), patch(
                "racing_ml.serving.jra_live_predict._fetch_live_win_odds",
                return_value=pd.DataFrame([
                    {"race_id": "202604050811", "gate_no": 1, "odds": 3.4, "popularity": 1, "odds_official_datetime": "2026-04-05 13:55:00"}
                ]),
            ), patch(
                "racing_ml.serving.jra_live_predict.load_training_table_for_feature_build",
                return_value=SimpleNamespace(
                    frame=pd.DataFrame([
                        {"date": "2026-04-04", "race_id": "hist1", "horse_id": "hh1", "f1": 0.2}
                    ]),
                    loaded_rows=1,
                    pre_feature_rows=1,
                    data_load_strategy="tail_cache",
                    primary_source_rows_total=1000,
                ),
            ), patch(
                "racing_ml.serving.jra_live_predict.build_features",
                side_effect=lambda frame: frame,
            ), patch(
                "racing_ml.serving.jra_live_predict.run_predict_from_frame",
                return_value={
                    "prediction_file": "artifacts/predictions/predictions_20260405_jra_live.csv",
                    "summary_file": "artifacts/predictions/predictions_20260405_jra_live.summary.json",
                },
            ), patch(
                "racing_ml.serving.jra_live_predict.summarize_runner_history_coverage",
                return_value=runner_history_coverage,
            ), patch(
                "racing_ml.serving.jra_live_predict.summarize_live_prediction_tradeoff",
                return_value={"top_score_horse": "サンプルA"},
            ), patch(
                "racing_ml.serving.jra_live_predict.summarize_policy_diagnostics",
                return_value=policy_diagnostics,
            ), patch(
                "racing_ml.serving.jra_live_predict.build_live_prediction_report",
                return_value="# test report\n",
            ), patch(
                "racing_ml.serving.jra_live_predict.write_text_file",
                side_effect=lambda path, content, **kwargs: written_text.append((path, content)),
            ), patch(
                "racing_ml.serving.jra_live_predict.write_json",
                side_effect=lambda path, payload, **kwargs: written_json.append((path, payload)),
            ):
                summary = run_jra_live_predict(
                    data_config_path="configs/data.yaml",
                    model_config_path="configs/model.yaml",
                    feature_config_path="configs/features.yaml",
                    crawl_config_path="configs/crawl.yaml",
                    race_date="2026-04-05",
                    profile_name="current_recommended_serving_2025_latest",
                )

        self.assertEqual(summary["target_date"], "2026-04-05")
        self.assertEqual(summary["source_version"], get_source_version())
        self.assertEqual(summary["historical_data_load"]["pre_feature_max_rows"], None)
        self.assertEqual(summary["historical_data_load"]["data_load_strategy"], "tail_cache")
        self.assertEqual(summary["historical_data_load"]["loaded_rows"], 1)
        self.assertEqual(summary["historical_data_load"]["primary_source_rows_total"], 1000)
        self.assertEqual(summary["policy_diagnostics"]["policy_selected_rows"], 1)
        self.assertEqual(summary["runner_history_coverage"]["profile_counts"]["history_rich"], 1)
        self.assertEqual(summary["tradeoff_summary"]["top_score_horse"], "サンプルA")
        self.assertIn("historical_load_seconds", summary["timings"])
        self.assertIn("feature_build_seconds", summary["timings"])
        self.assertIn("total_seconds", summary["timings"])
        self.assertEqual(len(written_json), 1)
        self.assertTrue(str(written_json[0][0]).endswith("predictions_20260405_jra_live.live.json"))
        self.assertEqual(written_json[0][1]["source_version"], get_source_version())
        self.assertEqual(len(written_text), 1)
        self.assertIn("# test report", written_text[0][1])

    def test_resolve_live_pre_feature_max_rows_reads_primary_tail_cache_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "artifacts" / "reports" / "primary_tail_cache_tail10000.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text('{"tail_rows": 10000, "status": "completed"}', encoding="utf-8")

            resolved = _resolve_live_pre_feature_max_rows(
                {"primary_tail_cache_manifest_file": "artifacts/reports/primary_tail_cache_tail10000.json"},
                workspace_root=root,
            )

            self.assertEqual(resolved, 10000)

    def test_merge_live_racecard_with_pedigree_keeps_live_odds_and_adds_metadata(self) -> None:
        racecard = pd.DataFrame(
            [
                {
                    "date": "2026-04-05",
                    "race_id": "202609020411",
                    "horse_id": "20260902041101",
                    "horse_key": "2019100001",
                    "horse_name": "サンプルA",
                    "gate_no": 1,
                    "odds": "3.4",
                    "popularity": "1",
                }
            ]
        )
        pedigree = pd.DataFrame(
            [
                {
                    "horse_key": "2019100001",
                    "owner_name": "owner-a",
                    "breeder_name": "breeder-a",
                    "sire_name": "sire-a",
                    "dam_name": "dam-a",
                    "damsire_name": "damsire-a",
                }
            ]
        )
        race_list = pd.DataFrame(
            [
                {
                    "date": "2026-04-05",
                    "race_id": "202609020411",
                    "race_no": 11,
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                }
            ]
        )

        merged = _merge_live_racecard_with_pedigree(
            racecard,
            pedigree,
            race_list,
            target_date="2026-04-05",
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "headline"], "11R 大阪杯 GI 15:40 芝2000m 15頭")
        self.assertEqual(merged.loc[0, "owner_name"], "owner-a")
        self.assertAlmostEqual(float(merged.loc[0, "odds"]), 3.4)
        self.assertEqual(int(merged.loc[0, "popularity"]), 1)
        self.assertTrue(pd.isna(merged.loc[0, "rank"]))
        self.assertEqual(int(merged.loc[0, "is_win"]), 0)

    def test_build_live_prediction_report_includes_headline_and_policy_state(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "race_id": "202609020411",
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                    "pred_rank": 1,
                    "horse_name": "サンプルA",
                    "score": 0.321,
                    "odds": 3.4,
                    "popularity": 1,
                    "expected_value": 1.091,
                    "ev_rank": 1,
                    "policy_selected": True,
                }
            ]
        )

        report = build_live_prediction_report(predictions, race_date="2026-04-05", odds_available_rows=1)

        self.assertIn("JRA live prediction report 2026-04-05", report)
        self.assertIn("11R 大阪杯 GI 15:40 芝2000m 15頭", report)
        self.assertIn("サンプルA", report)
        self.assertIn("yes", report)
        self.assertIn("Value-First", report)
        self.assertIn("Divergence", report)
        self.assertIn("Policy Diagnostics", report)

    def test_summarize_runner_history_coverage_reports_runner_profiles(self) -> None:
        feature_frame = pd.DataFrame(
            [
                {
                    "race_id": "202609020411",
                    "horse_id": "h1",
                    "horse_name": "サンプルA",
                    "horse_history_key_source": "horse_key",
                    "horse_last_3_avg_rank": 2.0,
                    "horse_last_5_win_rate": 0.2,
                    "horse_days_since_last_race": 21.0,
                    "owner_last_50_win_rate": 0.1,
                    "breeder_last_50_win_rate": 0.2,
                },
                {
                    "race_id": "202609020411",
                    "horse_id": "h2",
                    "horse_name": "サンプルB",
                    "horse_history_key_source": pd.NA,
                    "horse_last_3_avg_rank": pd.NA,
                    "horse_last_5_win_rate": pd.NA,
                    "horse_days_since_last_race": pd.NA,
                    "owner_last_50_win_rate": pd.NA,
                    "breeder_last_50_win_rate": pd.NA,
                },
            ]
        )

        summary = summarize_runner_history_coverage(feature_frame)

        self.assertTrue(summary["available"])
        self.assertEqual(summary["runner_count"], 2)
        self.assertEqual(summary["profile_counts"]["history_rich"], 1)
        self.assertEqual(summary["profile_counts"]["history_poor"], 1)

    def test_build_live_prediction_report_includes_runner_history_coverage(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "race_id": 202609020411,
                    "horse_id": "h1",
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                    "pred_rank": 1,
                    "horse_name": "サンプルA",
                    "score": 0.321,
                    "odds": 3.4,
                    "popularity": 1,
                    "expected_value": 1.091,
                    "ev_rank": 1,
                    "policy_selected": True,
                }
            ]
        )
        runner_history_coverage = {
            "available": True,
            "profile_counts": {"history_rich": 1},
            "group_columns": {"horse_history": ["horse_last_3_avg_rank"], "pedigree_history": ["owner_last_50_win_rate"]},
            "runners": [
                {
                    "race_id": "202609020411",
                    "horse_id": "h1",
                    "horse_name": "サンプルA",
                    "horse_history_key_source": "horse_key",
                    "horse_history_non_null": 1,
                    "horse_history_total": 1,
                    "pedigree_history_non_null": 1,
                    "pedigree_history_total": 1,
                    "history_profile": "history_rich",
                }
            ],
        }

        report = build_live_prediction_report(
            predictions,
            race_date="2026-04-05",
            odds_available_rows=1,
            runner_history_coverage=runner_history_coverage,
        )

        self.assertIn("Runner History Coverage", report)
        self.assertIn("| 1 | サンプルA | 1/1 (1.00) | 1/1 (1.00) | horse_key | history_rich |", report)
        self.assertIn("horse_key", report)
        self.assertIn("1/1 (1.00)", report)
        self.assertIn("history_rich", report)

    def test_build_live_prediction_report_handles_mixed_race_id_types_for_history_merge(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "race_id": 202609020411,
                    "horse_id": "h1",
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                    "pred_rank": 1,
                    "horse_name": "サンプルA",
                    "score": 0.321,
                    "odds": 3.4,
                    "popularity": 1,
                    "expected_value": 1.091,
                    "ev_rank": 1,
                    "policy_selected": True,
                }
            ]
        )
        runner_history_coverage = {
            "available": True,
            "profile_counts": {"history_rich": 1},
            "group_columns": {"horse_history": ["horse_last_3_avg_rank"], "pedigree_history": ["owner_last_50_win_rate"]},
            "runners": [
                {
                    "race_id": "202609020411",
                    "horse_id": "h1",
                    "horse_name": "サンプルA",
                    "horse_history_key_source": "horse_key",
                    "horse_history_non_null": 1,
                    "horse_history_total": 1,
                    "pedigree_history_non_null": 1,
                    "pedigree_history_total": 1,
                    "history_profile": "history_rich",
                }
            ],
        }

        report = build_live_prediction_report(
            predictions,
            race_date="2026-04-05",
            odds_available_rows=1,
            runner_history_coverage=runner_history_coverage,
        )

        self.assertIn("Runner History Coverage", report)
        self.assertIn("| 1 | サンプルA | 1/1 (1.00) | 1/1 (1.00) | horse_key | history_rich |", report)
        self.assertIn("1/1 (1.00)", report)

    def test_build_live_prediction_report_includes_reject_reason_breakdown_when_no_selection(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "race_id": "202609020411",
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                    "pred_rank": 1,
                    "horse_name": "サンプルA",
                    "score": 0.321,
                    "odds": 3.4,
                    "popularity": 1,
                    "expected_value": 0.891,
                    "ev_rank": 1,
                    "policy_selected": False,
                    "policy_reject_reason_primary": "expected_value_below_min_expected_value",
                },
                {
                    "race_id": "202609020411",
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                    "pred_rank": 2,
                    "horse_name": "サンプルB",
                    "score": 0.100,
                    "odds": 30.0,
                    "popularity": 8,
                    "expected_value": 3.000,
                    "ev_rank": 2,
                    "policy_selected": False,
                    "policy_reject_reason_primary": "odds_above_max",
                },
            ]
        )

        report = build_live_prediction_report(predictions, race_date="2026-04-05", odds_available_rows=2)
        summary = summarize_policy_diagnostics(predictions)

        self.assertEqual(summary["policy_selected_rows"], 0)
        self.assertEqual(summary["likely_blocker_reason"], "expected_value_below_min_expected_value")
        self.assertIn("likely_blocker_reason: expected_value_below_min_expected_value", report)
        self.assertIn("top reject reasons: expected_value_below_min_expected_value=1, odds_above_max=1", report)

    def test_parse_live_win_odds_payload_maps_gate_to_odds_and_popularity(self) -> None:
        payload = {
            "official_datetime": "2026-04-05 14:35:19",
            "odds": {
                "1": {
                    "01": ["222.0", "0.0", 15],
                    "04": ["4.3", "0.0", 2],
                    "15": ["2.8", "0.0", 1],
                }
            },
        }

        frame = _parse_live_win_odds_payload(payload, race_id="202609020411")

        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(float(frame.loc[frame["gate_no"] == 4, "odds"].iloc[0]), 4.3)
        self.assertEqual(int(frame.loc[frame["gate_no"] == 15, "popularity"].iloc[0]), 1)
        self.assertEqual(frame.loc[frame["gate_no"] == 1, "odds_official_datetime"].iloc[0], "2026-04-05 14:35:19")

    def test_summarize_live_prediction_tradeoff_reports_top_score_top_ev_and_gap(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "horse_name": "A",
                    "score": 0.30,
                    "expected_value": 0.90,
                    "pred_rank": 1,
                    "ev_rank": 3,
                    "policy_selected": False,
                },
                {
                    "horse_name": "B",
                    "score": 0.20,
                    "expected_value": 1.20,
                    "pred_rank": 3,
                    "ev_rank": 1,
                    "policy_selected": True,
                },
                {
                    "horse_name": "C",
                    "score": 0.25,
                    "expected_value": 1.00,
                    "pred_rank": 2,
                    "ev_rank": 2,
                    "policy_selected": False,
                },
            ]
        )

        summary = summarize_live_prediction_tradeoff(predictions)

        self.assertEqual(summary["top_score_horse"], "A")
        self.assertEqual(summary["top_expected_value_horse"], "B")
        self.assertIn(summary["largest_divergence_horse"], {"A", "B"})
        self.assertEqual(summary["policy_selected_rows"], 1)


if __name__ == "__main__":
    unittest.main()