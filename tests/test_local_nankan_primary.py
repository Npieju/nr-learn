from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

from racing_ml.data.local_nankan_id_prep import _build_horse_key_frame
from racing_ml.data.local_nankan_primary import materialize_local_nankan_primary_from_config


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pre_race_primary_script = _load_script_module(
    "test_run_materialize_local_nankan_pre_race_primary",
    "scripts/run_materialize_local_nankan_pre_race_primary.py",
)


class LocalNankanPrimaryMaterializeTest(unittest.TestCase):
    def test_materialize_keeps_jra_comparable_result_granularity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            result_path = base_dir / "data/external/local_nankan/results/local_race_result.csv"
            card_path = base_dir / "data/external/local_nankan/racecard/local_racecard.csv"
            pedigree_path = base_dir / "data/external/local_nankan/pedigree/local_pedigree.csv"
            output_path = base_dir / "data/local_nankan/raw/local_nankan_primary.csv"
            manifest_path = base_dir / "artifacts/reports/local_nankan_primary_materialize_manifest.json"
            result_keys_path = base_dir / "data/local_nankan/raw/local_nankan_race_result_keys.csv"
            race_card_output_path = base_dir / "data/local_nankan/raw/local_nankan_race_card.csv"
            pedigree_output_path = base_dir / "data/local_nankan/raw/local_nankan_pedigree.csv"

            result_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.parent.mkdir(parents=True, exist_ok=True)
            pedigree_path.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "race_id": "nar001",
                        "horse_id": "nar001:1",
                        "horse_key": "horse001",
                        "horse_name": "Alpha",
                        "rank": 1,
                        "odds": 2.5,
                        "popularity": 1,
                        "finish_time": "1:23.4",
                        "margin": "",
                        "closing_time_3f": 37.1,
                        "passing_order": "1-1-1-1",
                        "jockey_id": "j001",
                        "trainer_id": "t001",
                        "track": "大井",
                        "distance": 1400,
                        "weather": "晴",
                        "ground_condition": "良",
                        "sex": "牡",
                        "age": 3,
                        "carried_weight": 56.0,
                        "weight": 480,
                        "weight_change": -2,
                        "frame_no": 1,
                        "gate_no": 1,
                    }
                ]
            ).to_csv(result_path, index=False)

            pd.DataFrame(
                [
                    {
                        "race_id": "nar001",
                        "horse_id": "nar001:1",
                        "horse_key": "horse001",
                        "owner_name": "Owner A",
                        "breeder_name": "Breeder A",
                        "weight_change": -2,
                        "carried_weight": 56.0,
                        "popularity": 1,
                        "post_time": "15:20",
                        "scheduled_post_at": "2025-01-01T15:20:00+09:00",
                        "card_source_url": "https://www.nankankeiba.com/syousai/nar001.do",
                        "card_fetch_mode": "cache_manifest",
                        "card_snapshot_at": "2025-01-01T05:00:00Z",
                        "card_snapshot_relation": "pre_race",
                        "odds_source_url": "https://www.nankankeiba.com/oddsJS/nar001.do",
                        "odds_fetch_mode": "cache_manifest",
                        "odds_snapshot_at": "2025-01-01T05:05:00Z",
                        "odds_snapshot_relation": "pre_race",
                    }
                ]
            ).to_csv(card_path, index=False)

            pd.DataFrame(
                [
                    {
                        "horse_key": "horse001",
                        "sire_name": "Sire A",
                        "dam_name": "Dam A",
                        "damsire_name": "DamSire A",
                    }
                ]
            ).to_csv(pedigree_path, index=False)

            summary = materialize_local_nankan_primary_from_config(
                {"dataset": {"raw_dir": "data/local_nankan/raw"}},
                base_dir=base_dir,
                race_result_path=result_path,
                race_card_path=card_path,
                pedigree_path=pedigree_path,
                output_file=output_path,
                manifest_file=manifest_path,
                dry_run=False,
            )

            self.assertEqual(summary["status"], "completed")
            self.assertIn("generated_files", summary)

            frame = pd.read_csv(output_path)
            row = frame.iloc[0]
            self.assertEqual(frame.loc[0, "finish_time"], "1:23.4")
            self.assertEqual(float(str(row["closing_time_3f"])), 37.1)
            self.assertEqual(str(row["passing_order"]), "1-1-1-1")
            self.assertEqual(float(str(row["carried_weight"])), 56.0)
            self.assertEqual(int(str(row["weight_change"])), -2)
            self.assertEqual(str(row["owner_name"]), "Owner A")
            self.assertEqual(str(row["breeder_name"]), "Breeder A")
            self.assertEqual(str(row["sire_name"]), "Sire A")
            self.assertEqual(str(row["post_time"]), "15:20")
            self.assertEqual(str(row["card_fetch_mode"]), "cache_manifest")
            self.assertEqual(str(row["card_snapshot_relation"]), "pre_race")
            self.assertIn("oddsJS/nar001.do", str(row["odds_source_url"]))
            self.assertEqual(int(str(row["is_win"])), 1)

            result_keys = pd.read_csv(result_keys_path)
            race_card_output = pd.read_csv(race_card_output_path)
            pedigree_output = pd.read_csv(pedigree_output_path)

            self.assertEqual(str(result_keys.loc[0, "horse_key"]), "horse001")
            self.assertEqual(str(race_card_output.loc[0, "horse_id"]), "nar001:1")
            self.assertEqual(str(race_card_output.loc[0, "card_snapshot_relation"]), "pre_race")
            self.assertEqual(str(pedigree_output.loc[0, "sire_name"]), "Sire A")

    def test_materialize_falls_back_to_race_and_horse_key_for_non_runners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            result_path = base_dir / "data/external/local_nankan/results/local_race_result.csv"
            card_path = base_dir / "data/external/local_nankan/racecard/local_racecard.csv"
            output_path = base_dir / "data/local_nankan/raw/local_nankan_primary.csv"
            manifest_path = base_dir / "artifacts/reports/local_nankan_primary_materialize_manifest.json"

            result_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "race_id": "nar001",
                        "horse_id": "nar001:4",
                        "horse_key": "2000106224",
                        "horse_name": "Excluded Horse",
                        "rank": None,
                        "popularity": None,
                        "track": "大井",
                        "distance": 1400,
                    }
                ]
            ).to_csv(result_path, index=False)

            pd.DataFrame(
                [
                    {
                        "race_id": "nar001",
                        "horse_id": "nar001:除外",
                        "horse_key": "2000106224",
                        "horse_name": "Excluded Horse",
                        "owner_name": "Owner X",
                        "breeder_name": "Breeder X",
                        "odds": None,
                        "popularity": None,
                    }
                ]
            ).to_csv(card_path, index=False)

            summary = materialize_local_nankan_primary_from_config(
                {"dataset": {"raw_dir": "data/local_nankan/raw"}},
                base_dir=base_dir,
                race_result_path=result_path,
                race_card_path=card_path,
                output_file=output_path,
                manifest_file=manifest_path,
                dry_run=False,
            )

            self.assertEqual(summary["status"], "completed")
            frame = pd.read_csv(output_path)
            row = frame.iloc[0]
            self.assertEqual(str(row["owner_name"]), "Owner X")
            self.assertEqual(str(row["breeder_name"]), "Breeder X")
            self.assertEqual(float(row["odds"]), 0.0)


class LocalNankanHorseKeyPrepTest(unittest.TestCase):
    def test_build_horse_key_frame_excludes_non_numeric_seed_keys(self) -> None:
        seed_frame = pd.DataFrame(
            {
                "horse_key": ["LN_HORSE_0001", "LN_HORSE_0002", "2001109285"],
            }
        )
        target_output = Path("/tmp/nonexistent_local_nankan_race_result.csv")

        frame = _build_horse_key_frame(
            seed_frame,
            horse_key_column="horse_key",
            targets={
                "race_result": {"output_file": str(target_output)},
                "race_card": {"output_file": str(target_output)},
                "pedigree": {"output_file": str(target_output)},
            },
            base_dir=Path("/"),
            include_completed=False,
            limit=None,
        )

        self.assertEqual(frame["horse_key"].tolist(), ["2001109285"])


class LocalNankanPreRacePrimaryScriptTest(unittest.TestCase):
    def test_pre_race_primary_normalizes_not_ready_summary_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary_output = tmp_path / "local_nankan_pre_race_ready_summary.json"
            manifest_file = tmp_path / "local_nankan_primary_pre_race_ready_materialize_manifest.json"

            race_card_frame = pd.DataFrame([{"race_id": "nar001"}])
            race_result_frame = pd.DataFrame([{"race_id": "nar001"}])
            ready_card_frame = pd.DataFrame([{"race_id": "nar001"}])

            with patch.object(pre_race_primary_script, "load_yaml", return_value={"dataset": {}}), patch.object(
                pre_race_primary_script.pd,
                "read_csv",
                side_effect=[race_card_frame, race_result_frame],
            ), patch.object(
                pre_race_primary_script,
                "filter_result_ready_pre_race_only",
                return_value=ready_card_frame,
            ), patch.object(
                pre_race_primary_script,
                "build_pre_race_only_materialization_summary",
                return_value={
                    "result_ready_races": 0,
                    "pending_result_races": 24,
                },
            ), patch.object(
                pre_race_primary_script,
                "write_csv_file",
            ), patch.object(
                sys,
                "argv",
                [
                    "run_materialize_local_nankan_pre_race_primary.py",
                    "--filtered-race-card-output",
                    str(ROOT / "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_card.csv"),
                    "--filtered-race-result-output",
                    str(ROOT / "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_result.csv"),
                    "--primary-output-file",
                    str(ROOT / "data/local_nankan_pre_race_ready/raw/unit_not_ready_primary.csv"),
                    "--summary-output",
                    str(summary_output),
                    "--manifest-file",
                    str(manifest_file),
                ],
            ):
                exit_code = pre_race_primary_script.main()

            self.assertEqual(exit_code, 2)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(summary["filtered_race_card_output"], "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_card.csv")
            self.assertEqual(summary["filtered_race_result_output"], "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_result.csv")
            self.assertEqual(summary["primary_output_file"], "data/local_nankan_pre_race_ready/raw/unit_not_ready_primary.csv")
            self.assertEqual(summary["manifest_file"], str(manifest_file.relative_to(ROOT)))

    def test_pre_race_primary_normalizes_completed_summary_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary_output = tmp_path / "local_nankan_pre_race_ready_summary.json"
            manifest_file = tmp_path / "local_nankan_primary_pre_race_ready_materialize_manifest.json"

            race_card_frame = pd.DataFrame([{"race_id": "nar001"}])
            race_result_frame = pd.DataFrame([{"race_id": "nar001"}])
            ready_card_frame = pd.DataFrame([{"race_id": "nar001"}])

            with patch.object(pre_race_primary_script, "load_yaml", return_value={"dataset": {}}), patch.object(
                pre_race_primary_script.pd,
                "read_csv",
                side_effect=[race_card_frame, race_result_frame],
            ), patch.object(
                pre_race_primary_script,
                "filter_result_ready_pre_race_only",
                return_value=ready_card_frame,
            ), patch.object(
                pre_race_primary_script,
                "build_pre_race_only_materialization_summary",
                return_value={
                    "result_ready_races": 1,
                    "pending_result_races": 0,
                },
            ), patch.object(
                pre_race_primary_script,
                "write_csv_file",
            ), patch.object(
                pre_race_primary_script,
                "materialize_local_nankan_primary_from_config",
                return_value={
                    "status": "completed",
                    "current_phase": "primary_materialized",
                    "recommended_action": "review_pre_race_primary",
                    "generated_files": {
                        "output_file": str(ROOT / "data/local_nankan_pre_race_ready/raw/local_nankan_primary_pre_race_ready.csv"),
                        "manifest_file": str(ROOT / "artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json"),
                    },
                },
            ), patch.object(
                sys,
                "argv",
                [
                    "run_materialize_local_nankan_pre_race_primary.py",
                    "--summary-output",
                    str(summary_output),
                    "--manifest-file",
                    str(manifest_file),
                ],
            ):
                exit_code = pre_race_primary_script.main()

            self.assertEqual(exit_code, 0)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["filtered_race_card_output"],
                "data/local_nankan_pre_race_ready/raw/local_nankan_race_card_pre_race_ready.csv",
            )
            self.assertEqual(
                summary["manifest_file"],
                str(manifest_file.relative_to(ROOT)),
            )
            self.assertEqual(
                summary["primary_materialize_summary"]["generated_files"]["output_file"],
                "data/local_nankan_pre_race_ready/raw/local_nankan_primary_pre_race_ready.csv",
            )
            self.assertEqual(
                summary["primary_materialize_summary"]["generated_files"]["manifest_file"],
                "artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json",
            )


if __name__ == "__main__":
    unittest.main()
