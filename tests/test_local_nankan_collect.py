from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from racing_ml.data.local_nankan_collect import RequestSettings, _decode_html_bytes, _load_or_fetch_html, collect_local_nankan_from_config, parse_local_nankan_race_card_html, parse_local_nankan_race_result_html


class LocalNankanCollectParserTest(unittest.TestCase):
    def test_collect_dry_run_exposes_top_level_read_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            id_dir = base_dir / "data/external/local_nankan/ids"
            id_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"id": "2026041400000101"}]).to_csv(id_dir / "race_ids.csv", index=False)

            summary = collect_local_nankan_from_config(
                {
                    "crawl": {
                        "manifest_file": "artifacts/reports/local_nankan_crawl_manifest.json",
                        "targets": {
                            "race_result": {
                                "enabled": True,
                                "id_file": "data/external/local_nankan/ids/race_ids.csv",
                                "id_column": "id",
                                "output_file": "data/external/local_nankan/results/local_race_result.csv",
                            }
                        },
                    }
                },
                base_dir=base_dir,
                target_filter="race_result",
                dry_run=True,
            )

            self.assertEqual(summary["status"], "planned")
            self.assertEqual(summary["current_phase"], "planned")
            self.assertEqual(summary["recommended_action"], "review_collect_plan")
            self.assertEqual(
                summary["read_order"],
                [
                    "status",
                    "current_phase",
                    "recommended_action",
                    "selected_targets",
                    "highlights",
                    "targets[0].status",
                ],
            )
            manifest_path = base_dir / "artifacts/reports/local_nankan_crawl_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["read_order"], summary["read_order"])

    def test_load_or_fetch_html_persists_sidecar_metadata(self) -> None:
        class _DummyResponse:
            def __init__(self, content: bytes) -> None:
                self.content = content

            def raise_for_status(self) -> None:
                return None

        class _DummySession:
            def __init__(self) -> None:
                self.called = 0

            def get(self, url: str, timeout: float) -> _DummyResponse:
                del url, timeout
                self.called += 1
                return _DummyResponse("<html>ok</html>".encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sample.html"
            session = _DummySession()
            settings = RequestSettings(
                base_url="https://www.nankankeiba.com",
                user_agent="test-agent",
                timeout_sec=1.0,
                delay_sec=0.0,
                retry_count=1,
                retry_backoff_sec=0.0,
                overwrite=False,
            )

            html, fetched, last_fetch_at, provenance = _load_or_fetch_html(
                session=session,
                url="https://www.nankankeiba.com/syousai/sample.do",
                output_path=output_path,
                settings=settings,
                last_fetch_at=None,
            )

            self.assertEqual(html, "<html>ok</html>")
            self.assertTrue(fetched)
            self.assertIsNotNone(last_fetch_at)
            self.assertEqual(provenance["fetch_mode"], "fetched")
            self.assertTrue((Path(tmp_dir) / "sample.html.meta.json").exists())

            _, fetched_again, _, cached_provenance = _load_or_fetch_html(
                session=session,
                url="https://www.nankankeiba.com/syousai/sample.do",
                output_path=output_path,
                settings=settings,
                last_fetch_at=last_fetch_at,
            )

            self.assertFalse(fetched_again)
            self.assertEqual(session.called, 1)
            self.assertEqual(cached_provenance["fetch_mode"], "cache_manifest")
            self.assertEqual(cached_provenance["snapshot_at"], provenance["snapshot_at"])

    def test_parse_race_result_supports_legacy_nankan_layout(self) -> None:
        html_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "local_nankan"
            / "raw_html"
            / "race_result"
            / "2006032921140301.html"
        )
        html = _decode_html_bytes(html_path.read_bytes())

        frame = parse_local_nankan_race_result_html(html, "2006032921140301")

        self.assertEqual(len(frame), 10)
        self.assertEqual(frame.iloc[0]["date"], "2006-03-29")
        self.assertEqual(frame.iloc[0]["track"], "川崎")
        self.assertEqual(frame.iloc[0]["distance"], 1400)
        self.assertEqual(frame.iloc[0]["weather"], "晴")
        self.assertEqual(frame.iloc[0]["ground_condition"], "良")
        self.assertEqual(frame.iloc[0]["race_name"], "Ｃ３(一)(二)")
        self.assertEqual(frame.iloc[0]["horse_key"], "1999100679")
        self.assertEqual(frame.iloc[0]["horse_name"], "カソクソウチ")
        self.assertEqual(frame.iloc[0]["jockey_key"], "031113")
        self.assertEqual(frame.iloc[0]["trainer_key"], "010298")
        self.assertEqual(frame.iloc[0]["finish_time"], "1:30.8")
        self.assertEqual(frame.iloc[0]["closing_time_3f"], "40.8")

    def test_parse_race_result_handles_cancelled_race_page(self) -> None:
        html_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "local_nankan"
            / "raw_html"
            / "race_result"
            / "2007081820090301.html"
        )
        html = _decode_html_bytes(html_path.read_bytes())

        frame = parse_local_nankan_race_result_html(html, "2007081820090301")

        self.assertTrue(frame.empty)
        self.assertEqual(frame.columns.tolist()[0:7], [
            "date",
            "race_id",
            "track",
            "distance",
            "weather",
            "ground_condition",
            "race_name",
        ])

    def test_parse_race_card_persists_provenance_and_post_time(self) -> None:
        base_dir = Path(__file__).resolve().parents[1] / "data" / "external" / "local_nankan" / "raw_html"
        html = _decode_html_bytes((base_dir / "race_card" / "2025030421130207.html").read_bytes())
        odds_js = (base_dir / "race_card_odds" / "2025030421130207.js").read_text(encoding="utf-8")

        frame = parse_local_nankan_race_card_html(
            html,
            "2025030421130207",
            odds_js=odds_js,
            card_provenance={
                "source_url": "https://www.nankankeiba.com/syousai/2025030421130207.do",
                "fetch_mode": "cache_manifest",
                "snapshot_at": "2025-03-04T04:00:00Z",
            },
            odds_provenance={
                "source_url": "https://www.nankankeiba.com/oddsJS/2025030421130207.do",
                "fetch_mode": "cache_manifest",
                "snapshot_at": "2025-03-04T06:00:00Z",
            },
        )

        row = frame.iloc[0]
        self.assertEqual(str(row["post_time"]), "14:20")
        self.assertEqual(str(row["scheduled_post_at"]), "2025-03-04T14:20:00+09:00")
        self.assertEqual(str(row["card_fetch_mode"]), "cache_manifest")
        self.assertEqual(str(row["card_snapshot_relation"]), "pre_race")
        self.assertEqual(str(row["odds_snapshot_relation"]), "post_race")
        self.assertIn("oddsJS/2025030421130207.do", str(row["odds_source_url"]))
        self.assertEqual(str(row["horse_id"]), "2025030421130207:1")
        self.assertEqual(str(row["odds"]), "52.5")
        self.assertEqual(int(row["popularity"]), 5)


if __name__ == "__main__":
    unittest.main()
