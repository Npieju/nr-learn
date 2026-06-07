from __future__ import annotations

import importlib.util
from itertools import permutations
from pathlib import Path
import sys
import unittest


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


pages_script = _load_script_module("test_run_build_jra_live_pages", "scripts/publishing/run_build_jra_live_pages.py")


class BuildJraLivePagesTest(unittest.TestCase):
    def test_build_overview_picks_prefers_high_score_then_high_ev_longshot(self) -> None:
        frame = pages_script.pd.DataFrame(
            [
                {"horse_id": 20260503020105, "horse_name": "セイウンハーデス", "score": 0.42, "odds": 3.4, "expected_value": 1.05},
                {"horse_id": 20260503020108, "horse_name": "ライバル", "score": 0.31, "odds": 5.8, "expected_value": 0.96},
                {"horse_id": 20260503020112, "horse_name": "オオアナ", "score": 0.11, "odds": 18.6, "expected_value": 1.42},
            ]
        )

        picks = pages_script._build_overview_picks(frame)

        self.assertEqual(picks["favorite"]["horseLabel"], "5 セイウンハーデス")
        self.assertEqual(picks["contender"]["horseLabel"], "8 ライバル")
        self.assertEqual(picks["longshot"]["horseLabel"], "12 オオアナ")

    def test_build_harville_rows_for_wide_keeps_odds_range(self) -> None:
        config = next(item for item in pages_script.HARVILLE_MARKET_OPTIONS if item["key"] == "wide")
        rows = pages_script._build_harville_rows_for_market(
            config=config,
            actual_rows=[{"組み合わせ": "1-2", "オッズ": "10.5-12.8"}],
            horse_name_map={"1": "A", "2": "B", "3": "C"},
            horse_numbers=["1", "2", "3"],
            win_odds_map={"1": 2.4, "2": 3.1, "3": 6.8},
            win_probability_map={"1": 0.42, "2": 0.33, "3": 0.25},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_odds"], 10.5)
        self.assertEqual(rows[0]["market_odds_max"], 12.8)

    def test_parse_odds_value_ignores_dot_placeholders_in_ranges(self) -> None:
        self.assertIsNone(pages_script._parse_odds_value("."))
        self.assertIsNone(pages_script._parse_odds_value(".-."))
        self.assertEqual(pages_script._parse_odds_value("12.4-."), 12.4)
        self.assertEqual(pages_script._parse_odds_value(".-18.6"), 18.6)

    def test_render_root_page_applies_prefix_without_duplicate_segment(self) -> None:
        manifests = [
            {
                "target_date": "2026-06-07",
                "title": "JRA Live Predictions 2026-06-07",
                "source_version": "0.3.0",
                "race_count": 24,
                "row_count": 357,
                "policy_selected_rows": 0,
                "odds_official_datetime_max": "2026-06-06 17:15:53",
                "profile": "current_recommended_serving_baseline_r20260325_2025_latest",
                "model_artifact_suffix": "r20260325_current_recommended_serving_2025_latest_benchmark_refresh",
                "score_source_model_config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
                "relative_path": "2026-06-07/",
            },
            {
                "target_date": "2026-05-31",
                "title": "JRA Live Predictions 2026-05-31",
                "source_version": "0.2.9",
                "race_count": 24,
                "row_count": 341,
                "policy_selected_rows": 1,
                "odds_official_datetime_max": "2026-05-30 17:09:00",
                "profile": "current_recommended_serving_favonly_composite_budget_revision_scoped_2025_latest",
                "model_artifact_suffix": None,
                "score_source_model_config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_support_preserving_prob_favonly_composite_budget_revision_scoped.yaml",
                "relative_path": "jra-live/2026-05-31/",
            },
        ]

        root_html = pages_script.render_root_page(manifests=manifests, href_prefix="jra-live")
        self.assertIn('href="jra-live/2026-06-07/"', root_html)
        self.assertIn('href="jra-live/2026-05-31/"', root_html)
        self.assertNotIn('href="jra-live/jra-live/2026-06-07/"', root_html)
        self.assertNotIn('href="jra-live/jra-live/2026-05-31/"', root_html)
        self.assertEqual(root_html.count('class="card"'), 2)
        self.assertIn('profile=current_recommended_serving_baseline_r20260325_2025_latest', root_html)
        self.assertIn('artifact=r20260325_current_recommended_serving_2025_latest_benchmark_refresh', root_html)
        self.assertIn('model=model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml', root_html)
        self.assertIn('artifact=latest', root_html)

        jra_live_html = pages_script.render_root_page(manifests=manifests)
        self.assertIn('href="2026-06-07/"', jra_live_html)
        self.assertIn('href="2026-05-31/"', jra_live_html)
        self.assertNotIn('href="jra-live/2026-05-31/"', jra_live_html)

    def test_render_live_page_has_parent_navigation_and_focused_refresh_control(self) -> None:
        html = pages_script.render_live_page(page_title="JRA Live Predictions 2026-06-07")

        self.assertIn('class="hero-home-link" href="../"', html)
        self.assertIn('id="focused-refresh-button"', html)
        self.assertIn('id="focused-refresh-status"', html)
        self.assertIn('data-refresh-focused="current"', html)
        self.assertIn('id="harville-ignore-long-odds"', html)
        self.assertIn('id="harville-exclude-filters"', html)
        self.assertIn('>消し馬<', html)
        self.assertIn('.harville-anchor-select {', html)
        self.assertIn('min-width: 0;\n      }\n      .harville-filter-stack {', html)
        self.assertIn('class="harville-filter-grid"', html)
        self.assertNotIn('grid-auto-flow: column;', html)
        self.assertIn('grid-template-columns: repeat(${excludeOptions.length}, minmax(28px, max-content));', html)
        self.assertIn('function harvilleOverviewTableHtml', html)
        self.assertIn('const quinellaIndex = markets.findIndex((market) => market.key === "quinella");', html)
        self.assertIn('markets.splice(quinellaIndex, 0, winCompareMarket);', html)
        self.assertIn('table-layout: auto;', html)
        self.assertIn('min-width: 128px;', html)
        self.assertNotIn('min-width: 1180px;', html)
        self.assertIn('const filteredOverviewWinCompareRows = filteredHarvilleRows(harville.winCompareRows || [], [], state.harvilleExcludedHorses, false);', html)
        self.assertIn('const overviewSourceRowsByMarket = harville.overviewRowsByMarket || harville.rowsByMarket || {};', html)
        self.assertIn('filteredHarvilleRows(overviewSourceRowsByMarket?.[market.key] || [], selectedAnchors, state.harvilleExcludedHorses, false)', html)
        self.assertIn('label: "単勝"', html)
        self.assertIn('? harvilleOverviewTableHtml(race, filteredOverviewRowsByMarket, filteredOverviewWinCompareRows)', html)
        self.assertIn('>◎<', html)
        self.assertIn('>◯<', html)
        self.assertIn('>大穴<', html)
        self.assertIn('${formatOddsNumber(lower)} - ${formatOddsNumber(upper)}倍', html)

    def test_build_harville_payload_for_race_separates_detail_limit_and_overview_source(self) -> None:
        race_frame = pages_script.pd.DataFrame(
            [
                {"gate_no": horse_no, "horse_name": f"Horse {horse_no}", "score": 1.0 / 7, "expected_value": 1.0}
                for horse_no in range(1, 8)
            ]
        )
        trifecta_rows = [
            {"組み合わせ": f"{a}-{b}-{c}", "オッズ": f"{1000 + index * 10:.1f}"}
            for index, (a, b, c) in enumerate(permutations(range(1, 8), 3), start=1)
        ]
        analyze_payload = {
            "entries": [
                {"馬番": str(horse_no), "馬名": f"Horse {horse_no}"}
                for horse_no in range(1, 8)
            ],
            "odds": {
                "単勝": [
                    {"馬番": str(horse_no), "オッズ": f"{horse_no + 1}.0", "人気": horse_no}
                    for horse_no in range(1, 8)
                ],
                "三連単": trifecta_rows,
            },
            "race": {"odds_updated_at": "2026-06-07 12:00:00", "analyzed_at": "2026-06-07T12:00:00+00:00"},
        }

        original_fetch = pages_script._fetch_race_analyze_payload
        try:
            pages_script._fetch_race_analyze_payload = lambda race_id: analyze_payload
            payload = pages_script._build_harville_payload_for_race(race_frame, "202605030211")
        finally:
            pages_script._fetch_race_analyze_payload = original_fetch

        self.assertEqual(len(payload["rowsByMarket"]["trifecta"]), pages_script.HARVILLE_DETAIL_LIMIT)
        self.assertEqual(len(payload["overviewRowsByMarket"]["trifecta"]), len(trifecta_rows))
        self.assertEqual(next(item["rows"] for item in payload["marketOptions"] if item["key"] == "trifecta"), len(trifecta_rows))
        horse_three = next(item for item in payload["overviewRows"] if item["horseNo"] == "3")
        self.assertIsNotNone(horse_three["metrics"]["trifecta"]["actual"])


if __name__ == "__main__":
    unittest.main()