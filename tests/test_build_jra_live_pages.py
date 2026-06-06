from __future__ import annotations

import importlib.util
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
    def test_render_root_page_applies_prefix_without_duplicate_segment(self) -> None:
        manifests = [{
            "target_date": "2026-06-07",
            "title": "JRA Live Predictions 2026-06-07",
            "source_version": "0.3.0",
            "race_count": 24,
            "row_count": 357,
            "policy_selected_rows": 0,
            "odds_official_datetime_max": "2026-06-06 17:15:53",
            "relative_path": "2026-06-07/",
        }]

        root_html = pages_script.render_root_page(manifests=manifests, href_prefix="jra-live")
        self.assertIn('href="jra-live/2026-06-07/"', root_html)
        self.assertNotIn('href="jra-live/jra-live/2026-06-07/"', root_html)

        jra_live_html = pages_script.render_root_page(manifests=manifests)
        self.assertIn('href="2026-06-07/"', jra_live_html)

    def test_render_live_page_has_parent_navigation_and_focused_refresh_control(self) -> None:
        html = pages_script.render_live_page(page_title="JRA Live Predictions 2026-06-07")

        self.assertIn('class="hero-home-link" href="../"', html)
        self.assertIn('id="focused-refresh-button"', html)
        self.assertIn('id="focused-refresh-status"', html)
        self.assertIn('data-refresh-focused="current"', html)


if __name__ == "__main__":
    unittest.main()