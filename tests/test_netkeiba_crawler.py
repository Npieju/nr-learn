from __future__ import annotations

from pathlib import Path
import sys
import unittest

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.data.netkeiba_crawler import _extract_mobile_race_metadata


class NetkeibaCrawlerMetadataTest(unittest.TestCase):
    def test_extract_mobile_race_metadata_uses_title_and_race_data_fallback(self) -> None:
        html = """
        <html>
          <head>
            <title>大阪杯(G1) 5走表示 | 2026年4月5日 阪神11R レース情報(JRA) - netkeiba</title>
          </head>
          <body>
            <h1 class="Race_Name">大阪杯</h1>
            <div class="Race_Data">15:40 <span class="Turf">芝</span><span>2000m</span>(右 A) 15頭 <span class="WeatherData"> 晴</span><span class="Item03">良</span></div>
          </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")

        metadata = _extract_mobile_race_metadata(soup, "202609020411")

        self.assertEqual(metadata["date"], "2026-04-05")
        self.assertEqual(metadata["track"], "阪神")
        self.assertEqual(metadata["distance"], "2000")
        self.assertEqual(metadata["weather"], "晴")
        self.assertEqual(metadata["ground_condition"], "良")
        self.assertEqual(metadata["芝・ダート区分"], "芝")
        self.assertEqual(metadata["右左回り・直線区分"], "右")


if __name__ == "__main__":
    unittest.main()