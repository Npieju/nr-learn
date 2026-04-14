from __future__ import annotations

import unittest
from unittest.mock import patch

from racing_ml.data.local_nankan_race_list import _filter_upcoming_races, parse_local_nankan_program_html


class LocalNankanRaceListTest(unittest.TestCase):
    def test_program_parser_uses_result_links_when_required(self) -> None:
        html = """
        <html>
          <body>
            <a href="/race_info/2013011518100101.do">3歳</a>
            <a href="/syousai/2013011518100101.do">詳細</a>
          </body>
        </html>
        """

        frame = parse_local_nankan_program_html(
            html,
            meeting_id="20130115181001",
            source_page_url="https://www.nankankeiba.com/program/20130115181001.do",
            require_result_link=True,
        )

        self.assertTrue(frame.empty)

    def test_program_parser_falls_back_to_result_links(self) -> None:
        html = """
        <html>
          <body>
            <li class="nk23_c-block01__list__item">
              <div class="nk23_c-block01__list__top">
                <p class="nk23_c-block01__texts"><span class="nk23_c-block01__text">14:30</span></p>
              </div>
              <a href="/result/2014011518100201.do">結果</a>
              <a href="/syousai/2014011518100201.do">詳細</a>
            </li>
          </body>
        </html>
        """

        frame = parse_local_nankan_program_html(
            html,
            meeting_id="20140115181002",
            source_page_url="https://www.nankankeiba.com/program/20140115181002.do",
            require_result_link=True,
        )

        self.assertEqual(frame["race_id"].tolist(), ["2014011518100201"])
        self.assertEqual(frame["post_time"].tolist(), ["14:30"])
        self.assertEqual(frame["scheduled_post_at"].tolist(), ["2014-01-15T14:30:00+09:00"])

    def test_program_parser_ignores_result_links_from_other_meetings(self) -> None:
        html = """
        <html>
          <body>
            <a href="/result/2014011518100201.do">別meeting結果</a>
            <a href="/syousai/2014021019110101.do">詳細</a>
          </body>
        </html>
        """

        frame = parse_local_nankan_program_html(
            html,
            meeting_id="20140210191101",
            source_page_url="https://www.nankankeiba.com/program/20140210191101.do",
            require_result_link=True,
        )

        self.assertTrue(frame.empty)

    def test_filter_upcoming_races_keeps_only_future_post_time(self) -> None:
        html = """
        <html>
          <body>
            <li class="nk23_c-block01__list__item">
              <div class="nk23_c-block01__list__top">
                <span class="nk23_c-block01__label">1R</span>
                <p class="nk23_c-block01__texts"><span class="nk23_c-block01__text">14:30</span></p>
              </div>
              <a href="/syousai/2026041420010201.do">詳細</a>
            </li>
            <li class="nk23_c-block01__list__item">
              <div class="nk23_c-block01__list__top">
                <span class="nk23_c-block01__label">2R</span>
                <p class="nk23_c-block01__texts"><span class="nk23_c-block01__text">15:35</span></p>
              </div>
              <a href="/syousai/2026041420010202.do">詳細</a>
            </li>
          </body>
        </html>
        """

        frame = parse_local_nankan_program_html(
            html,
            meeting_id="20260414200102",
            source_page_url="https://www.nankankeiba.com/program/20260414200102.do",
        )
        filtered = _filter_upcoming_races(frame, as_of="2026-04-14T15:00:00+09:00")

        self.assertEqual(filtered["race_id"].tolist(), ["2026041420010202"])

    def test_filter_upcoming_races_defaults_as_of_to_current_tokyo_time(self) -> None:
        html = """
        <html>
          <body>
            <li class="nk23_c-block01__list__item">
              <div class="nk23_c-block01__list__top">
                <span class="nk23_c-block01__label">1R</span>
                <p class="nk23_c-block01__texts"><span class="nk23_c-block01__text">14:30</span></p>
              </div>
              <a href="/syousai/2026041420010201.do">詳細</a>
            </li>
            <li class="nk23_c-block01__list__item">
              <div class="nk23_c-block01__list__top">
                <span class="nk23_c-block01__label">2R</span>
                <p class="nk23_c-block01__texts"><span class="nk23_c-block01__text">15:35</span></p>
              </div>
              <a href="/syousai/2026041420010202.do">詳細</a>
            </li>
          </body>
        </html>
        """

        frame = parse_local_nankan_program_html(
            html,
            meeting_id="20260414200102",
            source_page_url="https://www.nankankeiba.com/program/20260414200102.do",
        )

        with patch("racing_ml.data.local_nankan_race_list.pd.Timestamp.now", return_value=parse_local_nankan_program_html.__globals__["pd"].Timestamp("2026-04-14T15:00:00+09:00")):
            filtered = _filter_upcoming_races(frame, as_of=None)

        self.assertEqual(filtered["race_id"].tolist(), ["2026041420010202"])


if __name__ == "__main__":
    unittest.main()