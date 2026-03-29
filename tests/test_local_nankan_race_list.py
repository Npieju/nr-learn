from __future__ import annotations

import unittest

from racing_ml.data.local_nankan_race_list import parse_local_nankan_program_html


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
            <a href="/result/2014011518100201.do">結果</a>
            <a href="/syousai/2014011518100201.do">詳細</a>
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


if __name__ == "__main__":
    unittest.main()