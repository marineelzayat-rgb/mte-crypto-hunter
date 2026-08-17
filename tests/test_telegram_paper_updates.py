from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mte_crypto.daemon import send_paper_telegram_updates
from mte_crypto.futures_shadow import open_futures_shadow, update_futures_shadow
from mte_crypto.paper_portfolio import open_paper_position, update_paper_portfolio


UTC = timezone.utc


class TelegramPaperUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.now = datetime(2026, 8, 17, 18, 5, tzinfo=UTC)
        self.environment = patch.dict(
            os.environ,
            {"MTE_TELEGRAM_BOT_TOKEN": "test", "MTE_TELEGRAM_CHAT_ID": "1"},
        )
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def _closed_trades(self):
        spot_open = open_paper_position(
            self.data_dir,
            {
                "id": "test|TESTUSDT",
                "symbol": "TESTUSDT",
                "state": "EARLY_PULSE",
                "price": 100.0,
            },
            now=self.now,
        )
        snapshot = {
            "bid": 99.9,
            "ask": 100.0,
            "mark": 99.95,
            "index": 99.94,
            "last_funding_rate": 0.0001,
            "next_funding_time": int((self.now + timedelta(hours=2)).timestamp() * 1000),
        }
        open_futures_shadow(
            self.data_dir, spot_open, snapshot, now=self.now
        )
        close_time = self.now + timedelta(minutes=5)
        spot_closed = update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 90.0}],
            now=close_time,
        )
        futures_closed = update_futures_shadow(
            self.data_dir,
            {"TESTUSDT": {**snapshot, "bid": 89.9, "ask": 90.0, "mark": 89.95}},
            spot_closed,
            now=close_time,
        )
        return spot_closed, futures_closed, close_time

    @patch("mte_crypto.daemon.send_telegram", return_value=True)
    def test_sends_balance_after_close_and_one_daily_summary(self, mocked_send):
        spot_closed, futures_closed, close_time = self._closed_trades()
        first = send_paper_telegram_updates(
            self.data_dir, spot_closed, futures_closed, now=close_time
        )
        self.assertEqual(len(first), 2)
        self.assertIn("Balances: Spot", first[0])
        self.assertIn("Futures 2x", first[0])
        self.assertIn("DAILY SUMMARY", first[1])

        second = send_paper_telegram_updates(
            self.data_dir, [], [], now=close_time + timedelta(minutes=5)
        )
        self.assertEqual(second, [])
        self.assertEqual(mocked_send.call_count, 2)


if __name__ == "__main__":
    unittest.main()
