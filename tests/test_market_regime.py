from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from mte_crypto.market_regime import detect_bull_regime


UTC = timezone.utc


class BullRegimeTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 8, 21, 12, tzinfo=UTC)
        self.markets = [
            {
                "symbol": f"COIN{i}USDT",
                "quote_volume_24h": 2_000_000.0,
                "return_24h": 0.04 if i < 8 else -0.01,
            }
            for i in range(10)
        ]

    def frame(self, *, confirmed=False):
        rows = []
        start = self.now - timedelta(days=21)
        for index in range(20):
            opened = start + timedelta(days=index)
            rows.append(
                {
                    "open_time": opened,
                    "open": 95.0,
                    "high": 100.0,
                    "low": 90.0,
                    "close": 96.0,
                    "close_time": opened + timedelta(hours=23, minutes=59),
                }
            )
        if confirmed:
            opened = self.now - timedelta(days=1)
            rows.append(
                {
                    "open_time": opened,
                    "open": 98.0,
                    "high": 106.0,
                    "low": 97.0,
                    "close": 105.0,
                    "close_time": self.now - timedelta(minutes=1),
                }
            )
            current_open = self.now
            current_close = 104.0
        else:
            current_open = self.now.replace(hour=0)
            current_close = 105.0
        rows.append(
            {
                "open_time": current_open,
                "open": 100.0 if not confirmed else 105.0,
                "high": 106.0,
                "low": 99.0,
                "close": current_close,
                "close_time": self.now + timedelta(hours=12),
            }
        )
        return pd.DataFrame(rows).set_index("open_time")

    def test_current_broad_breakout_activates_bull_mode(self):
        result = detect_bull_regime(
            self.markets,
            self.frame(),
            now=self.now,
        )
        self.assertTrue(result["active"])
        self.assertTrue(result["current_breakout"])
        self.assertEqual(result["state"], "BULL_BREAKOUT")

    def test_confirmed_close_keeps_bull_mode_active_next_session(self):
        result = detect_bull_regime(
            self.markets,
            self.frame(confirmed=True),
            now=self.now,
        )
        self.assertTrue(result["active"])
        self.assertTrue(result["confirmed_breakout"])

    def test_narrow_market_does_not_activate(self):
        narrow = [{**row, "return_24h": -0.01} for row in self.markets]
        result = detect_bull_regime(narrow, self.frame(), now=self.now)
        self.assertFalse(result["active"])


if __name__ == "__main__":
    unittest.main()
