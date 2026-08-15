import unittest

import numpy as np
import pandas as pd

from mte_crypto.hunter_mte_backtest import _first_passage
from mte_crypto.mte_ignition import compute_mte_ignitions


def synthetic_frame(rows: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=rows, freq="h", tz="UTC")
    x = np.arange(rows)
    close = 100.0 + 0.01 * x + 4.0 * np.sin(x / 31.0) + 1.5 * np.sin(x / 7.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    quote = 2_000_000.0 + 500_000.0 * (1.0 + np.sin(x / 11.0))
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": quote / close, "quote_volume": quote,
            "trades": 1000, "taker_buy_quote": quote * 0.52,
        },
        index=index,
    )


class MTEIgnitionTests(unittest.TestCase):
    def test_future_bars_do_not_change_past_signals(self):
        full = synthetic_frame(1200)
        prefix = full.iloc[:900]
        a = compute_mte_ignitions(prefix)
        b = compute_mte_ignitions(full).iloc[:900]
        pd.testing.assert_series_equal(a["bull_ignition"], b["bull_ignition"])
        pd.testing.assert_series_equal(a["trend_lock"], b["trend_lock"])

    def test_same_bar_stop_wins_over_target(self):
        frame = synthetic_frame(3)
        frame.iloc[1, frame.columns.get_loc("low")] = 90.0
        frame.iloc[1, frame.columns.get_loc("high")] = 140.0
        result = _first_passage(frame, 0, 95.0, 0.0, 2)
        self.assertFalse(result["hit_3r"])
        self.assertEqual(result["fixed_3r_exit_reason"], "STOP_BEFORE_3R")


if __name__ == "__main__":
    unittest.main()
