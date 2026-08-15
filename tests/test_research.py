import unittest

import numpy as np
import pandas as pd

from mte_crypto.features import confirmed_pivots, percentile_rank_last
from mte_crypto.research import forward_max_return, label_explosions
from mte_crypto.archive import _timestamp_to_datetime


class CausalityTests(unittest.TestCase):
    def test_pivot_appears_only_after_confirmation(self):
        idx = pd.date_range("2026-01-01", periods=9, freq="h", tz="UTC")
        values = pd.Series([1, 2, 3, 5, 3, 2, 1, 2, 3], index=idx, dtype=float)
        pivots = confirmed_pivots(values, strength=2, kind="high")
        self.assertTrue(np.isnan(pivots.iloc[3]))
        self.assertEqual(pivots.iloc[5], 5.0)

    def test_forward_label_excludes_current_high(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
        close = pd.Series(100.0, index=idx)
        high = pd.Series(100.0, index=idx)
        high.iloc[10] = 200.0
        fwd = forward_max_return(close, high, hours=24)
        self.assertEqual(fwd.iloc[10], 0.0)
        self.assertEqual(fwd.iloc[9], 1.0)

    def test_percentile_is_causal(self):
        series = pd.Series(range(20), dtype=float)
        before = percentile_rank_last(series, 10)
        changed = series.copy()
        changed.iloc[-1] = 10_000
        after = percentile_rank_last(changed, 10)
        pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])

    def test_mixed_millisecond_and_microsecond_archive_timestamps(self):
        values = pd.Series([1704067200000, 1735689600000000])
        parsed = _timestamp_to_datetime(values)
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertEqual(parsed.iloc[1], pd.Timestamp("2025-01-01", tz="UTC"))


if __name__ == "__main__":
    unittest.main()
