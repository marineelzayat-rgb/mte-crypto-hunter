import unittest

import numpy as np
import pandas as pd

from mte_crypto.flow import activation_flow_features, order_book_features


class FlowTests(unittest.TestCase):
    def test_activation_features_are_prefix_invariant(self):
        index = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": np.full(80, 100.0),
                "high": np.full(80, 102.0),
                "low": np.full(80, 99.0),
                "close": np.full(80, 101.0),
                "quote_volume": np.arange(1, 81, dtype=float) * 1000,
                "trades": np.arange(1, 81, dtype=float) * 10,
                "taker_buy_quote": np.arange(1, 81, dtype=float) * 550,
            },
            index=index,
        )
        before = activation_flow_features(frame, 60)
        changed = frame.copy()
        changed.iloc[61:, changed.columns.get_loc("quote_volume")] = 1e12
        after = activation_flow_features(changed, 60)
        self.assertEqual(before, after)

    def test_order_book_metrics(self):
        features = order_book_features(
            bids=[[100.0, 10.0], [99.9, 5.0]],
            asks=[[100.1, 4.0], [100.2, 3.0]],
        )
        self.assertGreater(features["book_top_imbalance"], 0)
        self.assertGreater(features["book_spread_bps"], 0)
        self.assertGreater(features["book_microprice_edge_bps"], 0)


if __name__ == "__main__":
    unittest.main()
