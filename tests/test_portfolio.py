import unittest

import pandas as pd

from mte_crypto.portfolio import PortfolioConfig, simulate_portfolio


class PortfolioTests(unittest.TestCase):
    def test_liquidity_order_and_position_cap(self):
        trades = pd.DataFrame(
            [
                {"symbol": "LOW", "entry_time": "2025-01-01T00:00:00Z", "exit_time": "2025-01-02T00:00:00Z", "entry_quote_volume_24h": 20, "r_multiple": 1},
                {"symbol": "HIGH", "entry_time": "2025-01-01T00:00:00Z", "exit_time": "2025-01-02T00:00:00Z", "entry_quote_volume_24h": 40, "r_multiple": 1},
            ]
        )
        accepted, _ = simulate_portfolio(trades, PortfolioConfig(max_positions=1))
        self.assertEqual(accepted["symbol"].tolist(), ["HIGH"])


if __name__ == "__main__":
    unittest.main()
