from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from mte_crypto.futures_shadow import (
    FuturesShadowConfig,
    ensure_futures_shadow,
    futures_shadow_payload,
    open_futures_shadow,
    update_futures_shadow,
)


UTC = timezone.utc


class FuturesShadowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.now = datetime(2026, 8, 17, 12, 5, tzinfo=UTC)
        self.cfg = FuturesShadowConfig(max_positions=2, taker_fee_rate=0.0005)

    def tearDown(self):
        self.temporary.cleanup()

    def spot_open(self, symbol="TESTUSDT", slot_id=1):
        return {
            "opened": True,
            "slot_id": slot_id,
            "symbol": symbol,
            "alert_id": f"pulse|{symbol}",
            "entry_price": 100.0,
        }

    def snapshot(
        self,
        *,
        bid=99.9,
        ask=100.0,
        mark=99.95,
        funding=0.0001,
        next_funding=1_776_668_800_000,
    ):
        return {
            "bid": bid,
            "ask": ask,
            "mark": mark,
            "index": 99.94,
            "last_funding_rate": funding,
            "next_funding_time": next_funding,
        }

    def test_initializes_matching_isolated_slots(self):
        state = ensure_futures_shadow(self.data_dir, now=self.now, cfg=self.cfg)
        self.assertEqual([slot["balance"] for slot in state["slots"]], [50.0, 50.0])
        self.assertEqual(state["mode"], "PAPER_ONLY_NO_ORDERS")

    def test_opens_at_ask_with_two_x_notional_and_fee(self):
        opened = open_futures_shadow(
            self.data_dir,
            self.spot_open(),
            self.snapshot(),
            now=self.now,
            cfg=self.cfg,
        )
        self.assertTrue(opened["opened"])
        self.assertEqual(opened["notional"], 100.0)
        self.assertEqual(opened["quantity"], 1.0)
        self.assertEqual(opened["entry_fee"], 0.05)
        self.assertAlmostEqual(opened["entry_spread"], 100.0 / 99.9 - 1.0)

    def test_mirrors_spot_close_at_bid_and_charges_both_fees(self):
        open_futures_shadow(
            self.data_dir,
            self.spot_open(),
            self.snapshot(),
            now=self.now,
            cfg=self.cfg,
        )
        closed = update_futures_shadow(
            self.data_dir,
            {"TESTUSDT": self.snapshot(bid=110.0, ask=110.1, mark=110.0)},
            [{"slot_id": 1, "exit_reason": "TRAIL", "exit_price": 109.8}],
            now=self.now + timedelta(hours=1),
            cfg=self.cfg,
        )[0]
        self.assertEqual(closed["exit_reason"], "TRAIL")
        self.assertEqual(closed["exit_price"], 110.0)
        self.assertAlmostEqual(closed["pnl"], 10.0 - 0.05 - 0.055)
        self.assertAlmostEqual(closed["return"], (10.0 - 0.05 - 0.055) / 50.0)

    def test_positive_funding_is_paid_by_long_once(self):
        old_funding_time = int(self.now.timestamp() * 1000) + 60_000
        open_futures_shadow(
            self.data_dir,
            self.spot_open(),
            self.snapshot(next_funding=old_funding_time),
            now=self.now,
            cfg=self.cfg,
        )
        after = self.now + timedelta(minutes=2)
        next_funding_time = old_funding_time + 8 * 3_600_000
        market = {
            "TESTUSDT": self.snapshot(
                bid=100.0,
                ask=100.1,
                mark=100.0,
                funding=0.001,
                next_funding=next_funding_time,
            )
        }
        update_futures_shadow(
            self.data_dir, market, [], now=after, cfg=self.cfg
        )
        first = futures_shadow_payload(self.data_dir, cfg=self.cfg)
        self.assertAlmostEqual(first["funding_pnl"], -0.1)
        update_futures_shadow(
            self.data_dir,
            market,
            [],
            now=after + timedelta(minutes=5),
            cfg=self.cfg,
        )
        second = futures_shadow_payload(self.data_dir, cfg=self.cfg)
        self.assertAlmostEqual(second["funding_pnl"], -0.1)

    def test_skips_symbol_without_usdm_quote(self):
        result = open_futures_shadow(
            self.data_dir,
            self.spot_open("SPOTONLYUSDT"),
            None,
            now=self.now,
            cfg=self.cfg,
        )
        self.assertFalse(result["opened"])
        self.assertEqual(result["reason"], "NO_USDM_PERPETUAL_QUOTE")


if __name__ == "__main__":
    unittest.main()
