from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from mte_crypto.paper_portfolio import (
    PaperPortfolioConfig,
    ensure_paper_portfolio,
    open_paper_position,
    paper_portfolio_payload,
    update_paper_portfolio,
)


UTC = timezone.utc


class PaperPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.now = datetime(2026, 8, 17, 12, 5, tzinfo=UTC)
        self.cfg = PaperPortfolioConfig(max_positions=2)

    def tearDown(self):
        self.temporary.cleanup()

    def alert(self, symbol="TESTUSDT", price=100.0, state="EARLY_PULSE"):
        return {
            "id": f"test|{symbol}|{state}",
            "symbol": symbol,
            "state": state,
            "price": price,
        }

    def test_initializes_equal_isolated_slots(self):
        state = ensure_paper_portfolio(self.data_dir, now=self.now, cfg=self.cfg)
        self.assertEqual(len(state["slots"]), 2)
        self.assertEqual([slot["balance"] for slot in state["slots"]], [50.0, 50.0])

    def test_opens_entry_states_and_rejects_duplicate_or_full_slots(self):
        first = open_paper_position(
            self.data_dir, self.alert(), now=self.now, cfg=self.cfg
        )
        self.assertTrue(first["opened"])
        self.assertEqual(first["slot_id"], 1)
        self.assertAlmostEqual(first["stop_price"], 92.5)

        duplicate = open_paper_position(
            self.data_dir,
            self.alert(),
            now=self.now + timedelta(minutes=1),
            cfg=self.cfg,
        )
        self.assertEqual(duplicate["reason"], "DUPLICATE_SYMBOL")

        bull = open_paper_position(
            self.data_dir,
            self.alert("BULLUSDT", state="BULL_CONTINUATION"),
            now=self.now + timedelta(minutes=2),
            cfg=self.cfg,
        )
        self.assertTrue(bull["opened"])

        ignored = open_paper_position(
            self.data_dir,
            self.alert("NOCHASEUSDT", state="RAPID_MOVE_NO_CHASE"),
            now=self.now + timedelta(minutes=3),
            cfg=self.cfg,
        )
        self.assertEqual(ignored["reason"], "NOT_ENTRY_SIGNAL")

        full = open_paper_position(
            self.data_dir,
            self.alert("THIRDUSDT"),
            now=self.now + timedelta(minutes=4),
            cfg=self.cfg,
        )
        self.assertEqual(full["reason"], "NO_FREE_SLOT")

    def test_activates_hourly_atr_trail_and_closes_causally_next_cycle(self):
        open_paper_position(self.data_dir, self.alert(), now=self.now, cfg=self.cfg)
        update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 105.0}],
            now=self.now + timedelta(hours=1),
            atr_provider=lambda _symbol: (2.0, "2026-08-17T12:00:00+00:00"),
            cfg=self.cfg,
        )
        position = paper_portfolio_payload(self.data_dir, cfg=self.cfg)[
            "open_positions"
        ][0]
        self.assertTrue(position["trail_active"])
        self.assertAlmostEqual(position["stop_price"], 102.0)
        self.assertAlmostEqual(position["profit_floor_return"], 0.02)

        closed = update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 101.0}],
            now=self.now + timedelta(hours=1, minutes=5),
            atr_provider=lambda _symbol: self.fail("same hourly candle was fetched twice"),
            cfg=self.cfg,
        )
        self.assertEqual(closed[0]["exit_reason"], "TRAIL")
        payload = paper_portfolio_payload(self.data_dir, cfg=self.cfg)
        self.assertEqual(payload["open_count"], 0)
        self.assertEqual(payload["available_slots"], 2)

    def test_profit_floor_ratchets_while_runner_stays_open_after_24_hours(self):
        open_paper_position(self.data_dir, self.alert(), now=self.now, cfg=self.cfg)
        update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 121.0}],
            now=self.now + timedelta(hours=2),
            atr_provider=lambda _symbol: (10.0, "2026-08-17T13:00:00+00:00"),
            bull_mode=True,
            cfg=self.cfg,
        )
        position = paper_portfolio_payload(self.data_dir, cfg=self.cfg)[
            "open_positions"
        ][0]
        self.assertAlmostEqual(position["profit_floor_return"], 0.12)
        self.assertAlmostEqual(position["stop_price"], 112.0)

        closed = update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 120.0}],
            now=self.now + timedelta(hours=24),
            cfg=self.cfg,
        )
        self.assertEqual(closed, [])
        self.assertEqual(
            paper_portfolio_payload(self.data_dir, cfg=self.cfg)["open_count"],
            1,
        )

    def test_trailing_stop_never_moves_down(self):
        open_paper_position(self.data_dir, self.alert(), now=self.now, cfg=self.cfg)
        update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 110.0}],
            now=self.now + timedelta(hours=1),
            atr_provider=lambda _symbol: (2.0, "2026-08-17T12:00:00+00:00"),
            cfg=self.cfg,
        )
        first_stop = paper_portfolio_payload(self.data_dir, cfg=self.cfg)[
            "open_positions"
        ][0]["stop_price"]
        update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 111.0}],
            now=self.now + timedelta(hours=2),
            atr_provider=lambda _symbol: (20.0, "2026-08-17T13:00:00+00:00"),
            cfg=self.cfg,
        )
        second_stop = paper_portfolio_payload(self.data_dir, cfg=self.cfg)[
            "open_positions"
        ][0]["stop_price"]
        self.assertEqual(second_stop, first_stop)

    def test_exits_immediately_if_new_hourly_trail_is_above_market(self):
        open_paper_position(self.data_dir, self.alert(), now=self.now, cfg=self.cfg)
        update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 110.0}],
            now=self.now + timedelta(hours=1),
            atr_provider=lambda _symbol: (5.0, "2026-08-17T12:00:00+00:00"),
            cfg=self.cfg,
        )
        closed = update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 105.0}],
            now=self.now + timedelta(hours=2),
            atr_provider=lambda _symbol: (1.0, "2026-08-17T13:00:00+00:00"),
            cfg=self.cfg,
        )
        self.assertEqual(closed[0]["exit_reason"], "TRAIL")

    def test_closes_stale_position_after_12_hours(self):
        open_paper_position(self.data_dir, self.alert(), now=self.now, cfg=self.cfg)
        closed = update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 101.0}],
            now=self.now + timedelta(hours=12),
            cfg=self.cfg,
        )
        self.assertEqual(closed[0]["exit_reason"], "STALE_12H")
        self.assertGreater(closed[0]["return"], 0.0)

    def test_closes_non_moving_loser_after_six_hours(self):
        open_paper_position(self.data_dir, self.alert(), now=self.now, cfg=self.cfg)
        closed = update_paper_portfolio(
            self.data_dir,
            [{"symbol": "TESTUSDT", "last_price": 99.0}],
            now=self.now + timedelta(hours=6),
            cfg=self.cfg,
        )
        self.assertEqual(closed[0]["exit_reason"], "STALE_LOSER")


if __name__ == "__main__":
    unittest.main()
