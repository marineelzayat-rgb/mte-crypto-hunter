from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

from mte_crypto.live_spot import (
    BinanceSpotPrivateClient,
    LIVE_CONFIRMATION,
    LiveSpotConfig,
    open_live_position,
    close_live_positions,
    live_spot_payload,
    refresh_connection_status,
)


UTC = timezone.utc


class FakeClient:
    def __init__(self, *, can_trade=True):
        self.can_trade = can_trade
        self.orders = []
        self.stop_status = {"status": "NEW"}

    def sync_time(self):
        return 0

    def account(self):
        return {
            "canTrade": self.can_trade,
            "canWithdraw": False,
            "canDeposit": True,
            "accountType": "SPOT",
            "balances": [{"asset": "USDT", "free": "100.00", "locked": "0"}],
        }

    def exchange_info(self, symbol):
        return {
            "symbols": [{
                "symbol": symbol,
                "baseAsset": "TEST",
                "orderTypes": ["MARKET", "STOP_LOSS"],
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                    {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                ],
            }]
        }

    def new_order(self, **params):
        self.orders.append(params)
        if params["side"] == "BUY":
            return {
                "orderId": 10,
                "executedQty": "1.1",
                "cummulativeQuoteQty": "11.0",
                "fills": [],
            }
        return {"orderId": 11, "status": "NEW"}

    def query_order(self, symbol, order_id=None, client_order_id=None):
        return {"orderId": order_id or 11, **self.stop_status}

    def cancel_order(self, symbol, order_id):
        return {"orderId": order_id, "status": "CANCELED"}


class LiveSpotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.now = datetime(2026, 8, 21, 18, tzinfo=UTC)

    def tearDown(self):
        self.temporary.cleanup()

    def test_ed25519_client_loads_generated_key(self):
        private = Ed25519PrivateKey.generate()
        pem = private.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        client = BinanceSpotPrivateClient("api", pem)
        self.assertEqual(client.api_key, "api")

    def test_connection_status_is_read_only_and_never_exposes_credentials(self):
        with patch.dict(
            "os.environ",
            {"BINANCE_API_KEY": "secret-api", "BINANCE_ED25519_PRIVATE_KEY": "present"},
            clear=False,
        ):
            status = refresh_connection_status(
                self.data_dir,
                now=self.now,
                cfg=LiveSpotConfig(),
                client=FakeClient(can_trade=False),
            )
        self.assertTrue(status["connected"])
        self.assertFalse(status["can_trade"])
        self.assertNotIn("secret-api", (self.data_dir / "live_spot_connection.json").read_text())

    def test_live_entry_requires_double_confirmation(self):
        result = open_live_position(
            self.data_dir,
            {"opened": True, "symbol": "TESTUSDT"},
            now=self.now,
            cfg=LiveSpotConfig(enabled=True, confirmation="wrong"),
            client=FakeClient(),
        )
        self.assertEqual(result["reason"], "SAFE_DISABLED")

    def test_armed_entry_buys_and_immediately_places_hard_stop(self):
        client = FakeClient()
        result = open_live_position(
            self.data_dir,
            {
                "opened": True,
                "symbol": "TESTUSDT",
                "slot_id": 1,
                "alert_id": "alert-1",
            },
            now=self.now,
            cfg=LiveSpotConfig(enabled=True, confirmation=LIVE_CONFIRMATION),
            client=client,
        )
        self.assertTrue(result["opened"])
        self.assertEqual(client.orders[0]["side"], "BUY")
        self.assertEqual(client.orders[0]["quoteOrderQty"], "11.00")
        self.assertEqual(client.orders[1]["type"], "STOP_LOSS")
        self.assertEqual(client.orders[1]["stopPrice"], "9.25")

    def test_exchange_stop_is_reconciled_without_waiting_for_paper_close(self):
        client = FakeClient()
        cfg = LiveSpotConfig(enabled=True, confirmation=LIVE_CONFIRMATION)
        open_live_position(
            self.data_dir,
            {"opened": True, "symbol": "TESTUSDT", "slot_id": 1},
            now=self.now,
            cfg=cfg,
            client=client,
        )
        client.stop_status = {
            "status": "FILLED",
            "executedQty": "1.1",
            "cummulativeQuoteQty": "10.175",
        }
        closed = close_live_positions(
            self.data_dir, [], now=self.now, cfg=cfg, client=client
        )
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "EXCHANGE_HARD_STOP")
        self.assertEqual(live_spot_payload(self.data_dir)["open_count"], 0)

    def test_public_payload_redacts_balance_positions_and_events(self):
        client = FakeClient()
        cfg = LiveSpotConfig(enabled=True, confirmation=LIVE_CONFIRMATION)
        open_live_position(
            self.data_dir,
            {"opened": True, "symbol": "TESTUSDT", "slot_id": 1},
            now=self.now,
            cfg=cfg,
            client=client,
        )
        payload = live_spot_payload(self.data_dir)
        serialized = str(payload)
        self.assertNotIn("quote_spent", serialized)
        self.assertNotIn("TESTUSDT", serialized)
        self.assertNotIn("usdt_free", serialized)

    def test_disabling_new_entries_still_manages_existing_position(self):
        client = FakeClient()
        armed = LiveSpotConfig(enabled=True, confirmation=LIVE_CONFIRMATION)
        open_live_position(
            self.data_dir,
            {"opened": True, "symbol": "TESTUSDT", "slot_id": 1},
            now=self.now,
            cfg=armed,
            client=client,
        )
        client.stop_status = {
            "status": "FILLED",
            "executedQty": "1.1",
            "cummulativeQuoteQty": "10.175",
        }
        closed = close_live_positions(
            self.data_dir,
            [],
            now=self.now,
            cfg=LiveSpotConfig(enabled=False),
            client=client,
        )
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["exit_reason"], "EXCHANGE_HARD_STOP")


if __name__ == "__main__":
    unittest.main()
