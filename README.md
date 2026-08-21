# MTE Crypto Hunter

See [REPORT.md](REPORT.md) for the frozen 2022/2025/2026 Hunter→MTE backtest.
The current result validates candidate discovery, but rejects automated entry
after the untouched 2022 holdout; order execution is intentionally disabled.
The live service uses the frozen `MTE_PRE_HUNTER_RF_V0_3` model at its validated
`0.6753` threshold. It records order-book/taker-flow data for forward research;
it does not use that data as a proven entry filter yet.

An auditable, causal research and scanning pipeline for finding Binance USDT altcoins in the transition from dormant/base-building behavior into structural expansion.

The project deliberately separates:

1. **Research labels** — what counts as an explosion, defined without using future data in features.
2. **Pre-explosion ranking** — volume wake-up, compression, structure, and relative strength.
3. **MTE ignition** — a confirmed expansion break with no mandatory retest.
4. **Execution state** — `WATCH`, `ARMED`, `HUNTER_ALERT`, or `NO_CHASE`.
5. **Crypto flow research** — activation-time taker flow plus forward-recorded
   order-book/tape data. This is not enabled as an entry filter until it passes
   out-of-sample validation.

The default scanner is long-only and uses public Binance Spot market data. No API key is required. Futures open interest, funding, and taker-flow are deliberately an optional confirmation layer rather than a dependency.

## Quick start

```bash
cd mte_crypto_hunter
python -m mte_crypto.scan --top 120 --output output/latest_scan.csv
```

The scanner first filters all Binance USDT pairs by 24-hour quote volume, then downloads one-hour candles for the liquid universe, resamples causally to H4/D1, and ranks candidates.

To inspect one symbol:

```bash
python -m mte_crypto.scan --symbols ACEUSDT --output output/ace_scan.csv
```

Run the stateful monitor once (prints alerts when Telegram environment
variables are absent):

```bash
python -m mte_crypto.monitor --once
```

Continuous five-minute monitoring:

```bash
python -m mte_crypto.monitor --every-minutes 5
```

Optional Telegram delivery uses `MTE_TELEGRAM_BOT_TOKEN` and
`MTE_TELEGRAM_CHAT_ID`. Secrets are read from environment variables and are
never stored in the project.

Run tests:

```bash
python -m unittest discover -s tests -v
```

Attach causal flow fields to a completed Hunter→MTE result:

```bash
python -m mte_crypto.enrich_activation_flow \
  --input output/pre_hunter_mte_pivot_2026.csv \
  --history-dirs data/archive_2026 \
  --output output/pre_hunter_mte_pivot_2026_flow.csv
```

Record live top-20 depth plus executed taker flow for forward validation:

```bash
python -m mte_crypto.book_collector ACEUSDT SOLUSDT \
  --output output/live_order_book.jsonl
```

The collector is read-only and never places an order. A single visible wall is
not treated as evidence; persistence and replenishment must be measured across
the recorded snapshots.

## Railway 24/7 service

The included `Dockerfile` starts `python -m mte_crypto.daemon`. It runs two
causal discovery layers:

1. every hour, scans the top 120 liquid Binance Spot USDT altcoins and applies
   the exact causal `PRE_IGNITION_HUNT` gate plus frozen RF ranker;
2. every two minutes, takes a lightweight price/volume pulse of every eligible
   Spot/USDT market, including cold markets below the top-120 liquidity cutoff;
3. labels a pulse `EARLY_PULSE` while its 24h return is at most 20%. During a
   broad BTC breakout regime, a non-parabolic 20–40% move becomes the separate
   `BULL_CONTINUATION` paper-entry state; faster/older extensions remain
   `RAPID_MOVE_NO_CHASE`;
4. keeps each `HUNTER_ALERT` active for 48 hours and each pulse candidate active
   for two hours;
5. records derived top-20 order-book and executed taker-flow snapshots for all
   open paper positions plus active hunter/pulse candidates. It samples every
   second for the first 30 minutes, every 5 seconds through two hours, and every
   15 seconds afterward so the full trade is covered without wasteful storage;
6. writes scans, pulse history, and compressed JSONL research data under `/data`.
7. exposes a read-only live ledger at `/status` and `/status.json` after a
   Railway public domain is generated for the service.
8. paper-trades `EARLY_PULSE` and controlled `BULL_CONTINUATION` candidates in
   one of 16 isolated slots that start with $6.25 each. The research-only Wave
   Rider uses a 7.5% initial stop, locks progressively larger profit floors at
   +5/+10/+20/+35/+50%, and follows the peak with a wide hourly Chandelier
   stop (3.5 ATR normally, 4.5 ATR during a broad bull breakout). Unactivated
   losers are recycled after six hours, other unactivated trades after 12
   hours, while an activated runner may continue for seven days. Active
   candidates are retried when a slot becomes free instead of being lost at
   their first `NO_FREE_SLOT` event;
9. mirrors every accepted paper entry and exit in a separate USD-M Futures 2x
   shadow ledger. It enters at the futures ask, exits at the bid, and records
   spread, mark/index basis, estimated taker fees, and observed funding.
10. includes an optional Ed25519-authenticated Binance Spot execution layer.
    It is `SAFE_DISABLED` by default and requires two separate environment
    gates before it can submit an order. A newly filled buy must receive an
    exchange-side 7.5% hard stop immediately or the position is flattened.
11. includes a separate optional RSA-authenticated Binance USD-M Futures layer.
    It opens one-way, isolated 4x longs only and compounds realized equity by
    allocating 11% of the current USDT wallet balance per position. It sends the
    hard stop to Binance's conditional-algo service immediately. Failure to
    attach that stop triggers an emergency reduce-only market close. It is
    independently `SAFE_DISABLED` by default.

Recommended Railway volume mount: `/data`. Optional environment variables:

- `MTE_SCAN_TOP` (default `120`)
- `MTE_SCAN_INTERVAL_SECONDS` (default `3600`)
- `MTE_PULSE_INTERVAL_SECONDS` (default `120`)
- `MTE_PULSE_TTL_MINUTES` (default `120`)
- `MTE_CANDIDATE_TTL_HOURS` (default `48`)
- `MTE_BOOK_SAMPLE_SECONDS` (default `1`)
- `MTE_MAX_BOOK_SYMBOLS` (default `32`, never below the 16 paper slots)
- `MTE_TELEGRAM_BOT_TOKEN` and `MTE_TELEGRAM_CHAT_ID` for alerts

Optional real-Spot variables (do not enable until read-only connection testing
has passed):

- `BINANCE_API_KEY`
- `BINANCE_ED25519_PRIVATE_KEY_B64` (PKCS#8 PEM encoded as base64)
- `MTE_LIVE_SPOT_ENABLED` (default `false`)
- `MTE_LIVE_SPOT_CONFIRMATION` (must exactly equal the separately documented
  arming phrase)
- `MTE_LIVE_MAX_POSITIONS` (default `8`)
- `MTE_LIVE_ORDER_USDT` (default `11`)
- `MTE_LIVE_RESERVE_USDT` (default `12`)
- `MTE_LIVE_DAILY_LOSS_LIMIT_USDT` (default `8`)

Optional real-Futures variables (use a separate RSA API key and do not enable
until the read-only connection check has passed):

- `BINANCE_FUTURES_API_KEY`
- `BINANCE_FUTURES_RSA_PRIVATE_KEY_B64` (PKCS#8 RSA PEM encoded as base64)
- `MTE_LIVE_FUTURES_ENABLED` (default `false`)
- `MTE_LIVE_FUTURES_CONFIRMATION` (must exactly equal the separately
  documented arming phrase)
- `MTE_LIVE_FUTURES_MAX_POSITIONS` (default `8`)
- `MTE_LIVE_FUTURES_MARGIN_FRACTION` (default `0.11`, or 11% of realized USDT
  wallet balance per entry)
- `MTE_LIVE_FUTURES_RESERVE_FRACTION` (default `0.10`)
- `MTE_LIVE_FUTURES_MINIMUM_WALLET_BALANCE_USDT` (default `10`; no new entries
  at or below this experiment floor)
- `MTE_LIVE_FUTURES_DAILY_LOSS_LIMIT_USDT` (default minimum `30`)
- `MTE_LIVE_FUTURES_DAILY_LOSS_LIMIT_FRACTION` (default `0.30`; the effective
  daily gate is the larger of this fraction of wallet balance or the USDT
  minimum)
- `MTE_LIVE_FUTURES_INITIAL_STOP_PCT` (default `0.075`)

The real Futures layer rejects hedge mode and multi-assets margin mode. It
requires USD-M one-way mode, uses isolated margin, and always requests exactly
4x leverage. If both real Spot and real Futures entry gates are armed, new real
entries go to Futures only so the same signal is not bought twice. Existing
positions in either layer continue to be reconciled even after new entries are
disabled.

The public status page exposes only connection/mode counts for the real accounts;
it deliberately redacts the Binance balance, symbols, quantities, order IDs,
events, API key, and private key. Real fills are reported privately by Telegram.

No Binance API key is needed for scanning, paper trading, or the futures shadow.
The optional real-Spot layer remains unable to place orders unless both live
gates are deliberately armed. The public status endpoints use strict redaction;
Telegram credentials, environment variables, account details, and raw
order-book files are never exposed.
The `$100` Wave Rider portfolio shown on the status page is simulated only;
entry and exit fees are estimated at 0.1% per side and no real funds are used.
The parallel `$100` Futures 2x shadow is also simulated only. It never submits
an order and uses public USD-M market data without exchange credentials.

Download historical monthly archives (includes delisted pairs when
`--all-historical` is used):

```bash
python -m mte_crypto.archive --start 2025-01 --end 2026-07 --all-historical --output-dir data/archive
python -m mte_crypto.research --data-dir data/archive --btc data/archive/BTCUSDT.csv
```

## Default research definition

An explosion is not a signal. It is an outcome label:

- forward 24-hour maximum return >= 40%, or
- forward 48-hour maximum return >= 60%.

Signals and features are computed only from information available at the timestamp being evaluated. The event-study layer compares explosion observations with time-matched non-explosion controls.

## Important limitations

- A current-symbol scan has survivorship bias. Full validation must include historical listings and delistings from the Binance public archive.
- A price/volume scanner cannot anticipate a completely untelegraphed listing or news event. Its job is to detect the first causal footprint before the terminal daily candle.
- `RESEARCH_IGNITION` is the frozen MTE Crypto v0.1 candidate: H1 break plus
  expansion quality, 5–10% prior 24h return, 5–10% structural risk, RVOL
  2–10x, 24h-volume ratio 1.5–10x, at least $20m rolling spot volume,
  H4 compression rank <= 0.20, green-volume share >= 60%, positive relative
  strength, and BTC 24h return >= -2%. It does **not** require a retest.
- `ARMED` is an early research watch, not an entry instruction.
- Paper and alert automation remain the default. The optional real-Spot layer
  reads credentials only from Railway environment variables and is disabled
  unless both independent live gates are armed.
- The optional real-Futures layer uses a separate RSA key and two independent
  live gates. Withdrawal and universal-transfer permissions are not required
  and should remain disabled.
- Real execution must include spread, slippage, fees, funding, and exchange-specific wick behavior.
