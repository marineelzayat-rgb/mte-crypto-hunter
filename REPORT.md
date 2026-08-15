# MTE Crypto Hunter — causal backtest report

## Frozen experiment

- Binance Spot USDT historical members, including delisted symbols.
- Closed H1 bars only; H4/D1 pivots are exposed only after right-side confirmation.
- Hunter model trained on 2023–2024.
- 2025 used to freeze the pre-ignition probability threshold (`0.6753`).
- 2022 was downloaded and evaluated last as an untouched regime holdout.
- Hunt lifetime: 48 hours; same-symbol cooldown: 72 hours.
- Entry: first qualified selected H4/D1 MTE break in the window, at the H1 close; no retest.
- MTE defaults preserved: previous-bar selected line, 0.10 ATR break buffer,
  EXP candle quality, H1 direction lock, and confirmed pivots.
- Conservative same-bar convention: stop is evaluated before target.
- Cost: 15 bps on entry and 15 bps on exit.
- Outcome experiment: fixed +3R versus -1R, with +5R potential also audited.
- Maximum hold: 30 days.

## Why the original ~50% Hunter did not trade

The high-precision Hunter was trained on `ARMED_IGNITION`, which was already a
break/expansion event. When a second MTE ignition was requested inside the next
48 hours, 84 of 89 activations occurred on the very same H1 bar. It was the same
timing event under two names, not a true two-stage setup.

| Year | Hunter alerts | Future explosions | Precision | MTE activations | Hit +3R | PF | Total R |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025 | 140 | 76 | 54.3% | 59 | 9 | 0.51 | -25.41R |
| 2026 Jan–Jul | 92 | 39 | 42.4% | 30 | 2 | 0.21 | -22.96R |

Conclusion: the ~50% statistic is a valid future-explosion classification
result, but it is not an entry expectancy result.

## Genuine pre-ignition Hunter

The earlier Hunter requires volume wake-up, positive BTC-relative strength,
liquidity, and proximity to resistance while price is still below the
confirmed-break threshold. It does not require the expansion candle.

| Year | Role | Hunts | Future explosions | Precision | Hunts/week |
|---|---|---:|---:|---:|---:|
| 2022 | Untouched holdout | 138 | 23 | 16.7% | 2.65 |
| 2025 | Threshold validation | 175 | 24 | 13.7% | 3.36 |
| 2026 Jan–Jul | Secondary test | 119 | 20 | 16.8% | 3.93 |

The raw base rate was roughly 3–4%, so the early Hunter retained a meaningful
discovery lift while genuinely preceding MTE ignition.

## Pre-ignition Hunter → MTE → H1 pivot stop

The crypto stop was placed below the last confirmed H1 pivot low minus 0.30 ATR.
Price risk was limited to 1–30%; account risk must be controlled by position
sizing rather than by using the full account notional.

| Year | Hunts | Activated | Hit +3R | Hit +5R | PF | Total R | Median delay | Median price risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022 holdout | 138 | 23 | 2 | 1 | 0.28 | -15.21R | 11h | 15.0% |
| 2025 | 175 | 43 | 10 | 6 | 1.07 | +2.10R | 17h | 13.8% |
| 2026 Jan–Jul | 119 | 15 | 5 | 1 | 1.77 | +6.50R | 18h | 14.2% |

At 0.25% account risk per trade and at most four concurrent positions, the
closed-equity approximations were -3.75% (2022), -0.25% (2025), and +1.62%
(2026 through July).

## Decision

The pre-ignition separation fixed the same-bar logic error and produced positive
results in 2025–2026, but it failed decisively on the untouched 2022 holdout.
Therefore it is **rejected as a robust auto-entry strategy**. Live order
execution remains disabled.

## Crypto flow layer (added after the frozen backtest)

The Hunter already used historical volume wake-up. The new experiment measures
fresh flow at the later MTE activation bar: 1/3/6/24-hour taker-buy ratio and
signed-volume proxy, H1 dollar-volume acceleration, trade-count acceleration,
close location, and six-hour flow persistence. These fields are causal and do
not change the frozen MTE structure or add a retest.

A regularized linear gate was fitted only on the small 2023–2024 activation
sample (33 trades, 8 winners), then frozen at its median training score. It is
diagnostic, not an accepted model:

| Year | Baseline trades/wins | Flow-gated trades/wins | Flow-gated PF | Flow-gated total R |
|---|---:|---:|---:|---:|
| 2025 | 43 / 10 | 20 / 5 | 1.28 | +3.71R |
| 2022 holdout | 23 / 2 | 8 / 0 | 0.00 | -8.07R |
| 2026 Jan–Jul | 15 / 5 | 10 / 3 | 1.48 | +2.88R |

Conclusion: simple candle-volume/taker confirmation does not generalize and
must not be promoted to a hard entry gate. At activation, extreme volume can be
either genuine demand or terminal distribution. The project now records live
top-20 depth and aggregate trades so that spread, visible depth, multi-band book
imbalance, microprice, executed-flow delta, persistence, pulling and
replenishment can be tested prospectively. Historical order-book results are
not claimed because those snapshots were not in the archive used here.

The next target must be path-aware from the beginning: train the Hunter on
whether a future MTE activation reaches +3R before structural invalidation,
instead of training it on a future maximum-price explosion. That makes the
learning target identical to the trade objective.

## Scope note

`mte_ignition.py` ports the causal long-entry signal path of the selected H4/D1
line bank, expansion quality, and H1 direction lock. The full TradingView
multi-leg projected target ladder is not used for this experiment; exits are the
explicit fixed +3R/-1R first-passage test described above.
