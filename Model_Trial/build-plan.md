
# Build Plan: 30-Minute Direction & Magnitude Model

**How I would build it, in the order I would do it, with the reasoning for each decision.**
Governing doctrine: the whitepaper's eight success conditions (C1–C8) are the build's acceptance tests. Every phase ends at a gate; each gate has a kill criterion. The riskiest assumption is always tested at the cheapest possible moment.

**Stack:** Python 3.11 · Polars/DuckDB (data) · PyTorch (model) · TA-Lib (indicators) · MLflow (experiments) · Parquet (storage) · single GPU is sufficient.

---

## Phase 0 — Test the linchpin before building anything (days 1–3)

The entire design stands on C1: *a tick-built OFI proxy must correlate with next-bar returns* [D1, D3]. If this fails, the architecture changes, so I test it with one day of data before writing any pipeline.

1. Download a few days of EURUSD ticks from Dukascopy (`datafeed.dukascopy.com/.../{HH}h_ticks.bi5`).
2. Decode: LZMA-decompress, parse 20-byte big-endian records `>iii ff` → ms-offset, ask(int ×10⁵), bid(int ×10⁵), askVol, bidVol. Sanity-check decoded prices against a known reference (e.g., a public 1-min close for that day).
3. Build a minimal OFI proxy per tick: when bid rises → +bidVol; bid falls → −bidVol; same for ask with opposite sign (Cont et al.'s construction, adapted to tick volumes as queue proxy).
4. Aggregate to 1-min bars; compute corr(OFI_proxy, next-bar mid return).

**Gate 0:** correlation strongly positive (expect ρ ≈ 0.3–0.6 based on [D1]'s R²≈65%).
- **Pass →** proceed. **Fail →** fall back to the whitepaper's contingency: trade-imbalance and tick-rule features replace G1; the linear core becomes HAR-style volatility + momentum instead of OFI. Everything else survives.

---

## Phase 1 — Data pipeline (week 1–2)

Build for correctness first, speed second. This pipeline is run once; the features built on it are run thousands of times.

1. **Bulk downloader:** async, resume-safe, per-hour files, checksum + retry. 3–5 liquid symbols (EURUSD, GBPUSD, USDJPY, XAUUSD), 8–10 years.
2. **Decoder → tick store:** stream-decode `.bi5` → per-day Parquet of ticks (ts_ms, bid, ask, bidVol, askVol). Validate every day against spot-checks.
3. **Bar builder:** aggregate ticks to 1-min bars, preserving *more than OHLCV* — bid/ask/mid OHLC, spread mean/max/last, tick count, ΣbidVol, ΣaskVol, signed tick flow, per-side quote-update counts (the OFI raw material). One code path, config-driven, because this exact function will be reused at serving time.
4. **Quality gates:** intra-session gap report (>5 min flagged), bad-print filter (jumps >10× ATR(1m) reverting within 3 ticks), UTC/DST audit, weekend/holiday calendar alignment.

**Gate 1:** <0.5% bad bars; quality report reviewed. Output: `bars.parquet`, partitioned `symbol/year/month`.

---

## Phase 2 — Labels (2–3 days)

Cheap to build, expensive to get wrong. All from mid-price, horizon = 30 bars.

1. `r30 = log(mid_{t+30}/mid_t)` — primary target.
2. `mag30 = log(|r30| + ε)` and `rv30 = log(√Σ r²_1m)` — magnitude targets in log space [E5].
3. `mfe30 / mae30` — excursion labels for later execution analysis.
4. Direction view with dead-zone |r30| < τ (τ from spread distribution) — evaluation only [C2].
5. Verify label stationarity per symbol; document the overlap structure (adjacent samples share 29/30 bars) — this dictates Phase 5's purge/embargo sizes.

**Gate 2:** label distributions sane (no NaN tails, dead-zone fraction reported per symbol).

---

## Phase 3 — Feature store (week 2–3)

Implement the whitepaper's six groups, in this order (highest expected value first):

1. **G1 microstructure:** OFI proxy at 1/3/5/10/30-min rolling sums; tick intensity; spread level/Δ; OFI÷depth-proxy ratio (running impact-coefficient estimate [D2]).
2. **G6 volatility state:** HAR-style log-RV at 5m/15m/30m/1h/1d, Parkinson, ATR(14) [E1, E2, E5].
3. **G2 momentum:** mid log-returns over 1/3/5/10/15/30/60 min.
4. **G3 indicators:** CCI, Stochastic, Bollinger %B, RSI, TRIX, MACD at 2–3 windows each — continuous values only [A1, A4, A5].
5. **G5 time/regime:** cyclical time-of-day, session flags, and the critical **time×OFI, time×ATR** interactions [A2, D4].
6. **G4 interactions/tails:** pairwise products among top singles (RSI×%B, OFI×spread, CCI×ATR…), |x| and distance-from-extreme transforms [A3, A6, A7].

Every feature registered in a feature registry (name → group → finding tag → formula). Rolling z-score normalization, **fit on train folds only**.

**Gate 3:** no NaN leakage across fold boundaries; feature count logged; registry complete.

---

## Phase 4 — Baselines and the linear core, before any deep learning (week 3–4)

I build the floor before the ceiling, because every later claim is "beats baseline X."

1. **B0 persistence:** r̂=0; magnitude = trailing mean |r30|.
2. **B1 HAR-RV:** linear HAR on G6 for rv30 — the volatility literature's standard [E1].
3. **B2 Branch A standalone:** ridge regression on G1 only → r̂_lin. Inspect coefficients: sign and magnitude should respect the 1/depth law [D2]. This is the whitepaper's linear core, and it is also the C2 test.

**Gate 4 (C2):** Branch A beats persistence out-of-sample. If the linear core carries no signal, stop and diagnose the OFI proxy before spending one GPU-hour.

---

## Phase 5 — Validation harness, before the deep model (week 4)

Counterintuitive ordering, deliberate: S6 says evaluation is the binding constraint, so the harness exists before the thing it judges.

1. **Purged walk-forward engine:** train 24m → purge 30 bars → embargo → validate 1m → roll. No random CV anywhere.
2. **Metric suite:** MCC/balanced-acc on dead-zoned direction; hit-rate conditioned on |r30| > spread; MAE on log-magnitude; QLIKE on rv30; quantile calibration.
3. **Per-regime slicing:** time-of-day buckets × volatility terciles [A2, D4].
4. **Economic gate:** minimal rule — act only when confident direction AND m̂30 > k× actual historical spread — net PnL per fold [C2, F2].
5. **Adversarial:** shuffle test, single-feature audits [C3].

**Gate 5:** harness reproduces known results (persistence ≈ chance; HAR beats persistence on rv30). A harness that can't rank obvious baselines can't judge a model.

---

## Phase 6 — The residual learner (week 5–6)

Only now does PyTorch get involved.

1. **Input:** 60-bar sequences × F features (lookback 30/60/120 ablated later).
2. **Branch B:** multi-width Conv1D blocks (kernel widths 3/5/7 — the cross-indicator interaction learner [F3]) → 2-layer BiLSTM [F3] → attention pooling. Small trunk, dropout + weight decay from the start [C1].
3. **Heads:** r̂30 = r̂_lin + residual̂; magnitude head consumes trunk ⊕ r̂30 ⊕ G6 (the cascade [E3]); quantile head q0.1–q0.9 on |r30|.
4. **Loss:** λ₁·Huber(r30) + λ₂·Huber(log|r30|) + λ₃·QLIKE(rv30); λ grid on validation.
5. **Training discipline:** early stopping on validation loss, every run in MLflow with feature-registry version + fold ID.

**Gate 6 (C3):** full model beats Branch A alone across folds. Fail → halve trunk capacity, shorten lookback, retry once; fail again → ship Branch A + magnitude head and document why.

---

## Phase 7 — Ablation battery: converting conditions into evidence (week 6–7)

Each ablation is one whitepaper condition:

| Ablation | Condition | Keep/kill rule |
|---|---|---|
| Remove G4 interactions | C4 [A3, A6, F3] | Kill group if val loss unchanged |
| Remove G5 time features | C5 [A2, D4] | Kill if per-bucket lift vanishes |
| Independent heads vs cascade | C6 [E3, A7] | Keep winner |
| Raw vs log magnitude target | C7 [E5] | Keep winner |
| Full vs Branch-A-only | C3 [D1, C1] | Confirms Gate 6 |
| Lookback 30/60/120 | — | Keep winner |
| Transfer test: train on 3 symbols, test on held-out 4th | universality [B2] | Report; informs monitoring |

**Gate 7 (C4–C7):** surviving configuration frozen. Every killed feature group is documented — a negative result here is a finding about this market, not a failure.

---

## Phase 8 — Economic gate and shadow serving (week 7–8)

1. Run the frozen model through the full walk-forward with the economic gate (C8). Report net edge per fold, per regime, per symbol.
2. **Serving:** cron at :00/:30 → same `features.py` code path as training (no train/serve skew) → model → JSON output `{r̂30, direction, P(up), m̂30, q10–q90, regime flags}`.
3. **Shadow mode 2 weeks:** log predictions vs. realized, no action taken.
4. **Monitoring:** PSI drift per feature group, OFI→return slope drift (the impact-coefficient health check [D2]), hit-rate by time bucket vs. validation baseline, quantile calibration curve.

**Gate 8 (C8):** economic gate positive on ≥70% of folds AND shadow performance statistically consistent with validation. Only then is the system what the whitepaper claims it can be.

---

## The build's logic in one paragraph

Test the linchpin with one day of data (C1) → build a pipeline that preserves microstructure → labels in log space → features in evidence-ranked order → baselines and the linear core before any deep learning (C2) → the validation harness before the model it judges (S6) → the smallest deep model that can explain the residual (C3) → ablate every claim into a kept or killed condition (C4–C7) → and admit the system to operation only through a spread-net economic gate and shadow consistency (C8). At every gate, the kill criterion is written down before the result is known.