
# A Mechanism for Predicting 30-Minute Price Direction and Magnitude from 1-Minute Data

**Whitepaper v1.0** — logic, mechanism, and success conditions
**Data foundation:** Dukascopy tick archives (bid/ask + volumes)
**Evidence base:** six peer-reviewed / working papers, cited as [A1–F3] per the findings registry in §10

---

## Abstract

This document sets out the complete logical and mechanistic basis for a model that, every 30 minutes, predicts the **direction** and **expected magnitude** of the next 30-minute price move using 1-minute data. The thesis is that short-horizon price changes are not noise to be filtered but the **mechanical output of order flow hitting a finite-depth book** — a relation that is empirically linear, depth-modulated, and intraday-seasonal — and that everything else in the model (technical indicators, volatility state, interactions, time structure) exists to explain the **residual** of that core relation and to govern the **magnitude** of the move. Success is achieved when eight falsifiable conditions (§8) hold simultaneously; each condition is grounded in a published finding and has a defined verification test. The document contains no project management content — only the logic of why this can work, the mechanism at each stage, and how each mechanism is confirmed or falsified.

---

## 1. Problem Formulation

At each decision time *t* (aligned to :00 and :30), observe the trailing window of 1-minute bars plus tick-derived microstructure, and produce:

- **r̂₃₀** — predicted log-return of mid-price over the next 30 minutes. **Direction = sign(r̂₃₀).**
- **m̂₃₀** — predicted magnitude: expected |r₃₀| and/or forward realized volatility RV₃₀, in log space.

Two formulation decisions precede all others:

1. **Regression-first, direction-derived.** Direction is not predicted as a class; it is read off a continuous return forecast. Binary direction labels discard magnitude information and constitute weak supervision; the continuous target preserves the full signal and lets the same output serve both tasks.
2. **Mid-price as the label basis.** Labels computed from mid avoid embedding the spread in the target. A model evaluated on mid-price moves must separately prove its edge exceeds the spread — this is enforced at validation (§7), not hidden in the labels [C2].

---

## 2. Why Prediction Is Possible: The Mechanism of Short-Horizon Price Formation

The entire edifice rests on one load-bearing empirical fact and its corollaries.

### 2.1 The core mechanism: order flow imbalance moves price linearly
Cont, Kukanov & Stoikov demonstrated that **order flow imbalance (OFI)** — the net change in queued volume at the best bid and ask — explains short-horizon mid-price changes **linearly, with R² ≈ 65%** across 50 NYSE stocks [D1]. Price formation at this horizon is therefore not primarily a statistical inference problem; it is a near-mechanical accounting identity: net queue pressure × impact coefficient = price change.

### 2.2 The modulator: depth
The impact coefficient is **inversely proportional to market depth** [D2]. The same order flow moves price more when the book is thin. Predictability is therefore *state-dependent*: the model must know the depth regime to scale its microstructure signal.

### 2.3 The consolidation: volume is a shadow of OFI
Trade imbalance explains only R² ≈ 32%, and once OFI is included, trade imbalance **loses significance** (t-stats drop 4×, significant in only 31% of subsamples) [D3]. Further, the celebrated price–volume relation is largely an **aggregation artifact**; the linear OFI model plus a scaling argument *derives* the empirical square-root volume law [D5]. Consequence: in OHLCV-only data, every volume feature is a noisy proxy for unobserved OFI — and Dukascopy tick data (bid/ask + volumes) lets us build a far closer proxy than candle volume ever could.

### 2.4 The clock: impact is intraday-seasonal
The impact coefficient varies through the day in line with known patterns in spread, depth and volatility [D4]. Independently, in indicator space, **relative time within the trading day ranks among the most important predictors**, with positive-return probability peaking near open and close [A2]. Two unrelated methodologies point at the same latent variable: **time indexes a regime — the depth/impact state of the book — not merely a timestamp** (derived node S1).

### 2.5 Independent confirmation from deep learning
DeepLOB's sensitivity analysis shows that a CNN+LSTM trained on raw order-book data concentrates its attention on **prices and sizes across book levels and horizons** [B3] — exactly the variables that constitute OFI. A completely different paradigm rediscovers the same sufficient statistic. And the learned features **transfer to instruments never seen in training** with stable accuracy [B2], indicating the mechanism is **structural to order-book markets**, not an artifact of one instrument.

**Conclusion of §2:** There exists a real, linear, transferable, depth-modulated, time-seasonal mechanism generating short-horizon price changes. The prediction problem decomposes into: (i) measure the mechanism's driver (OFI proxy), (ii) measure its modulators (depth proxy, time regime), (iii) learn only what the mechanism leaves unexplained.

---

## 3. The Logic Chain: Findings → Design Consequences

| # | Premise (finding) | Consequence (design) |
|---|---|---|
| P1 | OFI→price is linear, R²≈65% [D1] | The core predictor must be **linear in OFI**; spending deep-model capacity on a linear relation invites overfitting [C1] |
| P2 | Impact ∝ 1/depth [D2]; impact is intraday-seasonal [D4]; time-of-day is a top predictor [A2] | The model must **condition microstructure signals on time/regime** via explicit interaction features |
| P3 | Volume/trade imbalance subsumed by OFI [D3, D5] | Build an **OFI proxy from tick-level bid/ask/volume**; treat candle volume as its degraded shadow |
| P4 | Continuous indicators beat discrete rule signals [A1] | Feed **raw continuous indicator values**, never buy/sell/hold rule outputs |
| P5 | CCI, Stochastic, Bollinger strongest singles [A4]; MA/TRIX/RSI best combination partners [A5] | These six indicator families form the **indicator core** |
| P6 | Two-way combos add >15% accuracy, three-way +7% [A3]; weak singles (MACD, WPR, WMA, Chaikin) gain sharp importance in combos [A6]; CNN cross-indicator kernels independently rediscover interactions [F3] | **Interactions are first-class signal**: explicit pairwise products/quotients enter the feature set; the architecture must be able to learn the rest |
| P7 | Partial dependence is V-shaped — extreme indicator values in *either* tail produce large log-odds of positive returns [A7] | Signal is **non-monotone**; include \|indicator\| and distance-from-extreme transforms; magnitude and direction are coupled at extremes |
| P8 | RV direction is >80% predictable with RF + technical indicators [E1]; ATR ranks among the best [E2]; log(RV) beats raw RV [E5]; purified implied volatility improves forecasts [E4] | The **magnitude head** is a volatility-forecasting problem: HAR-style multi-horizon RV state + ATR, log-transformed target |
| P9 | Feeding predicted direction into the regression improves the magnitude (level) forecast [E3] | **Cascade architecture**: the magnitude head receives the direction estimate as an input — the heads are conditionally dependent, not parallel |
| P10 | LSTM outperforms MLP/CNN on 1-min bars with 18 indicators [F1]; 2-layer > 1-layer; multi-width CNN kernels capture what LSTM misses [F3]; CNN+Inception+LSTM is SOTA on book data [B1] | The residual learner is a **multi-width Conv1D → 2-layer BiLSTM** sequence model |
| P11 | Deep models beat classical ones but **overfit more**; feature-space size, dropout, early stopping are decisive [C1] | Small trunk, heavy regularization, and capacity spent only on the residual |
| P12 | >80% accuracy does not imply profit once costs and executable prices are considered [C2]; ~2% accuracy differences separate profitable from unprofitable strategies [F2]; standard benchmarks embed biases [C3] | **Validation is the binding constraint**: purged walk-forward, per-regime reporting, and a spread-net economic gate — accuracy alone is inadmissible evidence |

P1 + P11 jointly yield the central architectural theorem of this design (derived node S3): **a linear microstructure core with a nonlinear learner on the residual.** P7 + P9 yield the output theorem (S4): **direction and magnitude are coupled targets and must be modeled as such.**

---

## 4. Mechanism I — Target Construction

**Primary target:** r₃₀ = log(mid_{t+30} / mid_t). Continuous, unbounded, symmetric in log space.

**Magnitude targets (log space, per [E5]):**
- m₃₀ = log(|r₃₀| + ε) — direct magnitude of the move
- RV₃₀ = log(√Σ r²_1m over the next 30 bars) — forward realized volatility; the E-paper demonstrates this quantity's *direction* is >80% predictable [E1], making it the most forecastable magnitude proxy in the literature

**Auxiliary labels:** maximum favorable/adverse excursion (MFE/MAE) over the horizon — the magnitude that execution actually experiences.

**Direction view (evaluation only):** sign(r₃₀) with a dead-zone |r₃₀| < τ excluded, τ calibrated to the prevailing spread. This ensures the direction metric counts only moves large enough to be tradable [C2]. The V-shaped finding [A7] warns that indicator extremes precede *large* moves in both directions — hence magnitude information must reach the direction decision, which the cascade (§6) implements mechanically.

---

## 5. Mechanism II — Signal Extraction

Five feature groups, each a direct operationalization of findings:

**G1 — Microstructure core [D1, D2, D3, D5].** Per-bar OFI proxy from Dukascopy ticks: signed quote changes at bid/ask using tick volumes as queue-size proxy; rolling sums at 1/3/5/10/30 min; tick intensity; spread level and change; OFI ÷ depth-proxy ratio as a running estimate of the impact coefficient. *This group carries the linear core.*

**G2 — Multi-scale momentum [B3, F1].** Mid-price log returns over 1/3/5/10/15/30/60 min — the multi-horizon sensitivity that [B3] observed in the deep model's attention.

**G3 — Indicator core, continuous [A1, A4, A5, E2].** CCI, Stochastic %K/%D, Bollinger %B, RSI, TRIX, MACD, ATR — each at 2–3 window lengths, as raw continuous values. This is the exact subset the evidence singles out; the ten-indicator zoo is deliberately not imported wholesale.

**G4 — Interactions and tail transforms [A3, A6, A7, F3].** Explicit pairwise products/quotients among the strongest singles (e.g., RSI×%B, OFI×spread, CCI×ATR), plus |x| and distance-from-extreme transforms of every G3 feature to expose the V-shaped signal to linear layers. The finding that *weak singles become strong in combinations* [A6] means feature selection on singles alone would wrongly discard MACD/WPR/Chaikin — they enter through their interactions.

**G5 — Time and regime [A2, D4].** Cyclical time-of-day encoding, session flags, minutes-since-open/to-close — and, critically, **time × OFI** and **time × ATR** interactions, because the impact coefficient itself is intraday-seasonal [D4]: the model must be allowed to re-weight its microstructure signal by the clock.

**G6 — Volatility state [E1, E2, E4, E5].** HAR-style log-RV at 5m/15m/30m/1h/1d, Parkinson range, ATR(14). This group feeds primarily the magnitude head.

Input tensor: trailing **60 bars** × F features, rolling z-scored on train-fitted statistics only [C1].

---

## 6. Mechanism III — Model Architecture

```
                trailing 60 × F
                      │
        ┌─────────────┴──────────────┐
        │                            │
  BRANCH A (linear core)      BRANCH B (residual learner)
  Ridge on G1 only            multi-width Conv1D  [F3]
  r̂_lin = w·OFI-features      → BiLSTM ×2         [F3, B1]
  [D1, linear per P1]         → attention pooling
        │                     target: r₃₀ − r̂_lin
        │                            │
        └──────────┬─────────────────┘
                   ▼
        HEAD 1:  r̂₃₀ = r̂_lin + residual̂
                 direction = sign(r̂₃₀)
                   │  r̂₃₀ passed downstream [E3]
                   ▼
        HEAD 2:  m̂₃₀ = g( trunk ⊕ r̂₃₀ ⊕ G6 )      ← cascade [P9]
        HEAD 3:  quantiles q0.1…q0.9 of |r₃₀|       ← magnitude intervals
```

**Why this shape, mechanically:**

- **Branch A** encodes P1: the best-documented relation in short-horizon finance is linear; it gets a linear estimator. It is fit first and frozen, giving an interpretable, auditable backbone whose coefficients can be checked against the 1/depth law [D2].
- **Branch B** encodes P10 + P11: deep sequence models are the strongest known class on this data [B1, F1] *and* the most overfit-prone [C1] — so they are given the smallest possible job: explain what OFI linearity cannot. Capacity is rationed by design, not by hope.
- **The cascade** encodes P9: direction demonstrably improves magnitude forecasts [E3], and extremes carry two-tailed directional information [A7] — so the magnitude head consumes the direction estimate rather than re-deriving it.
- **Quantile head** converts magnitude from a point estimate into a tradable interval, which the economic gate (§7) uses directly.
- **Loss:** L = λ₁·Huber(r₃₀) + λ₂·Huber(log|r₃₀|) + λ₃·QLIKE(RV₃₀); λ tuned on validation folds.

---

## 7. Mechanism IV — Validation as the Binding Constraint

The evidence is unambiguous that evaluation design, not model power, is where such systems live or die [C1, C2, C3] — the derived node S6. The protocol:

1. **Purged walk-forward with embargo.** Labels overlap by construction (adjacent samples share 29 of 30 forward bars); random splits would leak the future into training. Train → embargo ≥ horizon → validate → roll, across years of folds.
2. **Per-regime reporting.** Metrics disaggregated by time-of-day bucket and volatility regime, because predictability is regime-dependent by mechanism [A2, D4]. Aggregate metrics conceal regime failure.
3. **The economic gate.** A minimal decision rule — act only when the direction is confident *and* predicted magnitude exceeds k× the *actual historical spread* (which Dukascopy data provides) — must be net-positive on the large majority of folds. This is the only admissible definition of success [C2]; the 2%-edge finding [F2] is why the gate uses real spreads rather than assumed ones.
4. **Adversarial checks [C3].** Shuffle test (score must collapse to chance); single-feature audits; and a **transfer test** on a held-out symbol — the mechanism claims universality [B2], so the model must demonstrate it.
5. **Component attribution.** Full model must beat Branch A alone, Branch B alone, persistence, and a HAR-RV baseline. Each component must pay for its capacity [P11].

---

## 8. The Eight Success Conditions

Success is the **conjunction** of the following, each falsifiable, each resting on cited evidence:

| # | Condition | Rests on | Verified by |
|---|---|---|---|
| C1 | The tick-built OFI proxy correlates with next-bar returns | [D1, D3] | corr(OFI_proxy, r₁m) strongly positive on raw data — *tested before any modeling* |
| C2 | The linear core alone beats persistence out-of-sample | [D1, D2] | Branch A vs. r̂=0 across walk-forward folds |
| C3 | The residual learner adds over the linear core | [B1, F1, C1] | Full model vs. Branch A; if not, P11 has dominated and capacity is cut |
| C4 | Interaction features carry independent signal | [A3, A6, F3] | Ablation of G4 must degrade validation loss |
| C5 | Time-regime conditioning carries independent signal | [A2, D4] | Ablation of G5; per-bucket metric lift |
| C6 | The cascade beats independent heads | [E3, A7] | Head-coupling ablation on magnitude error |
| C7 | Log-space magnitude outperforms raw | [E5] | Target-transform ablation |
| C8 | The edge survives real spreads on most folds, in most regimes | [C2, F2] | The economic gate (§7.3) — the terminal condition |

If C1 fails, the microstructure design is void on this data and the system falls back to G2+G3+G6 with cascade heads. If C8 fails while C1–C7 pass, the model is right and the market is simply too expensive to trade — a finding, not a bug.

---

## 9. Failure Modes and Their Signatures

| Failure | Mechanism | Signature | Remedy per evidence |
|---|---|---|---|
| Overfit residual learner | Deep class overfits most [C1] | Train ≪ validation gap; fold variance high | Shrink trunk, raise dropout, shorten lookback |
| Benchmark illusion | Dataset/label bias [C3] | Great aggregate, dead per-regime | Per-bucket reporting; shuffle test |
| Accuracy without profit | Costs consume thin edges [C2, F2] | Positive MCC, negative gate | Raise magnitude threshold k; trade fewer, larger-conviction events |
| Regime break | Impact coefficient shifts [D2, D4] | OFI coefficient drift; time-bucket decay | Re-estimate Branch A on rolling window; monitor OFI→r slope |
| Proxy decay | Tick volumes imperfectly proxy queue size [D3, D5] | C1 weakens over time/instruments | Rebuild proxy; fall back to trade-imbalance features |
| Magnitude–direction decoupling | Heads trained independently | Magnitude calibrated only in quiet regimes | Restore cascade [E3]; condition on \|r̂₃₀\| |

---

## 10. Findings Registry (Evidence Base)

- **[A1–A7]** *Equity Price Direction Prediction for Day Trading: Ensemble Classification Using Technical Analysis Indicators with Interaction Effects* — Ghent University. Minute-level RF; continuous > discrete indicators; time-of-day importance; combination gains; CCI/Stoch/BB and MA/TRIX/RSI rankings; weak-single/combo inversion; V-shaped partial dependence.
- **[B1–B3]** *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books* — Zhang, Zohren & Roberts, IEEE TSP 2019. SOTA architecture; cross-instrument transfer; LIME sensitivity to price/size.
- **[C1–C3]** *Predicting Stock Price Changes Based on the Limit Order Book: A Survey* — MDPI Mathematics 2022. Deep-vs-classical overfitting; accuracy≠profit; FI-2010 benchmark bias.
- **[D1–D5]** *The Price Impact of Order Book Events* — Cont, Kukanov & Stoikov, J. Financial Econometrics 2014. OFI linearity (R²≈65%); impact ∝ 1/depth; OFI dominance over trade imbalance; intraday impact seasonality; volume-relation artifact / square-root law.
- **[E1–E5]** *Forecasting of Realised Volatility with the Random Forests Algorithm* — MDPI JRFM 2018. RV direction >80%; ATR importance; direction→magnitude cascade gain; purified implied volatility; log-RV superiority.
- **[F1–F3]** *High-Frequency Trend Prediction of Bitcoin Using Technical Indicators and Deep Learning.* LSTM superiority on 1-min/18-indicator inputs; 2% edge economics; 2-layer LSTM and multi-width kernel effects.
- **Derived (S1–S6):** time-as-regime [A2∧D4]; interactions-as-signal [A6∧F3]; linear-core/nonlinear-residue [D1∧C1]; direction↔magnitude coupling [E3∧A7]; volume-as-OFI-shadow [D3∧D5]; evaluation-as-binding-constraint [B2∧C3].

---

## 11. Summary: The Complete Chain of Logic in One Paragraph

Order flow imbalance moves price linearly and explains most short-horizon variance [D1]; the conversion rate is set by depth [D2] and the clock [D4, A2]; volume is merely OFI's shadow [D3, D5]; deep models independently confirm these variables and show they transfer across instruments [B2, B3]. Therefore the model is a linear OFI core — nothing more is needed where the relation is known to be linear — plus a heavily regularized sequence learner confined to the residual [C1, B1, F1, F3], fed continuous indicators chosen by evidence [A1, A4, A5], their interactions [A3, A6, F3], their tail transforms [A7], and time-regime conditioning [A2, D4]. Direction is read off a continuous return forecast; magnitude is forecast in log space by a head that consumes the direction estimate [E3, E5, E1, E2]. And because accuracy is not profit [C2], edges are thin [F2], and benchmarks lie [C3], the system is admitted to operation only through purged walk-forward validation with per-regime reporting and a spread-net economic gate. Each link is a published finding; each finding has a test; the eight tests together define success.