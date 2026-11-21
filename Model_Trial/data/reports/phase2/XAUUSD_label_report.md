# Gate 2 Target & Label Verification Report — XAUUSD

**Gate Status:** ✅ **PASSED**
**Horizon:** 30 Minutes (6 Bars)
**Total Valid Labeled Rows:** `1,637,155`

---

## 1. Primary Target: $r_{30} = \log(\text{mid}_{t+30} / \text{mid}_t)$
| Statistic | Value | Theoretical Expectation | Status |
|---|---|---|---|
| **Mean** | `+0.000006` | $\approx 0$ (Martingale property) | ✅ Sane |
| **Standard Deviation** | `0.001632` | Multi-minute volatility | ✅ Normal |
| **Skewness (winsorized)** | `-0.2605` | $[-1.80, +1.80]$ post p0.010/p99.990 clip | ✅ Symmetric |
| **Excess Kurtosis (winsorized)** | `+15.84` | $> 0$ (Fat-tailed / Leptokurtic) | ✅ Realistic |
| **Range [Min, Max] (raw)** | `[-0.0573, +0.0459]` | Finite log-return bounds | ✅ Clean |

---

## 2. Magnitude & Volatility Targets (Log-Space per [E5])
- **Direct Magnitude $m_{30} = \log(|r_{30}| + \epsilon)$:**
  - Mean: `-7.5650` | Std: `1.4229`
- **Forward Realized Volatility $RV_{30} = \log(\sqrt{\sum r^2} + \epsilon)$:**
  - Mean: `-6.8936` | Std: `0.8957`

---

## 3. Excursions & Execution Boundary Metrics
- **Mean MFE ($MFE_{30}$):** `+0.000773` (Max upward excursion)
- **Mean MAE ($MAE_{30}$):** `-0.000780` (Max downward excursion)

---

## 4. Direction & Economic Dead-Zone ($|r_{30}| < \tau$)
- **Up-move Frequency:** `50.25%`
- **Down-move Frequency:** `49.32%`
- **Median Spread Threshold ($\tau$):** `0.000243`
- **Dead-Zone Fraction ($|r_{30}| < \text{Spread}$):** **`29.72%`** (Excluded from direction metrics per [C2])
- **Tradable Signal Fraction:** **`70.28%`**

---

## 5. Winsorization Metadata (Audit-Only; Parquet Labels Unchanged)
- **Clip Quantile:** p`0.010` / p`99.990`
- **Clip Bounds:** `[-0.019259, +0.018252]`
- **Rows Clipped:** `328` of `1,637,155` (`0.0200%`)
- **Note:** Winsorization applies to the audit skew metric only. Raw r30 values in the Parquet file are unmodified.

---

## 6. Gate 2 Verdict & Conclusion
**`GATE 2 PASS: Valid stationary label distributions (skew_w=-0.261, dead_zone=29.7%, tradable=70.3%, winsorized=328 rows at p0.010/p99.990)`**
