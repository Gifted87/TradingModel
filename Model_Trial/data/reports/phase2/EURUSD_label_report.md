# Gate 2 Target & Label Verification Report — EURUSD

**Gate Status:** ✅ **PASSED**
**Horizon:** 30 Minutes (6 Bars)
**Total Valid Labeled Rows:** `1,186,733`

---

## 1. Primary Target: $r_{30} = \log(\text{mid}_{t+30} / \text{mid}_t)$
| Statistic | Value | Theoretical Expectation | Status |
|---|---|---|---|
| **Mean** | `+0.000000` | $\approx 0$ (Martingale property) | ✅ Sane |
| **Standard Deviation** | `0.000685` | Multi-minute volatility | ✅ Normal |
| **Skewness (winsorized)** | `+0.0274` | $[-1.80, +1.80]$ post p0.010/p99.990 clip | ✅ Symmetric |
| **Excess Kurtosis (winsorized)** | `+15.32` | $> 0$ (Fat-tailed / Leptokurtic) | ✅ Realistic |
| **Range [Min, Max] (raw)** | `[-0.0157, +0.0203]` | Finite log-return bounds | ✅ Clean |

---

## 2. Magnitude & Volatility Targets (Log-Space per [E5])
- **Direct Magnitude $m_{30} = \log(|r_{30}| + \epsilon)$:**
  - Mean: `-8.7924` | Std: `2.2259`
- **Forward Realized Volatility $RV_{30} = \log(\sqrt{\sum r^2} + \epsilon)$:**
  - Mean: `-8.2100` | Std: `2.0592`

---

## 3. Excursions & Execution Boundary Metrics
- **Mean MFE ($MFE_{30}$):** `+0.000316` (Max upward excursion)
- **Mean MAE ($MAE_{30}$):** `-0.000315` (Max downward excursion)

---

## 4. Direction & Economic Dead-Zone ($|r_{30}| < \tau$)
- **Up-move Frequency:** `47.08%`
- **Down-move Frequency:** `46.75%`
- **Median Spread Threshold ($\tau$):** `0.000029`
- **Dead-Zone Fraction ($|r_{30}| < \text{Spread}$):** **`13.88%`** (Excluded from direction metrics per [C2])
- **Tradable Signal Fraction:** **`86.12%`**

---

## 5. Winsorization Metadata (Audit-Only; Parquet Labels Unchanged)
- **Clip Quantile:** p`0.010` / p`99.990`
- **Clip Bounds:** `[-0.008055, +0.007911]`
- **Rows Clipped:** `238` of `1,186,733` (`0.0201%`)
- **Note:** Winsorization applies to the audit skew metric only. Raw r30 values in the Parquet file are unmodified.

---

## 6. Gate 2 Verdict & Conclusion
**`GATE 2 PASS: Valid stationary label distributions (skew_w=+0.027, dead_zone=13.9%, tradable=86.1%, winsorized=238 rows at p0.010/p99.990)`**
