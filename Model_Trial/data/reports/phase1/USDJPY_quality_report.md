# Gate 1 Quality Audit Report — USDJPY

**Audit Status:** ✅ **PASSED**
**Audit Timestamp:** 2026-08-23 21:55:16 UTC

---

## 1. Dataset Dimensions
| Metric | Value |
|---|---|
| **Symbol** | `USDJPY` |
| **Total Bars** | `1,168,494` |
| **Start Date** | `2007-03-27 00:00:00` |
| **End Date** | `2026-08-23 21:35:00` |

---

## 2. Integrity & Anomaly Summary
| Test Category | Detected Issues | Rate (%) | Status |
|---|---|---|---|
| **Intra-Session Trading Gaps (>5m)** | 929 | 0.0795% | ✅ Clean |
| **Bad-Print Spikes (>10x ATR)** | 41 | 0.0035% | ✅ Clean |
| **Inverted / Negative Spreads** | 0 | 0.0000% | ✅ Zero |
| **Spread Blowouts (>50x Median)** | 140 | 0.0120% | Info |
| **Zero Volume Bars** | 68,440 | 5.8571% | Info |

---

## 3. Spread Statistics
- **Median Spread:** `0.00500`
- **Maximum Recorded Spread:** `0.53700`

---

## 4. Gate 1 Evaluation & Kill Criteria
- **Total Anomaly Count:** `970`
- **Composite Anomaly Ratio:** **`0.0830%`** (Kill Threshold: `< 0.50%`)
- **Conclusion:** **GATE 1 PASS: 0.083% anomalous bars (< 0.5% threshold)**
