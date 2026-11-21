# Gate 1 Quality Audit Report — EURUSD

**Audit Status:** ✅ **PASSED**
**Audit Timestamp:** 2026-08-23 21:55:10 UTC

---

## 1. Dataset Dimensions
| Metric | Value |
|---|---|
| **Symbol** | `EURUSD` |
| **Total Bars** | `1,191,112` |
| **Start Date** | `2007-01-01 00:00:00` |
| **End Date** | `2026-08-23 21:15:00` |

---

## 2. Integrity & Anomaly Summary
| Test Category | Detected Issues | Rate (%) | Status |
|---|---|---|---|
| **Intra-Session Trading Gaps (>5m)** | 965 | 0.0810% | ✅ Clean |
| **Bad-Print Spikes (>10x ATR)** | 13 | 0.0011% | ✅ Clean |
| **Inverted / Negative Spreads** | 0 | 0.0000% | ✅ Zero |
| **Spread Blowouts (>50x Median)** | 0 | 0.0000% | Info |
| **Zero Volume Bars** | 68,432 | 5.7452% | Info |

---

## 3. Spread Statistics
- **Median Spread:** `0.00003`
- **Maximum Recorded Spread:** `0.00326`

---

## 4. Gate 1 Evaluation & Kill Criteria
- **Total Anomaly Count:** `978`
- **Composite Anomaly Ratio:** **`0.0821%`** (Kill Threshold: `< 0.50%`)
- **Conclusion:** **GATE 1 PASS: 0.082% anomalous bars (< 0.5% threshold)**
