# Gate 5 Harness Validation Report — USDJPY

**Status:** ✅ PASSED

## Criterion 1 — Persistence (B0) MCC ≈ 0
| | Value | Threshold | Status |
|---|---|---|---|
| B0 Mean MCC | `+0.0000` | `|mcc| < 0.05` | ✅ |

## Criterion 2 — HAR-RV (B1) beats B0 on QLIKE in ≥ 60% of folds
| | Value | Threshold | Status |
|---|---|---|---|
| B1 QLIKE-lift folds | `94%` | `≥ 60%` | ✅ |

## Verdict
**`GATE 5 PASS: ✅ B0 MCC≈0 (mean=+0.0000); ✅ B1 beats B0 on QLIKE in 51/54 folds (94%)`**

---

> If Gate 5 FAILS: the harness is broken (data leakage, label misalignment).
> Do not proceed to Phase 6 until this is resolved.
