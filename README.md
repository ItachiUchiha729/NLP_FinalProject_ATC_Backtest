# Backtesting the ProntoNLP Earnings-Call ATC Signal

**Course:** LLM-Driven Quant Research  
**Dataset:** `Earnings_ATC_until_2026-04-21.csv` (~4.5 GB, 2.74M rows, 609 cols)

## One-Command Reproduction

```bash
# 1. Place Earnings_ATC_until_2026-04-21.csv in the project root
# 2. Run the full pipeline:
python run_all.py

# Or resume from a specific notebook:
python run_all.py --from 03

# Or run specific notebooks only:
python run_all.py --only 04 07

# Set project root explicitly:
ATC_PROJECT_ROOT=/path/to/project python run_all.py
```

**Kernel:** `python311` (Python 3.11). Install dependencies:
```bash
pip install pandas pyarrow yfinance lightgbm xgboost scikit-learn \
            scipy seaborn matplotlib pandas_market_calendars lxml beautifulsoup4 nbformat
```

## Notebook Pipeline

| Notebook | Description | Key Output |
|---|---|---|
| 01_data_pipeline | CSV → Parquet (chunked, FORCE_FLOAT, includes Fluff/Filler for placebo) | `data/signals.parquet` |
| 02_universe_and_prices | SP500 PIT (Wikipedia 1976+), yfinance prices, fwd returns 1/3/5/10/20d | `data/signals_with_returns.parquet` |
| 03_feature_engineering | 78 features: MWNS×45, speaker divergence, QoQ, sector_pct_rank (entry_date-sorted) | `data/features.parquet` |
| 04_backtest_baseline | IC heatmaps, IC by year, EventScore IC, quintile L/S, long/short decomp, placebo | `figures/`, `data/ic_by_year.csv` |
| 05_walkforward_model | LightGBM walk-forward (train ≤ 2019, quarterly folds, 5d target) | `data/model_predictions.parquet` |
| 06_portfolio_robustness | Sub-periods, sector-neutral, turnover frontier, production recommendation table | `data/summary_results.csv` |
| 07_horizon_sensitivity | 5d/10d/20d × Weekly/Monthly sensitivity — identifies Weekly+20d as optimal | `figures/horizon_sensitivity.png` |
| 08_regime_aware_model | XGBoost-DART + rolling-IC regime gating + sector normalisation | `data/regime_model_predictions.parquet` |

## Key Results (Net Sharpe, 5 bps TC)

### Full Period 2010–2026 (ATCClassifierScore)

| Universe | Weekly+5d | Weekly+20d |
|---|---|---|
| SP500 | −1.20 | +1.23 |
| SP1500 | −0.03 | **+1.42** |
| RU3K proxy | +0.57 | **+1.34** |

### Test Period 2020–2026 (walk-forward models)

| Universe | NB05 LGBM W+5d | NB07 ATC W+20d | NB08 Regime-XGB M+20d |
|---|---|---|---|
| SP500 | −1.20 | +0.56 | **+1.02** |
| SP1500 | −0.03 | +0.71 | −0.45 |
| RU3K proxy | +0.57 | +1.22 | −0.49 |

**Production recommendation (see `data/production_recommendation.csv`):**
- **SP500**: Regime-Aware XGBoost, Monthly+20d — Sharpe 1.02, ~120 bps TC/yr
- **SP1500**: ATCClassifierScore, Weekly+20d — Sharpe 1.42, ~520 bps TC/yr
- **RU3K proxy**: ATCClassifierScore, Weekly+20d — Sharpe 1.34, proxy universe caveat applies

## Look-Ahead Bias Audit

See `look_ahead_audit_checklist.txt`. 15 items reviewed; all pass.

Key methodology fixes (vs v1):
- `sentences_sector_z` uses `shift(1)` before expanding (fully PIT)
- `sector_pct_rank` sorted by `entry_date` not `DocDate` (avoids same-day leakage)
- Fluff/Filler columns now in `signals.parquet` → genuine non-zero placebo signal
- INGESTDATEUTC empirically verified as batch export timestamp (mean lag = 13 yr)

## Universe Caveats

- **SP500**: Fully PIT via Wikipedia history back to 1976 ✓
- **SP1500**: SP400 Wikipedia history from 2012; SP600 from 2019 — pre-2012/2019 uses current membership
- **RU3K proxy**: All US-listed ATC tickers (~7,372) vs true Russell 3000 (~3,000) — labelled as proxy throughout
