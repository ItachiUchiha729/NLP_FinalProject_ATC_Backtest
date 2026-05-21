# Backtesting the ProntoNLP Earnings-Call ATC Signal

**Course:** LLM-Driven Quant Research  
**Dataset:** Earnings_ATC_until_2026-04-21.csv (~4.5 GB, 2.74M rows, 609 cols)

## One-Command Reproduction

```bash
# 1. Place Earnings_ATC_until_2026-04-21.csv in the project root
# 2. Run notebooks in order (01→08) using the python311 kernel
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python311 \
  notebooks/01_data_pipeline.ipynb
# Repeat for 02→08
```

## Notebook Pipeline

| Notebook | Description | Key Output |
|---|---|---|
| 01_data_pipeline | CSV → Parquet (chunked, FORCE_FLOAT schema) | `data/signals.parquet` (304 MB) |
| 02_universe_and_prices | PIT universe membership + yfinance prices + fwd returns | `data/signals_with_returns.parquet` |
| 03_feature_engineering | 70+ features: MWNS, speaker divergence, QoQ, sector rank | `data/features.parquet` |
| 04_backtest_baseline | IC heatmaps, quintile L/S, decile drawdowns, placebo test | `figures/` |
| 05_walkforward_model | LightGBM walk-forward (train≤2019, test 2020→2026) | `data/model_predictions.parquet` |
| 06_portfolio_robustness | Sub-periods, sector-neutral, turnover-Sharpe frontier | `data/summary_results.csv` |
| 07_horizon_sensitivity | 5d/10d/20d × Weekly/Monthly — identifies optimal 20d | `figures/horizon_sensitivity.png` |
| 08_regime_aware_model | Regime-gated XGBoost-DART with sector normalisation | `data/regime_model_predictions.parquet` |

## Key Results (Test Period 2020–2026, 5bps TC)

| Model | SP500 | SP1500 | RU3K |
|---|---|---|---|
| NB05: LightGBM Weekly+5d | −1.20 | −0.03 | +0.57 |
| NB07: ATC Weekly+20d | +0.56 | +0.71 | +1.22 |
| NB08: Regime-XGB Monthly+20d | **+1.02** | −0.45 | −0.46 |

**Best production configuration:**
- SP500: Regime-Aware XGBoost, Monthly+20d (Sharpe 1.02)
- SP1500/RU3K: ATCClassifierScore, Weekly+20d (Sharpe 0.71 / 1.22)

## Look-Ahead Bias Audit

See `look_ahead_audit_checklist.txt`. All 12 items reviewed; 11 pass, 1 corrected
(INGESTDATEUTC is a batch export timestamp, not a per-call availability date).  
Three documented approximation caveats: SP1500 pre-2012/2019, RU3K proxy, INGESTDATEUTC.

## Environment

```bash
pip install pandas pyarrow yfinance lightgbm xgboost scikit-learn \
            scipy seaborn matplotlib pandas_market_calendars lxml beautifulsoup4
```
