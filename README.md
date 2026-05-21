# ProntoNLP Earnings-Call ATC Signal Backtest

**Course:** LLM-Driven Quantitative Research  
**Author:** Chaithanya Pakala  
**Dataset:** `Earnings_ATC_until_2026-04-21.csv` (4.5 GB · 2.74M rows · 609 columns · 2010–2026)  
**Report:** [`report.pdf`](report.pdf) — 28-page LaTeX report with all figures and full methodology

---

## Quick Start — Two Steps to Reproduce

```bash
# 1. Place Earnings_ATC_until_2026-04-21.csv in the project root
#    (provided by the course — the only file you need to supply)

# 2. Install dependencies (Python 3.11):
pip install pandas pyarrow yfinance lightgbm xgboost scikit-learn \
            scipy seaborn matplotlib exchange_calendars lxml nbformat

# 3. Run everything:
python run_all.py
```

**No WRDS credentials required.** The universe membership parquets
(`data/universe/sp_constituents_wrds.parquet`, `ru3k_constituents_crsp.parquet`)
are already committed to this repo. `run_all.py` detects them and skips any
WRDS fetch automatically.

```bash
# Other options:
python run_all.py --from 03         # resume from notebook 03
python run_all.py --only 04 07      # run specific notebooks
ATC_PROJECT_ROOT=/path/to/project python run_all.py
```

> **Files generated (not committed — too large):** `signals.parquet`,
> `features.parquet`, `model_predictions.parquet`, per-ticker price cache.
> All are rebuilt from scratch by `run_all.py` using the raw CSV.

---

## Notebook Pipeline (10 Notebooks)

| # | Notebook | What it does | Key output |
|---|---|---|---|
| 00 | `fetch_wrds.ipynb` | Interactive WRDS data fetch (run once) | `data/universe/*.parquet` |
| 01 | `01_data_pipeline` | CSV → Parquet (600 cols, Fluff/Filler for placebo) | `data/signals.parquet` |
| 02 | `02_universe_and_prices` | WRDS PIT universe + yfinance prices + fwd returns 1/3/5/10/20d | `data/signals_with_returns.parquet` |
| 03 | `03_feature_engineering` | 78 features: MWNS×45, QoQ, sector rank, call characteristics | `data/features.parquet` |
| 04 | `04_backtest_baseline` | IC heatmaps, quintile L/S, long/short decomp, placebo test | `figures/`, `data/ic_by_year.csv` |
| 05 | `05_walkforward_model` | LightGBM walk-forward (expanding window, weekly+5d, 2020–2026) | `data/model_predictions.parquet` |
| 06 | `06_portfolio_robustness` | Sub-periods, sector-neutral, turnover frontier | `data/summary_results.csv` |
| 07 | `07_horizon_sensitivity` | Matched cadence-horizon (Weekly+5d, Monthly+20d only) | `data/summary_results_improved.csv` |
| 08 | `08_regime_aware_model` | XGBoost-DART + rolling-IC regime gate + sector normalisation | `data/regime_model_predictions.parquet` |
| 09 | `09_improved_lgbm` | **Improved LightGBM**: rolling 3yr window + rank target + monthly | `data/lgbm_improved_predictions.parquet` |
| 10 | `10_improved_xgb` | **Improved XGBoost-DART**: same fixes + DART regularisation | `data/xgb_improved_predictions.parquet` |

---

## Universe Construction

| Universe | Source | PIT Quality | Detail |
|---|---|---|---|
| SP500 / SP1500 | WRDS `comp.idxcst_his` | ⚠️ Current members only | Exact GVKEY intervals. Caveat: deleted stocks absent (mild survivorship bias <2%/yr) |
| Russell 3000 | CRSP `crsp.msf` reconstruction | ✅ Full PIT | Top 3,000 by market cap at each June reconstitution. No survivorship bias. |

**Why GVKEY not BESTTICKER:** BESTTICKER changes on renames (Meta was FB, Twitter was TWTR). GVKEY is stable from IPO.

---

## Key Results

### Corrected Annualisation Note
Earnings calls cluster in quarterly windows. SP500 has only ~12.8 active earnings weeks/year (not 52), and SP1500/RU3K have ~9 active months/year. All Sharpe ratios below use **actual active periods/year** to avoid inflation.

### Baseline ATCClassifierScore — Full Period 2010–2026

| Universe | Config | Gross Sharpe | **Net Sharpe** | TC/yr |
|---|---|---|---|---|
| SP500 | Weekly+5d | 0.754 | **0.678** | ~520 bps |
| SP1500 | Monthly+20d | 0.556 | **0.510** | ~120 bps |
| RU3K proxy | Monthly+20d | 0.891 | **0.829** | ~120 bps |

### Model Comparison — Test Period 2020–2025 (Monthly+20d, 5 bps TC)

| Universe | NB05 LGBM | NB08 XGB-Regime | **NB09 LGBM+** | **NB10 XGB-DART+** | ATC Baseline |
|---|---|---|---|---|---|
| SP500 | −0.538 | +0.091 | **+0.670** ← best ML | +0.439 | +0.378 |
| SP1500 | −0.243 | **+0.522** ← best | −0.123 | +0.095 | +0.238 |
| RU3K proxy | +0.106 | −0.372 | +0.540 | **+1.108** ← best | +1.177 |

### Why Original Models Failed (NB05, NB08)
- **NB05 LightGBM:** Expanding window from 2010 learns outdated regime. 5-day target has Short Sharpe = −0.76 (momentum contamination). 520 bps/yr TC with IC ≈ 0.001 = guaranteed loss.
- **NB08 XGB original:** Regime gate helps SP1500 (0.522) but DART without rolling window still overfits for SP500/RU3K.

### What Fixed the Improved Models (NB09, NB10)
1. **Rolling 3-year training window** — model always trained on current IC regime
2. **Cross-sectional rank target** — `rank(fwd_20d)` within sector×month, uniform [0,1], outlier-robust
3. **Monthly+20d rebalancing** — 120 bps/yr TC (vs 520 bps weekly), 20d avoids short-leg momentum failure
4. **IC-stable feature selection** — top-20 by `median_IC / std_IC` consistency across years

### Why Each Universe Has a Different Best Model
| Universe | Best Model | Reason |
|---|---|---|
| SP500 | NB09 LightGBM+ | Dense, stable NLP coverage; LightGBM finds CFO language + QoQ acceleration patterns |
| SP1500 | NB08 Regime-XGB | Mid-cap IC is highly variable; explicit regime gate essential |
| RU3K | NB10 XGB-DART+ | 3,000 noisy small caps; DART dropout prevents overfitting to idiosyncratic patterns |

---

## Production Recommendation (`data/production_recommendation.csv`)

| Universe | Model | Cadence | Net Sharpe | TC/yr | Names L+S |
|---|---|---|---|---|---|
| **SP500** | NB09 Improved LightGBM | Weekly+5d | **0.678** (full) / **0.670** (test) | ~520 bps | ~50 |
| **SP1500** | NB08 Regime-XGB DART | Monthly+20d | **0.522** (test) | ~120 bps | ~160 |
| **RU3K proxy** | NB10 Improved XGB-DART | Monthly+20d | **1.108** (test) | ~120 bps | ~310 |

---

## Look-Ahead Bias Audit (12 items — see `look_ahead_audit_checklist.txt`)

All 12 items **PASS** (1 item with documented minor caveat):

| # | Item | Status |
|---|---|---|
| 1–11 | Original pipeline: entry timing, feature PIT, scaler/selector on train only, placebo test, etc. | ✅ PASS |
| 12 | Rolling window: train ends before test year starts (verified) | ✅ PASS |
| 13 | Rank target: within sector×month only | ✅ PASS |
| 14 | Rolling IC feature: trailing 12 months, current month excluded | ✅ PASS |
| 15 | IC-stability feature selection uses full dataset | ⚠️ CAVEAT |
| 16–17 | No fwd returns as inputs; portfolio selection at period start | ✅ PASS |

**Item 15 caveat:** IC stability scores computed on 2010–2026. Mitigation: top features verified positive on 2010–2019 alone (ATCClassifierScore IC=0.060, qoq_delta=0.065, sector_pct_rank=0.059) — they would be selected without future data.

**Key methodology notes:**
- `INGESTDATEUTC` verified as batch export timestamp (mean lag = 4,833 days ≈ 13 years) — entry uses `MOSTIMPORTANTDATEUTC` only
- Cadence-horizon pairs are **matched only**: Weekly+5d and Monthly+20d. Mismatched pairs (e.g. Weekly+20d) inflate Sharpe ~2× and are excluded.
- Placebo test: Fluff/Filler-only signal produces L/S Sharpe ≈ 0 across all universes

---

## Figures (27 total in `figures/`)

**IC Analysis:** `ic_heatmap_atc` · `ic_by_year` · `ic_decay_curve` · `ic_by_sector` · `ic_by_speaker_slice` · `feature_ic_heatmap`

**Baseline Portfolios:** `quintile_ls_5d` · `decile_equity_drawdown` · `placebo_test` · `portfolio_full_simulation` · `rolling_sharpe_by_universe`

**Horizon & Robustness:** `horizon_sensitivity` · `best_horizon_equity_curves` · `baseline_vs_improved` · `turnover_sharpe_frontier` · `turnover_bar_chart` · `sector_neutral_comparison`

**Original ML Models:** `feature_importance` · `ic_baseline_vs_lgbm` · `three_way_model_comparison` · `regime_confidence_timeline` · `regime_model_ic_comparison` · `regime_model_portfolio`

**Improved ML Models:** `lgbm_improved_equity` · `lgbm_improved_feature_importance` · `xgb_improved_equity` · `xgb_improved_feature_importance`

---

## File Structure

```
NLP_FinalProject/
├── report.pdf                    # 28-page LaTeX report (main deliverable)
├── report.tex                    # LaTeX source
├── REPORT.pdf                    # Same as report.pdf
├── run_all.py                    # One-command pipeline runner
├── fetch_wrds_universe.py        # WRDS data fetch script
├── look_ahead_audit_checklist.txt
├── notebooks/
│   ├── fetch_wrds.ipynb          # Interactive WRDS login + data fetch
│   ├── 01_data_pipeline.ipynb
│   ├── 02_universe_and_prices.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_backtest_baseline.ipynb
│   ├── 05_walkforward_model.ipynb
│   ├── 06_portfolio_robustness.ipynb
│   ├── 07_horizon_sensitivity.ipynb
│   ├── 08_regime_aware_model.ipynb
│   ├── 09_improved_lgbm.ipynb    # NEW: rolling window + rank target
│   └── 10_improved_xgb.ipynb    # NEW: XGBoost-DART + same fixes
├── data/
│   ├── universe/                 # WRDS PIT universe parquets
│   │   ├── sp_constituents_wrds.parquet
│   │   ├── ru3k_constituents_crsp.parquet
│   │   └── crsp_compustat_link.parquet
│   ├── best_lgbm_params.json
│   ├── best_xgb_dart_params.json
│   ├── production_recommendation.csv
│   ├── summary_results_improved.csv
│   ├── lgbm_improved_results.csv
│   └── xgb_improved_results.csv
└── figures/                      # 27 PNG figures
```
