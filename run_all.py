"""
run_all.py — One-command reproduction of the ATC backtest pipeline.

Usage:
    python run_all.py                   # run everything
    python run_all.py --from 03         # resume from notebook 03
    python run_all.py --only 04 07      # run specific notebooks only
    ATC_PROJECT_ROOT=/path/to/project python run_all.py

Prerequisites:
    1. Place Earnings_ATC_until_2026-04-21.csv in the project root
    2. pip install pandas pyarrow yfinance lightgbm xgboost scikit-learn
              scipy seaborn matplotlib exchange_calendars lxml nbformat

NOTE: WRDS universe data is already cached in data/universe/*.parquet
      (committed to the repo). No WRDS credentials needed to reproduce results.
      If you want to re-fetch from WRDS, run: notebooks/fetch_wrds.ipynb

Steps 1-10: all notebooks executed via jupyter nbconvert (non-interactive).
"""
import subprocess, sys, argparse, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NB_DIR  = PROJECT / 'notebooks'
UNIV_DIR = PROJECT / 'data' / 'universe'

NOTEBOOKS = [
    ('01', '01_data_pipeline.ipynb',        600,  'CSV → Parquet cache (600 cols incl. Fluff/Filler)'),
    ('02', '02_universe_and_prices.ipynb', 7200,  'WRDS PIT universe (cached) + yfinance prices + fwd returns'),
    ('03', '03_feature_engineering.ipynb', 1800,  'Feature engineering (MWNS×45, QoQ, sector rank)'),
    ('04', '04_backtest_baseline.ipynb',   1800,  'Baseline IC + quintile backtest + placebo test'),
    ('05', '05_walkforward_model.ipynb',   3600,  'Walk-forward LightGBM (original)'),
    ('06', '06_portfolio_robustness.ipynb',1800,  'Portfolio simulation + robustness checks'),
    ('07', '07_horizon_sensitivity.ipynb', 1800,  'Horizon sensitivity (matched Weekly+5d / Monthly+20d)'),
    ('08', '08_regime_aware_model.ipynb',  3600,  'Regime-aware XGBoost-DART'),
    ('09', '09_improved_lgbm.ipynb',       1800,  'Improved LightGBM (rolling window + rank target)'),
    ('10', '10_improved_xgb.ipynb',        1800,  'Improved XGBoost-DART (rolling window + rank target)'),
]

PYTHON = sys.executable


def wrds_cache_exists() -> bool:
    return ((UNIV_DIR / 'sp_constituents_wrds.parquet').exists() and
            (UNIV_DIR / 'ru3k_constituents_crsp.parquet').exists())


def fetch_wrds_data():
    """Run the WRDS fetch script interactively (stdin passed through so user can type credentials)."""
    fetch_script = PROJECT / 'fetch_wrds_universe.py'
    if not fetch_script.exists():
        print(f'ERROR: {fetch_script} not found.')
        sys.exit(1)

    print('\n' + '='*60)
    print('STEP 0: Fetching WRDS universe data (one-time setup)')
    print('='*60)
    print('You will be prompted for your WRDS username and password.')
    print('Credentials are saved to ~/.pgpass — all future runs are automatic.\n')

    result = subprocess.run(
        [PYTHON, str(fetch_script)],
        stdin=sys.stdin,    # pass terminal through so user can type credentials
        check=False,
    )
    if result.returncode != 0:
        print('\nERROR: WRDS fetch failed. Check credentials and try again.')
        print('You can also run manually: python fetch_wrds_universe.py')
        sys.exit(1)
    print('\n✓ WRDS universe data cached. Continuing with notebooks...\n')


def run_notebook(nb_file: Path, timeout: int, label: str) -> bool:
    print(f'\n{"="*60}')
    print(f'Running: {nb_file.name}')
    print(f'         {label}')
    print(f'{"="*60}')
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, '-m', 'nbconvert', '--to', 'notebook',
         '--execute', '--inplace',
         f'--ExecutePreprocessor.timeout={timeout}',
         '--ExecutePreprocessor.kernel_name=python311',
         str(nb_file)],
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f'  ✓  Completed in {elapsed/60:.1f} min')
        return True
    else:
        print(f'  ✗  FAILED after {elapsed/60:.1f} min (exit code {result.returncode})')
        return False


def main():
    parser = argparse.ArgumentParser(description='Run ATC backtest pipeline.')
    parser.add_argument('--from', dest='from_nb', default=None,
                        help='Resume from notebook number (e.g. 03)')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only these notebooks (e.g. 04 07)')
    parser.add_argument('--skip-wrds', action='store_true',
                        help='Skip WRDS fetch check (assume cache exists)')
    args = parser.parse_args()

    print(f'ATC Backtest Pipeline')
    print(f'Project root : {PROJECT}')

    # ── Step 0: WRDS universe data ───────────────────────────────────────────
    # The parquets in data/universe/ are committed to the repo — no WRDS
    # credentials needed. This check is only a safety net.
    if not args.skip_wrds and not args.only:
        if not wrds_cache_exists():
            print('\nWARNING: data/universe/ parquets not found.')
            print('Run notebooks/fetch_wrds.ipynb interactively to fetch them.')
            print('Or clone the repo fresh — they are committed to git.\n')
            sys.exit(1)
        else:
            print('\n✓ WRDS universe cache found (committed to repo) — no credentials needed.')

    # ── Steps 1-8: Notebooks (non-interactive via nbconvert) ──────────────────
    to_run = NOTEBOOKS
    if args.only:
        to_run = [nb for nb in NOTEBOOKS if nb[0] in args.only]
    elif args.from_nb:
        to_run = [nb for nb in NOTEBOOKS if nb[0] >= args.from_nb]

    print(f'\nRunning {len(to_run)} notebook(s)...')

    for nb_id, nb_name, timeout, label in to_run:
        nb_path = NB_DIR / nb_name
        if not nb_path.exists():
            print(f'  WARNING: {nb_name} not found, skipping')
            continue
        ok = run_notebook(nb_path, timeout, label)
        if not ok:
            print(f'\nPipeline stopped at {nb_name}.')
            print(f'Fix the error and resume with:  python run_all.py --from {nb_id}')
            sys.exit(1)

    print(f'\n{"="*60}')
    print('PIPELINE COMPLETE')
    print(f'Figures : {PROJECT}/figures/')
    print(f'Data    : {PROJECT}/data/')


if __name__ == '__main__':
    main()
