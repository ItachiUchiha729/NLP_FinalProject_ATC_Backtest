"""
run_all.py — One-command reproduction of the ATC backtest pipeline.

Usage:
    python run_all.py                        # run all notebooks
    python run_all.py --from 03              # resume from notebook 03
    python run_all.py --only 04 07           # run specific notebooks only
    ATC_PROJECT_ROOT=/path/to/project python run_all.py

Notebooks are executed in order using jupyter nbconvert.
Large data files (signals.parquet, prices/) are regenerated if absent.
"""
import subprocess, sys, argparse, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NB_DIR  = PROJECT / 'notebooks'

NOTEBOOKS = [
    ('01', '01_data_pipeline.ipynb',       600,  'CSV → Parquet cache'),
    ('02', '02_universe_and_prices.ipynb', 7200, 'Universe PIT + yfinance prices + fwd returns'),
    ('03', '03_feature_engineering.ipynb', 1800, 'Feature engineering (MWNS, QoQ, sector rank)'),
    ('04', '04_backtest_baseline.ipynb',   1800, 'Baseline IC + quintile backtest + placebo'),
    ('05', '05_walkforward_model.ipynb',   3600, 'Walk-forward LightGBM'),
    ('06', '06_portfolio_robustness.ipynb',1800, 'Portfolio simulation + robustness'),
    ('07', '07_horizon_sensitivity.ipynb', 1800, 'Horizon sensitivity (5d / 10d / 20d)'),
    ('08', '08_regime_aware_model.ipynb',  3600, 'Regime-aware XGBoost-DART'),
]

PYTHON = sys.executable


def run_notebook(nb_file: Path, timeout: int, label: str) -> bool:
    print(f'\n{"="*60}')
    print(f'Running: {nb_file.name}  ({label})')
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
    parser = argparse.ArgumentParser(description='Run ATC backtest pipeline notebooks.')
    parser.add_argument('--from', dest='from_nb', default=None,
                        help='Resume from this notebook number (e.g. 03)')
    parser.add_argument('--only', nargs='+', default=None,
                        help='Run only these notebook numbers (e.g. 04 07)')
    args = parser.parse_args()

    to_run = NOTEBOOKS
    if args.only:
        to_run = [nb for nb in NOTEBOOKS if nb[0] in args.only]
    elif args.from_nb:
        to_run = [nb for nb in NOTEBOOKS if nb[0] >= args.from_nb]

    print(f'ATC Backtest Pipeline — {len(to_run)} notebook(s) to run')
    print(f'Project root: {PROJECT}')

    failed = []
    for nb_id, nb_name, timeout, label in to_run:
        nb_path = NB_DIR / nb_name
        if not nb_path.exists():
            print(f'  WARNING: {nb_name} not found, skipping')
            continue
        ok = run_notebook(nb_path, timeout, label)
        if not ok:
            failed.append(nb_name)
            print(f'  Stopping pipeline due to failure in {nb_name}')
            break

    print(f'\n{"="*60}')
    if failed:
        print(f'PIPELINE FAILED at: {failed}')
        sys.exit(1)
    else:
        print('PIPELINE COMPLETE — all notebooks executed successfully.')
        print(f'Figures: {PROJECT}/figures/')
        print(f'Data   : {PROJECT}/data/')


if __name__ == '__main__':
    main()
