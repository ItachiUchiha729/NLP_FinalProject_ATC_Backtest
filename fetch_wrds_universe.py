"""
fetch_wrds_universe.py — One-time WRDS data fetch for universe membership.

Run this ONCE from a terminal before run_all.py:
    python fetch_wrds_universe.py

What it does:
  1. Connects to WRDS (prompts for username/password on first run;
     credentials are saved to ~/.pgpass automatically by the wrds library
     so subsequent runs are non-interactive).
  2. Fetches S&P 500 / 400 / 600 PIT membership from comp.idxcst_his (GVKEY-based).
  3. Fetches CRSP monthly market cap + PERMNO→GVKEY link for Russell 3000.
  4. Saves all three to data/universe/*.parquet (cached — never re-fetched).

After this script completes, run_all.py / nbconvert can run NB02 fully
non-interactively because NB02's WRDS cell loads from the cached parquets.
"""
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT   = Path(os.getenv("ATC_PROJECT_ROOT",
                            Path(__file__).resolve().parent)).resolve()
UNIV_DIR  = PROJECT / 'data' / 'universe'
UNIV_DIR.mkdir(parents=True, exist_ok=True)

SP_CACHE   = UNIV_DIR / 'sp_constituents_wrds.parquet'
RU3K_CACHE = UNIV_DIR / 'ru3k_constituents_crsp.parquet'
LINK_CACHE = UNIV_DIR / 'crsp_compustat_link.parquet'


def run():
    if SP_CACHE.exists() and RU3K_CACHE.exists():
        print("Cache files already exist — nothing to fetch.")
        print(f"  {SP_CACHE}")
        print(f"  {RU3K_CACHE}")
        print("Delete them to force a re-fetch.")
        return

    try:
        import wrds
    except ImportError:
        print("ERROR: wrds package not installed. Run: pip install wrds")
        sys.exit(1)

    # Show native macOS dialogs to collect WRDS credentials
    # (appears as a popup at the top of the screen — type and press OK)
    def _ask(prompt: str, hidden: bool = False) -> str:
        hidden_str = 'with hidden answer' if hidden else ''
        script = (
            f'set val to text returned of '
            f'(display dialog "{prompt}" default answer "" {hidden_str} '
            f'buttons {{"Cancel","OK"}} default button "OK" '
            f'with title "WRDS Login")'
            f'\nreturn val'
        )
        result = subprocess.run(
            ['osascript', '-e', script],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print('Dialog cancelled.')
            sys.exit(0)
        return result.stdout.strip()

    import subprocess
    print("A login dialog will appear — enter your WRDS credentials there.")
    _wrds_user = _ask('WRDS Username:')
    _wrds_pass = _ask('WRDS Password:', hidden=True)

    # Bypass wrds.Connection() (which falls back to interactive on failure).
    # Connect directly via SQLAlchemy — same driver, full error visibility.
    import sqlalchemy as sa, urllib.parse

    print(f"Connecting to WRDS as '{_wrds_user}'...")
    _enc_pass = urllib.parse.quote(_wrds_pass, safe='')
    _uri = f"postgresql://{_wrds_user}:{_enc_pass}@wrds-pgdata.wharton.upenn.edu:9737/wrds"
    try:
        engine = sa.create_engine(_uri, isolation_level='AUTOCOMMIT',
                                  connect_args={'connect_timeout': 20,
                                                'sslmode': 'require'})
        conn = engine.connect()
        print("Connected successfully.")
    except Exception as _e:
        print(f"\nConnection failed: {_e}")
        print("\nWRDS port 9737 is blocked from this network.")
        print("Fix: connect to your university VPN, then rerun this script.")
        sys.exit(1)

    def raw_sql(sql, date_cols=None):
        import pandas as _pd
        df = _pd.read_sql(sql, conn)
        if date_cols:
            for c in date_cols:
                if c in df.columns:
                    df[c] = _pd.to_datetime(df[c])
        return df

    # ── S&P 500 / 400 / 600 — Compustat PIT intervals ────────────────────────
    print("\nFetching comp.idxcst_his (S&P 500 / 400 / 600)...")
    sp_sql = """
        SELECT
            gvkey::bigint          AS gvkey,
            UPPER(indextype)       AS idx_type,
            "from"::date           AS start_dt,
            COALESCE(thru::date, '2035-12-31'::date) AS end_dt
        FROM comp.idxcst_his
        WHERE UPPER(indextype) IN ('SP500', 'SP400', 'SP600')
        ORDER BY gvkey, idx_type, start_dt
    """
    sp_hist = raw_sql(sp_sql, date_cols=['start_dt', 'end_dt'])
    sp_hist.to_parquet(SP_CACHE, index=False)
    print(f"  Saved {len(sp_hist):,} rows → {SP_CACHE.name}")
    for idx in ['SP500', 'SP400', 'SP600']:
        n = sp_hist[sp_hist['idx_type'] == idx]['gvkey'].nunique()
        yrs = (sp_hist[sp_hist['idx_type'] == idx]['start_dt'].min().year,
               sp_hist[sp_hist['idx_type'] == idx]['end_dt'].max().year)
        print(f"    {idx}: {n:,} unique GVKEYs, {yrs[0]}–{yrs[1]}")

    # ── CRSP monthly market cap for Russell 3000 ──────────────────────────────
    print("\nFetching crsp.msf (monthly market cap, eligible US stocks)...")
    me_sql = """
        SELECT msf.permno,
               msf.date,
               ABS(msf.prc) * msf.shrout AS me
        FROM crsp.msf AS msf
        JOIN crsp.msenames AS names
          ON msf.permno = names.permno
         AND msf.date BETWEEN names.namedt AND names.nameendt
        WHERE msf.date >= '2009-01-01'
          AND names.shrcd  IN (10, 11)
          AND names.exchcd IN (1, 2, 3)
          AND ABS(msf.prc) > 1.0
          AND msf.shrout  > 0
        ORDER BY msf.date, msf.permno
    """
    crsp_me = raw_sql(me_sql, date_cols=['date'])
    print(f"  {len(crsp_me):,} rows fetched")

    # ── CRSP–Compustat link (PERMNO → GVKEY) ─────────────────────────────────
    print("\nFetching crsp.ccmxpf_lnkhist (PERMNO → GVKEY mapping)...")
    link_sql = """
        SELECT
            lpermno::bigint AS permno,
            gvkey::bigint   AS gvkey,
            linkdt::date                                  AS link_start,
            COALESCE(linkenddt::date, '2035-12-31'::date) AS link_end
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LC', 'LU', 'LS')
          AND linkprim IN ('P', 'J', 'C')
        ORDER BY permno, link_start
    """
    crsp_link = raw_sql(link_sql, date_cols=['link_start', 'link_end'])
    crsp_link.to_parquet(LINK_CACHE, index=False)
    print(f"  Saved {len(crsp_link):,} rows → {LINK_CACHE.name}")

    conn.close()
    print("\nWRDS connection closed.")

    # ── Build Russell 3000 PIT membership ────────────────────────────────────
    print("\nBuilding Russell 3000 PIT membership from CRSP market cap...")
    print("  (Top 3000 eligible US stocks at each June annual reconstitution)")

    def last_friday_june(year):
        d = pd.Timestamp(f'{year}-06-30')
        return d - pd.Timedelta(days=(d.weekday() - 4) % 7)

    recon_dates = [last_friday_june(y) for y in range(2009, pd.Timestamp.now().year + 2)]
    crsp_me['month'] = crsp_me['date'].dt.to_period('M')

    ru3k_records = []
    for i, recon in enumerate(recon_dates):
        recon_month = recon.to_period('M')
        month_me    = crsp_me[crsp_me['month'] == recon_month]
        if month_me.empty:
            month_me = crsp_me[crsp_me['month'] == (recon_month - 1)]
        if month_me.empty:
            print(f"    Warning: no CRSP data near {recon.date()}, skipping")
            continue

        top3k_permnos = set(month_me.nlargest(3000, 'me')['permno'])

        link_at_recon = crsp_link[
            (crsp_link['link_start'] <= recon) & (crsp_link['link_end'] >= recon)
        ][['permno', 'gvkey']].drop_duplicates('permno')
        mapped = link_at_recon[link_at_recon['permno'].isin(top3k_permnos)]

        end_date = (recon_dates[i + 1] - pd.Timedelta(days=1)
                    if i + 1 < len(recon_dates) else pd.Timestamp('2035-12-31'))

        for _, row in mapped.iterrows():
            ru3k_records.append({
                'gvkey':    int(row['gvkey']),
                'start_dt': recon,
                'end_dt':   end_date,
            })

        print(f"    {recon.date()}: {len(mapped):,} GVKEYs mapped from {len(top3k_permnos):,} PERMNOs")

    ru3k_hist = pd.DataFrame(ru3k_records)
    ru3k_hist.to_parquet(RU3K_CACHE, index=False)
    print(f"\n  Saved {len(ru3k_hist):,} rows → {RU3K_CACHE.name}")
    print(f"  Unique GVKEYs ever in Russell 3000: {ru3k_hist['gvkey'].nunique():,}")

    print("\n✓ All WRDS universe data fetched and cached.")
    print("  You can now run: python run_all.py")


if __name__ == '__main__':
    run()
