"""
Direct WRDS fetch using psycopg2 (reads ~/.pgpass automatically — no interactive input).
Bypasses the wrds library entirely to avoid the immutabledict/params bug.
"""
import sys, os
from pathlib import Path
import pandas as pd
import psycopg2

PROJECT   = Path(os.getenv('ATC_PROJECT_ROOT', Path(__file__).resolve().parent)).resolve()
UNIV_DIR  = PROJECT / 'data' / 'universe'
UNIV_DIR.mkdir(parents=True, exist_ok=True)

SP_CACHE   = UNIV_DIR / 'sp_constituents_wrds.parquet'
RU3K_CACHE = UNIV_DIR / 'ru3k_constituents_crsp.parquet'
LINK_CACHE = UNIV_DIR / 'crsp_compustat_link.parquet'

if SP_CACHE.exists() and RU3K_CACHE.exists():
    print("Cache already exists — nothing to fetch.")
    sys.exit(0)

# Try both usernames from pgpass; psycopg2 reads the password automatically
for user in ('chaithanyapakala', 'arshdeep1111'):
    try:
        print(f"Connecting as '{user}' (password from ~/.pgpass)...")
        conn = psycopg2.connect(
            host='wrds-pgdata.wharton.upenn.edu',
            port=9737,
            dbname='wrds',
            user=user,
            sslmode='require',
            connect_timeout=20,
        )
        print(f"Connected as '{user}'.")
        break
    except Exception as e:
        print(f"  Failed: {e}")
        conn = None

if conn is None:
    print("\nCould not connect. Make sure you are on university VPN.")
    sys.exit(1)


def sql(query, date_cols=None):
    df = pd.read_sql_query(query, conn)
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
    return df


# ── S&P 500 / 400 / 600 ──────────────────────────────────────────────────────
print("\nFetching S&P index membership (comp.idxcst_his)...")
sp_hist = sql("""
    SELECT
        i.gvkey::bigint  AS gvkey,
        CASE
            WHEN co.conm ILIKE '%midcap 400%'   THEN 'SP400'
            WHEN co.conm ILIKE '%smallcap 600%' THEN 'SP600'
            WHEN co.conm ILIKE '%500%'           THEN 'SP500'
        END              AS idx_type,
        i."from"::date   AS start_dt,
        COALESCE(i.thru::date, '2035-12-31'::date) AS end_dt
    FROM comp.idxcst_his i
    JOIN comp.company co
      ON co.gvkey = i.gvkeyx
    WHERE co.conm ILIKE '%%s&p%%'
      AND (
            co.conm ILIKE '%%500%%'
         OR co.conm ILIKE '%%midcap 400%%'
         OR co.conm ILIKE '%%smallcap 600%%'
      )
    ORDER BY i.gvkey, start_dt
""", date_cols=['start_dt', 'end_dt'])

sp_hist = sp_hist[sp_hist['idx_type'].notna()].copy()
sp_hist.to_parquet(SP_CACHE, index=False)
print(f"  Saved {len(sp_hist):,} rows → {SP_CACHE.name}")
print(sp_hist['idx_type'].value_counts().to_string())

# ── CRSP monthly market cap ───────────────────────────────────────────────────
print("\nFetching CRSP monthly market cap (crsp.msf) — takes ~3 min...")
crsp_me = sql("""
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
""", date_cols=['date'])
print(f"  {len(crsp_me):,} rows fetched")

# ── PERMNO → GVKEY link ───────────────────────────────────────────────────────
print("\nFetching PERMNO → GVKEY link (crsp.ccmxpf_lnkhist)...")
crsp_link = sql("""
    SELECT
        lpermno::bigint AS permno,
        gvkey::bigint   AS gvkey,
        linkdt::date                                  AS link_start,
        COALESCE(linkenddt::date, '2035-12-31'::date) AS link_end
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LC', 'LU', 'LS')
      AND linkprim IN ('P', 'J', 'C')
    ORDER BY permno, link_start
""", date_cols=['link_start', 'link_end'])
crsp_link.to_parquet(LINK_CACHE, index=False)
print(f"  Saved {len(crsp_link):,} rows → {LINK_CACHE.name}")

conn.close()
print("\nWRDS connection closed.")

# ── Russell 3000 PIT ─────────────────────────────────────────────────────────
print("\nBuilding Russell 3000 PIT membership...")

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
        print(f"  Warning: no data near {recon.date()}, skipping")
        continue

    top3k = set(month_me.nlargest(3000, 'me')['permno'])
    link_at = crsp_link[
        (crsp_link['link_start'] <= recon) & (crsp_link['link_end'] >= recon)
    ][['permno', 'gvkey']].drop_duplicates('permno')
    mapped = link_at[link_at['permno'].isin(top3k)]

    end_date = (recon_dates[i + 1] - pd.Timedelta(days=1)
                if i + 1 < len(recon_dates) else pd.Timestamp('2035-12-31'))

    for _, row in mapped.iterrows():
        ru3k_records.append({'gvkey': int(row['gvkey']),
                             'start_dt': recon, 'end_dt': end_date})

    print(f"  {recon.date()}: {len(mapped):,} GVKEYs from {len(top3k):,} PERMNOs")

ru3k_hist = pd.DataFrame(ru3k_records)
ru3k_hist.to_parquet(RU3K_CACHE, index=False)
print(f"\nSaved {len(ru3k_hist):,} rows → {RU3K_CACHE.name}")
print(f"Unique GVKEYs ever in Russell 3000: {ru3k_hist['gvkey'].nunique():,}")
print("\n✓ All WRDS data cached. Run: python run_all.py")
