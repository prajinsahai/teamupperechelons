"""Pull tables from Supabase (PostgREST) into local CSVs — the app only ever reads CSVs.

    .venv/Scripts/python -m src.data.supabase_import claims policies treaties

Env: SUPABASE_URL, SUPABASE_KEY (the publishable/anon key is enough to read tables
that RLS exposes; it cannot list the schema, so table names must be given).
Output: data/imported/<table>.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd

IMPORT_DIR = Path("data/imported")
PAGE = 1000


def _client() -> tuple[httpx.Client, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set (see .env.example)")
    return httpx.Client(headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=60), url


def fetch_table(table: str, select: str = "*", max_rows: int = 200_000) -> pd.DataFrame:
    """Read a whole table with Range pagination. Raises with the PostgREST message on failure."""
    client, url = _client()
    rows: list[dict] = []
    with client:
        offset = 0
        while offset < max_rows:
            r = client.get(
                f"{url}/rest/v1/{table}",
                params={"select": select},
                headers={"Range": f"{offset}-{offset + PAGE - 1}", "Range-Unit": "items"},
            )
            if r.status_code == 404:
                raise LookupError(f"table '{table}' not found (or not exposed to this key)")
            if r.status_code >= 400:
                raise RuntimeError(f"{table}: HTTP {r.status_code} {r.text[:200]}")
            batch = r.json()
            rows.extend(batch)
            if len(batch) < PAGE:
                break
            offset += PAGE
    return pd.DataFrame(rows)


def import_tables(tables: list[str]) -> dict[str, Path]:
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for t in tables:
        df = fetch_table(t)
        path = IMPORT_DIR / f"{t}.csv"
        df.to_csv(path, index=False)
        out[t] = path
        print(f"{t}: {len(df):,} rows x {len(df.columns)} cols -> {path}")
    return out


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(".env")
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    import_tables(sys.argv[1:])
