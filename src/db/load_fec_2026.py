"""
load_fec_2026.py

Loads the FEC 2026 committee summary CSV into the existing
fec_committees table in PostgreSQL. Adds a cycle column to
distinguish 2024 vs 2026 data.
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "capstone_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")

FEC_2026_CSV = "/Users/battulasaimanikanta/Documents/capstone data sets /dataset 2/2025 /committee_summary_2026.csv"


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def load_2026(conn):
    print("Loading FEC 2026 CSV...")
    df = pd.read_csv(FEC_2026_CSV, dtype=str, low_memory=False)
    # preserve original column casing to match the table
    df.columns = [c.strip() for c in df.columns]
    df = df.fillna("")
    df["cycle"] = "2026"

    print(f"Loaded {len(df)} rows from CSV")

    cur = conn.cursor()

    # add cycle column if it does not exist yet
    cur.execute("""
        ALTER TABLE fec_committees
        ADD COLUMN IF NOT EXISTS cycle text DEFAULT '2024';
    """)
    conn.commit()

    # delete existing 2026 rows before reload
    cur.execute("SELECT COUNT(*) FROM fec_committees WHERE cycle = '2026'")
    existing = cur.fetchone()[0]
    if existing > 0:
        print(f"Found {existing} existing 2026 rows, deleting before reload")
        cur.execute("DELETE FROM fec_committees WHERE cycle = '2026'")
        conn.commit()

    # get current columns in the table with original casing
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'fec_committees'
        ORDER BY ordinal_position;
    """)
    table_cols = [row[0] for row in cur.fetchall()]
    print(f"Table has {len(table_cols)} columns")

    # only keep CSV columns that exist in the table
    csv_cols = [c for c in df.columns if c in table_cols]
    missing = [c for c in df.columns if c not in table_cols]
    if missing:
        print(f"Skipping {len(missing)} CSV columns not in table")

    print(f"Inserting {len(csv_cols)} columns")
    df_insert = df[csv_cols]

    rows = [tuple(row) for row in df_insert.itertuples(index=False)]

    # quote column names to preserve case in PostgreSQL
    quoted_cols = ", ".join(f'"{c}"' for c in csv_cols)
    insert_sql = f'INSERT INTO fec_committees ({quoted_cols}) VALUES %s ON CONFLICT DO NOTHING'

    execute_values(cur, insert_sql, rows, page_size=1000)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM fec_committees WHERE cycle = '2026'")
    inserted = cur.fetchone()[0]
    print(f"Inserted {inserted} rows for cycle 2026")

    cur.execute("SELECT COUNT(*) FROM fec_committees")
    total = cur.fetchone()[0]
    print(f"Total rows in fec_committees: {total}")

    cur.close()


def main():
    conn = get_connection()
    try:
        load_2026(conn)
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
