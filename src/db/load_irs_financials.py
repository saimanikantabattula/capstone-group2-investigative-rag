"""
load_irs_financials.py

Reads the 25k IRS XML files from the manifest, extracts key financial
figures from each filing, and loads them into a new PostgreSQL table
called irs_financials.

Fields extracted:
- Total revenue
- Total expenses
- Total assets (end of year)
- Total liabilities (end of year)
- Net assets
- Contributions and grants
- Program service revenue
- Officer compensation
- State, tax year, return type, org name, EIN
"""

import os
import re
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from lxml import etree


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "capstone_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")

MANIFEST_PATH = "data/manifests/irs_manifest_new.csv"
XML_BASE = "/Users/battulasaimanikanta/Documents/capstone data sets /dataset 1.0/unstructured"

# XML namespaces used in IRS 990 filings
NS = {
    "irs": "http://www.irs.gov/efile",
}


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )


def create_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS irs_financials (
            ein TEXT,
            org_name TEXT,
            return_type TEXT,
            tax_year TEXT,
            tax_period TEXT,
            state TEXT,
            total_revenue NUMERIC,
            total_expenses NUMERIC,
            total_assets NUMERIC,
            total_liabilities NUMERIC,
            net_assets NUMERIC,
            contributions_grants NUMERIC,
            program_service_revenue NUMERIC,
            officer_compensation NUMERIC,
            object_id TEXT PRIMARY KEY,
            xml_file TEXT
        );
    """)
    conn.commit()
    cur.close()
    print("Table irs_financials ready.")


def safe_numeric(value):
    if value is None:
        return None
    try:
        cleaned = re.sub(r"[^\d.-]", "", str(value))
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def find_text(root, *xpaths):
    for xpath in xpaths:
        try:
            results = root.xpath(xpath, namespaces=NS)
            if results:
                val = results[0]
                if hasattr(val, "text"):
                    return val.text
                return str(val).strip()
        except Exception:
            continue
    return None


def parse_xml(xml_path):
    try:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        return None

    # common paths for different 990 return types
    total_revenue = safe_numeric(find_text(root,
        ".//irs:TotalRevenueAmt",
        ".//irs:TotalRevenue",
        ".//irs:CYTotalRevenueAmt",
    ))

    total_expenses = safe_numeric(find_text(root,
        ".//irs:TotalExpensesAmt",
        ".//irs:TotalExpenses",
        ".//irs:CYTotalExpensesAmt",
    ))

    total_assets = safe_numeric(find_text(root,
        ".//irs:TotalAssetsEOYAmt",
        ".//irs:TotalAssets",
        ".//irs:BookValueAssetsEOYAmt",
    ))

    total_liabilities = safe_numeric(find_text(root,
        ".//irs:TotalLiabilitiesEOYAmt",
        ".//irs:TotalLiabilities",
    ))

    net_assets = safe_numeric(find_text(root,
        ".//irs:NetAssetsOrFundBalancesEOYAmt",
        ".//irs:NetAssets",
        ".//irs:TotNetAstOrFundBalancesEOYAmt",
    ))

    contributions = safe_numeric(find_text(root,
        ".//irs:CYContributionsGrantsAmt",
        ".//irs:ContributionsGrantsAmt",
        ".//irs:TotalContributionsAmt",
    ))

    program_revenue = safe_numeric(find_text(root,
        ".//irs:CYProgramServiceRevenueAmt",
        ".//irs:ProgramServiceRevenueAmt",
    ))

    officer_comp = safe_numeric(find_text(root,
        ".//irs:OfficerDirectorTrusteeKeyEmplTotCompAmt",
        ".//irs:CompCurrentOfcrDirectorsAmt",
        ".//irs:TotalCompensation",
    ))

    state = find_text(root,
        ".//irs:USAddress/irs:StateAbbreviationCd",
        ".//irs:StateAbbreviationCd",
    )

    org_name = find_text(root,
        ".//irs:BusinessName/irs:BusinessNameLine1Txt",
        ".//irs:OrganizationName",
    )

    return {
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "net_assets": net_assets,
        "contributions_grants": contributions,
        "program_service_revenue": program_revenue,
        "officer_compensation": officer_comp,
        "state": state,
        "org_name_xml": org_name,
    }


def find_xml_file(xml_base, object_id, xml_batch_id):
    # files are stored as XML_BATCH_ID/OBJECT_ID_public.xml
    if object_id and xml_batch_id:
        # Try double-nested path first (XML_BATCH_ID/XML_BATCH_ID/OBJECT_ID_public.xml)
        path = Path(xml_base) / xml_batch_id / xml_batch_id / f"{object_id}_public.xml"
        if path.exists():
            return path
        path = Path(xml_base) / xml_batch_id / f"{object_id}_public.xml"
        if path.exists():
            return path

    # fallback: search batch folder for object_id
    if xml_batch_id and object_id:
        batch_dir = Path(xml_base) / xml_batch_id
        if batch_dir.exists():
            candidate = batch_dir / f"{object_id}_public.xml"
            if candidate.exists():
                return candidate

    return None


def main():
    print("Loading manifest...")
    df = pd.read_csv(MANIFEST_PATH, dtype=str, low_memory=False)
    df = df.fillna("")
    print(f"Manifest rows: {len(df)}")

    conn = get_connection()
    create_table(conn)

    # check existing rows
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM irs_financials")
    existing = cur.fetchone()[0]
    cur.close()

    if existing > 0:
        print(f"Found {existing} existing rows, clearing table for fresh load...")
        cur = conn.cursor()
        # cur.execute("TRUNCATE TABLE irs_financials")  # disabled to preserve existing data
        conn.commit()
        cur.close()

    rows = []
    processed = 0
    skipped = 0
    batch_size = 500

    print(f"Processing {len(df)} filings...")

    for i, row in df.iterrows():
        object_id = row.get("OBJECT_ID", "").strip()
        xml_batch_id = row.get("XML_BATCH_ID", "").strip()
        ein = row.get("EIN", "").strip()
        org_name = row.get("TAXPAYER_NAME", "").strip()
        return_type = row.get("RETURN_TYPE", "").strip()
        tax_period = row.get("TAX_PERIOD", "").strip()
        sub_date = row.get("SUB_DATE", "").strip()

        tax_year = sub_date[:4] if sub_date else ""

        xml_path = find_xml_file(XML_BASE, object_id, xml_batch_id)
        if not xml_path:
            skipped += 1
            continue

        financials = parse_xml(xml_path)
        if not financials:
            skipped += 1
            continue

        rows.append((
            ein,
            financials.get("org_name_xml") or org_name,
            return_type,
            tax_year,
            tax_period,
            financials.get("state"),
            financials.get("total_revenue"),
            financials.get("total_expenses"),
            financials.get("total_assets"),
            financials.get("total_liabilities"),
            financials.get("net_assets"),
            financials.get("contributions_grants"),
            financials.get("program_service_revenue"),
            financials.get("officer_compensation"),
            object_id,
            str(xml_path),
        ))

        processed += 1

        if len(rows) >= batch_size:
            insert_batch(conn, rows)
            rows = []
            print(f"Progress: {processed}/{len(df)} processed, {skipped} skipped")

    if rows:
        insert_batch(conn, rows)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM irs_financials")
    total = cur.fetchone()[0]
    cur.close()

    print(f"\nDone.")
    print(f"Processed: {processed}")
    print(f"Skipped (no XML found): {skipped}")
    print(f"Total rows in irs_financials: {total}")

    conn.close()


def insert_batch(conn, rows):
    cur = conn.cursor()
    insert_sql = """
        INSERT INTO irs_financials (
            ein, org_name, return_type, tax_year, tax_period, state,
            total_revenue, total_expenses, total_assets, total_liabilities,
            net_assets, contributions_grants, program_service_revenue,
            officer_compensation, object_id, xml_file
        ) VALUES %s
        ON CONFLICT (object_id) DO NOTHING
    """
    execute_values(cur, insert_sql, rows, page_size=500)
    conn.commit()
    cur.close()


if __name__ == "__main__":
    main()
