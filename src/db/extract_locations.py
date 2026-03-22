"""
extract_locations.py

Extracts state, city, zip, and address from all IRS 990 XML files
and loads them into a new irs_locations table in PostgreSQL.
Also updates irs_financials state column where missing.
"""

import os
import sys
import glob
import psycopg2
import psycopg2.extras
import lxml.etree as ET
from pathlib import Path

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "capstone_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")

XML_BASE = "/Users/battulasaimanikanta/Documents/capstone data sets /dataset 1.0/unstructured"
NS = {"irs": "http://www.irs.gov/efile"}


def get_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def get_text(root, path):
    el = root.find(path, NS)
    return el.text.strip() if el is not None and el.text else ""


def extract_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        name = get_text(root, ".//irs:BusinessName/irs:BusinessNameLine1Txt")
        ein_raw = get_text(root, ".//irs:EIN")
        ein = ein_raw.replace("-", "").strip()
        state = get_text(root, ".//irs:USAddress/irs:StateAbbreviationCd")
        city = get_text(root, ".//irs:USAddress/irs:CityNm")
        zip_code = get_text(root, ".//irs:USAddress/irs:ZIPCd")
        address = get_text(root, ".//irs:USAddress/irs:AddressLine1Txt")
        return_type = get_text(root, ".//irs:ReturnTypeCd")
        tax_year = get_text(root, ".//irs:TaxYr")
        object_id = Path(xml_path).stem.replace("_public", "")

        if not ein or not state:
            return None

        return {
            "ein": ein,
            "org_name": name,
            "state": state,
            "city": city,
            "zip_code": zip_code,
            "address": address,
            "return_type": return_type,
            "tax_year": tax_year,
            "object_id": object_id,
        }
    except Exception as e:
        return None


def create_table(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS irs_locations (
            ein TEXT,
            org_name TEXT,
            state TEXT,
            city TEXT,
            zip_code TEXT,
            address TEXT,
            return_type TEXT,
            tax_year TEXT,
            object_id TEXT PRIMARY KEY
        );
        CREATE INDEX IF NOT EXISTS idx_locations_state ON irs_locations(state);
        CREATE INDEX IF NOT EXISTS idx_locations_ein ON irs_locations(ein);
        CREATE INDEX IF NOT EXISTS idx_locations_city ON irs_locations(city);
        CREATE INDEX IF NOT EXISTS idx_locations_org ON irs_locations(LOWER(org_name));
    """)
    conn.commit()
    cur.close()
    print("Table irs_locations created/verified.")


def main():
    print("Finding XML files...")
    xml_files = glob.glob(f"{XML_BASE}/**/*.xml", recursive=True)
    print(f"Found {len(xml_files)} XML files")

    conn = get_db()
    create_table(conn)
    cur = conn.cursor()

    batch = []
    processed = 0
    skipped = 0
    total = len(xml_files)

    for i, xml_path in enumerate(xml_files, 1):
        record = extract_from_xml(xml_path)
        if record:
            batch.append(record)
            processed += 1
        else:
            skipped += 1

        # Batch insert every 500 records
        if len(batch) >= 500:
            psycopg2.extras.execute_values(cur, """
                INSERT INTO irs_locations
                    (ein, org_name, state, city, zip_code, address, return_type, tax_year, object_id)
                VALUES %s
                ON CONFLICT (object_id) DO UPDATE SET
                    state = EXCLUDED.state,
                    city = EXCLUDED.city,
                    zip_code = EXCLUDED.zip_code
            """, [(
                r["ein"], r["org_name"], r["state"], r["city"],
                r["zip_code"], r["address"], r["return_type"],
                r["tax_year"], r["object_id"]
            ) for r in batch])
            conn.commit()
            print(f"Progress: {i}/{total} | Inserted: {processed} | Skipped: {skipped}")
            batch = []

    # Final batch
    if batch:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO irs_locations
                (ein, org_name, state, city, zip_code, address, return_type, tax_year, object_id)
            VALUES %s
            ON CONFLICT (object_id) DO UPDATE SET
                state = EXCLUDED.state,
                city = EXCLUDED.city,
                zip_code = EXCLUDED.zip_code
        """, [(
            r["ein"], r["org_name"], r["state"], r["city"],
            r["zip_code"], r["address"], r["return_type"],
            r["tax_year"], r["object_id"]
        ) for r in batch])
        conn.commit()

    # Also update irs_financials state where missing
    print("\nUpdating irs_financials state from irs_locations...")
    cur.execute("""
        UPDATE irs_financials f
        SET state = l.state
        FROM irs_locations l
        WHERE f.ein = l.ein
          AND (f.state IS NULL OR f.state = '')
    """)
    updated = cur.rowcount
    conn.commit()
    print(f"Updated {updated} rows in irs_financials with state data")

    # Summary
    cur.execute("SELECT COUNT(*) FROM irs_locations")
    total_rows = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT state) FROM irs_locations WHERE state != ''")
    states = cur.fetchone()[0]
    cur.execute("""
        SELECT state, COUNT(*) as cnt
        FROM irs_locations
        WHERE state != ''
        GROUP BY state
        ORDER BY cnt DESC
        LIMIT 5
    """)
    top_states = cur.fetchall()

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"Total rows in irs_locations: {total_rows}")
    print(f"States covered: {states}")
    print(f"Top 5 states:")
    for state, cnt in top_states:
        print(f"  {state}: {cnt} organizations")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
