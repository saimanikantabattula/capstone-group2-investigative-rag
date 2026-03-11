"""
fec_csv_ingest.py

Ingests FEC committee summary CSV files into ChromaDB.
Converts each committee row into a readable text chunk
so the RAG can answer questions about FEC committees.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions


EMBED_MODEL = "all-MiniLM-L6-v2"


def format_money(val):
    try:
        v = float(val)
        if v >= 1_000_000_000:
            return f"${v/1_000_000_000:.2f} billion"
        elif v >= 1_000_000:
            return f"${v/1_000_000:.2f} million"
        elif v >= 1_000:
            return f"${v/1_000:.1f} thousand"
        else:
            return f"${v:.2f}"
    except:
        return str(val) if val else "N/A"


def row_to_text(row, cycle):
    """Convert a FEC committee CSV row into a readable text chunk."""
    name = row.get("CMTE_NM", "Unknown Committee")
    cmte_id = row.get("CMTE_ID", "")
    cmte_type = row.get("CMTE_TP", "")
    state = row.get("CMTE_ST", "")
    party = row.get("CAND_PTY_AFFILIATION", row.get("CMTE_PTY_AFFILIATION", ""))
    receipts = format_money(row.get("TTL_RECEIPTS", ""))
    disbursements = format_money(row.get("TTL_DISB", ""))
    cash = format_money(row.get("DEBTS_OWED_BY_CMTE", ""))
    indiv = format_money(row.get("TTL_INDIV_CONTRIB", ""))
    candidate = row.get("CAND_NAME", "")

    type_map = {
        "P": "Presidential Campaign Committee",
        "S": "Senate Campaign Committee",
        "H": "House Campaign Committee",
        "N": "Non-Party Independent Expenditure Committee (Super PAC)",
        "Q": "Non-Qualified Party Committee",
        "X": "Non-Qualified Non-Party Committee",
        "Y": "Non-Qualified Non-Party Committees",
        "Z": "Non-Qualified Non-Party Committees",
        "D": "Delegate Committee",
        "E": "Electioneering Communication",
        "I": "Independent Expenditure (Not a committee)",
        "O": "Super PAC (Independent Expenditure Only)",
        "U": "Single Candidate Independent Expenditure",
        "V": "Lobbyist/Registrant PAC",
        "W": "Lobbyist/Registrant PAC",
    }
    cmte_type_label = type_map.get(cmte_type, cmte_type)

    lines = [
        f"FEC Committee: {name}",
        f"Committee ID: {cmte_id}",
        f"Type: {cmte_type_label}",
        f"State: {state}",
        f"Election Cycle: {cycle}",
    ]

    if party:
        lines.append(f"Party Affiliation: {party}")
    if candidate:
        lines.append(f"Associated Candidate: {candidate}")

    lines.append(f"Total Receipts (Money Raised): {receipts}")
    lines.append(f"Total Disbursements (Money Spent): {disbursements}")

    if indiv and indiv != "N/A":
        lines.append(f"Individual Contributions: {indiv}")
    if cash and cash != "N/A":
        lines.append(f"Debts Owed: {cash}")

    return "\n".join(lines)


def ingest_csv(csv_path, col, cycle, chunk_count):
    """Ingest one CSV file into ChromaDB collection."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    print(f"  Loaded {len(df)} rows from {csv_path.name}")

    docs, metas, ids = [], [], []
    processed = 0

    for _, row in df.iterrows():
        cmte_id = row.get("CMTE_ID", "")
        name = row.get("CMTE_NM", "Unknown")

        if not cmte_id or not name:
            continue

        text = row_to_text(row, cycle)
        doc_id = f"fec_csv_{cycle}_{cmte_id}"

        docs.append(text)
        metas.append({
            "source": "FEC",
            "file_name": f"FEC_{cycle}_committee_{cmte_id}",
            "committee_id": cmte_id,
            "committee_name": name,
            "cycle": str(cycle),
            "state": row.get("CMTE_ST", ""),
            "committee_type": row.get("CMTE_TP", ""),
            "total_receipts": row.get("TTL_RECEIPTS", ""),
            "total_disbursements": row.get("TTL_DISB", ""),
        })
        ids.append(doc_id)
        processed += 1

        # Batch upsert every 500 rows
        if len(docs) >= 500:
            # Deduplicate within batch
            seen = {}
            for d, m, i in zip(docs, metas, ids):
                seen[i] = (d, m)
            u_ids = list(seen.keys())
            u_docs = [seen[i][0] for i in u_ids]
            u_metas = [seen[i][1] for i in u_ids]
            col.upsert(documents=u_docs, metadatas=u_metas, ids=u_ids)
            chunk_count += len(u_ids)
            print(f"    Upserted batch | total so far: {chunk_count}")
            docs, metas, ids = [], [], []

    # Final batch
    if docs:
        seen = {}
        for d, m, i in zip(docs, metas, ids):
            seen[i] = (d, m)
        u_ids = list(seen.keys())
        u_docs = [seen[i][0] for i in u_ids]
        u_metas = [seen[i][1] for i in u_ids]
        col.upsert(documents=u_docs, metadatas=u_metas, ids=u_ids)
        chunk_count += len(u_ids)

    print(f"  Done: {processed} committees indexed from {csv_path.name}")
    return chunk_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_2024", required=True, help="Path to committee_summary_2024.csv")
    ap.add_argument("--csv_2026", required=True, help="Path to committee_summary_2026.csv")
    ap.add_argument("--chroma_path", default="chroma_db", help="ChromaDB folder path")
    ap.add_argument("--collection", default="fec_filings", help="ChromaDB collection name")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = client.get_or_create_collection(name=args.collection, embedding_function=ef)

    print(f"Collection '{args.collection}' currently has {col.count()} chunks")
    print("Starting FEC CSV ingestion...\n")

    chunk_count = 0

    # Ingest 2024
    csv_2024 = Path(args.csv_2024)
    if csv_2024.exists():
        print(f"Processing 2024 CSV: {csv_2024}")
        chunk_count = ingest_csv(csv_2024, col, 2024, chunk_count)
    else:
        print(f"WARNING: 2024 CSV not found at {csv_2024}")

    # Ingest 2026
    csv_2026 = Path(args.csv_2026)
    if csv_2026.exists():
        print(f"\nProcessing 2026 CSV: {csv_2026}")
        chunk_count = ingest_csv(csv_2026, col, 2026, chunk_count)
    else:
        print(f"WARNING: 2026 CSV not found at {csv_2026}")

    print(f"\nDONE.")
    print(f"Total new chunks added: {chunk_count}")
    print(f"Collection '{args.collection}' now has {col.count()} total chunks")


if __name__ == "__main__":
    main()
