import argparse
import chromadb
import re


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[▲▼■◆•]+", " ", s)  # remove common PDF symbols
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_snippet(text: str, max_chars: int = 260) -> str:
    t = clean_text(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(" ", 1)[0] + "..."


def print_section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_top_unique(results: dict, k_unique_files: int = 2):
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    seen = set()
    shown = 0

    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        file_name = meta.get("file_name") or ""
        path = meta.get("path") or ""
        key = (file_name or path or doc_id)

        if key in seen:
            continue
        seen.add(key)

        source = meta.get("source", "UNKNOWN")
        object_id = meta.get("object_id", "")
        ein = meta.get("ein", "")
        org = meta.get("org_name", "")

        shown += 1
        print(f"\nResult {shown}  (distance={dist:.4f})")
        if source == "IRS":
            print(f"- Source: IRS 990 XML")
            print(f"- Org: {org}" if org else "- Org: (unknown)")
            print(f"- EIN: {ein}" if ein else "- EIN: (unknown)")
            print(f"- OBJECT_ID: {object_id}" if object_id else "- OBJECT_ID: (unknown)")
            print(f"- File: {file_name}")
            print(f"- Citation: {file_name} (OBJECT_ID={object_id})")
        else:
            print(f"- Source: FEC Filing PDF")
            print(f"- File: {file_name}")
            if path:
                print(f"- Path: {path}")
            print(f"- Citation: {file_name}")

        print(f"- Evidence: {format_snippet(doc)}")

        if shown >= k_unique_files:
            break

    if shown == 0:
        print("\nNo results found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="Question/search query")
    ap.add_argument("--n", type=int, default=8, help="raw top-k per collection (before dedupe)")
    ap.add_argument("--show", type=int, default=2, help="how many UNIQUE files to show per dataset")
    ap.add_argument("--chroma_path", default="chroma_db")
    ap.add_argument("--irs_collection", default="irs_filings_5000")
    ap.add_argument("--fec_collection", default="fec_filings_clean")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_path)

    # IRS
    print_section("IRS RESULTS (990 XML)")
    try:
        irs = client.get_collection(args.irs_collection)
        irs_res = irs.query(query_texts=[args.q], n_results=args.n)
        print_top_unique(irs_res, k_unique_files=args.show)
    except Exception as e:
        print("IRS query error:", e)

    # FEC
    print_section("FEC RESULTS (Filing PDFs)")
    try:
        fec = client.get_collection(args.fec_collection)
        fec_res = fec.query(query_texts=[args.q], n_results=args.n)
        print_top_unique(fec_res, k_unique_files=args.show)
    except Exception as e:
        print("FEC query error:", e)


if __name__ == "__main__":
    main()