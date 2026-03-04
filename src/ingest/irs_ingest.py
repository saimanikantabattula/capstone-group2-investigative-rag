import argparse
import re
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from lxml import etree


def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_words(text: str, chunk_size: int = 250, overlap: int = 40):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = max(0, j - overlap)
    return chunks


def safe_parse_xml(xml_path: Path):
    parser = etree.XMLParser(recover=True, huge_tree=True)
    return etree.parse(str(xml_path), parser)


def extract_ein(tree) -> str:
    try:
        vals = tree.xpath("//*[contains(translate(local-name(), 'ein', 'EIN'), 'EIN')]/text()")
    except Exception:
        vals = []
    for v in vals:
        digits = re.sub(r"\D", "", str(v))
        if len(digits) == 9:
            return digits
    return ""


def extract_org_name(tree) -> str:
    xpaths = [
        "//*[contains(local-name(), 'BusinessNameLine1')]/text()",
        "//*[contains(local-name(), 'NameLine1')]/text()",
        "//*[contains(local-name(), 'BusinessName')]/text()",
    ]
    for xp in xpaths:
        try:
            vals = tree.xpath(xp)
        except Exception:
            vals = []
        for v in vals:
            t = collapse_ws(str(v))
            if t:
                return t
    return ""


def extract_all_text(tree) -> str:
    try:
        text_nodes = tree.xpath("//text()")
    except Exception:
        text_nodes = []
    return collapse_ws(" ".join([str(t) for t in text_nodes]))


def resolve_batch_dir(unstructured_root: Path, batch_id: str) -> Path:
    d = unstructured_root / batch_id
    nested = d / batch_id
    if nested.exists():
        return nested
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unstructured", required=True, help="IRS unstructured folder (contains 2024_TEOS_XML_* etc.)")
    ap.add_argument("--manifest", required=True, help="CSV with OBJECT_ID and XML_BATCH_ID (5,000 rows)")
    ap.add_argument("--chroma_path", default="chroma_db", help="Local chroma folder")
    ap.add_argument("--collection", default="irs_filings_5000", help="Chroma collection name")
    ap.add_argument("--chunk_size", type=int, default=250, help="Words per chunk")
    ap.add_argument("--overlap", type=int, default=40, help="Word overlap")
    args = ap.parse_args()

    unstructured_root = Path(args.unstructured)

    df = pd.read_csv(args.manifest, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    if "OBJECT_ID" not in df.columns:
        raise SystemExit("Manifest must contain OBJECT_ID column")
    if "XML_BATCH_ID" not in df.columns:
        raise SystemExit("Manifest must contain XML_BATCH_ID column")

    df = df.dropna(subset=["OBJECT_ID", "XML_BATCH_ID"])
    df["OBJECT_ID"] = df["OBJECT_ID"].astype(str).str.strip()
    df["XML_BATCH_ID"] = df["XML_BATCH_ID"].astype(str).str.strip()

    missing = 0
    selected = []

    for batch_id, g in df.groupby("XML_BATCH_ID"):
        batch_dir = resolve_batch_dir(unstructured_root, batch_id)
        if not batch_dir.exists():
            missing += len(g)
            continue

        for obj in g["OBJECT_ID"].tolist():
            xml_name = f"{obj}_public.xml"
            p = batch_dir / xml_name
            if p.exists():
                selected.append((obj, p))
            else:
                hits = list(batch_dir.rglob(xml_name))
                if hits:
                    selected.append((obj, hits[0]))
                else:
                    missing += 1

    if not selected:
        print("No XMLs found for the manifest. Check paths and batch folders.")
        return

    print(f"Manifest rows: {len(df)}")
    print(f"Found XML files for: {len(selected)}")
    print(f"Missing XML files: {missing}")

    client = chromadb.PersistentClient(path=args.chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = client.get_or_create_collection(name=args.collection, embedding_function=ef)

    total_chunks = 0
    processed = 0

    for i, (object_id, xml_path) in enumerate(selected, start=1):
        try:
            tree = safe_parse_xml(xml_path)
            ein = extract_ein(tree)
            org = extract_org_name(tree)
            full_text = extract_all_text(tree)
        except Exception as e:
            print("Skip (parse error):", xml_path.name, "->", e)
            continue

        if len(full_text) < 200:
            continue

        chunks = chunk_words(full_text, chunk_size=args.chunk_size, overlap=args.overlap)
        docs, metas, ids = [], [], []

        for c_i, c in enumerate(chunks):
            doc_id = f"irs_{object_id}_{c_i}"
            docs.append(c)
            metas.append(
                {
                    "source": "IRS",
                    "object_id": object_id,
                    "ein": ein,
                    "org_name": org,
                    "file_name": xml_path.name,
                }
            )
            ids.append(doc_id)

        col.upsert(documents=docs, metadatas=metas, ids=ids)
        total_chunks += len(chunks)
        processed += 1

        if i % 100 == 0:
            print(f"Indexed {i}/{len(selected)} filings | processed={processed} | chunks={total_chunks}")

    print("DONE.")
    print("Filings processed:", processed)
    print("Total chunks indexed:", total_chunks)
    print("Chroma collection:", args.collection)
    print("Chroma path:", args.chroma_path)


if __name__ == "__main__":
    main()