import argparse
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF


def collapse_ws(s: str) -> str:
    s = re.sub(r"[▲▼■◆•]+", " ", s)   # remove common PDF symbols
    s = re.sub(r"\s+", " ", s)
    return s.strip()
def remove_fec_boilerplate(text: str) -> str:
    bad_phrases = [
        "Statements may not be sold or used",
        "for the purpose of soliciting contributions",
        "commercial purposes",
    ]
    lines = text.splitlines()
    keep = []
    for line in lines:
        if any(p.lower() in line.lower() for p in bad_phrases):
            continue
        keep.append(line)
    return collapse_ws("\n".join(keep))



def chunk_words(text: str, chunk_size: int = 220, overlap: int = 30):
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


def pdf_to_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return remove_fec_boilerplate(collapse_ws("\n".join(parts)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="FEC unstructured PDF folder (repeatable)")
    ap.add_argument("--chroma_path", default="chroma_db", help="Local chroma folder")
    ap.add_argument("--collection", default="fec_filings", help="Chroma collection name")
    ap.add_argument("--max_files", type=int, default=60, help="Limit PDFs for first run")
    ap.add_argument("--chunk_size", type=int, default=220, help="Words per chunk")
    ap.add_argument("--overlap", type=int, default=30, help="Word overlap")
    args = ap.parse_args()

    pdf_files = []
    for folder in args.input:
        pdf_files.extend(sorted(Path(folder).rglob("*.pdf")))

    if not pdf_files:
        print("No PDFs found in the provided folders.")
        return

    pdf_files = pdf_files[: args.max_files]
    print(f"Found {len(pdf_files)} PDFs (using max_files={args.max_files}).")

    client = chromadb.PersistentClient(path=args.chroma_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = client.get_or_create_collection(name=args.collection, embedding_function=ef)

    total_chunks = 0
    processed = 0

    for idx, pdf_path in enumerate(pdf_files, start=1):
        try:
            text = pdf_to_text(pdf_path)
        except Exception as e:
            print("Skip (PDF error):", pdf_path.name, "->", e)
            continue

        if len(text) < 50:
            continue

        chunks = chunk_words(text, chunk_size=args.chunk_size, overlap=args.overlap)

        docs, metas, ids = [], [], []
        for c_i, c in enumerate(chunks):
            doc_id = f"fec_{pdf_path.stem}_{c_i}"
            docs.append(c)
            metas.append(
                {
                    "source": "FEC",
                    "file_name": pdf_path.name,
                    "path": str(pdf_path),
                }
            )
            ids.append(doc_id)

        col.upsert(documents=docs, metadatas=metas, ids=ids)
        total_chunks += len(chunks)
        processed += 1

        if idx % 10 == 0:
            print(f"Indexed {idx}/{len(pdf_files)} PDFs | processed={processed} | chunks={total_chunks}")

    print("DONE.")
    print("PDFs processed:", processed)
    print("Total chunks indexed:", total_chunks)
    print("Chroma collection:", args.collection)
    print("Chroma path:", args.chroma_path)


if __name__ == "__main__":
    main()