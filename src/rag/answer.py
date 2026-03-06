"""
answer.py

Retrieves relevant chunks from ChromaDB for a given query,
builds a context block, and calls the language model API to generate
a cited answer. Returns structured output with answer text
and source citations.
"""

import os
import re
from dataclasses import dataclass, field

import anthropic as _llm_client
import chromadb
from chromadb.utils import embedding_functions


CHROMA_PATH = os.getenv("CHROMA_PATH", "/Users/battulasaimanikanta/Documents/capstone-group2-investigative-rag/chroma_db")
IRS_COLLECTION = os.getenv("IRS_COLLECTION", "irs_filings_25k")
FEC_COLLECTION = os.getenv("FEC_COLLECTION", "fec_filings")
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class Citation:
    source: str
    file_name: str
    org_name: str = ""
    ein: str = ""
    object_id: str = ""
    snippet: str = ""
    distance: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def clean_text(text, max_chars=300):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def retrieve(collection_name, query, k=TOP_K):
    try:
        client = get_chroma_client()
        ef = get_embedding_function()
        col = client.get_collection(name=collection_name, embedding_function=ef)
        results = col.query(query_texts=[query], n_results=k)
    except Exception as e:
        print(f"Could not query collection {collection_name}: {e}")
        return []

    citations = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    seen = set()
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        file_name = meta.get("file_name", doc_id)
        if file_name in seen:
            continue
        seen.add(file_name)
        citations.append(Citation(
            source=meta.get("source", "UNKNOWN"),
            file_name=file_name,
            org_name=meta.get("org_name", ""),
            ein=meta.get("ein", ""),
            object_id=meta.get("object_id", ""),
            snippet=clean_text(doc),
            distance=round(dist, 4),
        ))

    return citations


def build_context(citations):
    parts = []
    for i, c in enumerate(citations, start=1):
        if c.source == "IRS":
            header = f"[{i}] IRS 990 | Org: {c.org_name or 'Unknown'} | EIN: {c.ein or 'N/A'} | File: {c.file_name}"
        else:
            header = f"[{i}] FEC Filing | File: {c.file_name}"
        parts.append(f"{header}\n{c.snippet}")
    return "\n\n".join(parts)


def build_citation_list(citations):
    lines = []
    for i, c in enumerate(citations, start=1):
        if c.source == "IRS":
            lines.append(f"[{i}] {c.file_name} — {c.org_name} (EIN: {c.ein})")
        else:
            lines.append(f"[{i}] {c.file_name}")
    return "\n".join(lines)


def generate_answer(context, citation_list, query):
    system_prompt = (
        "You are an investigative intelligence analyst specializing in nonprofit finance "
        "(IRS 990 filings) and political finance (FEC filings).\n\n"
        "Answer questions using only the provided source documents.\n"
        "Use inline citations like [1], [2] when referencing sources.\n"
        "Be concise, factual, and professional.\n"
        "If the documents do not contain enough information, say so clearly.\n"
        "Never fabricate numbers or organization names.\n"
        "End every answer with a Sources section."
    )

    user_message = (
        f"Question: {query}\n\n"
        f"--- SOURCE DOCUMENTS ---\n{context}\n\n"
        f"--- CITATION REFERENCES ---\n{citation_list}\n\n"
        "Answer using only the source documents above. Use inline citations."
    )

    api = _llm_client.Anthropic(api_key=LLM_API_KEY)
    response = api.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def ask(query, dataset="both", top_k=TOP_K):
    all_citations = []
    sources_used = []

    if dataset in ("irs", "both"):
        irs = retrieve(IRS_COLLECTION, query, k=top_k)
        all_citations.extend(irs)
        if irs:
            sources_used.append("IRS 990")

    if dataset in ("fec", "both"):
        fec = retrieve(FEC_COLLECTION, query, k=top_k)
        all_citations.extend(fec)
        if fec:
            sources_used.append("FEC Filings")

    if not all_citations:
        return RAGResponse(
            answer="No relevant documents found. Try rephrasing your question or selecting a different dataset.",
            citations=[],
            sources_used=[],
        )

    context = build_context(all_citations)
    citation_list = build_citation_list(all_citations)

    try:
        answer_text = generate_answer(context, citation_list, query)
    except Exception as e:
        answer_text = f"Could not generate answer: {e}"

    return RAGResponse(
        answer=answer_text,
        citations=all_citations,
        sources_used=sources_used,
    )
