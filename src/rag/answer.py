"""
answer.py

RAG engine using Pinecone for vector search.
Uses all-MiniLM-L6-v2 for embeddings (same model used during ingestion).
Implements Reciprocal Rank Fusion (RRF) to combine IRS and FEC results.
"""

import os
from dataclasses import dataclass, field
from typing import List

import anthropic as _llm_client

# Pinecone
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# ChromaDB (fallback)
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "investigative-rag")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
IRS_COLLECTION = os.getenv("IRS_COLLECTION", "irs_filings_25k")
FEC_COLLECTION = os.getenv("FEC_COLLECTION", "fec_filings")
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60

_embed_model = None
_pinecone_index = None


def get_embed_model():
    global _embed_model
    if _embed_model is None and EMBEDDINGS_AVAILABLE:
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            print(f"Could not load embedding model: {e}")
    return _embed_model

def get_embedding_via_api(text):
    """Get embedding via HuggingFace API - no local model needed."""
    import urllib.request, json
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    data = json.dumps({"inputs": text, "options": {"wait_for_model": True}}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if isinstance(result, list) and isinstance(result[0], list):
                return result[0]
            return result
    except Exception as e:
        print(f"HuggingFace API error: {e}")
        return None


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None and PINECONE_AVAILABLE and PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


@dataclass
class Citation:
    source: str
    file_name: str
    org_name: str
    ein: str
    object_id: str
    snippet: str
    distance: float


@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)


def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())[:800]


def search_pinecone(query, namespace, k=TOP_K):
    """Search Pinecone index for similar chunks."""
    try:
        model = get_embed_model()
        index = get_pinecone_index()
        if model is None or index is None:
            return []

        if model:
            query_vector = model.encode(query).tolist()
        else:
            query_vector = get_embedding_via_api(query)
            if not query_vector:
                return []
        results = index.query(
            vector=query_vector,
            top_k=k,
            namespace=namespace,
            include_metadata=True
        )

        citations = []
        seen = set()
        for match in results.matches:
            meta = match.metadata or {}
            file_name = meta.get("file_name", match.id)
            if file_name in seen:
                continue
            seen.add(file_name)
            citations.append(Citation(
                source=meta.get("source", namespace.upper()),
                file_name=file_name,
                org_name=meta.get("org_name", ""),
                ein=meta.get("ein", ""),
                object_id=meta.get("object_id", ""),
                snippet=clean_text(meta.get("text", "")),
                distance=round(1 - match.score, 4),
            ))
        return citations
    except Exception as e:
        print(f"Pinecone search error ({namespace}): {e}")
        return []


def search_chromadb(collection_name, query, k=TOP_K):
    """Fallback to ChromaDB if Pinecone not available."""
    if not CHROMADB_AVAILABLE:
        return []
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col = client.get_collection(name=collection_name, embedding_function=ef)
        results = col.query(query_texts=[query], n_results=k)

        citations = []
        seen = set()
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

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
    except Exception as e:
        print(f"ChromaDB search error ({collection_name}): {e}")
        return []


def retrieve(collection_name, query, k=TOP_K):
    """Retrieve from Pinecone (primary) or ChromaDB (fallback)."""
    namespace = "irs" if "irs" in collection_name.lower() else "fec"

    if PINECONE_AVAILABLE and PINECONE_API_KEY:
        results = search_pinecone(query, namespace, k)
        if results:
            return results

    return search_chromadb(collection_name, query, k)


def reciprocal_rank_fusion(irs_citations, fec_citations, k=RRF_K):
    """Combine IRS and FEC results using RRF."""
    scores = {}
    all_citations = {}

    for rank, citation in enumerate(irs_citations, start=1):
        key = citation.file_name
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        all_citations[key] = citation

    for rank, citation in enumerate(fec_citations, start=1):
        key = citation.file_name
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        all_citations[key] = citation

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [all_citations[k] for k in ranked_keys]


def build_context(citations):
    parts = []
    for i, c in enumerate(citations, start=1):
        if c.source == "IRS":
            header = f"[{i}] IRS 990 | Org: {c.org_name or 'Unknown'} | EIN: {c.ein or 'N/A'}"
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
        "Never fabricate numbers or organization names.\n"
        "End every answer with a Sources section."
    )
    user_message = (
        f"Question: {query}\n\n"
        f"--- SOURCE DOCUMENTS ---\n{context}\n\n"
        f"--- CITATION REFERENCES ---\n{citation_list}\n\n"
        "Answer using only the source documents above."
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

    irs_citations = []
    fec_citations = []

    if dataset in ("irs", "both"):
        irs_citations = retrieve(IRS_COLLECTION, query, k=top_k)
        if irs_citations:
            sources_used.append("IRS 990 (Pinecone)" if PINECONE_AVAILABLE and PINECONE_API_KEY else "IRS 990")

    if dataset in ("fec", "both"):
        fec_citations = retrieve(FEC_COLLECTION, query, k=top_k)
        if fec_citations:
            sources_used.append("FEC Filings (Pinecone)" if PINECONE_AVAILABLE and PINECONE_API_KEY else "FEC Filings")

    if irs_citations and fec_citations:
        all_citations = reciprocal_rank_fusion(irs_citations, fec_citations)
    else:
        all_citations = irs_citations + fec_citations

    if not all_citations:
        return RAGResponse(
            answer="No relevant documents found. Try rephrasing your question.",
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
        citations=all_citations[:10],
        sources_used=sources_used,
    )
