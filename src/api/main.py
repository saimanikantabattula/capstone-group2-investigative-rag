"""
main.py

FastAPI backend for the Investigative RAG system.
Exposes endpoints for querying IRS 990 and FEC filing data
using hybrid retrieval — PostgreSQL for financial queries,
ChromaDB for document search.
"""

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from src.rag.hybrid import hybrid_ask
    RAG_AVAILABLE = True
except Exception as e:
    print(f"RAG not available: {e}")
    RAG_AVAILABLE = False


app = FastAPI(
    title="Investigative RAG API",
    description="Multi-agent RAG over IRS 990 and FEC filings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    dataset: Literal["irs", "fec", "both"] = "both"
    top_k: int = Field(5, ge=1, le=20)


class CitationOut(BaseModel):
    source: str
    file_name: str
    org_name: str
    ein: str
    object_id: str
    snippet: str
    distance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]
    sources_used: list[str]


@app.get("/test-pinecone")
def test_pinecone():
    """Test Pinecone connection."""
    import os
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index = pc.Index(os.getenv("PINECONE_INDEX", "investigative-rag"))
        stats = index.describe_index_stats()
        return {"status": "ok", "total_vectors": stats.total_vector_count, "namespaces": list(stats.namespaces.keys())}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test-embedding")
def test_embedding():
    """Test HuggingFace embedding API."""
    import urllib.request, json, os
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    hf_token = os.getenv("HF_TOKEN", "")
    data = json.dumps({"inputs": "test query"}).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {hf_token}"
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            return {"status": "ok", "embedding_dim": len(result), "token_set": bool(hf_token)}
    except Exception as e:
        return {"status": "error", "error": str(e), "token_set": bool(hf_token)}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/collections")
def list_collections():
    try:
        import chromadb
        chroma_path = os.getenv(
            "CHROMA_PATH",
            "/Users/battulasaimanikanta/Documents/capstone-group2-investigative-rag/chroma_db"
        )
        client = chromadb.PersistentClient(path=chroma_path)
        cols = [{"name": c.name, "count": c.count()} for c in client.list_collections()]
        return {"collections": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not RAG_AVAILABLE:
        return {"answer": "RAG system not available in this deployment.", "citations": [], "sources_used": []}
    result = hybrid_ask(
        question=req.question,
        dataset=req.dataset,
        top_k=req.top_k,
    )

    citations_out = [
        CitationOut(
            source=c.source,
            file_name=c.file_name,
            org_name=c.org_name,
            ein=c.ein,
            object_id=c.object_id,
            snippet=c.snippet,
            distance=c.distance,
        )
        for c in result.citations
    ]

    return QueryResponse(
        question=req.question,
        answer=result.answer,
        citations=citations_out,
        sources_used=result.sources_used,
    )
