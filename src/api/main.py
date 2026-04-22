"""
main.py

FastAPI backend for the Investigative RAG system.
Exposes endpoints for querying IRS 990 and FEC filing data
using hybrid retrieval — PostgreSQL for financial queries,
ChromaDB for document search.
"""

import os
from typing import Literal

import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - start, 3)
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {elapsed}s")
    return response

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


@app.get("/test-search2")
def test_search2():
    """Debug vector search."""
    import os, urllib.request, json
    try:
        # Test embedding
        url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        hf_token = os.getenv("HF_TOKEN", "")
        data = json.dumps({"inputs": "Gates Foundation mission"}).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {hf_token}"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            vector = json.loads(resp.read())
        
        # Test Pinecone query with this vector
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index = pc.Index(os.getenv("PINECONE_INDEX", "investigative-rag"))
        results = index.query(vector=vector, top_k=3, namespace="irs", include_metadata=True)
        
        return {
            "status": "ok",
            "embedding_dim": len(vector),
            "pinecone_results": len(results.matches),
            "first_match": results.matches[0].metadata.get("org_name") if results.matches else None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test-search")
def test_search():
    """Test full vector search pipeline."""
    try:
        from src.rag.answer import retrieve
        results = retrieve("irs_filings_25k", "Gates Foundation mission", k=3)
        return {
            "status": "ok",
            "results_count": len(results),
            "first_result": results[0].org_name if results else None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

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

@app.get("/dashboard")
def get_dashboard_data():
    """Returns aggregated data for dashboard charts."""
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME", "capstone_rag"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASS", "")
        )
        cur = conn.cursor()

        cur.execute("""
            SELECT state, COUNT(*) as count FROM irs_locations
            WHERE state IS NOT NULL AND state != ''
            GROUP BY state ORDER BY count DESC LIMIT 10
        """)
        top_states = [{"state": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute("""
            SELECT org_name, state, total_revenue FROM irs_financials
            WHERE total_revenue IS NOT NULL
            ORDER BY total_revenue DESC LIMIT 10
        """)
        top_revenue = [{"org": r[0], "state": r[1], "revenue": float(r[2])} for r in cur.fetchall()]

        cur.execute("""
            SELECT return_type, COUNT(*) as count FROM irs_index
            WHERE return_type IS NOT NULL
            GROUP BY return_type ORDER BY count DESC
        """)
        return_types = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute("""
            SELECT "CMTE_NM", "TTL_RECEIPTS", "TTL_DISB" FROM fec_committees
            WHERE "TTL_RECEIPTS" IS NOT NULL AND "TTL_RECEIPTS" != ''
            ORDER BY CAST("TTL_RECEIPTS" AS FLOAT) DESC LIMIT 10
        """)
        top_fec = [{"name": r[0], "receipts": float(r[1] or 0), "disbursements": float(r[2] or 0)} for r in cur.fetchall()]

        cur.execute("SELECT COUNT(*), SUM(total_revenue) FROM irs_financials WHERE total_revenue IS NOT NULL")
        row = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM fec_committees")
        total_fec = cur.fetchone()[0]

        conn.close()
        return {
            "top_states": top_states,
            "top_revenue": top_revenue,
            "return_types": return_types,
            "top_fec": top_fec,
            "summary": {
                "total_orgs": row[0],
                "total_revenue": float(row[1] or 0),
                "total_fec": total_fec,
                "total_vectors": 100835
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/suggestions")
def get_suggestions(body: QueryRequest):
    """Generate truly contextual related questions based on the question."""
    q = body.question.lower()
    suggestions = []

    # State-based — suggest other states
    STATE_SUGGESTIONS = {
        "california": ["Which nonprofits are based in New York?", "Which nonprofits are based in Texas?", "Which nonprofits are based in Florida?", "Which nonprofits are based in Illinois?"],
        "new york": ["Which nonprofits are based in California?", "Which nonprofits are based in Texas?", "Which nonprofits are based in New Jersey?", "Which nonprofits are based in Massachusetts?"],
        "texas": ["Which nonprofits are based in California?", "Which nonprofits are based in Florida?", "Which nonprofits are based in Georgia?", "Which nonprofits are based in North Carolina?"],
        "florida": ["Which nonprofits are based in Texas?", "Which nonprofits are based in Georgia?", "Which nonprofits are based in California?", "Which nonprofits are based in North Carolina?"],
        "illinois": ["Which nonprofits are based in Ohio?", "Which nonprofits are based in Michigan?", "Which nonprofits are based in Indiana?", "Which nonprofits are based in Wisconsin?"],
        "boston": ["Which nonprofits are based in Massachusetts?", "Which nonprofits are based in New York?", "Which nonprofits are based in Chicago?", "Which nonprofits are based in Seattle?"],
        "chicago": ["Which nonprofits are based in Boston?", "Which nonprofits are based in Illinois?", "Which nonprofits are based in Detroit?", "Which nonprofits are based in Milwaukee?"],
        "massachusetts": ["Which nonprofits are based in New York?", "Which nonprofits are based in Connecticut?", "Which nonprofits are based in Pennsylvania?", "Which nonprofits are based in Maryland?"],
    }
    for state, sugg in STATE_SUGGESTIONS.items():
        if state in q:
            return {"suggestions": [s for s in sugg if s.lower() != q]}

    # Specific committee — suggest related committees
    if "actblue" in q:
        return {"suggestions": ["How much did WinRed raise in 2024?", "What did Harris Victory Fund report in 2024?", "How much did the DNC raise in 2024?", "Which PACs spent the most in 2024?"]}
    if "winred" in q:
        return {"suggestions": ["How much did ActBlue raise in 2024?", "How much did the RNC raise in 2024?", "Which Republican committees raised the most?", "Which PACs spent the most in 2024?"]}
    if "harris" in q:
        return {"suggestions": ["How much did ActBlue raise in 2024?", "What did the DNC raise in 2024?", "Which PACs spent the most in 2024?", "Which Democratic committees raised the most?"]}
    if "dnc" in q or "democratic" in q:
        return {"suggestions": ["How much did the RNC raise in 2024?", "Which Democratic committees raised the most?", "How much did ActBlue raise in 2024?", "Which PACs spent the most in 2024?"]}
    if "rnc" in q or "republican" in q:
        return {"suggestions": ["How much did the DNC raise in 2024?", "Which Republican committees raised the most?", "How much did WinRed raise in 2024?", "Which PACs spent the most in 2024?"]}

    # City-based suggestions
    if "boston" in q:
        return {"suggestions": ["Which nonprofits are based in Massachusetts?", "Which nonprofits are based in New York?", "Which nonprofits are based in Chicago?", "Which nonprofits are based in Seattle?"]}
    if "chicago" in q:
        return {"suggestions": ["Which nonprofits are based in Boston?", "Which nonprofits are based in Illinois?", "Which nonprofits are based in Detroit?", "Which nonprofits are based in Milwaukee?"]}
    if "seattle" in q:
        return {"suggestions": ["Which nonprofits are based in Washington?", "Which nonprofits are based in Oregon?", "Which nonprofits are based in California?", "Which nonprofits are based in Boston?"]}
    if "atlanta" in q:
        return {"suggestions": ["Which nonprofits are based in Georgia?", "Which nonprofits are based in Florida?", "Which nonprofits are based in North Carolina?", "Which nonprofits are based in Tennessee?"]}
    if "los angeles" in q:
        return {"suggestions": ["Which nonprofits are based in California?", "Which nonprofits are based in New York?", "Which nonprofits are based in Chicago?", "Which nonprofits are based in Houston?"]}

    # Revenue questions — suggest other financial metrics
    if any(w in q for w in ["raised", "revenue", "most money"]) and "based in" not in q and "located in" not in q:
        return {"suggestions": [
            "Which nonprofits have the most total assets?",
            "Which nonprofits pay their officers the most?",
            "Which nonprofits reported the most expenses?",
            "Which nonprofits have the highest net assets?",
        ]}

    # Asset questions — suggest revenue and other metrics
    if "asset" in q:
        return {"suggestions": [
            "Which nonprofits raised the most money?",
            "Which nonprofits have the highest liabilities?",
            "Which nonprofits pay officers the most?",
            "Which foundations have the most net assets?",
        ]}

    # Hospital/health — suggest related health orgs
    if any(w in q for w in ["hospital", "health system", "medical center"]):
        return {"suggestions": [
            "Which hospitals have the most assets?",
            "Which health systems have the highest revenue?",
            "Which medical centers spent the most money?",
            "Which nonprofit hospitals are the largest?",
        ]}

    # University/college
    if any(w in q for w in ["universit", "college"]):
        return {"suggestions": [
            "Which universities have the most assets?",
            "Which colleges raised the most money?",
            "Which universities have the most debt?",
            "Which research institutes have the most assets?",
        ]}

    # PAC/FEC general
    if any(w in q for w in ["pac", "committee", "fec", "spent", "political"]):
        return {"suggestions": [
            "How much did ActBlue raise in 2024?",
            "Which committees have the most cash on hand?",
            "Which Democratic committees raised the most?",
            "Which Republican committees raised the most?",
        ]}

    # 990PF / filing type
    if "990pf" in q or "private foundation" in q:
        return {"suggestions": [
            "Which foundations have the most net assets?",
            "Which foundations raised the most in contributions?",
            "Which foundations spent the most money?",
            "Which nonprofits filed 990EZ returns?",
        ]}

    # Cross dataset
    if any(w in q for w in ["connection", "both", "political"]):
        return {"suggestions": [
            "Which nonprofits have the most total assets?",
            "Which PACs spent the most in 2024?",
            "Which health organizations have the most revenue?",
            "Which foundations have the most assets?",
        ]}

    # Officer compensation
    if any(w in q for w in ["officer", "compensation", "executive", "salary"]):
        return {"suggestions": [
            "Which nonprofits raised the most money?",
            "Which hospitals have the most assets?",
            "Which universities have the most assets?",
            "Which nonprofits have the highest revenue?",
        ]}

    # Year-based — suggest same question for other years
    import re
    years = re.findall(r'202[0-4]', q)
    if years:
        year = int(years[0])
        other_years = [y for y in [2021, 2022, 2023, 2024] if y != year]
        return {"suggestions": [
            f"Which nonprofits raised the most money in {other_years[0]}?",
            f"Which nonprofits raised the most money in {other_years[1]}?",
            "Which nonprofits raised the most money overall?",
            "Which nonprofits had the most assets in recent years?",
        ]}

    # Default — truly different from Try hints
    return {"suggestions": [
        "Which hospitals have the most assets?",
        "Which foundations have the most net assets?",
        "Which nonprofits have connections to political committees?",
        "Which nonprofits pay their officers the most?",
    ]}

@app.api_route("/health", methods=["GET", "HEAD"])
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

    logger.info(f"Query: '{req.question[:60]}' | dataset={req.dataset}")
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
