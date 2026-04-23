"""
main.py
=======
This is the main backend file for our Investigative RAG system.
It is built using FastAPI — a Python web framework that creates API endpoints.

What this file does:
- Receives questions from the frontend (React app)
- Sends those questions to our RAG system (hybrid.py)
- Returns the answer with citations back to the frontend

How to run locally:
    uvicorn src.api.main:app --port 8000
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────
# os: used to read environment variables like DB_PASS, ANTHROPIC_API_KEY
import os

# Literal: used to restrict dataset field to only "irs", "fec", or "both"
from typing import Literal

# logging: used to print logs with timestamps so we can monitor the system
import logging

# time: used to measure how long each request takes
import time

# FastAPI: the main web framework — creates our API
# HTTPException: used to return error messages with proper HTTP status codes
# Request: gives us access to incoming HTTP request details
from fastapi import FastAPI, HTTPException, Request

# CORSMiddleware: allows our React frontend (different port) to call this API
# Without this, the browser blocks cross-origin requests
from fastapi.middleware.cors import CORSMiddleware

# BaseModel, Field: used to define the structure of request and response data
# Field adds validation like min/max length
from pydantic import BaseModel, Field


# ── LOGGING SETUP ────────────────────────────────────────────────────────
# This sets up our logging system
# Every request will be logged with: timestamp | level | message
# Example log: 2024-01-15 10:30:45 | INFO | POST /query | 200 | 1.23s
logging.basicConfig(
    level=logging.INFO,                          # show INFO and above messages
    format="%(asctime)s | %(levelname)s | %(message)s",   # log format
    datefmt="%Y-%m-%d %H:%M:%S"                 # timestamp format
)
logger = logging.getLogger(__name__)             # create logger for this file


# ── LOAD RAG SYSTEM ──────────────────────────────────────────────────────
# We try to import our RAG system (hybrid_ask function from hybrid.py)
# If it fails (e.g. missing libraries on Render), we set RAG_AVAILABLE = False
# This way the API still starts up even if RAG has import errors
try:
    from src.rag.hybrid import hybrid_ask
    RAG_AVAILABLE = True    # RAG loaded successfully
except Exception as e:
    print(f"RAG not available: {e}")
    RAG_AVAILABLE = False   # RAG failed to load — API will return fallback message


# ── CREATE FASTAPI APP ────────────────────────────────────────────────────
# This creates our FastAPI application
# title, description, version show up in the auto-generated API docs at /docs
app = FastAPI(
    title="Investigative RAG API",
    description="Multi-agent RAG over IRS 990 and FEC filings",
    version="1.0.0",
)


# ── REQUEST LOGGING MIDDLEWARE ─────────────────────────────────────────────
# This middleware runs for EVERY request that comes in
# It records: which endpoint was called, what status code was returned, how long it took
# Example: POST /query | 200 | 1.23s
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()                                    # record start time
    response = await call_next(request)                    # process the request
    elapsed = round(time.time() - start, 3)               # calculate time taken
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {elapsed}s")
    return response


# ── CORS MIDDLEWARE ────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Our React frontend runs on a different domain (Vercel) than our backend (Render)
# Browsers block requests between different domains by default for security
# This middleware tells the browser: "it is okay, allow all origins to call this API"
# allow_origins=["*"] means any website can call our API
# allow_credentials=False because we do not use cookies or sessions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow requests from any domain
    allow_credentials=False,    # no cookies or auth headers needed
    allow_methods=["*"],        # allow GET, POST, PUT, DELETE etc.
    allow_headers=["*"],        # allow any request headers
)


# ── REQUEST/RESPONSE MODELS ────────────────────────────────────────────────
# These classes define the structure of data coming in and going out
# Pydantic automatically validates the data — if something is wrong it returns 422 error

class QueryRequest(BaseModel):
    """
    This is the structure of every question request from the frontend.
    Example JSON: {"question": "Which nonprofits raised the most?", "dataset": "irs", "top_k": 5}
    """
    question: str = Field(..., min_length=3, max_length=500)  # question must be 3-500 chars
    dataset: Literal["irs", "fec", "both"] = "both"           # which dataset to search
    top_k: int = Field(5, ge=1, le=20)                        # how many results to retrieve


class CitationOut(BaseModel):
    """
    This is the structure of each citation in the response.
    Every answer includes citations so users can verify where the data came from.
    """
    source: str       # "IRS" or "FEC"
    file_name: str    # which file or table the data came from
    org_name: str     # organization name
    ein: str          # IRS Employer Identification Number (unique ID for nonprofits)
    object_id: str    # unique document ID
    snippet: str      # short preview of the relevant data
    distance: float   # how similar this result is to the question (lower = more similar)


class QueryResponse(BaseModel):
    """
    This is the structure of every response we send back to the frontend.
    It includes the original question, the generated answer, citations, and sources used.
    """
    question: str               # the original question asked
    answer: str                 # the AI-generated answer with citations
    citations: list[CitationOut]  # list of source documents used
    sources_used: list[str]     # which databases were queried (PostgreSQL, Pinecone etc.)


# ── DEBUG ENDPOINTS ────────────────────────────────────────────────────────
# These endpoints are used for testing individual components during development
# They are not used by the frontend in production

@app.get("/test-search2")
def test_search2():
    """
    Tests the full embedding + Pinecone search pipeline directly.
    Step 1: Calls HuggingFace API to convert text into a vector
    Step 2: Queries Pinecone with that vector to find similar documents
    Useful for checking if HuggingFace and Pinecone are both working correctly.
    """
    import urllib.request, json
    try:
        # Step 1: Get embedding from HuggingFace API
        # We send text and get back a 384-dimensional vector (list of 384 numbers)
        url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        hf_token = os.getenv("HF_TOKEN", "")
        data = json.dumps({"inputs": "Gates Foundation mission"}).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {hf_token}"
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            vector = json.loads(resp.read())

        # Step 2: Search Pinecone using the vector we got above
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index = pc.Index(os.getenv("PINECONE_INDEX", "investigative-rag"))
        results = index.query(vector=vector, top_k=3, namespace="irs", include_metadata=True)

        return {
            "status": "ok",
            "embedding_dim": len(vector),           # should be 384
            "pinecone_results": len(results.matches),
            "first_match": results.matches[0].metadata.get("org_name") if results.matches else None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/test-search")
def test_search():
    """
    Tests the retrieve() function from answer.py directly.
    This is a simpler test than test-search2 — it uses our own helper function.
    """
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
    """
    Tests if Pinecone connection is working and shows index statistics.
    Returns total vector count and namespace names.
    Expected output: ~100,835 total vectors, namespaces: ["irs", "fec"]
    """
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index = pc.Index(os.getenv("PINECONE_INDEX", "investigative-rag"))
        stats = index.describe_index_stats()
        return {
            "status": "ok",
            "total_vectors": stats.total_vector_count,
            "namespaces": list(stats.namespaces.keys())
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/test-embedding")
def test_embedding():
    """
    Tests if HuggingFace embedding API is reachable and working.
    Sends a simple test text and checks if we get back a vector.
    Expected output: embedding_dim = 384 (the model produces 384-dimensional vectors)
    """
    import urllib.request, json
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


# ── DASHBOARD ENDPOINT ─────────────────────────────────────────────────────
@app.get("/dashboard")
def get_dashboard_data():
    """
    Returns aggregated statistics for analytics dashboards.
    Queries all 4 tables and returns summary data for charts:
    - Top 10 states by nonprofit count
    - Top 10 nonprofits by revenue
    - IRS filing type distribution (990, 990EZ, 990PF, 990T)
    - Top 10 FEC committees by total receipts
    - Summary stats (total orgs, total revenue, total FEC committees, total vectors)
    """
    import psycopg2
    try:
        # Connect to our PostgreSQL database using environment variables
        # On Render: these are set in the environment variables dashboard
        # Locally: these are set when starting the server
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            dbname=os.getenv("DB_NAME", "capstone_rag"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASS", "")
        )
        cur = conn.cursor()

        # Query 1: Top 10 states by number of nonprofits
        # Uses irs_locations table which has location data for all 1.2M organizations
        cur.execute("""
            SELECT state, COUNT(*) as count FROM irs_locations
            WHERE state IS NOT NULL AND state != ''
            GROUP BY state ORDER BY count DESC LIMIT 10
        """)
        top_states = [{"state": r[0], "count": r[1]} for r in cur.fetchall()]

        # Query 2: Top 10 nonprofits by total revenue
        # Uses irs_financials table which has financial data for 378,272 organizations
        cur.execute("""
            SELECT org_name, state, total_revenue FROM irs_financials
            WHERE total_revenue IS NOT NULL
            ORDER BY total_revenue DESC LIMIT 10
        """)
        top_revenue = [{"org": r[0], "state": r[1], "revenue": float(r[2])} for r in cur.fetchall()]

        # Query 3: Count of each IRS filing type (990, 990EZ, 990PF, 990T)
        # 990 = standard nonprofit, 990EZ = small nonprofit, 990PF = private foundation
        cur.execute("""
            SELECT return_type, COUNT(*) as count FROM irs_index
            WHERE return_type IS NOT NULL
            GROUP BY return_type ORDER BY count DESC
        """)
        return_types = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]

        # Query 4: Top 10 FEC committees by total receipts (money raised)
        # Note: FEC column names are in ALL CAPS because that is how FEC publishes their data
        # TTL_RECEIPTS is stored as text so we cast to FLOAT for sorting
        cur.execute("""
            SELECT "CMTE_NM", "TTL_RECEIPTS", "TTL_DISB" FROM fec_committees
            WHERE "TTL_RECEIPTS" IS NOT NULL AND "TTL_RECEIPTS" != ''
            ORDER BY CAST("TTL_RECEIPTS" AS FLOAT) DESC LIMIT 10
        """)
        top_fec = [{"name": r[0], "receipts": float(r[1] or 0), "disbursements": float(r[2] or 0)} for r in cur.fetchall()]

        # Query 5: Overall summary stats
        # Total organizations and total revenue tracked across all IRS records
        cur.execute("SELECT COUNT(*), SUM(total_revenue) FROM irs_financials WHERE total_revenue IS NOT NULL")
        row = cur.fetchone()

        # Total number of FEC committees in our database
        cur.execute("SELECT COUNT(*) FROM fec_committees")
        total_fec = cur.fetchone()[0]

        conn.close()  # always close the database connection when done

        return {
            "top_states": top_states,
            "top_revenue": top_revenue,
            "return_types": return_types,
            "top_fec": top_fec,
            "summary": {
                "total_orgs": row[0],
                "total_revenue": float(row[1] or 0),
                "total_fec": total_fec,
                "total_vectors": 100835   # hardcoded — this is our Pinecone vector count
            }
        }
    except Exception as e:
        return {"error": str(e)}   # return error message if anything goes wrong


# ── SUGGESTIONS ENDPOINT ───────────────────────────────────────────────────
@app.post("/suggestions")
def get_suggestions(body: QueryRequest):
    """
    Returns 4 contextually relevant follow-up questions based on what was just asked.
    This powers the Related Questions feature in the frontend.

    Logic:
    - If user asked about California nonprofits → suggest other states (NY, TX, FL)
    - If user asked about ActBlue → suggest WinRed, Harris Victory Fund, DNC
    - If user asked about revenue → suggest assets, officer pay, expenses
    - If user asked about a year (2023) → suggest other years (2021, 2022, 2024)
    - Default → suggest hospital, foundation, cross-dataset questions

    This is intentionally different from the Try hints shown on the homepage
    so users always see fresh, relevant suggestions after each answer.
    """
    q = body.question.lower()  # convert to lowercase for easier matching

    # ── State-based suggestions ──
    # If user asked about nonprofits in a specific state,
    # suggest other similar states they might want to explore next
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
            # return suggestions for this state, but exclude the current question itself
            return {"suggestions": [s for s in sugg if s.lower() != q]}

    # ── Specific FEC committee suggestions ──
    # If user asked about a specific political committee,
    # suggest related committees from the same political ecosystem
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

    # ── City-based suggestions ──
    # If user asked about nonprofits in a specific city,
    # suggest other nearby cities or the same state
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

    # ── Revenue questions → suggest other financial metrics ──
    # If user asked about revenue/money raised, suggest other ways to measure financial health
    if any(w in q for w in ["raised", "revenue", "most money"]) and "based in" not in q and "located in" not in q:
        return {"suggestions": [
            "Which nonprofits have the most total assets?",
            "Which nonprofits pay their officers the most?",
            "Which nonprofits reported the most expenses?",
            "Which nonprofits have the highest net assets?",
        ]}

    # ── Asset questions → suggest related financial metrics ──
    if "asset" in q:
        return {"suggestions": [
            "Which nonprofits raised the most money?",
            "Which nonprofits have the highest liabilities?",
            "Which nonprofits pay officers the most?",
            "Which foundations have the most net assets?",
        ]}

    # ── Healthcare questions → suggest related healthcare orgs ──
    if any(w in q for w in ["hospital", "health system", "medical center"]):
        return {"suggestions": [
            "Which hospitals have the most assets?",
            "Which health systems have the highest revenue?",
            "Which medical centers spent the most money?",
            "Which nonprofit hospitals are the largest?",
        ]}

    # ── University/college questions → suggest related education orgs ──
    if any(w in q for w in ["universit", "college"]):
        return {"suggestions": [
            "Which universities have the most assets?",
            "Which colleges raised the most money?",
            "Which universities have the most debt?",
            "Which research institutes have the most assets?",
        ]}

    # ── General FEC/PAC questions → suggest specific committees ──
    if any(w in q for w in ["pac", "committee", "fec", "spent", "political"]):
        return {"suggestions": [
            "How much did ActBlue raise in 2024?",
            "Which committees have the most cash on hand?",
            "Which Democratic committees raised the most?",
            "Which Republican committees raised the most?",
        ]}

    # ── Private foundation / 990PF questions ──
    if "990pf" in q or "private foundation" in q:
        return {"suggestions": [
            "Which foundations have the most net assets?",
            "Which foundations raised the most in contributions?",
            "Which foundations spent the most money?",
            "Which nonprofits filed 990EZ returns?",
        ]}

    # ── Cross-dataset questions → suggest related investigation angles ──
    if any(w in q for w in ["connection", "both", "political"]):
        return {"suggestions": [
            "Which nonprofits have the most total assets?",
            "Which PACs spent the most in 2024?",
            "Which health organizations have the most revenue?",
            "Which foundations have the most assets?",
        ]}

    # ── Officer compensation questions ──
    if any(w in q for w in ["officer", "compensation", "executive", "salary"]):
        return {"suggestions": [
            "Which nonprofits raised the most money?",
            "Which hospitals have the most assets?",
            "Which universities have the most assets?",
            "Which nonprofits have the highest revenue?",
        ]}

    # ── Year-based questions → suggest same question for other years ──
    # Example: user asked about 2023 → suggest 2021, 2022, 2024
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

    # ── Default suggestions ──
    # If no category matched, return suggestions that are different
    # from the homepage Try hints so user always sees something new
    return {"suggestions": [
        "Which hospitals have the most assets?",
        "Which foundations have the most net assets?",
        "Which nonprofits have connections to political committees?",
        "Which nonprofits pay their officers the most?",
    ]}


# ── HEALTH CHECK ENDPOINT ──────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    """
    Simple health check endpoint.
    Returns {"status": "ok"} if the server is running.

    We accept both GET and HEAD methods because:
    - GET: normal browser/curl check
    - HEAD: UptimeRobot monitoring service uses HEAD to keep Render awake
    UptimeRobot pings this endpoint every 5 minutes to prevent Render from sleeping.
    """
    return {"status": "ok"}


# ── COLLECTIONS ENDPOINT ───────────────────────────────────────────────────
@app.get("/collections")
def list_collections():
    """
    Lists all ChromaDB collections (local vector database).
    Note: We migrated from ChromaDB to Pinecone for cloud deployment.
    This endpoint still works locally but is not used in production.
    """
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


# ── MAIN QUERY ENDPOINT ────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    This is the MAIN endpoint — the one the frontend calls when user asks a question.

    Flow:
    1. Validate the question (Pydantic does this automatically)
    2. Log the question for monitoring
    3. Call hybrid_ask() from hybrid.py — this does all the smart routing:
       - Financial questions → PostgreSQL SQL queries
       - Document questions → Pinecone vector search
       - Geographic questions → irs_locations table
       - Cross-dataset → SQL JOIN between IRS and FEC
    4. Format the citations into CitationOut objects
    5. Return the complete QueryResponse to the frontend

    Example request:
        POST /query
        {"question": "Which nonprofits raised the most money?", "dataset": "both", "top_k": 5}

    Example response:
        {"question": "...", "answer": "...", "citations": [...], "sources_used": [...]}
    """
    # Check that the question is not just empty spaces
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Log the incoming question so we can monitor what users are asking
    logger.info(f"Query: '{req.question[:60]}' | dataset={req.dataset}")

    # If RAG failed to load (e.g. missing libraries), return a fallback message
    if not RAG_AVAILABLE:
        return {"answer": "RAG system not available in this deployment.", "citations": [], "sources_used": []}

    # Call the main RAG function — this does all the work
    # hybrid_ask() in hybrid.py handles routing, retrieval, and answer generation
    result = hybrid_ask(
        question=req.question,
        dataset=req.dataset,
        top_k=req.top_k,
    )

    # Convert Citation objects from hybrid.py into CitationOut format for the API response
    # This is necessary because Pydantic needs a specific structure to serialize to JSON
    citations_out = [
        CitationOut(
            source=c.source,        # "IRS" or "FEC"
            file_name=c.file_name,  # which table or file the data came from
            org_name=c.org_name,    # organization name
            ein=c.ein,              # IRS tax ID number
            object_id=c.object_id,  # unique document ID
            snippet=c.snippet,      # short preview of the data
            distance=c.distance,    # similarity score
        )
        for c in result.citations
    ]

    # Return the final response to the frontend
    return QueryResponse(
        question=req.question,
        answer=result.answer,
        citations=citations_out,
        sources_used=result.sources_used,
    )
