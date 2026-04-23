"""
answer.py
=========
This is the Pinecone RAG (Retrieval Augmented Generation) engine.

When hybrid.py cannot find an answer using SQL (PostgreSQL),
it falls back to this file which does semantic vector search using Pinecone.

What this file does:
1. Takes the user question
2. Converts it into a 384-dimensional vector using HuggingFace API
3. Searches Pinecone for the most similar document chunks
4. Applies Reciprocal Rank Fusion (RRF) to combine IRS and FEC results
5. Passes the best chunks to Claude to generate a cited answer

Two caches are implemented here for performance:
- Embedding cache: avoids calling HuggingFace API for repeated questions
- Answer cache: avoids running the full RAG pipeline for repeated questions
Result: 493x faster on repeated queries (4.93s → 0.0s)
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# os: read environment variables (API keys, collection names etc.)
import os

# dataclass, field: Python built-in for creating clean data container classes
# We use these for Citation and RAGResponse objects
from dataclasses import dataclass, field

# List: type hint for lists
from typing import List

# anthropic: official Anthropic Python library to call Claude API
import anthropic as _llm_client


# ── PINECONE IMPORT ───────────────────────────────────────────────────────────
# Try to import Pinecone — if not installed, fall back to ChromaDB
# On Render (cloud): Pinecone is available
# Locally without install: falls back to ChromaDB
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


# ── SENTENCE TRANSFORMERS IMPORT ─────────────────────────────────────────────
# SentenceTransformer: local embedding model (500MB, not used on Render)
# On Render we removed this library to reduce build size
# Instead we call HuggingFace API for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# ── CHROMADB IMPORT (FALLBACK) ────────────────────────────────────────────────
# ChromaDB: local vector database — used as fallback if Pinecone is not available
# We migrated from ChromaDB to Pinecone for cloud deployment
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# All configuration from environment variables so values are never hardcoded
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "investigative-rag")  # our index name in Pinecone
CHROMA_PATH      = os.getenv("CHROMA_PATH", "chroma_db")             # local ChromaDB path (fallback)
IRS_COLLECTION   = os.getenv("IRS_COLLECTION", "irs_filings_25k")    # IRS namespace/collection name
FEC_COLLECTION   = os.getenv("FEC_COLLECTION", "fec_filings")        # FEC namespace/collection name
LLM_API_KEY      = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL        = "claude-haiku-4-5-20251001"  # Claude Haiku = fast and affordable
TOP_K            = int(os.getenv("TOP_K", "5")) # how many results to retrieve per namespace
EMBED_MODEL      = "all-MiniLM-L6-v2"           # embedding model — produces 384-dimensional vectors
RRF_K            = 60                            # RRF constant — standard value used in research papers


# ── GLOBAL VARIABLES ──────────────────────────────────────────────────────────
# These are module-level variables that persist across function calls
# They are initialized as None and created on first use (lazy initialization)

_embed_model    = None   # local SentenceTransformer model (only used locally)
_pinecone_index = None   # Pinecone index connection (reused across requests)

# Cache for HuggingFace embeddings
# Key = question text, Value = 384-dimensional vector
# Benefit: same question → return cached vector instantly, no API call needed
_embedding_cache = {}

# Cache for complete RAG answers
# Key = "question|dataset", Value = RAGResponse object
# Benefit: same question → return cached answer instantly, no DB/LLM calls needed
_answer_cache = {}

MAX_CACHE_SIZE = 100  # maximum number of entries to keep in each cache


# ── DATA CLASSES ─────────────────────────────────────────────────────────────
@dataclass
class Citation:
    """
    Represents a single source document used in an answer.
    Each citation has metadata about where the data came from.

    source:    "IRS" or "FEC"
    file_name: original XML filename or FEC committee ID
    org_name:  organization name
    ein:       IRS Employer Identification Number (unique tax ID)
    object_id: unique document identifier
    snippet:   short text preview of the relevant passage
    distance:  similarity score (0 = identical, 1 = completely different)
    """
    source:    str
    file_name: str
    org_name:  str
    ein:       str
    object_id: str
    snippet:   str
    distance:  float


@dataclass
class RAGResponse:
    """
    The complete response object returned to main.py for every question.

    answer:       Claude's generated answer with [1][2] citation markers
    citations:    list of Citation objects (the sources used)
    sources_used: list of strings like ["IRS 990 (Pinecone)", "FEC Filings (Pinecone)"]
    """
    answer:       str
    citations:    List[Citation] = field(default_factory=list)
    sources_used: List[str]     = field(default_factory=list)


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def get_embed_model():
    """
    Returns the local SentenceTransformer model (lazy initialization).
    Only used when running locally with the full sentence-transformers library.
    On Render (cloud), this always returns None because the library is not installed.
    In that case, we fall back to get_embedding_via_api() instead.
    """
    global _embed_model
    if _embed_model is None and EMBEDDINGS_AVAILABLE:
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            print(f"Could not load embedding model: {e}")
    return _embed_model


def get_embedding_via_api(text):
    """
    Converts text into a 384-dimensional vector using HuggingFace Inference API.
    This is our primary embedding method on Render (cloud deployment).

    CACHING: If we already computed this embedding before, return it instantly
    from the in-memory cache instead of calling the API again.
    This gives 493x speedup on repeated queries.

    The model: sentence-transformers/all-MiniLM-L6-v2
    - Produces vectors of size 384
    - Fast and lightweight (good for our use case)
    - Same model used during data ingestion, so vectors are compatible

    HuggingFace API returns either:
    - Flat format:   [0.1, 0.2, 0.3, ...] (384 numbers)
    - Nested format: [[0.1, 0.2, 0.3, ...]] (list inside list)
    We handle both formats.
    """
    global _embedding_cache

    # Check cache first — if we already computed this embedding, return it instantly
    if text in _embedding_cache:
        return _embedding_cache[text]

    import urllib.request, json

    # HuggingFace API endpoint for our embedding model
    url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    data    = json.dumps({"inputs": text, "options": {"wait_for_model": True}}).encode()
    hf_token = os.getenv("HF_TOKEN", "")

    # Build the HTTP request with authorization header
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {hf_token}"
    })

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())

            # Handle both flat [0.1, ...] and nested [[0.1, ...]] response formats
            if isinstance(result, list):
                if isinstance(result[0], list):
                    vector = result[0]   # nested format — take inner list
                else:
                    vector = result      # flat format — use directly

                # Save to cache for future calls (only if cache is not full)
                if len(_embedding_cache) < MAX_CACHE_SIZE:
                    _embedding_cache[text] = vector
                return vector

            return None  # unexpected format

    except Exception as e:
        print(f"HuggingFace API error: {e}")
        return None


def get_pinecone_index():
    """
    Returns the Pinecone index connection (lazy initialization).
    Creates the connection on first call and reuses it for all subsequent calls.
    Our index: "investigative-rag" with 100,835 vectors in two namespaces:
    - "irs" namespace: 74,529 IRS 990 text chunks
    - "fec" namespace: 26,306 FEC committee text chunks
    """
    global _pinecone_index
    if _pinecone_index is None and PINECONE_AVAILABLE and PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


# ── TEXT CLEANING ─────────────────────────────────────────────────────────────
def clean_text(text):
    """
    Cleans a text snippet for display in citations.
    Removes extra whitespace and truncates to 800 characters.
    IRS XML files often have extra spaces and line breaks that we clean here.
    """
    if not text:
        return ""
    return " ".join(text.split())[:800]


# ── PINECONE SEARCH ───────────────────────────────────────────────────────────
def search_pinecone(query, namespace, k=TOP_K):
    """
    Searches our Pinecone index for the most similar document chunks.

    How it works:
    1. Convert the query to a 384-dimensional vector
    2. Pinecone finds the top-K vectors closest to our query vector
    3. Returns the matching document chunks with their metadata

    namespace = "irs" → searches IRS 990 XML text chunks
    namespace = "fec" → searches FEC committee description text chunks

    Deduplication: if the same file appears multiple times in results,
    we keep only the first (highest similarity) occurrence.
    """
    try:
        model = get_embed_model()   # try local model first
        index = get_pinecone_index()
        if index is None:
            return []

        # Get the query vector
        if model:
            # Use local SentenceTransformer model (only available locally)
            query_vector = model.encode(query).tolist()
        else:
            # Use HuggingFace API (used on Render cloud)
            query_vector = get_embedding_via_api(query)
            if not query_vector:
                return []

        # Search Pinecone with the query vector
        results = index.query(
            vector=query_vector,
            top_k=k,
            namespace=namespace,         # "irs" or "fec"
            include_metadata=True        # return metadata (org name, EIN etc.)
        )

        # Convert Pinecone results to Citation objects
        citations = []
        seen = set()  # track seen file names to avoid duplicates

        for match in results.matches:
            meta = match.metadata or {}
            file_name = meta.get("file_name", match.id)

            # Skip if we already have this file
            if file_name in seen:
                continue
            seen.add(file_name)

            citations.append(Citation(
                source    = meta.get("source", namespace.upper()),
                file_name = file_name,
                org_name  = meta.get("org_name", ""),
                ein       = meta.get("ein", ""),
                object_id = meta.get("object_id", ""),
                snippet   = clean_text(meta.get("text", "")),
                distance  = round(1 - match.score, 4),  # convert similarity to distance
            ))

        return citations

    except Exception as e:
        print(f"Pinecone search error ({namespace}): {e}")
        return []


# ── CHROMADB SEARCH (FALLBACK) ────────────────────────────────────────────────
def search_chromadb(collection_name, query, k=TOP_K):
    """
    Fallback search using local ChromaDB when Pinecone is not available.
    We migrated to Pinecone for cloud deployment but keep this as a backup.

    ChromaDB stores vectors locally on disk (5-6 GB for our full dataset).
    This is why we cannot use it on Render (1GB build size limit).
    """
    if not CHROMADB_AVAILABLE:
        return []
    try:
        # Create embedding function for ChromaDB
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL)

        # Connect to local ChromaDB and get the collection
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col    = client.get_collection(name=collection_name, embedding_function=ef)

        # Search ChromaDB
        results = col.query(query_texts=[query], n_results=k)

        # Convert results to Citation objects
        citations = []
        seen = set()
        ids   = results.get("ids",        [[]])[0]
        docs  = results.get("documents",  [[]])[0]
        metas = results.get("metadatas",  [[]])[0]
        dists = results.get("distances",  [[]])[0]

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            file_name = meta.get("file_name", doc_id)
            if file_name in seen:
                continue
            seen.add(file_name)
            citations.append(Citation(
                source    = meta.get("source", "UNKNOWN"),
                file_name = file_name,
                org_name  = meta.get("org_name", ""),
                ein       = meta.get("ein", ""),
                object_id = meta.get("object_id", ""),
                snippet   = clean_text(doc),
                distance  = round(dist, 4),
            ))
        return citations

    except Exception as e:
        print(f"ChromaDB search error ({collection_name}): {e}")
        return []


# ── RETRIEVE FUNCTION ─────────────────────────────────────────────────────────
def retrieve(collection_name, query, k=TOP_K):
    """
    Retrieves relevant document chunks from either Pinecone or ChromaDB.

    Priority:
    1. Try Pinecone first (cloud — used on Render)
    2. Fall back to ChromaDB if Pinecone fails or is unavailable

    The namespace is determined from the collection name:
    - "irs_filings_25k" → namespace "irs"
    - "fec_filings"     → namespace "fec"
    """
    namespace = "irs" if "irs" in collection_name.lower() else "fec"

    # Try Pinecone first
    if PINECONE_AVAILABLE and PINECONE_API_KEY:
        results = search_pinecone(query, namespace, k)
        if results:
            return results

    # Fall back to ChromaDB
    return search_chromadb(collection_name, query, k)


# ── RECIPROCAL RANK FUSION ────────────────────────────────────────────────────
def reciprocal_rank_fusion(irs_citations, fec_citations, k=RRF_K):
    """
    Combines IRS and FEC search results into a single ranked list using RRF.

    Why RRF?
    When a user asks a cross-dataset question, we get two separate ranked lists:
    - IRS results: [doc_A rank 1, doc_B rank 2, doc_C rank 3 ...]
    - FEC results: [doc_X rank 1, doc_Y rank 2, doc_A rank 3 ...]

    A document that ranks highly in BOTH lists is more relevant than one
    that ranks highly in only one list. RRF captures this.

    Formula: score(doc) = sum of 1/(k + rank) across all lists
    - k=60 is the standard constant from the original RRF research paper
    - A document ranked 1st in a list gets score: 1/(60+1) = 0.0164
    - A document ranked 10th gets score: 1/(60+10) = 0.0143
    - A document appearing in both lists gets both scores added together

    Result: documents relevant to both IRS and FEC questions rank highest.
    """
    scores       = {}  # key = file_name, value = cumulative RRF score
    all_citations = {} # key = file_name, value = Citation object

    # Score IRS results — rank 1 gets highest score
    for rank, citation in enumerate(irs_citations, start=1):
        key = citation.file_name
        scores[key]        = scores.get(key, 0) + 1 / (k + rank)
        all_citations[key] = citation

    # Score FEC results — documents in both lists get scores added
    for rank, citation in enumerate(fec_citations, start=1):
        key = citation.file_name
        scores[key]        = scores.get(key, 0) + 1 / (k + rank)
        all_citations[key] = citation

    # Sort by RRF score descending — highest score = most relevant
    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [all_citations[k] for k in ranked_keys]


# ── BUILD CONTEXT ─────────────────────────────────────────────────────────────
def build_context(citations):
    """
    Formats citations into numbered text context for Claude.

    Example output:
    [1] IRS 990 | Org: Gates Foundation | EIN: 56-2618866
    The foundation was established to... [text from XML]

    [2] FEC Filing | File: FEC_2024_committee_C00401224
    Committee: ACTBLUE | Type: Non-Party | Receipts: $3.8 billion...

    The numbers [1], [2] become the citation references in Claude's answer.
    """
    parts = []
    for i, c in enumerate(citations, start=1):
        if c.source == "IRS":
            header = f"[{i}] IRS 990 | Org: {c.org_name or 'Unknown'} | EIN: {c.ein or 'N/A'}"
        else:
            header = f"[{i}] FEC Filing | File: {c.file_name}"
        parts.append(f"{header}\n{c.snippet}")
    return "\n\n".join(parts)


def build_citation_list(citations):
    """
    Builds a simple numbered citation list for the end of the answer.
    Example:
    [1] 202312345_public.xml — Gates Foundation (EIN: 56-2618866)
    [2] FEC_2024_committee_C00401224
    """
    lines = []
    for i, c in enumerate(citations, start=1):
        if c.source == "IRS":
            lines.append(f"[{i}] {c.file_name} — {c.org_name} (EIN: {c.ein})")
        else:
            lines.append(f"[{i}] {c.file_name}")
    return "\n".join(lines)


# ── GENERATE ANSWER ───────────────────────────────────────────────────────────
def generate_answer(context, citation_list, query):
    """
    Calls Claude to generate a professional cited answer from retrieved documents.

    This is the "Generation" step in RAG (Retrieval Augmented Generation).
    We pass:
    1. The user question
    2. The retrieved document chunks (numbered context)
    3. The citation list

    Claude reads all of this and generates a clear, cited answer.
    It uses [1], [2] etc. to reference the specific source documents.

    System prompt instructions:
    - Be specific and cite with [1][2]
    - Format numbers clearly ($3.1 billion, $450 million)
    - Never fabricate numbers or organization names
    - End with a Sources section
    """
    system_prompt = (
        "You are an investigative intelligence analyst specializing in nonprofit finance "
        "(IRS 990 filings) and political finance (FEC filings).\n\n"
        "Answer questions using only the provided source documents.\n"
        "Use inline citations like [1], [2] when referencing sources.\n"
        "Be concise, factual, and professional.\n"
        "Never fabricate numbers or organization names.\n"  # critical — no hallucinations
        "End every answer with a Sources section."
    )

    user_message = (
        f"Question: {query}\n\n"
        f"--- SOURCE DOCUMENTS ---\n{context}\n\n"
        f"--- CITATION REFERENCES ---\n{citation_list}\n\n"
        "Answer using only the source documents above."
    )

    # Call Anthropic API
    api = _llm_client.Anthropic(api_key=LLM_API_KEY)
    response = api.messages.create(
        model=LLM_MODEL,          # claude-haiku-4-5-20251001
        max_tokens=1024,          # max answer length
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ── MAIN ASK FUNCTION ─────────────────────────────────────────────────────────
def ask(query, dataset="both", top_k=TOP_K):
    """
    Main entry point for Pinecone-based RAG.
    Called by hybrid.py as the fallback when no SQL route matches.

    ANSWER CACHING:
    We cache complete answers in memory. If the same question is asked again,
    we return the cached answer instantly without any database or API calls.
    The cache key is "question|dataset" (e.g. "what is united way mission?|irs").

    Steps:
    1. Check answer cache — return instantly if found
    2. Search Pinecone IRS namespace (if dataset = "irs" or "both")
    3. Search Pinecone FEC namespace (if dataset = "fec" or "both")
    4. Apply RRF fusion if both IRS and FEC results exist
    5. Build context string from top results
    6. Call Claude to generate cited answer
    7. Cache the answer for future identical queries
    8. Return RAGResponse with answer, citations, sources_used

    Parameters:
    - query:   user question string
    - dataset: "irs", "fec", or "both" — which namespaces to search
    - top_k:   how many results to retrieve per namespace (default 5)
    """
    global _answer_cache

    # ── Step 1: Check answer cache ────────────────────────────────────────────
    # Create a unique key for this exact question + dataset combination
    cache_key = f"{query.lower().strip()}|{dataset}"
    if cache_key in _answer_cache:
        print(f"[Cache] Returning cached answer for: {query[:50]}")
        return _answer_cache[cache_key]  # instant return — no API calls

    # ── Step 2 & 3: Retrieve from Pinecone ───────────────────────────────────
    all_citations = []
    sources_used  = []
    irs_citations = []
    fec_citations = []

    if dataset in ("irs", "both"):
        # Search IRS namespace in Pinecone
        irs_citations = retrieve(IRS_COLLECTION, query, k=top_k)
        if irs_citations:
            label = "IRS 990 (Pinecone)" if PINECONE_AVAILABLE and PINECONE_API_KEY else "IRS 990"
            sources_used.append(label)

    if dataset in ("fec", "both"):
        # Search FEC namespace in Pinecone
        fec_citations = retrieve(FEC_COLLECTION, query, k=top_k)
        if fec_citations:
            label = "FEC Filings (Pinecone)" if PINECONE_AVAILABLE and PINECONE_API_KEY else "FEC Filings"
            sources_used.append(label)

    # ── Step 4: Apply RRF fusion ──────────────────────────────────────────────
    if irs_citations and fec_citations:
        # Both namespaces returned results — merge with RRF
        all_citations = reciprocal_rank_fusion(irs_citations, fec_citations)
    else:
        # Only one namespace returned results — just combine them
        all_citations = irs_citations + fec_citations

    # ── Handle no results ─────────────────────────────────────────────────────
    if not all_citations:
        return RAGResponse(
            answer="No relevant documents found. Try rephrasing your question.",
            citations=[],
            sources_used=[],
        )

    # ── Step 5: Build context ─────────────────────────────────────────────────
    # Format the retrieved chunks as numbered context for Claude
    context       = build_context(all_citations)
    citation_list = build_citation_list(all_citations)

    # ── Step 6: Generate answer ───────────────────────────────────────────────
    # Call Claude to read the context and write a cited answer
    try:
        answer_text = generate_answer(context, citation_list, query)
    except Exception as e:
        answer_text = f"Could not generate answer: {e}"

    # ── Step 7: Build response and cache it ───────────────────────────────────
    response = RAGResponse(
        answer       = answer_text,
        citations    = all_citations[:10],  # return top 10 citations max
        sources_used = sources_used,
    )

    # Save to answer cache for future identical queries
    if len(_answer_cache) < MAX_CACHE_SIZE:
        _answer_cache[cache_key] = response

    return response
