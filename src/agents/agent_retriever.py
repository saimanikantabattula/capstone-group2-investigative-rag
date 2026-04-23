"""
agent_retriever.py
==================
The Retriever Agent handles all unstructured document text search queries.

When a user asks a question that needs document text (not financial numbers),
this agent searches our vector database for relevant text passages.

Example questions handled by this agent:
- "What is the mission of United Way?"
- "What programs does the Gates Foundation fund?"
- "Tell me about Harvard's research programs"

How it works:
1. Takes the user question as input
2. Searches ChromaDB (local) for the most semantically similar text chunks
3. Applies Reciprocal Rank Fusion (RRF) to combine IRS and FEC results
4. Returns ranked document chunks with metadata

Important: On our cloud deployment (Render), we use Pinecone instead of ChromaDB.
This agent uses ChromaDB directly — it is kept for local testing and as a reference.
The cloud version (answer.py) uses Pinecone with the same logic.

Embedding Model: all-MiniLM-L6-v2 (384 dimensions)
Vector Search:   HNSW algorithm with cosine similarity
Fusion:          Reciprocal Rank Fusion (RRF) with k=60
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chromadb
from chromadb.utils import embedding_functions

# Configuration from environment variables
CHROMA_PATH    = os.getenv("CHROMA_PATH", "chroma_db")         # path to local ChromaDB files
IRS_COLLECTION = os.getenv("IRS_COLLECTION", "irs_filings_25k") # name of IRS collection
FEC_COLLECTION = os.getenv("FEC_COLLECTION", "fec_filings")     # name of FEC collection
EMBED_MODEL    = "all-MiniLM-L6-v2"  # same model used during data ingestion — must match!
RRF_K          = 60                   # standard RRF constant from research paper


class RetrieverAgent:
    """
    Handles semantic (meaning-based) search across ChromaDB vector collections.
    Uses Reciprocal Rank Fusion (RRF) to combine IRS and FEC results.

    Vector dimensions: 384 (produced by all-MiniLM-L6-v2 model)
    Search algorithm:  HNSW (Hierarchical Navigable Small World) — fast approximate nearest neighbor
    Similarity metric: Cosine similarity (measures angle between vectors)
    """

    def __init__(self):
        # Connect to local ChromaDB database
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Create the embedding function that converts text to 384-dimensional vectors
        # Must use the SAME model that was used when we indexed the documents
        # If models don't match, search results will be meaningless
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

    def search_collection(self, collection_name, query, k=5):
        """
        Searches a single ChromaDB collection for the most similar text chunks.

        How ChromaDB search works:
        1. Convert query text to a 384-dimensional vector using the embedding model
        2. Find the k vectors in our collection closest to the query vector
        3. "Closest" means smallest cosine distance (most similar meaning)
        4. Return the matching document chunks with their metadata

        Deduplication: if the same document file appears multiple times,
        we keep only the first (most similar) occurrence.

        Parameters:
        - collection_name: "irs_filings_25k" or "fec_filings"
        - query: user question text
        - k: number of results to return

        Returns: list of citation dictionaries
        """
        try:
            col = self.client.get_collection(
                name=collection_name,
                embedding_function=self.ef
            )
            results = col.query(query_texts=[query], n_results=k)
        except Exception as e:
            print(f"[RetrieverAgent] Could not query {collection_name}: {e}")
            return []

        # Extract results from ChromaDB response format
        citations = []
        ids   = results.get("ids",        [[]])[0]  # document IDs
        docs  = results.get("documents",  [[]])[0]  # actual text content
        metas = results.get("metadatas",  [[]])[0]  # metadata (org name, EIN etc.)
        dists = results.get("distances",  [[]])[0]  # similarity distances

        seen = set()  # track seen file names to avoid duplicate documents

        for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
            meta      = meta or {}
            file_name = meta.get("file_name", doc_id)

            # Skip if we already have a result from this file
            if file_name in seen:
                continue
            seen.add(file_name)

            citations.append({
                "source":    meta.get("source", "UNKNOWN"),   # "IRS" or "FEC"
                "file_name": file_name,                        # XML filename or FEC ID
                "org_name":  meta.get("org_name", ""),        # organization name
                "ein":       meta.get("ein", ""),             # IRS tax ID
                "object_id": meta.get("object_id", ""),       # unique document ID
                "snippet":   doc[:500] if doc else "",         # first 500 chars of text
                "distance":  round(dist, 4),                   # similarity score
            })

        return citations

    def reciprocal_rank_fusion(self, irs_results, fec_results):
        """
        Merges IRS and FEC result lists using Reciprocal Rank Fusion (RRF).

        Why RRF?
        When we search both IRS and FEC collections separately, we get two ranked lists.
        We need to combine them into one final ranked list intelligently.

        RRF Formula: score(document) = sum of 1/(k + rank) for each list it appears in
        - k=60 is the standard constant (from the 2009 Cormack et al. paper)
        - A document ranked #1 gets score: 1/(60+1) = 0.0164
        - A document ranked #10 gets score: 1/(60+10) = 0.0143
        - A document appearing in BOTH lists gets scores ADDED together
          → Documents relevant to both IRS and FEC rank highest

        Example:
        IRS list:  [doc_A rank 1, doc_B rank 2]
        FEC list:  [doc_C rank 1, doc_A rank 2]
        RRF result: doc_A wins because it appears in both lists

        Parameters:
        - irs_results: ranked list of IRS citation dicts
        - fec_results: ranked list of FEC citation dicts

        Returns: single merged and re-ranked list
        """
        scores        = {}  # key = file_name, value = cumulative RRF score
        all_citations = {}  # key = file_name, value = citation dict

        # Calculate RRF scores for IRS results
        for rank, citation in enumerate(irs_results, start=1):
            key = citation["file_name"]
            scores[key]        = scores.get(key, 0) + 1 / (RRF_K + rank)
            all_citations[key] = citation

        # Calculate RRF scores for FEC results (scores add to existing if doc appeared in IRS too)
        for rank, citation in enumerate(fec_results, start=1):
            key = citation["file_name"]
            scores[key]        = scores.get(key, 0) + 1 / (RRF_K + rank)
            all_citations[key] = citation

        # Sort all documents by their final RRF score (highest first)
        ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [all_citations[k] for k in ranked_keys]

    def run(self, question, dataset="both", top_k=5):
        """
        Main entry point for RetrieverAgent.
        Searches ChromaDB and returns ranked document chunks.

        Steps:
        1. If dataset includes IRS: search IRS collection
        2. If dataset includes FEC: search FEC collection
        3. If both have results: apply RRF fusion
        4. If only one has results: return those directly
        5. Return citations with source metadata

        Parameters:
        - question: user question text
        - dataset:  "irs", "fec", or "both"
        - top_k:    how many results to get per collection

        Returns a dict with:
        - citations:    list of citation dicts with source, snippet, metadata
        - sources_used: list of collection names that returned results
        """
        print(f"[RetrieverAgent] Searching ChromaDB for: {question[:55]}...")

        irs_results  = []
        fec_results  = []
        sources_used = []

        # Search IRS collection if needed
        if dataset in ("irs", "both"):
            irs_results = self.search_collection(IRS_COLLECTION, question, k=top_k)
            if irs_results:
                sources_used.append("IRS 990 (ChromaDB)")
                print(f"[RetrieverAgent] Found {len(irs_results)} IRS chunks")

        # Search FEC collection if needed
        if dataset in ("fec", "both"):
            fec_results = self.search_collection(FEC_COLLECTION, question, k=top_k)
            if fec_results:
                sources_used.append("FEC Filings (ChromaDB)")
                print(f"[RetrieverAgent] Found {len(fec_results)} FEC chunks")

        # Merge results using RRF if both collections returned results
        if irs_results and fec_results:
            print(f"[RetrieverAgent] Applying RRF fusion...")
            citations = self.reciprocal_rank_fusion(irs_results, fec_results)
        else:
            # Only one collection had results — no fusion needed
            citations = irs_results + fec_results

        print(f"[RetrieverAgent] Returning {len(citations)} fused chunks")
        return {"citations": citations, "sources_used": sources_used}


# ── Direct usage for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    agent  = RetrieverAgent()
    result = agent.run("What is the mission of the Gates Foundation?", dataset="irs")
    print(f"Found {len(result['citations'])} results")
    for c in result["citations"][:2]:
        print(f"  [{c['source']}] {c['org_name']} — distance: {c['distance']}")
        print(f"  {c['snippet'][:150]}...")
