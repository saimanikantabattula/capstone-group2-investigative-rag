"""
retriever_agent.py

The Retriever Agent handles all unstructured document search queries.
It is responsible for:
- Semantic search across IRS 990 text chunks (74,529 chunks)
- Semantic search across FEC filing text chunks (26,306 chunks)
- Reciprocal Rank Fusion (RRF) to combine results from both collections
- Returning ranked document chunks with metadata and citations

This agent uses ChromaDB vector search with all-MiniLM-L6-v2 embeddings.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
IRS_COLLECTION = os.getenv("IRS_COLLECTION", "irs_filings_25k")
FEC_COLLECTION = os.getenv("FEC_COLLECTION", "fec_filings")
EMBED_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60  # RRF constant


class RetrieverAgent:
    """
    Handles semantic search across ChromaDB vector collections.
    Uses Reciprocal Rank Fusion (RRF) to combine IRS and FEC results.

    Embedding Model: all-MiniLM-L6-v2 (384 dimensions)
    Vector Search: HNSW with cosine similarity
    Fusion: Reciprocal Rank Fusion (RRF)
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )

    def search_collection(self, collection_name, query, k=5):
        """
        Search a single ChromaDB collection for the top-K most similar chunks.

        Args:
            collection_name: Name of the ChromaDB collection
            query: User question string
            k: Number of results to return

        Returns:
            List of citation dicts with source, snippet, distance, metadata
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
            citations.append({
                "source": meta.get("source", "UNKNOWN"),
                "file_name": file_name,
                "org_name": meta.get("org_name", ""),
                "ein": meta.get("ein", ""),
                "object_id": meta.get("object_id", ""),
                "snippet": doc[:500] if doc else "",
                "distance": round(dist, 4),
            })

        return citations

    def reciprocal_rank_fusion(self, irs_results, fec_results):
        """
        Reciprocal Rank Fusion (RRF) combines ranked lists from IRS and FEC.

        Formula: RRF_score(doc) = sum(1 / (k + rank)) for each list
        Higher score = more relevant across both sources.

        Args:
            irs_results: Ranked list of IRS citation dicts
            fec_results: Ranked list of FEC citation dicts

        Returns:
            Single merged and re-ranked list of citations
        """
        scores = {}
        all_citations = {}

        # Score IRS results
        for rank, citation in enumerate(irs_results, start=1):
            key = citation["file_name"]
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank)
            all_citations[key] = citation

        # Score FEC results
        for rank, citation in enumerate(fec_results, start=1):
            key = citation["file_name"]
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank)
            all_citations[key] = citation

        # Sort by RRF score descending
        ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [all_citations[k] for k in ranked_keys]

    def run(self, question, dataset="both", top_k=5):
        """
        Main entry point for RetrieverAgent.
        Searches ChromaDB and returns ranked document chunks.

        Args:
            question: User question string
            dataset: 'irs', 'fec', or 'both'
            top_k: Number of results per collection

        Returns:
            dict with keys: citations, sources_used
        """
        print(f"[RetrieverAgent] Searching ChromaDB for: {question[:55]}...")

        irs_results = []
        fec_results = []
        sources_used = []

        if dataset in ("irs", "both"):
            irs_results = self.search_collection(IRS_COLLECTION, question, k=top_k)
            if irs_results:
                sources_used.append("IRS 990 (ChromaDB)")
                print(f"[RetrieverAgent] Found {len(irs_results)} IRS chunks")

        if dataset in ("fec", "both"):
            fec_results = self.search_collection(FEC_COLLECTION, question, k=top_k)
            if fec_results:
                sources_used.append("FEC Filings (ChromaDB)")
                print(f"[RetrieverAgent] Found {len(fec_results)} FEC chunks")

        # Apply RRF fusion when both sources have results
        if irs_results and fec_results:
            print(f"[RetrieverAgent] Applying RRF fusion...")
            citations = self.reciprocal_rank_fusion(irs_results, fec_results)
        else:
            citations = irs_results + fec_results

        print(f"[RetrieverAgent] Returning {len(citations)} fused chunks")
        return {"citations": citations, "sources_used": sources_used}


if __name__ == "__main__":
    agent = RetrieverAgent()
    result = agent.run("What is the mission of the Gates Foundation?", dataset="irs")
    print(f"Found {len(result['citations'])} results")
    for c in result["citations"][:2]:
        print(f"  [{c['source']}] {c['org_name']} — distance: {c['distance']}")
        print(f"  {c['snippet'][:150]}...")
