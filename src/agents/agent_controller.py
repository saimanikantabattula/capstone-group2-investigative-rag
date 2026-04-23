"""
agent_controller.py
===================
The Controller Agent is the main coordinator of the multi-agent system.

Think of it like a manager in an office:
- It receives every question from the user
- It decides which team member (agent) is best suited to answer
- It sends the question to that agent
- It gets the result back and passes it to the Writer Agent
- It returns the final answer to the user

Flow:
1. Receive question + dataset preference from main.py
2. Classify question type (financial, geographic, document, cross-dataset)
3. Route to Filter Agent (PostgreSQL) or Retriever Agent (ChromaDB)
4. Pass results to Writer Agent to generate cited answer
5. Return final answer with citations and timing info
"""

import os
import sys
import time

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the three specialized agents
from src.agents.agent_filter    import FilterAgent    # handles PostgreSQL queries
from src.agents.agent_retriever import RetrieverAgent # handles ChromaDB vector search
from src.agents.agent_writer    import WriterAgent    # handles answer generation


class ControllerAgent:
    """
    Main coordinator agent.
    Receives questions and routes them to the appropriate specialized agent.

    This implements the "orchestrator" pattern in multi-agent systems:
    - One central agent (controller) coordinates multiple specialized agents
    - Each specialized agent has one job and does it well
    - The controller combines their outputs into a final answer
    """

    def __init__(self):
        # Create instances of all three specialized agents
        # These are created once and reused for all questions
        self.filter_agent    = FilterAgent()    # PostgreSQL queries
        self.retriever_agent = RetrieverAgent() # ChromaDB vector search
        self.writer_agent    = WriterAgent()    # LLM answer generation

        # Keywords that indicate a financial/structured database query
        # If a question contains ANY of these → route to Filter Agent (PostgreSQL)
        self.financial_keywords = [
            "most money", "highest revenue", "most revenue", "largest",
            "most assets", "most expenses", "top 10", "top 5", "raised",
            "which nonprofits", "which organizations", "which hospitals",
            "which universities", "which foundations", "which committees",
            "which PACs", "over 100 million", "over 1 billion",
            "based in", "located in", "cash on hand", "spent the most",
        ]

        # Keywords that indicate a cross-dataset query (IRS + FEC JOIN)
        # Example: "Which nonprofits have connections to political committees?"
        self.cross_dataset_keywords = [
            "connections to", "linked to", "associated with",
            "both irs and fec", "nonprofit and political",
            "appear in both", "cross dataset",
        ]

        # Keywords that indicate a political finance (FEC) query
        # Example: "Which PACs spent the most in 2024?"
        self.fec_keywords = [
            "pac", "committee", "campaign", "political", "fec",
            "donation", "expenditure", "disbursement", "election",
            "actblue", "winred", "harris", "trump", "dnc", "rnc",
            "democratic", "republican",
        ]

    def classify_question(self, question, dataset):
        """
        Decides which agent should handle this question.

        Returns one of:
        - "filter_cross" → cross-dataset SQL JOIN query
        - "filter"       → PostgreSQL financial/geographic query
        - "retriever"    → ChromaDB vector search (document text)

        Decision logic:
        1. If cross-dataset keywords found → filter_cross
        2. If financial keywords found → filter
        3. If FEC keywords found AND dataset includes FEC → filter
        4. Otherwise → retriever (document text search)
        """
        q = question.lower()

        # Cross-dataset queries always use the Filter Agent (SQL JOIN)
        if any(kw in q for kw in self.cross_dataset_keywords):
            return "filter_cross"

        # Financial/ranking questions use the Filter Agent (PostgreSQL ORDER BY)
        if any(kw in q for kw in self.financial_keywords):
            return "filter"

        # FEC questions with financial intent use the Filter Agent
        if any(kw in q for kw in self.fec_keywords) and dataset in ("fec", "both"):
            return "filter"

        # All other questions (e.g. "What is the mission of United Way?") use the Retriever Agent
        return "retriever"

    def run(self, question, dataset="both", top_k=5):
        """
        Main entry point. Routes question to the right agent and returns the answer.

        Parameters:
        - question: user's question string
        - dataset:  "irs", "fec", or "both"
        - top_k:    number of results to retrieve from ChromaDB

        Returns a dict with:
        - answer:        the generated answer string
        - citations:     list of source documents used
        - sources_used:  which databases were queried
        - response_time: how long it took in seconds
        - agent_used:    which agents were used (for transparency)
        """
        start_time = time.time()

        print(f"[ControllerAgent] Question: {question[:60]}...")
        print(f"[ControllerAgent] Dataset: {dataset}")

        # Step 1: Classify the question to decide which agent to use
        route = self.classify_question(question, dataset)
        print(f"[ControllerAgent] Routing to: {route}")

        # Step 2: Route to the appropriate agent
        if route in ("filter", "filter_cross"):

            # ── Route A: Filter Agent (PostgreSQL) ────────────────────────────
            # Good for: financial rankings, geographic search, specific committee lookups
            filter_result = self.filter_agent.run(question, dataset)

            if filter_result["rows"] and len(filter_result["rows"]) > 0:
                # Step 3: Pass PostgreSQL results to Writer Agent to generate answer
                answer, citations = self.writer_agent.run(
                    question     = question,
                    data         = filter_result["rows"],
                    source_label = filter_result["source_label"],
                    source_type  = "postgresql"
                )
                sources_used = [filter_result["source_label"]]
                agent_used   = "filter_agent + writer_agent"

            else:
                # No PostgreSQL results found → fall back to Retriever Agent
                print(f"[ControllerAgent] No PostgreSQL results, falling back to retriever")
                retriever_result = self.retriever_agent.run(question, dataset, top_k)
                answer, citations = self.writer_agent.run(
                    question     = question,
                    data         = retriever_result["citations"],
                    source_label = "ChromaDB RAG",
                    source_type  = "chromadb"
                )
                sources_used = retriever_result["sources_used"]
                agent_used   = "retriever_agent + writer_agent"

        else:
            # ── Route B: Retriever Agent (ChromaDB vector search) ─────────────
            # Good for: document text questions, mission statements, program descriptions
            retriever_result = self.retriever_agent.run(question, dataset, top_k)
            answer, citations = self.writer_agent.run(
                question     = question,
                data         = retriever_result["citations"],
                source_label = "ChromaDB RAG",
                source_type  = "chromadb"
            )
            sources_used = retriever_result["sources_used"]
            agent_used   = "retriever_agent + writer_agent"

        # Calculate total response time
        response_time = round(time.time() - start_time, 2)
        print(f"[ControllerAgent] Done in {response_time}s using {agent_used}")

        return {
            "answer":        answer,
            "citations":     citations,
            "sources_used":  sources_used,
            "response_time": response_time,
            "agent_used":    agent_used,
        }


# ── Direct usage for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    agent  = ControllerAgent()
    result = agent.run("Which nonprofits raised the most money?", dataset="irs")
    print(result["answer"][:300])
