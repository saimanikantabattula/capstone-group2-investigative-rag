"""
controller_agent.py

The Controller Agent is the main coordinator of the Investigative RAG system.
It receives the user question, decides which agents to call, and returns
the final cited answer.

Flow:
1. Receive question + dataset preference
2. Classify question type (financial, geographic, document, cross-dataset)
3. Route to Filter Agent (PostgreSQL) or Retriever Agent (ChromaDB)
4. Pass results to Writer Agent to generate cited answer
5. Return final answer with citations
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.agent_filter import FilterAgent
from src.agents.agent_retriever import RetrieverAgent
from src.agents.agent_writer import WriterAgent


class ControllerAgent:
    """
    Main coordinator agent. Receives questions and routes them to the
    appropriate specialized agent based on question type.
    """

    def __init__(self):
        self.filter_agent = FilterAgent()
        self.retriever_agent = RetrieverAgent()
        self.writer_agent = WriterAgent()

        # Keywords that indicate a financial/structured query
        self.financial_keywords = [
            "most money", "highest revenue", "most revenue", "largest",
            "most assets", "most expenses", "top 10", "top 5", "raised",
            "which nonprofits", "which organizations", "which hospitals",
            "which universities", "which foundations", "which committees",
            "which PACs", "over 100 million", "over 1 billion",
            "based in", "located in", "cash on hand", "spent the most",
        ]

        # Keywords that indicate a cross-dataset query
        self.cross_dataset_keywords = [
            "connections to", "linked to", "associated with",
            "both irs and fec", "nonprofit and political",
            "appear in both", "cross dataset",
        ]

        # Keywords that indicate an FEC query
        self.fec_keywords = [
            "pac", "committee", "campaign", "political", "fec",
            "donation", "expenditure", "disbursement", "election",
            "actblue", "winred", "harris", "trump", "dnc", "rnc",
            "democratic", "republican",
        ]

    def classify_question(self, question, dataset):
        """
        Classify the question type to route to the right agent.
        Returns: 'filter' (PostgreSQL) or 'retriever' (ChromaDB)
        """
        q = question.lower()

        # Cross-dataset queries always go to filter agent
        if any(kw in q for kw in self.cross_dataset_keywords):
            return "filter_cross"

        # Financial questions go to filter agent
        if any(kw in q for kw in self.financial_keywords):
            return "filter"

        # FEC questions with financial intent go to filter agent
        if any(kw in q for kw in self.fec_keywords) and dataset in ("fec", "both"):
            return "filter"

        # All other questions go to retriever agent
        return "retriever"

    def run(self, question, dataset="both", top_k=5):
        """
        Main entry point. Routes question to appropriate agent and returns answer.

        Args:
            question: User's question string
            dataset: 'irs', 'fec', or 'both'
            top_k: Number of results to retrieve

        Returns:
            dict with keys: answer, citations, sources_used, response_time, agent_used
        """
        start_time = time.time()

        print(f"[ControllerAgent] Question: {question[:60]}...")
        print(f"[ControllerAgent] Dataset: {dataset}")

        # Step 1 — Classify question
        route = self.classify_question(question, dataset)
        print(f"[ControllerAgent] Routing to: {route}")

        # Step 2 — Route to appropriate agent
        if route in ("filter", "filter_cross"):
            # Use Filter Agent (PostgreSQL)
            filter_result = self.filter_agent.run(question, dataset)

            if filter_result["rows"] and len(filter_result["rows"]) > 0:
                # Step 3 — Pass to Writer Agent
                answer, citations = self.writer_agent.run(
                    question=question,
                    data=filter_result["rows"],
                    source_label=filter_result["source_label"],
                    source_type="postgresql"
                )
                sources_used = [filter_result["source_label"]]
                agent_used = "filter_agent + writer_agent"
            else:
                # Fall back to retriever if no PostgreSQL results
                print(f"[ControllerAgent] No PostgreSQL results, falling back to retriever")
                retriever_result = self.retriever_agent.run(question, dataset, top_k)
                answer, citations = self.writer_agent.run(
                    question=question,
                    data=retriever_result["citations"],
                    source_label="ChromaDB RAG",
                    source_type="chromadb"
                )
                sources_used = retriever_result["sources_used"]
                agent_used = "retriever_agent + writer_agent"

        else:
            # Use Retriever Agent (ChromaDB)
            retriever_result = self.retriever_agent.run(question, dataset, top_k)
            answer, citations = self.writer_agent.run(
                question=question,
                data=retriever_result["citations"],
                source_label="ChromaDB RAG",
                source_type="chromadb"
            )
            sources_used = retriever_result["sources_used"]
            agent_used = "retriever_agent + writer_agent"

        response_time = round(time.time() - start_time, 2)
        print(f"[ControllerAgent] Done in {response_time}s using {agent_used}")

        return {
            "answer": answer,
            "citations": citations,
            "sources_used": sources_used,
            "response_time": response_time,
            "agent_used": agent_used,
        }


# ── Direct usage ──
if __name__ == "__main__":
    agent = ControllerAgent()
    result = agent.run("Which nonprofits raised the most money?", dataset="irs")
    print(result["answer"][:300])
