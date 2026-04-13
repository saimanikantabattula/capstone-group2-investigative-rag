"""
writer_agent.py

The Writer Agent generates clear, professional, cited answers from
data retrieved by the Filter Agent or Retriever Agent.

It is responsible for:
- Formatting retrieved rows or chunks into readable context
- Calling the language model API to generate a cited answer
- Formatting the final answer with numbered citations [1], [2]
- Returning the answer and citation list

This agent uses the language model as the final synthesis step.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import anthropic as _llm_client

LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"


class WriterAgent:
    """
    Generates cited answers from structured or unstructured data.
    Acts as the final synthesis step in the multi-agent pipeline.

    Input: Data rows (PostgreSQL) or document chunks (ChromaDB)
    Output: Professional cited answer + citation list
    """

    def __init__(self):
        self.api = _llm_client.Anthropic(api_key=LLM_API_KEY)

    def format_sql_context(self, rows, source_label):
        """
        Format PostgreSQL rows as numbered context for the language model.

        Args:
            rows: List of dicts from PostgreSQL query
            source_label: Human readable source name

        Returns:
            Formatted context string with [1], [2] citations
        """
        if not rows:
            return "No data found."

        lines = []
        for i, row in enumerate(rows, 1):
            parts = [f"[{i}] {source_label}"]
            for k, v in row.items():
                if v is not None and v != "":
                    if isinstance(v, (int, float)) and v > 1000:
                        v = f"${v:,.0f}"
                    parts.append(f"{k}: {v}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def format_chromadb_context(self, citations):
        """
        Format ChromaDB document chunks as numbered context.

        Args:
            citations: List of citation dicts from RetrieverAgent

        Returns:
            Formatted context string with [1], [2] citations
        """
        if not citations:
            return "No relevant documents found."

        parts = []
        for i, c in enumerate(citations, 1):
            if c["source"] == "IRS":
                header = f"[{i}] IRS 990 | Org: {c['org_name'] or 'Unknown'} | EIN: {c['ein'] or 'N/A'}"
            else:
                header = f"[{i}] FEC Filing | File: {c['file_name']}"
            parts.append(f"{header}\n{c['snippet']}")
        return "\n\n".join(parts)

    def build_citations(self, data, source_type):
        """
        Build citation list from data.

        Returns:
            List of citation dicts with source, org_name, snippet, link
        """
        citations = []
        if source_type == "postgresql":
            for row in data[:5]:
                org_name = (row.get("org_name") or row.get("irs_name") or
                           row.get("CMTE_NM") or row.get("fec_name") or "")
                source = "FEC" if "CMTE_NM" in row or "fec_name" in row else "IRS"
                ein = row.get("ein", "")
                snippet = " | ".join(
                    f"{k}: {v}" for k, v in row.items()
                    if v is not None and v != ""
                )[:200]
                citations.append({
                    "source": source,
                    "file_name": f"{source} PostgreSQL",
                    "org_name": org_name,
                    "ein": ein,
                    "object_id": "",
                    "snippet": snippet,
                    "distance": 0.0,
                })
        else:
            citations = data[:5]
        return citations

    def generate(self, question, context, source_label):
        """
        Call language model to generate a cited answer.

        Args:
            question: User's question
            context: Formatted context string with numbered citations
            source_label: Source description for the system prompt

        Returns:
            Generated answer string
        """
        system_prompt = (
            f"You are an investigative analyst specializing in {source_label} data.\n"
            "Answer the question using ONLY the data provided below.\n"
            "Be specific and cite data with [1], [2] etc.\n"
            "Format numbers clearly (e.g. $3.1 billion, $450 million).\n"
            "Present findings confidently — say 'Based on our dataset, the top organizations are...'\n"
            "Never say 'I cannot answer' if there is any relevant data.\n"
            "Be concise and professional.\n"
            "End with a brief Sources section."
        )

        user_message = (
            f"Question: {question}\n\n"
            f"--- DATA ---\n{context}\n\n"
            "Answer the question using the data above."
        )

        response = self.api.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def run(self, question, data, source_label, source_type="postgresql"):
        """
        Main entry point for WriterAgent.

        Args:
            question: User's question
            data: Rows (PostgreSQL) or citations (ChromaDB)
            source_label: Human readable source name
            source_type: 'postgresql' or 'chromadb'

        Returns:
            Tuple of (answer_string, citations_list)
        """
        print(f"[WriterAgent] Generating answer from {source_type} data...")

        if not data:
            return (
                "No relevant data found. Try rephrasing your question or selecting a different dataset.",
                []
            )

        # Format context based on source type
        if source_type == "postgresql":
            context = self.format_sql_context(data, source_label)
        else:
            context = self.format_chromadb_context(data)

        # Generate answer
        try:
            answer = self.generate(question, context, source_label)
        except Exception as e:
            answer = f"Could not generate answer: {e}"

        # Build citations
        citations = self.build_citations(data, source_type)

        print(f"[WriterAgent] Answer generated ({len(answer)} chars)")
        return answer, citations


if __name__ == "__main__":
    agent = WriterAgent()
    sample_rows = [
        {"org_name": "Mass General Brigham", "state": "MA", "total_revenue": 23474745033},
        {"org_name": "Fidelity Investments Charitable", "state": "MA", "total_revenue": 19000000000},
    ]
    answer, citations = agent.run(
        question="Which nonprofits raised the most money?",
        data=sample_rows,
        source_label="IRS nonprofit financial data",
        source_type="postgresql"
    )
    print(answer[:300])
