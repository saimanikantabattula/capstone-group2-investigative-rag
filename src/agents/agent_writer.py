"""
agent_writer.py
===============
The Writer Agent generates clear, professional, cited answers.

This is the final step in our multi-agent pipeline.
It takes data retrieved by the Filter Agent or Retriever Agent
and uses Claude AI to write a well-formatted, cited answer.

Think of it like a journalist:
- The Filter Agent (or Retriever Agent) gathers the raw facts
- The Writer Agent writes the final story from those facts
- Every fact in the answer gets a citation number [1], [2] etc.

Responsibilities:
- Formatting PostgreSQL rows as readable numbered context
- Formatting ChromaDB document chunks as readable numbered context
- Calling Claude API to generate the answer
- Building the citation list for the frontend
- Returning the answer and citations

Input:  Raw data rows (from PostgreSQL) or document chunks (from ChromaDB)
Output: Professional answer string + list of citation objects
"""

import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Anthropic Python library for calling Claude API
import anthropic as _llm_client

# Configuration from environment variables
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL   = "claude-haiku-4-5-20251001"  # Claude Haiku: fast and affordable


class WriterAgent:
    """
    Generates cited answers from structured or unstructured data.
    This is the final synthesis step in the multi-agent pipeline.

    The Writer Agent works with two types of input data:
    1. PostgreSQL rows   (from FilterAgent)  → structured financial numbers
    2. ChromaDB chunks   (from RetrieverAgent) → unstructured document text
    """

    def __init__(self):
        # Create Anthropic API client — reused for all requests
        self.api = _llm_client.Anthropic(api_key=LLM_API_KEY)

    def format_sql_context(self, rows, source_label):
        """
        Converts PostgreSQL row dictionaries into numbered text for Claude.

        Why we need to format rows as text:
        Claude cannot directly read Python dictionaries.
        We convert the rows into a readable numbered format that Claude understands.

        Example input (a list of row dicts):
        [{"org_name": "Mass General", "total_revenue": 23474745033, "state": "MA"}]

        Example output (formatted context):
        [1] IRS Financials (PostgreSQL) | org_name: Mass General | total_revenue: $23,474,745,033 | state: MA
        [2] IRS Financials (PostgreSQL) | org_name: Fidelity Investments | total_revenue: $19,858,151,933 | state: MA

        The [1], [2] numbers become citation markers in Claude's answer.
        Large numbers (> 1000) are formatted as dollar amounts with commas.

        Parameters:
        - rows:         list of dicts from PostgreSQL query
        - source_label: human readable name like "IRS Financials (PostgreSQL)"

        Returns: formatted string to pass to Claude
        """
        if not rows:
            return "No data found."

        lines = []
        for i, row in enumerate(rows, 1):
            parts = [f"[{i}] {source_label}"]
            for k, v in row.items():
                if v is not None and v != "":
                    # Format large numbers as dollar amounts for readability
                    # Example: 23474745033 → "$23,474,745,033"
                    if isinstance(v, (int, float)) and v > 1000:
                        v = f"${v:,.0f}"
                    parts.append(f"{k}: {v}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def format_chromadb_context(self, citations):
        """
        Converts ChromaDB document chunks into numbered text for Claude.

        ChromaDB chunks contain raw text from IRS XML files or FEC documents.
        We format them with a header showing the source and organization name.

        Example output:
        [1] IRS 990 | Org: Gates Foundation | EIN: 56-2618866
        The foundation was established to... [actual XML text]

        [2] FEC Filing | File: FEC_2024_committee_C00401224
        Committee: ACTBLUE | Type: Non-Party Independent Expenditure...

        Parameters:
        - citations: list of citation dicts from RetrieverAgent

        Returns: formatted string to pass to Claude
        """
        if not citations:
            return "No relevant documents found."

        parts = []
        for i, c in enumerate(citations, 1):
            if c["source"] == "IRS":
                # IRS documents: show org name and EIN
                header = f"[{i}] IRS 990 | Org: {c['org_name'] or 'Unknown'} | EIN: {c['ein'] or 'N/A'}"
            else:
                # FEC documents: show file name
                header = f"[{i}] FEC Filing | File: {c['file_name']}"
            parts.append(f"{header}\n{c['snippet']}")
        return "\n\n".join(parts)

    def build_citations(self, data, source_type):
        """
        Builds a list of citation objects for the frontend to display.

        The frontend shows citation cards below each answer.
        Each card shows: source (IRS/FEC), org name, EIN, and a snippet.
        Users can click to view the original document.

        Parameters:
        - data:        rows (PostgreSQL) or citation dicts (ChromaDB)
        - source_type: "postgresql" or "chromadb"

        Returns: list of citation dicts with source, org_name, ein, snippet
        """
        citations = []

        if source_type == "postgresql":
            # Build citations from PostgreSQL row data
            for row in data[:5]:  # limit to top 5 citations
                # Get org name from either IRS or FEC column names
                org_name = (
                    row.get("org_name")  or   # IRS financials column
                    row.get("irs_name")  or   # cross-dataset IRS column
                    row.get("CMTE_NM")   or   # FEC committee name column
                    row.get("fec_name")  or   # cross-dataset FEC column
                    ""
                )

                # Determine if this is IRS or FEC data based on column names present
                source = "FEC" if "CMTE_NM" in row or "fec_name" in row else "IRS"
                ein    = row.get("ein", "")

                # Build a readable snippet from all non-empty row values
                snippet = " | ".join(
                    f"{k}: {v}" for k, v in row.items()
                    if v is not None and v != ""
                )[:200]  # truncate to 200 chars

                citations.append({
                    "source":    source,
                    "file_name": f"{source} PostgreSQL",
                    "org_name":  org_name,
                    "ein":       ein,
                    "object_id": "",
                    "snippet":   snippet,
                    "distance":  0.0,  # no distance for SQL results
                })

        else:
            # ChromaDB citations already have the right format
            citations = data[:5]

        return citations

    def generate(self, question, context, source_label):
        """
        Calls Claude to generate the final cited answer.

        This is the AI generation step — we give Claude:
        1. A system prompt that tells it how to behave
        2. The user's question
        3. The retrieved data (formatted as numbered context)

        Claude reads all of this and writes a professional answer
        using [1], [2] to cite specific pieces of data.

        System prompt instructions:
        - Answer ONLY from the provided data (no hallucination)
        - Use [1], [2] citation markers
        - Format numbers clearly ($3.1 billion, $450 million)
        - Be confident and professional
        - End with a Sources section

        Parameters:
        - question:     user's question
        - context:      formatted numbered context string
        - source_label: describes the data source (for system prompt)

        Returns: generated answer string from Claude
        """
        system_prompt = (
            f"You are an investigative analyst specializing in {source_label} data.\n"
            "Answer the question using ONLY the data provided below.\n"
            "Be specific and cite data with [1], [2] etc.\n"     # always cite sources
            "Format numbers clearly (e.g. $3.1 billion, $450 million).\n"
            "Present findings confidently.\n"
            "Never say 'I cannot answer' if there is any relevant data.\n"  # always give best answer
            "Be concise and professional.\n"
            "End with a brief Sources section."
        )

        user_message = (
            f"Question: {question}\n\n"
            f"--- DATA ---\n{context}\n\n"
            "Answer the question using the data above."
        )

        # Call Claude API
        response = self.api.messages.create(
            model      = LLM_MODEL,   # claude-haiku-4-5-20251001
            max_tokens = 1024,        # max answer length (~750 words)
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_message}],
        )

        # Return just the text content from Claude's response
        return response.content[0].text

    def run(self, question, data, source_label, source_type="postgresql"):
        """
        Main entry point for WriterAgent.

        Steps:
        1. Check if data is empty → return "no data found" message
        2. Format data into readable context based on source type
        3. Call Claude to generate the cited answer
        4. Build citation objects for the frontend
        5. Return (answer, citations) tuple

        Parameters:
        - question:     user's question
        - data:         rows from PostgreSQL OR citation dicts from ChromaDB
        - source_label: human readable source name ("IRS Financials (PostgreSQL)" etc.)
        - source_type:  "postgresql" or "chromadb"

        Returns: tuple of (answer_string, citations_list)
        """
        print(f"[WriterAgent] Generating answer from {source_type} data...")

        # Handle case where no data was found
        if not data:
            return (
                "No relevant data found. Try rephrasing your question or selecting a different dataset.",
                []
            )

        # Format data into context string based on what type of data it is
        if source_type == "postgresql":
            # PostgreSQL rows → numbered formatted text
            context = self.format_sql_context(data, source_label)
        else:
            # ChromaDB document chunks → numbered text with headers
            context = self.format_chromadb_context(data)

        # Generate the answer using Claude
        try:
            answer = self.generate(question, context, source_label)
        except Exception as e:
            # If Claude API fails, return error message instead of crashing
            answer = f"Could not generate answer: {e}"

        # Build citations for the frontend to display
        citations = self.build_citations(data, source_type)

        print(f"[WriterAgent] Answer generated ({len(answer)} chars)")
        return answer, citations


# ── Direct usage for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    agent = WriterAgent()

    # Test with sample PostgreSQL data
    sample_rows = [
        {"org_name": "Mass General Brigham", "state": "MA", "total_revenue": 23474745033},
        {"org_name": "Fidelity Investments Charitable", "state": "MA", "total_revenue": 19000000000},
    ]
    answer, citations = agent.run(
        question     = "Which nonprofits raised the most money?",
        data         = sample_rows,
        source_label = "IRS nonprofit financial data",
        source_type  = "postgresql"
    )
    print(answer[:300])
