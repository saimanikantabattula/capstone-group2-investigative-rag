"""
hybrid.py

Hybrid query engine that routes questions to the right data source:
- Financial/ranking questions → PostgreSQL (irs_financials, fec_committees)
- Document/text questions → ChromaDB RAG
- Combines both when needed
"""

import os
import re
import psycopg2
import psycopg2.extras
import anthropic as _llm_client

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "capstone_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"

# Keywords that indicate a financial/ranking question best answered by PostgreSQL
FINANCIAL_KEYWORDS = [
    "most money", "highest revenue", "most revenue", "top nonprofit",
    "largest", "biggest", "most expensive", "highest expenses",
    "most assets", "richest", "top earning", "raised the most",
    "spent the most", "most funding", "highest compensation",
    "officer compensation", "executive pay", "salary",
    "total revenue", "total expenses", "total assets",
    "most receipts", "top pac", "top committee",
    "which nonprofits", "which organizations", "which hospitals",
    "which universities", "which foundations", "which charities",
    "rank", "ranking", "top 10", "top 5", "list of",
]

FEC_KEYWORDS = [
    "pac", "committee", "campaign", "political", "fec",
    "donation", "contribution", "expenditure", "disbursement",
    "election", "candidate", "party", "super pac",
]


def get_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def is_financial_question(question):
    q = question.lower()
    return any(kw in q for kw in FINANCIAL_KEYWORDS)


def is_fec_question(question):
    q = question.lower()
    return any(kw in q for kw in FEC_KEYWORDS)


def query_irs_financials(question):
    """Query PostgreSQL irs_financials table based on question intent."""
    q = question.lower()
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        if any(w in q for w in ["revenue", "raised", "money", "income", "funding"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE total_revenue IS NOT NULL
                ORDER BY total_revenue DESC
                LIMIT 15
            """)
        elif any(w in q for w in ["expense", "spent", "spending", "cost"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials
                WHERE total_expenses IS NOT NULL
                ORDER BY total_expenses DESC
                LIMIT 15
            """)
        elif any(w in q for w in ["asset", "worth", "wealth", "large"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(net_assets) as net_assets
                FROM irs_financials
                WHERE total_assets IS NOT NULL
                ORDER BY total_assets DESC
                LIMIT 15
            """)
        elif any(w in q for w in ["compensation", "salary", "officer", "executive", "pay"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(officer_compensation) as officer_compensation,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials
                WHERE officer_compensation IS NOT NULL
                ORDER BY officer_compensation DESC
                LIMIT 15
            """)
        elif any(w in q for w in ["hospital", "medical", "health"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE total_revenue IS NOT NULL
                AND (LOWER(org_name) LIKE '%hospital%'
                     OR LOWER(org_name) LIKE '%medical%'
                     OR LOWER(org_name) LIKE '%health%')
                ORDER BY total_revenue DESC
                LIMIT 15
            """)
        elif any(w in q for w in ["university", "college", "school", "education"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE total_revenue IS NOT NULL
                AND (LOWER(org_name) LIKE '%university%'
                     OR LOWER(org_name) LIKE '%college%'
                     OR LOWER(org_name) LIKE '%school%'
                     OR LOWER(org_name) LIKE '%institute%')
                ORDER BY total_revenue DESC
                LIMIT 15
            """)
        elif any(w in q for w in ["foundation", "charity", "charitable"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE total_revenue IS NOT NULL
                AND (LOWER(org_name) LIKE '%foundation%'
                     OR return_type = '990PF')
                ORDER BY total_revenue DESC
                LIMIT 15
            """)
        else:
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE total_revenue IS NOT NULL
                ORDER BY total_revenue DESC
                LIMIT 15
            """)

        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def query_fec_committees(question):
    """Query PostgreSQL fec_committees table."""
    q = question.lower()
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        if any(w in q for w in ["spent", "spending", "expenditure", "disbursement"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST", "TTL_DISB", "TTL_RECEIPTS", "cycle"
                FROM fec_committees
                WHERE "TTL_DISB" IS NOT NULL AND "TTL_DISB" != ''
                  AND "TTL_DISB" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_DISB" AS NUMERIC) DESC
                LIMIT 15
            """)
        else:
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST", "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE "TTL_RECEIPTS" IS NOT NULL AND "TTL_RECEIPTS" != ''
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC
                LIMIT 15
            """)

        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def format_rows_as_context(rows, source="IRS"):
    """Format database rows as readable context for the LLM."""
    if not rows:
        return "No data found."

    lines = []
    for i, row in enumerate(rows, 1):
        parts = [f"[{i}] {source}"]
        for k, v in row.items():
            if v is not None and v != "":
                # Format large numbers with commas
                if isinstance(v, (int, float)) and v > 1000:
                    v = f"${v:,.0f}"
                parts.append(f"{k}: {v}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def generate_answer_from_data(question, context, source_label):
    """Send structured data context to LLM for answer generation."""
    system_prompt = (
        f"You are an investigative analyst specializing in {source_label} financial data.\n"
        "Answer the question using ONLY the data provided below.\n"
        "Be specific, cite the data with [1], [2] etc.\n"
        "Format numbers clearly (e.g. $3.1 billion, $450 million).\n"
        "Be concise and professional.\n"
        "End with a brief Sources section."
    )

    user_message = (
        f"Question: {question}\n\n"
        f"--- DATA ---\n{context}\n\n"
        "Answer the question using the data above."
    )

    api = _llm_client.Anthropic(api_key=LLM_API_KEY)
    response = api.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def hybrid_ask(question, dataset="both", top_k=5):
    """
    Routes the question to the best data source:
    - Financial/ranking questions → PostgreSQL
    - Document questions → ChromaDB RAG
    - Returns consistent response format
    """
    from src.rag.answer import ask as rag_ask, RAGResponse, Citation

    use_db = is_financial_question(question)
    use_fec = is_fec_question(question)

    # Financial question about IRS data
    if use_db and dataset in ("irs", "both") and not use_fec:
        try:
            rows = query_irs_financials(question)
            if rows:
                context = format_rows_as_context(rows, "IRS Financials")
                answer = generate_answer_from_data(question, context, "IRS nonprofit finance")
                citations = [
                    Citation(
                        source="IRS",
                        file_name="irs_financials (PostgreSQL)",
                        org_name=r.get("org_name", ""),
                        ein="",
                        object_id="",
                        snippet=f"Revenue: {r.get('total_revenue', 'N/A')} | Expenses: {r.get('total_expenses', 'N/A')} | State: {r.get('state', 'N/A')}",
                        distance=0.0,
                    )
                    for r in rows[:5]
                ]
                return RAGResponse(
                    answer=answer,
                    citations=citations,
                    sources_used=["IRS Financials (PostgreSQL)"],
                )
        except Exception as e:
            print(f"PostgreSQL query failed: {e}, falling back to RAG")

    # Financial question about FEC data
    if use_db and use_fec and dataset in ("fec", "both"):
        try:
            rows = query_fec_committees(question)
            if rows:
                context = format_rows_as_context(rows, "FEC")
                answer = generate_answer_from_data(question, context, "FEC political finance")
                citations = [
                    Citation(
                        source="FEC",
                        file_name="fec_committees (PostgreSQL)",
                        org_name=r.get("CMTE_NM", ""),
                        ein="",
                        object_id="",
                        snippet=f"Receipts: {r.get('TTL_RECEIPTS', 'N/A')} | Disbursements: {r.get('TTL_DISB', 'N/A')} | Cycle: {r.get('cycle', 'N/A')}",
                        distance=0.0,
                    )
                    for r in rows[:5]
                ]
                return RAGResponse(
                    answer=answer,
                    citations=citations,
                    sources_used=["FEC Committees (PostgreSQL)"],
                )
        except Exception as e:
            print(f"FEC PostgreSQL query failed: {e}, falling back to RAG")

    # Default: use ChromaDB RAG
    return rag_ask(question, dataset=dataset, top_k=top_k)
