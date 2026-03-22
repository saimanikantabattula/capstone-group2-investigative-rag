"""
hybrid.py - v2

Hybrid query engine that routes questions to the right data source:
- Geographic questions → PostgreSQL irs_index
- Specific committee lookups → PostgreSQL fec_committees
- Threshold questions → PostgreSQL fec_committees
- Financial/ranking questions → PostgreSQL irs_financials or fec_committees
- Document/text questions → ChromaDB RAG
"""

import os
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
    "how much did", "how much has", "how much money",
    "over 100 million", "over 1 billion", "over 10 million",
    "more than", "at least", "based in", "located in", "cash on hand", "how much cash", "most cash",
]

FEC_KEYWORDS = [
    "pac", "committee", "campaign", "political", "fec",
    "donation", "contribution", "expenditure", "disbursement",
    "election", "candidate", "party", "super pac",
    "actblue", "winred", "harris", "trump", "dnc", "rnc",
    "democratic", "republican",
]

STATE_MAP = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}

SPECIFIC_COMMITTEES = [
    "actblue", "winred", "harris for president", "harris victory",
    "trump", "biden", "dnc", "rnc", "lincoln project",
    "emily", "planned parenthood", "nra", "america first",
    "priorities usa", "club for growth",
]

THRESHOLD_MAP = {
    "over 1 billion": 1_000_000_000, "more than 1 billion": 1_000_000_000,
    "over 100 million": 100_000_000, "more than 100 million": 100_000_000,
    "over 50 million": 50_000_000, "more than 50 million": 50_000_000,
    "over 10 million": 10_000_000, "more than 10 million": 10_000_000,
}


def get_db():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def detect_state(question):
    q = question.lower()
    for state_name, abbr in STATE_MAP.items():
        if state_name in q:
            return abbr, state_name
    return None, None


def is_financial_question(question):
    q = question.lower()
    return any(kw in q for kw in FINANCIAL_KEYWORDS)


def is_fec_question(question):
    q = question.lower()
    return any(kw in q for kw in FEC_KEYWORDS)


def query_irs_by_state(state_abbr):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT f.org_name, f.state, l.city, f.return_type, f.tax_year,
                   ROUND(f.total_revenue) as total_revenue,
                   ROUND(f.total_assets) as total_assets
            FROM irs_financials f
            LEFT JOIN irs_locations l USING (ein)
            WHERE UPPER(f.state) = %s
              AND f.total_revenue IS NOT NULL
            ORDER BY f.total_revenue DESC
            LIMIT 10
        """, (state_abbr,))
        financial_rows = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT org_name, state, city, return_type, tax_year
            FROM irs_locations
            WHERE UPPER(state) = %s
            ORDER BY org_name
            LIMIT 15
        """, (state_abbr,))
        location_rows = [dict(r) for r in cur.fetchall()]

        seen = set(r["org_name"] for r in financial_rows)
        extra = [r for r in location_rows if r["org_name"] not in seen]
        return financial_rows + extra[:10]
    finally:
        cur.close()
        conn.close()


def query_fec_specific_committee(committee_name):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                   "TTL_RECEIPTS", "TTL_DISB", "cycle"
            FROM fec_committees
            WHERE LOWER("CMTE_NM") LIKE %s
              AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
            ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC
            LIMIT 5
        """, (f"%{committee_name.lower()}%",))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def query_fec_threshold(amount, field="TTL_RECEIPTS"):
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(f"""
            SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                   "TTL_RECEIPTS", "TTL_DISB", "cycle"
            FROM fec_committees
            WHERE "{field}" IS NOT NULL AND "{field}" != ''
              AND "{field}" ~ '^[0-9.]+$'
              AND CAST("{field}" AS NUMERIC) >= %s
            ORDER BY CAST("{field}" AS NUMERIC) DESC
            LIMIT 20
        """, (amount,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def query_irs_financials(question):
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
                ORDER BY total_revenue DESC LIMIT 15
            """)
        elif any(w in q for w in ["expense", "spent", "spending", "cost"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials
                WHERE total_expenses IS NOT NULL
                ORDER BY total_expenses DESC LIMIT 15
            """)
        elif any(w in q for w in ["asset", "worth", "wealth", "large"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(net_assets) as net_assets
                FROM irs_financials
                WHERE total_assets IS NOT NULL
                ORDER BY total_assets DESC LIMIT 15
            """)
        elif any(w in q for w in ["compensation", "salary", "officer", "executive", "pay"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(officer_compensation) as officer_compensation,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials
                WHERE officer_compensation IS NOT NULL
                ORDER BY officer_compensation DESC LIMIT 15
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
                ORDER BY total_revenue DESC LIMIT 15
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
                ORDER BY total_revenue DESC LIMIT 15
            """)
        elif any(w in q for w in ["foundation", "charity", "charitable"]):
            order_col = "net_assets" if "net asset" in q else "total_assets" if "asset" in q else "total_revenue"
            cur.execute(f"""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets,
                       ROUND(net_assets) as net_assets
                FROM irs_financials
                WHERE (LOWER(org_name) LIKE '%foundation%' OR return_type = '990PF')
                  AND {order_col} IS NOT NULL
                ORDER BY {order_col} DESC LIMIT 15
            """)
        else:
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE total_revenue IS NOT NULL
                ORDER BY total_revenue DESC LIMIT 15
            """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def query_fec_committees(question):
    q = question.lower()
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        if any(w in q for w in ["spent", "spending", "expenditure", "disbursement"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST", "CMTE_CITY",
                       "TTL_DISB", "TTL_RECEIPTS", "INDV_CONTB",
                       "COH_COP", "DEBTS_OWED_BY_CMTE", "cycle"
                FROM fec_committees
                WHERE "TTL_DISB" IS NOT NULL AND "TTL_DISB" != ''
                  AND "TTL_DISB" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_DISB" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["cash", "hand", "balance"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "COH_COP", "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE "COH_COP" IS NOT NULL AND "COH_COP" != ''
                  AND "COH_COP" ~ '^[0-9.]+$'
                ORDER BY CAST("COH_COP" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["debt", "owe", "liability"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "DEBTS_OWED_BY_CMTE", "TTL_RECEIPTS", "cycle"
                FROM fec_committees
                WHERE "DEBTS_OWED_BY_CMTE" IS NOT NULL AND "DEBTS_OWED_BY_CMTE" != ''
                  AND "DEBTS_OWED_BY_CMTE" ~ '^[0-9.]+$'
                  AND CAST("DEBTS_OWED_BY_CMTE" AS NUMERIC) > 0
                ORDER BY CAST("DEBTS_OWED_BY_CMTE" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["individual", "donor", "person"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "INDV_CONTB", "TTL_RECEIPTS", "cycle"
                FROM fec_committees
                WHERE "INDV_CONTB" IS NOT NULL AND "INDV_CONTB" != ''
                  AND "INDV_CONTB" ~ '^[0-9.]+$'
                ORDER BY CAST("INDV_CONTB" AS NUMERIC) DESC LIMIT 15
            """)
        else:
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST", "CMTE_CITY",
                       "TTL_RECEIPTS", "TTL_DISB", "INDV_CONTB",
                       "COH_COP", "cycle"
                FROM fec_committees
                WHERE "TTL_RECEIPTS" IS NOT NULL AND "TTL_RECEIPTS" != ''
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


def format_rows_as_context(rows, source="IRS"):
    if not rows:
        return "No data found."
    lines = []
    for i, row in enumerate(rows, 1):
        parts = [f"[{i}] {source}"]
        for k, v in row.items():
            if v is not None and v != "":
                if isinstance(v, (int, float)) and v > 1000:
                    v = f"${v:,.0f}"
                parts.append(f"{k}: {v}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def generate_answer_from_data(question, context, source_label):
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
    from src.rag.answer import ask as rag_ask, RAGResponse, Citation

    use_db = is_financial_question(question)
    use_fec = is_fec_question(question)
    q_lower = question.lower()

    # ── 1. Geographic: query by state ──
    state_abbr, state_name = detect_state(question)
    if state_abbr:
        # FEC geographic question
        if use_fec and dataset in ("fec", "both"):
            try:
                conn = get_db()
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "TTL_RECEIPTS", "TTL_DISB", "cycle"
                    FROM fec_committees
                    WHERE UPPER("CMTE_ST") = %s
                      AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                    ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC
                    LIMIT 20
                """, (state_abbr,))
                rows = [dict(r) for r in cur.fetchall()]
                cur.close()
                conn.close()
                if rows:
                    context = format_rows_as_context(rows, "FEC")
                    answer = generate_answer_from_data(question, context, "FEC political finance")
                    citations = [Citation(
                        source="FEC", file_name="fec_committees (PostgreSQL)",
                        org_name=r.get("CMTE_NM", ""), ein="", object_id="",
                        snippet=f"State: {r.get('CMTE_ST','N/A')} | Receipts: {r.get('TTL_RECEIPTS','N/A')} | Cycle: {r.get('cycle','N/A')}",
                        distance=0.0,
                    ) for r in rows[:5]]
                    return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
            except Exception as e:
                print(f"FEC state query failed: {e}, falling back to RAG")

        # IRS geographic question
        elif dataset in ("irs", "both"):
            try:
                rows = query_irs_by_state(state_abbr)
                if rows:
                    context = format_rows_as_context(rows, "IRS Financials")
                    answer = generate_answer_from_data(question, context, "IRS nonprofit filings")
                    citations = [Citation(
                        source="IRS", file_name="irs_financials (PostgreSQL)",
                        org_name=r.get("org_name", ""), ein="", object_id="",
                        snippet=f"State: {r.get('state','N/A')} | Revenue: {r.get('total_revenue','N/A')} | Return: {r.get('return_type','N/A')}",
                        distance=0.0,
                    ) for r in rows[:5]]
                    return RAGResponse(answer=answer, citations=citations, sources_used=["IRS Financials (PostgreSQL)"])
            except Exception as e:
                print(f"IRS state query failed: {e}, falling back to RAG")

    # ── 2. Specific FEC committee lookup ──
    matched_committee = next((c for c in SPECIFIC_COMMITTEES if c in q_lower), None)
    if matched_committee and use_fec and dataset in ("fec", "both"):
        try:
            rows = query_fec_specific_committee(matched_committee)
            if rows:
                context = format_rows_as_context(rows, "FEC")
                answer = generate_answer_from_data(question, context, "FEC political finance")
                citations = [Citation(
                    source="FEC", file_name="fec_committees (PostgreSQL)",
                    org_name=r.get("CMTE_NM", ""), ein="", object_id="",
                    snippet=f"Receipts: {r.get('TTL_RECEIPTS','N/A')} | Disbursements: {r.get('TTL_DISB','N/A')} | Cycle: {r.get('cycle','N/A')}",
                    distance=0.0,
                ) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"Specific committee query failed: {e}, falling back to RAG")

    # ── 3. Threshold questions: over 100 million etc ──
    matched_threshold = next((v for k, v in THRESHOLD_MAP.items() if k in q_lower), None)
    if matched_threshold and use_fec and dataset in ("fec", "both"):
        try:
            field = "TTL_DISB" if any(w in q_lower for w in ["spent", "disbursement"]) else "TTL_RECEIPTS"
            rows = query_fec_threshold(matched_threshold, field)
            if rows:
                context = format_rows_as_context(rows, "FEC")
                answer = generate_answer_from_data(question, context, "FEC political finance")
                citations = [Citation(
                    source="FEC", file_name="fec_committees (PostgreSQL)",
                    org_name=r.get("CMTE_NM", ""), ein="", object_id="",
                    snippet=f"Receipts: {r.get('TTL_RECEIPTS','N/A')} | Disbursements: {r.get('TTL_DISB','N/A')} | Cycle: {r.get('cycle','N/A')}",
                    distance=0.0,
                ) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"Threshold query failed: {e}, falling back to RAG")

    # ── 4. Financial IRS questions ──
    if use_db and dataset in ("irs", "both") and not use_fec:
        try:
            rows = query_irs_financials(question)
            if rows:
                context = format_rows_as_context(rows, "IRS Financials")
                answer = generate_answer_from_data(question, context, "IRS nonprofit finance")
                citations = [Citation(
                    source="IRS", file_name="irs_financials (PostgreSQL)",
                    org_name=r.get("org_name", ""), ein="", object_id="",
                    snippet=f"Revenue: {r.get('total_revenue','N/A')} | Expenses: {r.get('total_expenses','N/A')} | State: {r.get('state','N/A')}",
                    distance=0.0,
                ) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations, sources_used=["IRS Financials (PostgreSQL)"])
        except Exception as e:
            print(f"PostgreSQL query failed: {e}, falling back to RAG")

    # ── 5. Financial FEC questions ──
    if use_db and use_fec and dataset in ("fec", "both"):
        try:
            rows = query_fec_committees(question)
            if rows:
                context = format_rows_as_context(rows, "FEC")
                answer = generate_answer_from_data(question, context, "FEC political finance")
                citations = [Citation(
                    source="FEC", file_name="fec_committees (PostgreSQL)",
                    org_name=r.get("CMTE_NM", ""), ein="", object_id="",
                    snippet=f"Receipts: {r.get('TTL_RECEIPTS','N/A')} | Disbursements: {r.get('TTL_DISB','N/A')} | Cycle: {r.get('cycle','N/A')}",
                    distance=0.0,
                ) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"FEC PostgreSQL query failed: {e}, falling back to RAG")

    # ── 6. Default: ChromaDB RAG ──
    return rag_ask(question, dataset=dataset, top_k=top_k)
