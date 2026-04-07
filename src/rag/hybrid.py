"""
hybrid.py - v3

Hybrid query engine that routes questions to the right data source:
- Geographic questions → PostgreSQL irs_locations + irs_financials
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
    "which universit", "which foundations", "which charities",
    "which colleges", "which schools", "which health", "which medical",
    "which arts", "which housing", "which youth", "which children",
    "which veterans", "which environmental", "which research",
    "which community", "which social", "which education",
    "rank", "ranking", "top 10", "top 5", "list of",
    "how much did", "how much has", "how much money",
    "over 100 million", "over 1 billion", "over 10 million",
    "more than", "at least", "based in", "located in",
    "cash on hand", "how much cash", "most cash", "individual contribution", "most individual",
    "contributions and grants", "program service revenue", "executives over", "pay executives",
    "filed 990", "990pf", "990ez", "990t",
    "connections to", "linked to", "associated with",
]

FEC_KEYWORDS = [
    "pac", "committee", "campaign", "political", "fec",
    "donation", "expenditure", "disbursement",
    "election", "candidate", "party", "super pac",
    "actblue", "winred", "harris", "trump", "dnc", "rnc",
    "democratic", "republican", "lobbyist", "house campaign",
    "senate campaign", "presidential campaign", "lincoln project",
    "individual contributions", "most individual", "individual contribution", "lobbyist pac", "lobbyist", "fec committees",
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
    "vermont": "VT", "virginia": "VA", "washington state": "WA", "nonprofits in washington": "WA", "based in washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}

SPECIFIC_COMMITTEES = [
    "actblue", "winred", "harris for president", "harris victory",
    "trump", "biden", "dnc", "rnc", "lincoln project",
    "emily", "planned parenthood", "nra", "america first",
    "priorities usa", "club for growth", "maga inc", "fairshake", "rnc", "republican national committee", "republican national",
    "democracy pac", "slf pac", "harris victory fund",
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


def query_cross_dataset(question):
    """Find organizations that appear in both IRS and FEC datasets."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT 
                f.org_name as irs_name,
                f.state,
                ROUND(f.total_revenue) as irs_revenue,
                ROUND(f.total_assets) as irs_assets,
                c."CMTE_NM" as fec_name,
                c."CMTE_TP" as committee_type,
                c."TTL_RECEIPTS" as fec_receipts,
                c."TTL_DISB" as fec_disbursements,
                c."cycle"
            FROM irs_financials f
            JOIN fec_committees c 
                ON LOWER(f.org_name) = LOWER(c."CMTE_NM")
            WHERE f.total_revenue IS NOT NULL
              AND c."TTL_RECEIPTS" ~ '^[0-9.]+$'
            ORDER BY f.total_revenue DESC
            LIMIT 15
        """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        cur.close()
        conn.close()


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
            WHERE UPPER(f.state) = %s AND f.total_revenue IS NOT NULL
            ORDER BY f.total_revenue DESC LIMIT 10
        """, (state_abbr,))
        financial_rows = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT org_name, state, city, return_type, tax_year
            FROM irs_locations
            WHERE UPPER(state) = %s
            ORDER BY org_name LIMIT 15
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
            ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 5
        """, (f"%{committee_name.lower()}%",))
        return [dict(r) for r in cur.fetchall()]
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
            ORDER BY CAST("{field}" AS NUMERIC) DESC LIMIT 20
        """, (amount,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def query_irs_financials(question):
    q = question.lower()
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Return type queries — use irs_index
        if any(w in q for w in ["990pf", "990ez", "990t", "990 return"]) or "filed 990" in q:
            if "990pf" in q: rt = "990PF"
            elif "990ez" in q: rt = "990EZ"
            elif "990t" in q: rt = "990T"
            else: rt = "990"
            cur.execute("""
                SELECT "TAXPAYER_NAME" as org_name, "RETURN_TYPE" as return_type,
                       "TAX_PERIOD" as tax_year, "EIN" as ein
                FROM irs_index WHERE UPPER("RETURN_TYPE") = %s
                ORDER BY "TAXPAYER_NAME" LIMIT 20
            """, (rt,))

        # Contributions and grants
        elif any(w in q for w in ["contributions and grants", "most contributions", "most grants", "giving", "contributions and grant"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(contributions_grants) as contributions_grants,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE contributions_grants IS NOT NULL
                ORDER BY contributions_grants DESC LIMIT 15
            """)

        # Program service revenue
        elif any(w in q for w in ["program service", "program revenue", "service revenue"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(program_service_revenue) as program_service_revenue,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE program_service_revenue IS NOT NULL
                ORDER BY program_service_revenue DESC LIMIT 15
            """)

        # Officer compensation with threshold
        elif any(w in q for w in ["compensation", "salary", "officer", "executive", "pay"]):
            if any(w in q for w in ["over 1 million", "1 million dollar", "officers over", "over a million"]):
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials
                    WHERE officer_compensation IS NOT NULL AND officer_compensation >= 1000000
                    ORDER BY officer_compensation DESC LIMIT 15
                """)
            elif any(w in q for w in ["500 thousand", "500k", "over 500", "executives over"]):
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials
                    WHERE officer_compensation IS NOT NULL AND officer_compensation >= 500000
                    ORDER BY officer_compensation DESC LIMIT 15
                """)
            else:
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE officer_compensation IS NOT NULL
                    ORDER BY officer_compensation DESC LIMIT 15
                """)

        # Arts organizations
        elif any(w in q for w in ["arts", "museum", "theater", "theatre", "cultural", "symphony", "opera"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_assets IS NOT NULL
                AND (LOWER(org_name) LIKE '%art%' OR LOWER(org_name) LIKE '%museum%'
                     OR LOWER(org_name) LIKE '%theater%' OR LOWER(org_name) LIKE '%symphony%'
                     OR LOWER(org_name) LIKE '%cultural%' OR LOWER(org_name) LIKE '%theatre%'
                     OR LOWER(org_name) LIKE '%opera%')
                ORDER BY total_assets DESC LIMIT 15
            """)

        # Housing nonprofits
        elif any(w in q for w in ["housing", "shelter", "homeless"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_liabilities IS NOT NULL
                AND (LOWER(org_name) LIKE '%housing%' OR LOWER(org_name) LIKE '%shelter%'
                     OR LOWER(org_name) LIKE '%homeless%' OR LOWER(org_name) LIKE '%habitat%')
                ORDER BY total_liabilities DESC LIMIT 15
            """)

        # Youth / children organizations
        elif any(w in q for w in ["youth", "children", "child", "kids", "juvenile"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_assets IS NOT NULL
                AND (LOWER(org_name) LIKE '%youth%' OR LOWER(org_name) LIKE '%children%'
                     OR LOWER(org_name) LIKE '%child%' OR LOWER(org_name) LIKE '%boys%'
                     OR LOWER(org_name) LIKE '%girls%' OR LOWER(org_name) LIKE '%kids%'
                     OR LOWER(org_name) LIKE '%juvenile%')
                ORDER BY total_assets DESC LIMIT 15
            """)

        # Educational institutions with debt
        elif ("debt" in q or "liabilit" in q) and any(w in q for w in ["education", "school", "university", "college", "institution"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_liabilities IS NOT NULL
                AND (LOWER(org_name) LIKE '%school%' OR LOWER(org_name) LIKE '%university%'
                     OR LOWER(org_name) LIKE '%college%' OR LOWER(org_name) LIKE '%academy%'
                     OR LOWER(org_name) LIKE '%institute%')
                ORDER BY total_liabilities DESC LIMIT 15
            """)

        # Community foundations
        elif "community foundation" in q:
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets,
                       ROUND(contributions_grants) as contributions_grants
                FROM irs_financials WHERE total_revenue IS NOT NULL
                AND LOWER(org_name) LIKE '%community foundation%'
                ORDER BY total_revenue DESC LIMIT 15
            """)

        # University / college — must come before generic asset check
        elif any(w in q for w in ["universit", "college"]):
            order_col = "total_assets" if "asset" in q else "total_revenue"
            cur.execute(f"""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE {order_col} IS NOT NULL
                AND (LOWER(org_name) LIKE '%universit%' OR LOWER(org_name) LIKE '%college%')
                ORDER BY {order_col} DESC LIMIT 15
            """)
        # Veterans — must come before generic asset check
        elif any(w in q for w in ["veteran", "vfw", "american legion"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_assets IS NOT NULL
                AND (LOWER(org_name) LIKE '%veteran%' OR LOWER(org_name) LIKE '%vfw%'
                     OR LOWER(org_name) LIKE '%american legion%')
                ORDER BY total_assets DESC LIMIT 15
            """)
        # Environmental — must come before generic revenue check
        elif any(w in q for w in ["environment", "conservation", "wildlife"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_revenue IS NOT NULL
                AND (LOWER(org_name) LIKE '%environment%' OR LOWER(org_name) LIKE '%conservation%'
                     OR LOWER(org_name) LIKE '%wildlife%' OR LOWER(org_name) LIKE '%nature%')
                ORDER BY total_revenue DESC LIMIT 15
            """)
        # Revenue queries
        elif any(w in q for w in ["revenue", "raised", "money", "income", "funding"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_revenue IS NOT NULL
                ORDER BY total_revenue DESC LIMIT 15
            """)

        # Expense queries
        elif any(w in q for w in ["expense", "spent", "spending", "cost"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_expenses IS NOT NULL
                ORDER BY total_expenses DESC LIMIT 15
            """)

        # Asset queries
        elif any(w in q for w in ["asset", "worth", "wealth", "large"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(net_assets) as net_assets
                FROM irs_financials WHERE total_assets IS NOT NULL
                ORDER BY total_assets DESC LIMIT 15
            """)

        # Liabilities queries
        elif any(w in q for w in ["liabilit", "debt", "owe"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_liabilities IS NOT NULL
                  AND total_liabilities > 0
                ORDER BY total_liabilities DESC LIMIT 15
            """)

        # Surplus / deficit
        elif any(w in q for w in ["surplus", "profit", "loss", "deficit"]):
            if "loss" in q or "deficit" in q:
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(total_revenue) as total_revenue,
                           ROUND(total_expenses) as total_expenses,
                           ROUND(total_revenue - total_expenses) as net_income
                    FROM irs_financials
                    WHERE total_revenue IS NOT NULL AND total_expenses IS NOT NULL
                      AND total_revenue < total_expenses
                    ORDER BY (total_revenue - total_expenses) ASC LIMIT 15
                """)
            else:
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(total_revenue) as total_revenue,
                           ROUND(total_expenses) as total_expenses,
                           ROUND(total_revenue - total_expenses) as net_income
                    FROM irs_financials
                    WHERE total_revenue IS NOT NULL AND total_expenses IS NOT NULL
                      AND total_revenue > total_expenses
                    ORDER BY (total_revenue - total_expenses) DESC LIMIT 15
                """)

        # Hospital / medical / health
        elif any(w in q for w in ["hospital", "medical", "health system", "healthcare"]):
            order_col = "total_assets" if "asset" in q else "total_liabilities" if "liabilit" in q else "total_revenue"
            cur.execute(f"""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_liabilities) as total_liabilities
                FROM irs_financials WHERE {order_col} IS NOT NULL
                AND (LOWER(org_name) LIKE '%hospital%' OR LOWER(org_name) LIKE '%medical%'
                     OR LOWER(org_name) LIKE '%health%' OR LOWER(org_name) LIKE '%healthcare%')
                ORDER BY {order_col} DESC LIMIT 15
            """)

        # University / college
        elif any(w in q for w in ["universit", "college", "school", "education"]):
            order_col = "total_assets" if "asset" in q else "total_revenue"
            cur.execute(f"""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE {order_col} IS NOT NULL
                AND (LOWER(org_name) LIKE '%university%' OR LOWER(org_name) LIKE '%college%'
                     OR LOWER(org_name) LIKE '%school%' OR LOWER(org_name) LIKE '%institute%')
                ORDER BY {order_col} DESC LIMIT 15
            """)

        # Foundation / charity
        elif any(w in q for w in ["foundation", "charity", "charitable"]):
            if "net asset" in q:
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(net_assets) as net_assets,
                           ROUND(total_assets) as total_assets
                    FROM irs_financials
                    WHERE (LOWER(org_name) LIKE '%foundation%' OR return_type = '990PF')
                      AND net_assets IS NOT NULL
                    ORDER BY net_assets DESC LIMIT 15
                """)
            elif "asset" in q:
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(total_assets) as total_assets,
                           ROUND(net_assets) as net_assets
                    FROM irs_financials
                    WHERE (LOWER(org_name) LIKE '%foundation%' OR return_type = '990PF')
                      AND total_assets IS NOT NULL
                    ORDER BY total_assets DESC LIMIT 15
                """)
            elif "contribution" in q or "grant" in q:
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(contributions_grants) as contributions_grants,
                           ROUND(total_assets) as total_assets
                    FROM irs_financials
                    WHERE (LOWER(org_name) LIKE '%foundation%' OR return_type = '990PF')
                      AND contributions_grants IS NOT NULL
                    ORDER BY contributions_grants DESC LIMIT 15
                """)
            else:
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(total_revenue) as total_revenue,
                           ROUND(total_assets) as total_assets,
                           ROUND(net_assets) as net_assets
                    FROM irs_financials
                    WHERE (LOWER(org_name) LIKE '%foundation%' OR return_type = '990PF')
                      AND total_revenue IS NOT NULL
                    ORDER BY total_revenue DESC LIMIT 15
                """)

        # Research organizations
        elif any(w in q for w in ["research", "institute", "science", "technology"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_assets IS NOT NULL
                AND (LOWER(org_name) LIKE '%research%' OR LOWER(org_name) LIKE '%institute%'
                     OR LOWER(org_name) LIKE '%science%' OR LOWER(org_name) LIKE '%technology%')
                ORDER BY total_assets DESC LIMIT 15
            """)

        # Veterans organizations
        elif any(w in q for w in ["veteran", "military", "vfw", "american legion"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_assets IS NOT NULL
                AND (LOWER(org_name) LIKE '%veteran%' OR LOWER(org_name) LIKE '%vfw%'
                     OR LOWER(org_name) LIKE '%american legion%' OR LOWER(org_name) LIKE '%military%')
                ORDER BY total_assets DESC LIMIT 15
            """)

        # Environmental organizations
        elif any(w in q for w in ["environment", "conservation", "wildlife", "nature"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_revenue IS NOT NULL
                AND (LOWER(org_name) LIKE '%environment%' OR LOWER(org_name) LIKE '%conservation%'
                     OR LOWER(org_name) LIKE '%wildlife%' OR LOWER(org_name) LIKE '%nature%'
                     OR LOWER(org_name) LIKE '%ecology%')
                ORDER BY total_revenue DESC LIMIT 15
            """)

        # Social service organizations
        elif any(w in q for w in ["social service", "human service", "community service"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_revenue IS NOT NULL
                AND (LOWER(org_name) LIKE '%service%' OR LOWER(org_name) LIKE '%community%'
                     OR LOWER(org_name) LIKE '%social%' OR LOWER(org_name) LIKE '%human%')
                ORDER BY total_revenue DESC LIMIT 15
            """)

        # Default — top by revenue
        else:
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_revenue IS NOT NULL
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
        if any(w in q for w in ["house campaign", "house committee"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" = 'H'
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["senate campaign", "senate committee"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" = 'S'
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["presidential campaign", "presidential committee"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" = 'P'
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["lobbyist", "registrant"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE (LOWER("CMTE_NM") LIKE '%lobbyist%'
                       OR LOWER("CMTE_NM") LIKE '%registrant%'
                       OR "CMTE_TP" IN ('V', 'W'))
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["independent expenditure", "super pac"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" IN ('O', 'U', 'N')
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        elif any(w in q for w in ["spent", "spending", "expenditure", "disbursement"]):
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
        elif any(w in q for w in ["individual contribution", "individual donor", "indv"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "INDV_CONTB", "TTL_RECEIPTS", "cycle"
                FROM fec_committees
                WHERE "INDV_CONTB" IS NOT NULL AND "INDV_CONTB" != ''
                  AND "INDV_CONTB" ~ '^[0-9.]+$'
                ORDER BY CAST("INDV_CONTB" AS NUMERIC) DESC LIMIT 15
            """)
        elif "democratic" in q:
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE ("CMTE_PTY_AFFILIATION" = 'DEM' OR "CAND_PTY_AFFILIATION" = 'DEM'
                       OR LOWER("CMTE_NM") LIKE '%democrat%')
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        elif "republican" in q:
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE ("CMTE_PTY_AFFILIATION" = 'REP' OR "CAND_PTY_AFFILIATION" = 'REP'
                       OR LOWER("CMTE_NM") LIKE '%republican%')
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        else:
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST", "CMTE_CITY",
                       "TTL_RECEIPTS", "TTL_DISB", "INDV_CONTB", "COH_COP", "cycle"
                FROM fec_committees
                WHERE "TTL_RECEIPTS" IS NOT NULL AND "TTL_RECEIPTS" != ''
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)
        return [dict(r) for r in cur.fetchall()]
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
        "You are answering from a curated sample of government filings.\n"
        "Answer the question using the data provided below.\n"
        "Be specific and cite data with [1], [2] etc.\n"
        "Format numbers clearly (e.g. $3.1 billion, $450 million).\n"
        "Present findings confidently — say 'Based on our dataset, the top organizations are...' not 'only X organizations are in this dataset'.\n"
        "If data is limited, say 'Among the organizations in our sample...' and still give the best answer.\n"
        "Never say 'I cannot answer' if there is any relevant data — always provide the best available answer.\n"
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

    # ── 0. Cross-dataset queries ──
    cross_keywords = ["connections to", "linked to", "associated with", "both irs and fec",
                      "nonprofit and political", "appear in both", "cross dataset", "overlap"]
    if any(kw in q_lower for kw in cross_keywords):
        try:
            rows = query_cross_dataset(question)
            if rows:
                context = format_rows_as_context(rows, "IRS+FEC")
                answer = generate_answer_from_data(question, context,
                    "cross-dataset IRS nonprofit and FEC political finance")
                citations = [Citation(
                    source="IRS",
                    file_name="irs_financials + fec_committees (PostgreSQL)",
                    org_name=r.get("irs_name", ""),
                    ein="", object_id="",
                    snippet=f"IRS Revenue: {r.get('irs_revenue','N/A')} | FEC Receipts: {r.get('fec_receipts','N/A')} | State: {r.get('state','N/A')}",
                    distance=0.0,
                ) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations,
                    sources_used=["IRS + FEC Cross-Dataset (PostgreSQL)"])
        except Exception as e:
            print(f"Cross-dataset query failed: {e}, falling back to RAG")

    # ── 1. Geographic: query by state ──
    state_abbr, state_name = detect_state(question)
    if state_abbr:
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
                    ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 20
                """, (state_abbr,))
                rows = [dict(r) for r in cur.fetchall()]
                cur.close(); conn.close()
                if rows:
                    context = format_rows_as_context(rows, "FEC")
                    answer = generate_answer_from_data(question, context, "FEC political finance")
                    citations = [Citation(source="FEC", file_name="fec_committees (PostgreSQL)",
                        org_name=r.get("CMTE_NM",""), ein="", object_id="",
                        snippet=f"State: {r.get('CMTE_ST','N/A')} | Receipts: {r.get('TTL_RECEIPTS','N/A')}",
                        distance=0.0) for r in rows[:5]]
                    return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
            except Exception as e:
                print(f"FEC state query failed: {e}, falling back to RAG")
        elif dataset in ("irs", "both"):
            try:
                rows = query_irs_by_state(state_abbr)
                if rows:
                    context = format_rows_as_context(rows, "IRS Financials")
                    answer = generate_answer_from_data(question, context, "IRS nonprofit filings")
                    citations = [Citation(source="IRS", file_name="irs_financials (PostgreSQL)",
                        org_name=r.get("org_name",""), ein="", object_id="",
                        snippet=f"State: {r.get('state','N/A')} | Revenue: {r.get('total_revenue','N/A')}",
                        distance=0.0) for r in rows[:5]]
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
                citations = [Citation(source="FEC", file_name="fec_committees (PostgreSQL)",
                    org_name=r.get("CMTE_NM",""), ein="", object_id="",
                    snippet=f"Receipts: {r.get('TTL_RECEIPTS','N/A')} | Disbursements: {r.get('TTL_DISB','N/A')}",
                    distance=0.0) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"Specific committee query failed: {e}, falling back to RAG")

    # ── 3. Threshold questions ──
    matched_threshold = next((v for k, v in THRESHOLD_MAP.items() if k in q_lower), None)
    if matched_threshold and use_fec and dataset in ("fec", "both"):
        try:
            field = "TTL_DISB" if any(w in q_lower for w in ["spent", "disbursement"]) else "TTL_RECEIPTS"
            rows = query_fec_threshold(matched_threshold, field)
            if rows:
                context = format_rows_as_context(rows, "FEC")
                answer = generate_answer_from_data(question, context, "FEC political finance")
                citations = [Citation(source="FEC", file_name="fec_committees (PostgreSQL)",
                    org_name=r.get("CMTE_NM",""), ein="", object_id="",
                    snippet=f"Receipts: {r.get('TTL_RECEIPTS','N/A')} | Disbursements: {r.get('TTL_DISB','N/A')}",
                    distance=0.0) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"Threshold query failed: {e}, falling back to RAG")

    # ── 4. Financial IRS questions ──
    if use_db and dataset in ("irs", "both") and (not use_fec or "executive" in q_lower or "officer" in q_lower):
        try:
            rows = query_irs_financials(question)
            if rows:
                context = format_rows_as_context(rows, "IRS Financials")
                answer = generate_answer_from_data(question, context, "IRS nonprofit finance")
                citations = [Citation(source="IRS", file_name="irs_financials (PostgreSQL)",
                    org_name=r.get("org_name",""), ein="", object_id="",
                    snippet=f"Revenue: {r.get('total_revenue','N/A')} | State: {r.get('state','N/A')}",
                    distance=0.0) for r in rows[:5]]
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
                citations = [Citation(source="FEC", file_name="fec_committees (PostgreSQL)",
                    org_name=r.get("CMTE_NM",""), ein="", object_id="",
                    snippet=f"Receipts: {r.get('TTL_RECEIPTS','N/A')} | Disbursements: {r.get('TTL_DISB','N/A')}",
                    distance=0.0) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations, sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"FEC PostgreSQL query failed: {e}, falling back to RAG")

    # ── 6. Default: ChromaDB RAG ──
    return rag_ask(question, dataset=dataset, top_k=top_k)
