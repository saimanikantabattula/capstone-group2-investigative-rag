"""
agent_filter.py
===============
The Filter Agent handles all structured data queries against PostgreSQL.

Think of it like a specialized accountant who knows exactly which SQL query
to run for every type of financial question.

Responsibilities:
- Financial ranking queries (top nonprofits by revenue, assets, expenses etc.)
- Geographic queries (nonprofits in California, FEC committees in New York)
- Threshold queries (organizations that raised over 100 million)
- Specific committee lookups (ActBlue, WinRed, Harris for President)
- Cross-dataset queries (find orgs that appear in BOTH IRS and FEC data)

This agent uses SQL directly — no vector embeddings, no AI, just database queries.
It is fast and accurate for structured/numeric data questions.
"""

import os
import sys
import psycopg2
import psycopg2.extras

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── DATABASE CONNECTION SETTINGS ──────────────────────────────────────────────
# Read from environment variables — never hardcode passwords in code
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "capstone_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")

# ── STATE NAME TO ABBREVIATION MAP ────────────────────────────────────────────
# Converts full state names to 2-letter abbreviations for SQL queries
# Our database stores states as "CA", "NY", "TX" etc.
STATE_MAP = {
    "california": "CA", "new york": "NY", "texas": "TX", "florida": "FL",
    "illinois": "IL", "pennsylvania": "PA", "ohio": "OH", "georgia": "GA",
    "michigan": "MI", "north carolina": "NC", "new jersey": "NJ",
    "virginia": "VA", "washington": "WA", "massachusetts": "MA",
    "tennessee": "TN", "colorado": "CO", "maryland": "MD",
    "indiana": "IN", "minnesota": "MN", "arizona": "AZ",
}

# ── SPECIFIC COMMITTEE NAMES ──────────────────────────────────────────────────
# Well-known political committee names for direct name lookup
# If any of these appear in the question, we do a targeted name search
SPECIFIC_COMMITTEES = [
    "actblue", "winred", "harris for president", "harris victory",
    "trump", "biden", "dnc", "rnc", "lincoln project",
    "emily", "planned parenthood", "nra", "america first",
    "priorities usa", "club for growth",
]


class FilterAgent:
    """
    Handles structured data queries against PostgreSQL.
    Routes different question types to the most appropriate SQL query.

    Our database tables:
    - irs_financials  (378,272 rows) — financial data for nonprofits
    - irs_locations   (1,216,026 rows) — location data for all 1.2M orgs
    - irs_index       (100,000 rows) — filing type and EIN data
    - fec_committees  (38,793 rows) — political committee financial data
    """

    def __init__(self):
        self.conn = None  # database connection (created per query)

    def get_connection(self):
        """
        Creates a fresh PostgreSQL connection.
        We create a new connection for each query to avoid connection issues
        on the Supabase free tier which has limited concurrent connections.
        """
        return psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS
        )

    def detect_state(self, question):
        """
        Checks if a US state name appears in the question.
        Returns the 2-letter abbreviation or None.
        Example: "Which nonprofits are in California?" → "CA"
        """
        q = question.lower()
        for state_name, abbr in STATE_MAP.items():
            if state_name in q:
                return abbr
        return None

    def detect_committee(self, question):
        """
        Checks if a specific political committee name appears in the question.
        Returns the matched committee name or None.
        Example: "How much did ActBlue raise?" → "actblue"
        """
        q = question.lower()
        return next((c for c in SPECIFIC_COMMITTEES if c in q), None)

    def query_irs_financial(self, question):
        """
        Runs the appropriate IRS financial SQL query based on question keywords.

        Handles these question types:
        - Contributions and grants  → ORDER BY contributions_grants
        - Program service revenue   → ORDER BY program_service_revenue
        - Officer compensation      → ORDER BY officer_compensation
        - Total assets              → ORDER BY total_assets
        - Total expenses            → ORDER BY total_expenses
        - Liabilities/debt          → ORDER BY total_liabilities
        - Default (revenue)         → ORDER BY total_revenue

        Returns: list of row dictionaries with org financial data
        """
        q    = question.lower()
        conn = self.get_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            if any(w in q for w in ["contribution", "grant", "giving"]):
                # Questions like "Which nonprofits received the most donations?"
                cur.execute("""
                    SELECT org_name, state,
                           ROUND(contributions_grants) as contributions_grants,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE contributions_grants IS NOT NULL
                    ORDER BY contributions_grants DESC LIMIT 15
                """)

            elif any(w in q for w in ["program service", "program revenue"]):
                # Questions like "Which nonprofits have the most program service revenue?"
                cur.execute("""
                    SELECT org_name, state,
                           ROUND(program_service_revenue) as program_service_revenue,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE program_service_revenue IS NOT NULL
                    ORDER BY program_service_revenue DESC LIMIT 15
                """)

            elif any(w in q for w in ["compensation", "salary", "officer", "executive", "pay"]):
                # Questions like "Which nonprofits pay executives the most?"
                cur.execute("""
                    SELECT org_name, state,
                           ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE officer_compensation IS NOT NULL
                    ORDER BY officer_compensation DESC LIMIT 15
                """)

            elif any(w in q for w in ["asset", "worth", "wealth"]):
                # Questions like "Which nonprofits have the most assets?"
                cur.execute("""
                    SELECT org_name, state,
                           ROUND(total_assets) as total_assets,
                           ROUND(total_liabilities) as total_liabilities,
                           ROUND(net_assets) as net_assets
                    FROM irs_financials WHERE total_assets IS NOT NULL
                    ORDER BY total_assets DESC LIMIT 15
                """)

            elif any(w in q for w in ["expense", "spent", "spending"]):
                # Questions like "Which nonprofits spent the most money?"
                cur.execute("""
                    SELECT org_name, state,
                           ROUND(total_expenses) as total_expenses,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE total_expenses IS NOT NULL
                    ORDER BY total_expenses DESC LIMIT 15
                """)

            elif any(w in q for w in ["liabilit", "debt"]):
                # Questions like "Which nonprofits have the most debt?"
                cur.execute("""
                    SELECT org_name, state,
                           ROUND(total_liabilities) as total_liabilities,
                           ROUND(total_assets) as total_assets
                    FROM irs_financials
                    WHERE total_liabilities IS NOT NULL AND total_liabilities > 0
                    ORDER BY total_liabilities DESC LIMIT 15
                """)

            else:
                # Default: rank by total revenue
                # Handles: "Which nonprofits raised the most money?"
                cur.execute("""
                    SELECT org_name, state,
                           ROUND(total_revenue) as total_revenue,
                           ROUND(total_expenses) as total_expenses,
                           ROUND(total_assets) as total_assets
                    FROM irs_financials WHERE total_revenue IS NOT NULL
                    ORDER BY total_revenue DESC LIMIT 15
                """)

            return [dict(r) for r in cur.fetchall()]

        finally:
            cur.close()
            conn.close()

    def query_fec_financial(self, question):
        """
        Runs the appropriate FEC financial SQL query based on question keywords.

        Handles:
        - Spending/disbursements → ORDER BY TTL_DISB
        - Cash on hand           → ORDER BY COH_COP
        - Default (receipts)     → ORDER BY TTL_RECEIPTS

        Note: FEC column names are ALL CAPS because that is how the FEC data comes.
        TTL_RECEIPTS = Total Receipts (money raised)
        TTL_DISB     = Total Disbursements (money spent)
        COH_COP      = Cash on Hand at Close of Period
        """
        q    = question.lower()
        conn = self.get_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            if any(w in q for w in ["spent", "spending", "disbursement"]):
                # Questions like "Which PACs spent the most money?"
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "TTL_DISB", "TTL_RECEIPTS", "cycle"
                    FROM fec_committees
                    WHERE "TTL_DISB" ~ '^[0-9.]+$'    -- only valid numeric values
                    ORDER BY CAST("TTL_DISB" AS NUMERIC) DESC LIMIT 15
                """)

            elif any(w in q for w in ["cash", "hand", "balance"]):
                # Questions like "Which committees have the most cash on hand?"
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "COH_COP", "TTL_RECEIPTS", "cycle"
                    FROM fec_committees
                    WHERE "COH_COP" ~ '^[0-9.]+$'
                    ORDER BY CAST("COH_COP" AS NUMERIC) DESC LIMIT 15
                """)

            else:
                # Default: rank by total receipts (money raised)
                # Handles: "Which PACs raised the most money?"
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "TTL_RECEIPTS", "TTL_DISB", "COH_COP", "cycle"
                    FROM fec_committees
                    WHERE "TTL_RECEIPTS" ~ '^[0-9.]+$'
                    ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
                """)

            return [dict(r) for r in cur.fetchall()]

        finally:
            cur.close()
            conn.close()

    def query_by_state(self, state_abbr, dataset):
        """
        Returns organizations from a specific US state.

        If dataset includes FEC → queries fec_committees by CMTE_ST
        Otherwise              → queries irs_financials by state column

        Returns: (rows_list, source_label_string)
        """
        conn = self.get_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            if dataset in ("fec", "both"):
                # Get FEC committees in this state
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "TTL_RECEIPTS", "TTL_DISB", "cycle"
                    FROM fec_committees
                    WHERE UPPER("CMTE_ST") = %s         -- match 2-letter state code
                    AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                    ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 20
                """, (state_abbr,))
                return [dict(r) for r in cur.fetchall()], "FEC Committees (PostgreSQL)"

            else:
                # Get IRS nonprofits in this state
                # JOIN with irs_locations to get city data
                cur.execute("""
                    SELECT f.org_name, f.state, l.city,
                           ROUND(f.total_revenue) as total_revenue,
                           ROUND(f.total_assets) as total_assets
                    FROM irs_financials f
                    LEFT JOIN irs_locations l USING (ein)  -- join on EIN (unique org ID)
                    WHERE UPPER(f.state) = %s
                    AND f.total_revenue IS NOT NULL
                    ORDER BY f.total_revenue DESC LIMIT 20
                """, (state_abbr,))
                return [dict(r) for r in cur.fetchall()], "IRS Financials (PostgreSQL)"

        finally:
            cur.close()
            conn.close()

    def query_cross_dataset(self):
        """
        Finds organizations that appear in BOTH IRS and FEC datasets.
        This is our key investigative feature.

        Does a SQL JOIN on org_name between irs_financials and fec_committees.
        When a nonprofit also has a political committee, it appears in both tables.

        Example finding: "Development Now for Chicago"
        - IRS:  $81.5M nonprofit revenue
        - FEC: $102M political receipts
        This kind of connection is what investigative journalists need.
        """
        conn = self.get_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("""
                SELECT
                    f.org_name as irs_name,                    -- name from IRS data
                    f.state,
                    ROUND(f.total_revenue) as irs_revenue,     -- revenue from IRS
                    ROUND(f.total_assets) as irs_assets,       -- assets from IRS
                    c."CMTE_NM" as fec_name,                   -- name from FEC data
                    c."CMTE_TP",                               -- committee type
                    c."TTL_RECEIPTS" as fec_receipts,          -- money raised from FEC
                    c."TTL_DISB" as fec_disbursements,         -- money spent from FEC
                    c."cycle"                                  -- election year
                FROM irs_financials f
                JOIN fec_committees c
                    ON LOWER(f.org_name) = LOWER(c."CMTE_NM") -- case-insensitive name match
                WHERE f.total_revenue IS NOT NULL
                  AND c."TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY f.total_revenue DESC LIMIT 15
            """)
            return [dict(r) for r in cur.fetchall()], "IRS + FEC Cross-Dataset (PostgreSQL)"
        finally:
            cur.close()
            conn.close()

    def run(self, question, dataset="both"):
        """
        Main entry point for FilterAgent.
        Classifies the question and runs the appropriate SQL query.

        Routing priority:
        1. Cross-dataset keywords → query_cross_dataset()
        2. State name detected    → query_by_state()
        3. Specific committee     → direct name lookup in fec_committees
        4. FEC keywords           → query_fec_financial()
        5. Default                → query_irs_financial()

        Returns a dict with:
        - rows:         list of database row dictionaries
        - source_label: human readable source name for citations
        """
        q = question.lower()
        print(f"[FilterAgent] Processing: {question[:55]}...")

        # ── Route 1: Cross-dataset query ──────────────────────────────────────
        # Example: "Which nonprofits have connections to political committees?"
        cross_keywords = ["connections to", "linked to", "associated with", "appear in both"]
        if any(kw in q for kw in cross_keywords):
            rows, label = self.query_cross_dataset()
            return {"rows": rows, "source_label": label}

        # ── Route 2: State-based geographic query ─────────────────────────────
        # Example: "Which nonprofits are based in California?"
        state = self.detect_state(question)
        if state:
            rows, label = self.query_by_state(state, dataset)
            return {"rows": rows, "source_label": label}

        # ── Route 3: Specific FEC committee lookup ─────────────────────────────
        # Example: "How much did ActBlue raise in 2024?"
        committee = self.detect_committee(question)
        if committee and dataset in ("fec", "both"):
            conn = self.get_connection()
            cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE LOWER("CMTE_NM") LIKE %s          -- partial name match
                AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 5
            """, (f"%{committee}%",))
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {"rows": rows, "source_label": "FEC Committees (PostgreSQL)"}

        # ── Route 4: FEC financial query ──────────────────────────────────────
        # Example: "Which PACs raised the most money in 2024?"
        fec_keywords = ["pac", "committee", "campaign", "fec", "actblue", "winred",
                        "democratic", "republican", "dnc", "rnc"]
        if any(kw in q for kw in fec_keywords) and dataset in ("fec", "both"):
            rows = self.query_fec_financial(question)
            return {"rows": rows, "source_label": "FEC Committees (PostgreSQL)"}

        # ── Route 5: IRS financial query (default) ─────────────────────────────
        # Example: "Which nonprofits raised the most money?"
        rows = self.query_irs_financial(question)
        return {"rows": rows, "source_label": "IRS Financials (PostgreSQL)"}


# ── Direct usage for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    agent  = FilterAgent()
    result = agent.run("Which nonprofits raised the most money?", dataset="irs")
    print(f"Got {len(result['rows'])} rows from {result['source_label']}")
    for r in result["rows"][:3]:
        print(r)
