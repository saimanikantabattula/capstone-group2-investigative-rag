"""
filter_agent.py

The Filter Agent handles all structured data queries against PostgreSQL.
It is responsible for:
- Financial ranking queries (top nonprofits by revenue, assets, etc.)
- Geographic queries (nonprofits in California, FEC committees in NY)
- Threshold queries (organizations over 100 million)
- Specific committee lookups (ActBlue, WinRed, Harris)
- Cross-dataset queries (IRS + FEC JOIN)

This agent uses SQL directly — no vector embeddings needed.
"""

import os
import sys
import psycopg2
import psycopg2.extras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "capstone_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")

# State name to abbreviation mapping
STATE_MAP = {
    "california": "CA", "new york": "NY", "texas": "TX", "florida": "FL",
    "illinois": "IL", "pennsylvania": "PA", "ohio": "OH", "georgia": "GA",
    "michigan": "MI", "north carolina": "NC", "new jersey": "NJ",
    "virginia": "VA", "washington": "WA", "massachusetts": "MA",
    "tennessee": "TN", "colorado": "CO", "maryland": "MD",
    "indiana": "IN", "minnesota": "MN", "arizona": "AZ",
}

# Specific committees to look up by name
SPECIFIC_COMMITTEES = [
    "actblue", "winred", "harris for president", "harris victory",
    "trump", "biden", "dnc", "rnc", "lincoln project",
    "emily", "planned parenthood", "nra", "america first",
    "priorities usa", "club for growth",
]


class FilterAgent:
    """
    Handles structured data queries against PostgreSQL.
    Routes different question types to appropriate SQL queries.
    """

    def __init__(self):
        self.conn = None

    def get_connection(self):
        return psycopg2.connect(
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS
        )

    def detect_state(self, question):
        q = question.lower()
        for state_name, abbr in STATE_MAP.items():
            if state_name in q:
                return abbr
        return None

    def detect_committee(self, question):
        q = question.lower()
        return next((c for c in SPECIFIC_COMMITTEES if c in q), None)

    def query_irs_financial(self, question):
        """Query irs_financials table for financial ranking questions."""
        q = question.lower()
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            if any(w in q for w in ["contribution", "grant", "giving"]):
                cur.execute("""
                    SELECT org_name, state, ROUND(contributions_grants) as contributions_grants,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE contributions_grants IS NOT NULL
                    ORDER BY contributions_grants DESC LIMIT 15
                """)
            elif any(w in q for w in ["program service", "program revenue"]):
                cur.execute("""
                    SELECT org_name, state, ROUND(program_service_revenue) as program_service_revenue,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE program_service_revenue IS NOT NULL
                    ORDER BY program_service_revenue DESC LIMIT 15
                """)
            elif any(w in q for w in ["compensation", "salary", "officer", "executive", "pay"]):
                cur.execute("""
                    SELECT org_name, state, ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE officer_compensation IS NOT NULL
                    ORDER BY officer_compensation DESC LIMIT 15
                """)
            elif any(w in q for w in ["asset", "worth", "wealth"]):
                cur.execute("""
                    SELECT org_name, state, ROUND(total_assets) as total_assets,
                           ROUND(total_liabilities) as total_liabilities,
                           ROUND(net_assets) as net_assets
                    FROM irs_financials WHERE total_assets IS NOT NULL
                    ORDER BY total_assets DESC LIMIT 15
                """)
            elif any(w in q for w in ["expense", "spent", "spending"]):
                cur.execute("""
                    SELECT org_name, state, ROUND(total_expenses) as total_expenses,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE total_expenses IS NOT NULL
                    ORDER BY total_expenses DESC LIMIT 15
                """)
            elif any(w in q for w in ["liabilit", "debt"]):
                cur.execute("""
                    SELECT org_name, state, ROUND(total_liabilities) as total_liabilities,
                           ROUND(total_assets) as total_assets
                    FROM irs_financials WHERE total_liabilities IS NOT NULL AND total_liabilities > 0
                    ORDER BY total_liabilities DESC LIMIT 15
                """)
            else:
                cur.execute("""
                    SELECT org_name, state, ROUND(total_revenue) as total_revenue,
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
        """Query fec_committees table for FEC financial questions."""
        q = question.lower()
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            if any(w in q for w in ["spent", "spending", "disbursement"]):
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "TTL_DISB", "TTL_RECEIPTS", "cycle"
                    FROM fec_committees
                    WHERE "TTL_DISB" ~ '^[0-9.]+$'
                    ORDER BY CAST("TTL_DISB" AS NUMERIC) DESC LIMIT 15
                """)
            elif any(w in q for w in ["cash", "hand", "balance"]):
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "COH_COP", "TTL_RECEIPTS", "cycle"
                    FROM fec_committees
                    WHERE "COH_COP" ~ '^[0-9.]+$'
                    ORDER BY CAST("COH_COP" AS NUMERIC) DESC LIMIT 15
                """)
            else:
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
        """Query organizations by state."""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            if dataset in ("fec", "both"):
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "TTL_RECEIPTS", "TTL_DISB", "cycle"
                    FROM fec_committees
                    WHERE UPPER("CMTE_ST") = %s AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                    ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 20
                """, (state_abbr,))
                return [dict(r) for r in cur.fetchall()], "FEC Committees (PostgreSQL)"
            else:
                cur.execute("""
                    SELECT f.org_name, f.state, l.city,
                           ROUND(f.total_revenue) as total_revenue,
                           ROUND(f.total_assets) as total_assets
                    FROM irs_financials f
                    LEFT JOIN irs_locations l USING (ein)
                    WHERE UPPER(f.state) = %s AND f.total_revenue IS NOT NULL
                    ORDER BY f.total_revenue DESC LIMIT 20
                """, (state_abbr,))
                return [dict(r) for r in cur.fetchall()], "IRS Financials (PostgreSQL)"
        finally:
            cur.close()
            conn.close()

    def query_cross_dataset(self):
        """Find organizations appearing in both IRS and FEC datasets."""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("""
                SELECT f.org_name as irs_name, f.state,
                       ROUND(f.total_revenue) as irs_revenue,
                       ROUND(f.total_assets) as irs_assets,
                       c."CMTE_NM" as fec_name, c."CMTE_TP",
                       c."TTL_RECEIPTS" as fec_receipts,
                       c."TTL_DISB" as fec_disbursements, c."cycle"
                FROM irs_financials f
                JOIN fec_committees c ON LOWER(f.org_name) = LOWER(c."CMTE_NM")
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
        Routes question to appropriate SQL query.

        Returns:
            dict with keys: rows, source_label
        """
        q = question.lower()

        print(f"[FilterAgent] Processing: {question[:55]}...")

        # Cross-dataset query
        cross_keywords = ["connections to", "linked to", "associated with", "appear in both"]
        if any(kw in q for kw in cross_keywords):
            rows, label = self.query_cross_dataset()
            return {"rows": rows, "source_label": label}

        # State-based query
        state = self.detect_state(question)
        if state:
            rows, label = self.query_by_state(state, dataset)
            return {"rows": rows, "source_label": label}

        # Specific FEC committee
        committee = self.detect_committee(question)
        if committee and dataset in ("fec", "both"):
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE LOWER("CMTE_NM") LIKE %s AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 5
            """, (f"%{committee}%",))
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            conn.close()
            return {"rows": rows, "source_label": "FEC Committees (PostgreSQL)"}

        # FEC financial query
        fec_keywords = ["pac", "committee", "campaign", "fec", "actblue", "winred",
                        "democratic", "republican", "dnc", "rnc"]
        if any(kw in q for kw in fec_keywords) and dataset in ("fec", "both"):
            rows = self.query_fec_financial(question)
            return {"rows": rows, "source_label": "FEC Committees (PostgreSQL)"}

        # IRS financial query
        rows = self.query_irs_financial(question)
        return {"rows": rows, "source_label": "IRS Financials (PostgreSQL)"}


if __name__ == "__main__":
    agent = FilterAgent()
    result = agent.run("Which nonprofits raised the most money?", dataset="irs")
    print(f"Got {len(result['rows'])} rows from {result['source_label']}")
    for r in result["rows"][:3]:
        print(r)
