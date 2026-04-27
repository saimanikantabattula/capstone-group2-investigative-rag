"""
hybrid.py
=========
This is the BRAIN of our Investigative RAG system.

When a user asks a question, this file decides:
  - Should we query PostgreSQL (for financial/numeric data)?
  - Should we query Pinecone (for document text search)?
  - Should we search IRS data, FEC data, or both?

Think of it like a smart receptionist who listens to your question
and sends you to the right department automatically.

Routing logic (in order):
  1. Cross-dataset questions  → SQL JOIN between IRS + FEC tables
  2. City-level search        → irs_locations + irs_financials JOIN
  3. Year/date filter         → irs_financials WHERE tax_year = X
  4. State geographic search  → irs_locations or fec_committees by state
  5. Specific FEC committee   → fec_committees WHERE name LIKE X
  6. Threshold questions      → fec_committees WHERE receipts > X billion
  7. General IRS financial    → irs_financials ORDER BY revenue/assets
  8. General FEC financial    → fec_committees ORDER BY receipts
  9. Everything else          → Pinecone vector search (document text)
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
# os: used to read environment variables (DB password, API keys etc.)
import os

# psycopg2: Python library to connect to PostgreSQL database
# psycopg2.extras: gives us RealDictCursor which returns rows as dictionaries
import psycopg2
import psycopg2.extras

# anthropic: the official Anthropic Python library
# We use it to call Claude to generate the final answer
import anthropic as _llm_client


# ── DATABASE CONNECTION SETTINGS ──────────────────────────────────────────────
# These are read from environment variables so we never hardcode passwords in code
# On Render (cloud): these are set in the environment variables dashboard
# Locally: these are set when starting the server with DB_PASS='...' uvicorn ...
DB_HOST = os.getenv("DB_HOST", "localhost")       # database server address
DB_PORT = os.getenv("DB_PORT", "5432")            # PostgreSQL default port
DB_NAME = os.getenv("DB_NAME", "capstone_rag")    # database name
DB_USER = os.getenv("DB_USER", "postgres")        # database username
DB_PASS = os.getenv("DB_PASS", "")               # database password
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Anthropic API key for Claude
LLM_MODEL = "claude-haiku-4-5-20251001"           # which Claude model to use (Haiku = fast + cheap)


# ── FINANCIAL KEYWORDS LIST ───────────────────────────────────────────────────
# This is a list of words that tell us the question needs a financial/database answer
# If a question contains ANY of these words, we route it to PostgreSQL
# Example: "Which nonprofits raised the most money?" contains "most money" → PostgreSQL
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

# ── FEC KEYWORDS LIST ─────────────────────────────────────────────────────────
# This is a list of words that tell us the question is about political finance (FEC data)
# If a question contains ANY of these words, we route it to fec_committees table
# Example: "Which PACs spent the most?" contains "pac" → FEC data
FEC_KEYWORDS = [
    "pac", "committee", "campaign", "political", "fec",
    "donation", "expenditure", "disbursement",
    "election", "candidate", "party", "super pac",
    "actblue", "winred", "harris", "trump", "dnc", "rnc",
    "democratic", "republican", "lobbyist", "house campaign",
    "senate campaign", "presidential campaign", "lincoln project",
    "individual contributions", "most individual", "individual contribution",
    "lobbyist pac", "lobbyist", "fec committees",
]

# ── STATE NAME TO ABBREVIATION MAP ───────────────────────────────────────────
# This dictionary converts full state names to 2-letter abbreviations
# We need this because our database stores states as abbreviations (CA, NY, TX etc.)
# Example: user types "California" → we convert to "CA" for the SQL query
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
    "vermont": "VT", "virginia": "VA",
    "washington state": "WA", "nonprofits in washington": "WA", "based in washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# ── SPECIFIC FEC COMMITTEE NAMES ──────────────────────────────────────────────
# This is a list of well-known political committees
# If a question mentions any of these names exactly, we do a direct name lookup
# instead of a general search — this gives more accurate results
# Example: "How much did ActBlue raise?" → direct lookup for "actblue"
SPECIFIC_COMMITTEES = [
    "actblue", "winred", "harris for president", "harris victory",
    "trump", "biden", "dnc", "rnc", "lincoln project",
    "emily", "planned parenthood", "nra", "america first",
    "priorities usa", "club for growth", "maga inc", "fairshake",
    "republican national committee", "republican national",
    "democracy pac", "slf pac", "harris victory fund",
]

# ── THRESHOLD AMOUNTS MAP ─────────────────────────────────────────────────────
# This maps plain English threshold phrases to actual dollar amounts
# When user says "over 1 billion", we convert to 1000000000 for the SQL WHERE clause
# Example: "Which PACs raised over 100 million?" → WHERE TTL_RECEIPTS >= 100000000
THRESHOLD_MAP = {
    "over 1 billion": 1_000_000_000,
    "more than 1 billion": 1_000_000_000,
    "over 100 million": 100_000_000,
    "more than 100 million": 100_000_000,
    "over 50 million": 50_000_000,
    "more than 50 million": 50_000_000,
    "over 10 million": 10_000_000,
    "more than 10 million": 10_000_000,
}


# ── DATABASE CONNECTION HELPER ────────────────────────────────────────────────
def get_db():
    """
    Creates and returns a new PostgreSQL database connection.
    We always create a fresh connection and close it after each query
    to avoid connection leaks on the free tier Render/Supabase setup.
    """
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )


# ── CROSS DATASET QUERY ───────────────────────────────────────────────────────
def query_cross_dataset(question):
    """
    Finds organizations that appear in BOTH IRS and FEC datasets.
    This is the key investigative feature — finding nonprofits that also have
    political committee connections.

    How it works:
    - Does a SQL JOIN between irs_financials and fec_committees tables
    - Matches on organization name (case-insensitive)
    - Returns combined data from both tables

    Example result:
    "Development Now for Chicago" appears in:
    - IRS data: $81.5M in nonprofit revenue
    - FEC data: $102M in political receipts
    This kind of connection is exactly what investigative journalists need.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT
                f.org_name as irs_name,          -- organization name from IRS data
                f.state,                          -- state from IRS data
                ROUND(f.total_revenue) as irs_revenue,   -- total revenue from IRS
                ROUND(f.total_assets) as irs_assets,     -- total assets from IRS
                c."CMTE_NM" as fec_name,          -- committee name from FEC data
                c."CMTE_TP" as committee_type,    -- committee type (P=presidential, S=senate etc.)
                c."TTL_RECEIPTS" as fec_receipts, -- total money raised from FEC
                c."TTL_DISB" as fec_disbursements,-- total money spent from FEC
                c."cycle"                         -- election cycle year
            FROM irs_financials f
            JOIN fec_committees c
                ON LOWER(f.org_name) = LOWER(c."CMTE_NM")  -- match on name (case-insensitive)
            WHERE f.total_revenue IS NOT NULL
              AND c."TTL_RECEIPTS" ~ '^[0-9.]+$'  -- only include FEC rows with valid numeric receipts
            ORDER BY f.total_revenue DESC
            LIMIT 15
        """)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        # Always close connection even if an error occurs
        cur.close()
        conn.close()


# ── STATE DETECTION ───────────────────────────────────────────────────────────
def detect_state(question):
    """
    Checks if a question mentions a US state name.
    Converts the state name to its 2-letter abbreviation for SQL queries.

    Returns: (abbreviation, state_name) tuple or (None, None) if no state found
    Example: "Which nonprofits are in California?" → ("CA", "california")
    """
    q = question.lower()
    for state_name, abbr in STATE_MAP.items():
        if state_name in q:
            return abbr, state_name
    return None, None  # no state found in the question


# ── CITY DETECTION ────────────────────────────────────────────────────────────
def detect_city(question):
    """
    Checks if a question mentions a major US city name.
    We use this to search the irs_locations table which has city-level data
    for all 1.2 million organizations.

    Returns: city name (title case) or None if no city found
    Example: "Which nonprofits are in Boston?" → "Boston"

    Note: We check for city before state because some cities share names with states
    (e.g. "New York" could be city or state — we handle city first)
    """
    q = question.lower()
    # List of 50 major US cities we support
    CITIES = [
        "new york city", "new york", "los angeles", "chicago", "houston",
        "phoenix", "philadelphia", "san antonio", "san diego", "dallas",
        "san jose", "austin", "jacksonville", "fort worth", "columbus",
        "charlotte", "indianapolis", "san francisco", "seattle", "denver",
        "boston", "nashville", "baltimore", "atlanta", "miami",
        "minneapolis", "portland", "las vegas", "detroit", "memphis",
        "louisville", "milwaukee", "albuquerque", "tucson", "fresno",
        "sacramento", "kansas city", "mesa", "omaha", "cleveland",
        "raleigh", "colorado springs", "virginia beach", "long beach",
        "tampa", "new orleans", "pittsburgh", "cincinnati", "st louis",
        "orlando", "buffalo", "richmond", "madison", "birmingham",
    ]
    for city in CITIES:
        if city in q:
            return city.title()  # return as "Boston" not "boston"
    return None  # no city found in the question


# ── YEAR DETECTION ────────────────────────────────────────────────────────────
def detect_year(question):
    """
    Checks if a question mentions a specific year for filtering.
    If found, we add a WHERE tax_year = X filter to the SQL query.

    Returns: year as string ("2023", "2024") or "latest" or None
    Examples:
    - "Which nonprofits raised most in 2023?" → "2023"
    - "Show me the latest filings" → "latest"
    - "Which nonprofits raised the most?" → None (no year filter)
    """
    import re
    # Look for years between 2015 and 2024
    years = re.findall(r'20(1[5-9]|2[0-4])', question)
    if years:
        return f"20{years[0]}"
    # Check for specific years directly
    if "2024" in question: return "2024"
    if "2023" in question: return "2023"
    if "2022" in question: return "2022"
    if "2021" in question: return "2021"
    # "latest", "recent", "current" → return most recent data
    if "latest" in question.lower() or "recent" in question.lower() or "current" in question.lower():
        return "latest"
    return None  # no year filter needed


# ── QUERY IRS BY CITY ─────────────────────────────────────────────────────────
def query_irs_by_city(city):
    """
    Searches for nonprofits in a specific city.
    Does a JOIN between irs_financials (financial data) and irs_locations (location data)
    on the EIN field (Employer Identification Number = unique tax ID for every nonprofit).

    Step 1: Try exact city match (e.g. city = "Boston" exactly)
    Step 2: If no results, try partial match (e.g. city LIKE "%Boston%")

    Returns: list of organizations with their financial and location data
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Step 1: Exact city name match
        cur.execute("""
            SELECT f.org_name, f.state, l.city, f.return_type, f.tax_year,
                   ROUND(f.total_revenue) as total_revenue,
                   ROUND(f.total_assets) as total_assets
            FROM irs_financials f
            JOIN irs_locations l ON f.ein = l.ein   -- join on EIN (unique org ID)
            WHERE LOWER(l.city) = LOWER(%s)          -- case-insensitive exact match
            AND f.total_revenue IS NOT NULL
            ORDER BY f.total_revenue DESC LIMIT 15
        """, (city,))
        rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            # Step 2: Partial match if exact match found nothing
            cur.execute("""
                SELECT f.org_name, f.state, l.city, f.return_type, f.tax_year,
                       ROUND(f.total_revenue) as total_revenue,
                       ROUND(f.total_assets) as total_assets
                FROM irs_financials f
                JOIN irs_locations l ON f.ein = l.ein
                WHERE LOWER(l.city) LIKE LOWER(%s)   -- partial match using LIKE
                AND f.total_revenue IS NOT NULL
                ORDER BY f.total_revenue DESC LIMIT 15
            """, (f"%{city}%",))
            rows = [dict(r) for r in cur.fetchall()]

        return rows
    finally:
        cur.close()
        conn.close()


# ── FUZZY NAME SEARCH ─────────────────────────────────────────────────────────
def query_irs_fuzzy_name(org_name):
    """
    Searches for an organization by partial or approximate name.
    This handles cases where user doesn't know the exact name.

    Step 1: Try LIKE '%name%' — finds anything containing the search term
    Step 2: If no results, split into individual words and search word by word
            Example: "Gates Foundation" → search for rows containing both "Gates" AND "Foundation"

    This is our fuzzy matching solution without needing any special libraries.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Step 1: Simple partial match — find names containing the search term
        cur.execute("""
            SELECT org_name, state, return_type, tax_year,
                   ROUND(total_revenue) as total_revenue,
                   ROUND(total_assets) as total_assets
            FROM irs_financials
            WHERE LOWER(org_name) LIKE LOWER(%s)    -- case-insensitive partial match
            AND total_revenue IS NOT NULL
            ORDER BY total_revenue DESC LIMIT 10
        """, (f"%{org_name}%",))
        rows = [dict(r) for r in cur.fetchall()]

        if not rows:
            # Step 2: Word-by-word search — split name into individual words
            # Filter out short words (less than 4 characters) to avoid noise
            # Example: "Mass General Brigham" → ["Mass", "General", "Brigham"]
            words = [w for w in org_name.split() if len(w) > 3]
            if words:
                # Build SQL condition: org_name must contain ALL important words
                conditions = " AND ".join([
                    f"LOWER(org_name) LIKE '%{w.lower()}%'" for w in words[:3]
                ])
                cur.execute(f"""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(total_revenue) as total_revenue,
                           ROUND(total_assets) as total_assets
                    FROM irs_financials
                    WHERE {conditions}
                    AND total_revenue IS NOT NULL
                    ORDER BY total_revenue DESC LIMIT 10
                """)
                rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        cur.close()
        conn.close()


# ── QUERY IRS BY YEAR ─────────────────────────────────────────────────────────
def query_irs_by_year(year, metric="total_revenue"):
    """
    Returns top nonprofits filtered by a specific tax year.
    The metric parameter decides whether to rank by revenue or assets.

    year = "latest" → returns most recently filed data (highest tax_year first)
    year = "2023"   → returns only filings from 2023 tax year
    metric = "total_revenue" → rank by revenue (default)
    metric = "total_assets"  → rank by assets (when user asks about assets)
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        if year == "latest":
            # No year filter — just get most recent filings sorted by tax_year DESC
            cur.execute(f"""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE {metric} IS NOT NULL
                ORDER BY tax_year DESC, {metric} DESC LIMIT 15
            """)
        else:
            # Filter by specific year using WHERE tax_year = year
            cur.execute(f"""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials
                WHERE {metric} IS NOT NULL
                AND tax_year = %s          -- filter by specific year
                ORDER BY {metric} DESC LIMIT 15
            """, (year,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ── IS FINANCIAL QUESTION? ────────────────────────────────────────────────────
def is_financial_question(question):
    """
    Returns True if the question needs a financial/database answer.
    Checks if any word from FINANCIAL_KEYWORDS appears in the question.

    Example:
    "Which nonprofits raised the most money?" → True (contains "raised the most")
    "What is the mission of United Way?" → False (no financial keywords)
    """
    q = question.lower()
    return any(kw in q for kw in FINANCIAL_KEYWORDS)


# ── IS FEC QUESTION? ──────────────────────────────────────────────────────────
def is_fec_question(question):
    """
    Returns True if the question is about political finance (FEC data).
    Checks if any word from FEC_KEYWORDS appears in the question.

    Example:
    "Which PACs spent the most in 2024?" → True (contains "pac")
    "Which nonprofits raised the most?" → False (no FEC keywords)
    """
    q = question.lower()
    return any(kw in q for kw in FEC_KEYWORDS)


# ── QUERY IRS BY STATE ────────────────────────────────────────────────────────
def query_irs_by_state(state_abbr):
    """
    Returns nonprofits from a specific US state.

    Strategy:
    1. First query irs_financials to get top organizations by revenue in that state
       (these have financial data but only ~378K organizations)
    2. Then query irs_locations to get additional organizations in that state
       (this has all 1.2M organizations but less financial detail)
    3. Combine both results, removing duplicates

    This gives the best coverage — rich financial data for top orgs,
    plus location-only data for smaller orgs.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Step 1: Get organizations with financial data for this state
        cur.execute("""
            SELECT f.org_name, f.state, l.city, f.return_type, f.tax_year,
                   ROUND(f.total_revenue) as total_revenue,
                   ROUND(f.total_assets) as total_assets
            FROM irs_financials f
            LEFT JOIN irs_locations l USING (ein)   -- LEFT JOIN keeps all financial rows
            WHERE UPPER(f.state) = %s               -- match state abbreviation
            AND f.total_revenue IS NOT NULL
            ORDER BY f.total_revenue DESC LIMIT 10
        """, (state_abbr,))
        financial_rows = [dict(r) for r in cur.fetchall()]

        # Step 2: Get additional organizations from location table
        cur.execute("""
            SELECT org_name, state, city, return_type, tax_year
            FROM irs_locations
            WHERE UPPER(state) = %s
            ORDER BY org_name LIMIT 15
        """, (state_abbr,))
        location_rows = [dict(r) for r in cur.fetchall()]

        # Step 3: Combine results — avoid showing same org twice
        seen = set(r["org_name"] for r in financial_rows)
        extra = [r for r in location_rows if r["org_name"] not in seen]
        return financial_rows + extra[:10]  # return financial rows + up to 10 extra
    finally:
        cur.close()
        conn.close()


# ── QUERY SPECIFIC FEC COMMITTEE ──────────────────────────────────────────────
def query_fec_specific_committee(committee_name):
    """
    Looks up a specific political committee by name.
    Uses LIKE search so partial names work (e.g. "actblue" finds "ACTBLUE CHARITIES").

    Only returns rows where TTL_RECEIPTS is a valid number
    (some rows have empty or null values in the FEC data).
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                   "TTL_RECEIPTS", "TTL_DISB", "cycle"
            FROM fec_committees
            WHERE LOWER("CMTE_NM") LIKE %s               -- partial name match
              AND "TTL_RECEIPTS" ~ '^[0-9.]+$'           -- only valid numeric receipts
            ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 5
        """, (f"%{committee_name.lower()}%",))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ── QUERY FEC BY THRESHOLD ────────────────────────────────────────────────────
def query_fec_threshold(amount, field="TTL_RECEIPTS"):
    """
    Returns FEC committees that raised or spent more than a specific dollar amount.

    field = "TTL_RECEIPTS" → filter by total receipts (money raised)
    field = "TTL_DISB"     → filter by total disbursements (money spent)

    Note: FEC financial data is stored as TEXT not numbers in our database,
    so we use CAST(field AS NUMERIC) to convert it for comparison.
    The regex '^[0-9.]+$' filters out any rows with invalid non-numeric values.
    """
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute(f"""
            SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                   "TTL_RECEIPTS", "TTL_DISB", "cycle"
            FROM fec_committees
            WHERE "{field}" IS NOT NULL
              AND "{field}" != ''
              AND "{field}" ~ '^[0-9.]+$'           -- validate it is a number
              AND CAST("{field}" AS NUMERIC) >= %s  -- filter by threshold amount
            ORDER BY CAST("{field}" AS NUMERIC) DESC LIMIT 20
        """, (amount,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


# ── QUERY IRS FINANCIALS (MAIN IRS FUNCTION) ──────────────────────────────────
def query_irs_financials(question):
    """
    The main IRS financial query function.
    Handles all types of financial questions about nonprofits.

    This function uses a long if-elif chain to detect what kind of financial
    question is being asked and then runs the appropriate SQL query.

    Categories handled:
    - Return type queries (990, 990EZ, 990PF, 990T)
    - Contributions and grants
    - Program service revenue
    - Officer compensation (with optional threshold filters)
    - Organization type queries (hospital, university, foundation, arts, housing etc.)
    - Financial metric queries (revenue, expenses, assets, liabilities, surplus/deficit)
    - Default: top by total revenue
    """
    q = question.lower()  # convert to lowercase for keyword matching
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:

        # ── Return type queries ──
        # Handles: "Which orgs filed 990PF?" "Which orgs filed 990EZ returns?"
        # Uses irs_index table which has return type data for 100,000 organizations
        if any(w in q for w in ["990pf", "990ez", "990t", "990 return"]) or "filed 990" in q:
            if "990pf" in q: rt = "990PF"       # Private Foundation returns
            elif "990ez" in q: rt = "990EZ"     # Small nonprofit returns
            elif "990t" in q: rt = "990T"       # Unrelated business income returns
            else: rt = "990"                    # Standard nonprofit returns
            cur.execute("""
                SELECT org_name, return_type, tax_period as tax_year, ein
                FROM irs_index WHERE UPPER(return_type) = %s
                AND org_name IS NOT NULL
                ORDER BY org_name LIMIT 20
            """, (rt,))

        # ── Contributions and grants ──
        # Handles: "Which nonprofits received the most contributions?"
        elif any(w in q for w in ["contributions and grants", "most contributions", "most grants", "giving"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(contributions_grants) as contributions_grants,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE contributions_grants IS NOT NULL
                ORDER BY contributions_grants DESC LIMIT 15
            """)

        # ── Program service revenue ──
        # Handles: "Which nonprofits have the most program service revenue?"
        elif any(w in q for w in ["program service", "program revenue", "service revenue"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(program_service_revenue) as program_service_revenue,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE program_service_revenue IS NOT NULL
                ORDER BY program_service_revenue DESC LIMIT 15
            """)

        # ── Officer compensation ──
        # Handles: "Which nonprofits pay officers the most?"
        # Also handles threshold versions: "Which nonprofits pay officers over $1 million?"
        elif any(w in q for w in ["compensation", "salary", "officer", "executive", "pay"]):
            if any(w in q for w in ["over 1 million", "1 million dollar", "officers over", "over a million"]):
                # Officers paid more than $1 million
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials
                    WHERE officer_compensation IS NOT NULL
                    AND officer_compensation >= 1000000     -- $1 million threshold
                    ORDER BY officer_compensation DESC LIMIT 15
                """)
            elif any(w in q for w in ["500 thousand", "500k", "over 500", "executives over"]):
                # Officers paid more than $500 thousand
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials
                    WHERE officer_compensation IS NOT NULL
                    AND officer_compensation >= 500000      -- $500K threshold
                    ORDER BY officer_compensation DESC LIMIT 15
                """)
            else:
                # No threshold — just return top by officer compensation
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(officer_compensation) as officer_compensation,
                           ROUND(total_revenue) as total_revenue
                    FROM irs_financials WHERE officer_compensation IS NOT NULL
                    ORDER BY officer_compensation DESC LIMIT 15
                """)

        # ── Arts organizations ──
        # Handles: "Which arts organizations have the most assets?"
        # Uses LIKE to search for keywords in the org name
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

        # ── Housing nonprofits ──
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

        # ── Youth and children organizations ──
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

        # ── Educational institutions with debt ──
        # Special case: only matches when BOTH debt AND education keywords are present
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

        # ── Community foundations ──
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

        # ── Universities and colleges ──
        # Note: This must come BEFORE the generic asset/revenue check below
        # to avoid college questions being caught by the generic handler
        elif any(w in q for w in ["universit", "college"]):
            # Choose which metric to sort by based on what user is asking
            order_col = "total_assets" if "asset" in q else "total_revenue"
            cur.execute(f"""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE {order_col} IS NOT NULL
                AND (LOWER(org_name) LIKE '%universit%' OR LOWER(org_name) LIKE '%college%')
                ORDER BY {order_col} DESC LIMIT 15
            """)

        # ── Veterans organizations ──
        # Note: Must come before generic asset check
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

        # ── Environmental organizations ──
        # Note: Must come before generic revenue check
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

        # ── Revenue queries (generic) ──
        # Handles: "Which nonprofits raised the most money?" "Top nonprofits by revenue"
        elif any(w in q for w in ["revenue", "raised", "money", "income", "funding"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_revenue) as total_revenue,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_revenue IS NOT NULL
                ORDER BY total_revenue DESC LIMIT 15
            """)

        # ── Expense queries ──
        # Handles: "Which nonprofits spent the most money?"
        elif any(w in q for w in ["expense", "spent", "spending", "cost"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_expenses) as total_expenses,
                       ROUND(total_revenue) as total_revenue
                FROM irs_financials WHERE total_expenses IS NOT NULL
                ORDER BY total_expenses DESC LIMIT 15
            """)

        # ── Asset queries ──
        # Handles: "Which nonprofits have the most assets?"
        elif any(w in q for w in ["asset", "worth", "wealth", "large"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_assets) as total_assets,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(net_assets) as net_assets
                FROM irs_financials WHERE total_assets IS NOT NULL
                ORDER BY total_assets DESC LIMIT 15
            """)

        # ── Liabilities queries ──
        # Handles: "Which nonprofits have the most debt?"
        elif any(w in q for w in ["liabilit", "debt", "owe"]):
            cur.execute("""
                SELECT org_name, state, return_type, tax_year,
                       ROUND(total_liabilities) as total_liabilities,
                       ROUND(total_assets) as total_assets
                FROM irs_financials WHERE total_liabilities IS NOT NULL
                  AND total_liabilities > 0       -- exclude zero liabilities
                ORDER BY total_liabilities DESC LIMIT 15
            """)

        # ── Surplus or deficit queries ──
        # Handles: "Which nonprofits had losses?" "Which nonprofits had a surplus?"
        elif any(w in q for w in ["surplus", "profit", "loss", "deficit"]):
            if "loss" in q or "deficit" in q:
                # Find organizations where expenses > revenue (running at a loss)
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(total_revenue) as total_revenue,
                           ROUND(total_expenses) as total_expenses,
                           ROUND(total_revenue - total_expenses) as net_income
                    FROM irs_financials
                    WHERE total_revenue IS NOT NULL AND total_expenses IS NOT NULL
                      AND total_revenue < total_expenses  -- expenses exceed revenue = loss
                    ORDER BY (total_revenue - total_expenses) ASC LIMIT 15
                """)
            else:
                # Find organizations where revenue > expenses (running at a surplus)
                cur.execute("""
                    SELECT org_name, state, return_type, tax_year,
                           ROUND(total_revenue) as total_revenue,
                           ROUND(total_expenses) as total_expenses,
                           ROUND(total_revenue - total_expenses) as net_income
                    FROM irs_financials
                    WHERE total_revenue IS NOT NULL AND total_expenses IS NOT NULL
                      AND total_revenue > total_expenses  -- revenue exceeds expenses = surplus
                    ORDER BY (total_revenue - total_expenses) DESC LIMIT 15
                """)

        # ── Hospital and healthcare organizations ──
        elif any(w in q for w in ["hospital", "medical", "health system", "healthcare"]):
            # Dynamically choose sort column based on what user asked
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

        # ── University and education (second handler — broader) ──
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

        # ── Foundations and charities ──
        # Handles multiple sub-cases: net assets, total assets, contributions, or general
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

        # ── Research organizations ──
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

        # ── Veterans organizations (second handler — broader) ──
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

        # ── Environmental organizations (second handler — broader) ──
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

        # ── Social service organizations ──
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

        # ── DEFAULT: top nonprofits by revenue ──
        # If no specific category matched, return top organizations by total revenue
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


# ── QUERY FEC COMMITTEES (MAIN FEC FUNCTION) ──────────────────────────────────
def query_fec_committees(question):
    """
    The main FEC financial query function.
    Handles all types of questions about political committees.

    Categories handled:
    - House campaign committees (CMTE_TP = 'H')
    - Senate campaign committees (CMTE_TP = 'S')
    - Presidential campaign committees (CMTE_TP = 'P')
    - Lobbyist PACs
    - Independent expenditure committees / Super PACs
    - Spending/disbursement rankings
    - Cash on hand rankings
    - Debt rankings
    - Individual contributions rankings
    - Democratic committees
    - Republican committees
    - Default: all committees by total receipts

    Note: FEC column names are ALL CAPS because that is how the FEC publishes raw data.
    """
    q = question.lower()
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # ── House campaign committees ──
        if any(w in q for w in ["house campaign", "house committee"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" = 'H'   -- H = House campaign committee
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Senate campaign committees ──
        elif any(w in q for w in ["senate campaign", "senate committee"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" = 'S'   -- S = Senate campaign committee
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Presidential campaign committees ──
        elif any(w in q for w in ["presidential campaign", "presidential committee"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" = 'P'   -- P = Presidential campaign committee
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Lobbyist PACs ──
        elif any(w in q for w in ["lobbyist", "registrant"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE (LOWER("CMTE_NM") LIKE '%lobbyist%'
                       OR LOWER("CMTE_NM") LIKE '%registrant%'
                       OR "CMTE_TP" IN ('V', 'W'))          -- V/W = lobbyist committee types
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Independent expenditure committees and Super PACs ──
        elif any(w in q for w in ["independent expenditure", "super pac"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees WHERE "CMTE_TP" IN ('O', 'U', 'N')  -- O/U/N = Super PAC types
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Spending / disbursements ranking ──
        # Handles: "Which PACs spent the most money?"
        elif any(w in q for w in ["spent", "spending", "expenditure", "disbursement"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST", "CMTE_CITY",
                       "TTL_DISB", "TTL_RECEIPTS", "INDV_CONTB",
                       "COH_COP", "DEBTS_OWED_BY_CMTE", "cycle"
                FROM fec_committees
                WHERE "TTL_DISB" IS NOT NULL AND "TTL_DISB" != ''
                  AND "TTL_DISB" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_DISB" AS NUMERIC) DESC LIMIT 15  -- sort by disbursements
            """)

        # ── Cash on hand ranking ──
        # Handles: "Which committees have the most cash on hand?"
        elif any(w in q for w in ["cash", "hand", "balance"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "COH_COP", "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE "COH_COP" IS NOT NULL AND "COH_COP" != ''
                  AND "COH_COP" ~ '^[0-9.]+$'
                ORDER BY CAST("COH_COP" AS NUMERIC) DESC LIMIT 15   -- COH_COP = Cash on Hand
            """)

        # ── Debt ranking ──
        # Handles: "Which committees have the most debt?"
        elif any(w in q for w in ["debt", "owe", "liability"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "DEBTS_OWED_BY_CMTE", "TTL_RECEIPTS", "cycle"
                FROM fec_committees
                WHERE "DEBTS_OWED_BY_CMTE" IS NOT NULL AND "DEBTS_OWED_BY_CMTE" != ''
                  AND "DEBTS_OWED_BY_CMTE" ~ '^[0-9.]+$'
                  AND CAST("DEBTS_OWED_BY_CMTE" AS NUMERIC) > 0     -- only positive debt
                ORDER BY CAST("DEBTS_OWED_BY_CMTE" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Individual contributions ranking ──
        elif any(w in q for w in ["individual contribution", "individual donor", "indv"]):
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "INDV_CONTB", "TTL_RECEIPTS", "cycle"
                FROM fec_committees
                WHERE "INDV_CONTB" IS NOT NULL AND "INDV_CONTB" != ''
                  AND "INDV_CONTB" ~ '^[0-9.]+$'
                ORDER BY CAST("INDV_CONTB" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Democratic committees ──
        elif "democratic" in q:
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE ("CMTE_PTY_AFFILIATION" = 'DEM'           -- party affiliation field
                       OR "CAND_PTY_AFFILIATION" = 'DEM'
                       OR LOWER("CMTE_NM") LIKE '%democrat%')   -- or name contains "democrat"
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)

        # ── Republican committees ──
        elif "republican" in q:
            cur.execute("""
                SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                       "TTL_RECEIPTS", "TTL_DISB", "cycle"
                FROM fec_committees
                WHERE ("CMTE_PTY_AFFILIATION" = 'REP'           -- party affiliation field
                       OR "CAND_PTY_AFFILIATION" = 'REP'
                       OR LOWER("CMTE_NM") LIKE '%republican%') -- or name contains "republican"
                  AND "TTL_RECEIPTS" ~ '^[0-9.]+$'
                ORDER BY CAST("TTL_RECEIPTS" AS NUMERIC) DESC LIMIT 15
            """)

        # ── DEFAULT: all committees by total receipts ──
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


# ── FORMAT ROWS AS CONTEXT ────────────────────────────────────────────────────
def format_rows_as_context(rows, source="IRS"):
    """
    Converts database rows into a numbered text format that Claude can read.
    This is the "context" we pass to Claude when asking it to generate an answer.

    Input: list of database row dictionaries
    Output: numbered text like:
        [1] IRS | org_name: Mass General Brigham | total_revenue: $23,474,745,033 | state: MA
        [2] IRS | org_name: Fidelity Investments | total_revenue: $19,858,151,933 | state: MA

    The numbers [1], [2] etc. become the citation references in the final answer.
    Dollar amounts over $1000 are formatted with $ sign and commas.
    """
    if not rows:
        return "No data found."

    lines = []
    for i, row in enumerate(rows, 1):
        parts = [f"[{i}] {source}"]
        for k, v in row.items():
            if v is not None and v != "":
                # Format large numbers as dollar amounts for readability
                if isinstance(v, (int, float)) and v > 1000:
                    v = f"${v:,.0f}"
                parts.append(f"{k}: {v}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


# ── GENERATE ANSWER FROM DATA ─────────────────────────────────────────────────
def generate_answer_from_data(question, context, source_label):
    """
    Calls Claude (Anthropic AI) to generate a professional cited answer.

    How it works:
    1. We provide a system prompt that tells Claude how to behave
       (be specific, cite data, format numbers, be confident)
    2. We provide the question + the database context (formatted rows)
    3. Claude reads both and writes a professional answer with [1], [2] citations

    This is the "Generation" part of RAG (Retrieval Augmented Generation)
    - Retrieval = getting data from PostgreSQL or Pinecone
    - Augmented = adding that data as context
    - Generation = Claude writing the final answer

    We use claude-haiku model because:
    - It is the fastest Claude model
    - It is the cheapest ($0.25 per million tokens)
    - It is accurate enough for our structured data answers
    """
    # System prompt: tells Claude exactly how to behave
    system_prompt = (
        f"You are an investigative analyst specializing in {source_label} financial data.\n"
        "You are answering from a curated sample of government filings.\n"
        "Answer the question using the data provided below.\n"
        "Be specific and cite data with [1], [2] etc.\n"             # requires citations
        "Format numbers clearly (e.g. $3.1 billion, $450 million).\n"
        "Present findings confidently.\n"
        "If data is limited, say 'Among the organizations in our sample...' and still give the best answer.\n"
        "Never say 'I cannot answer' if there is any relevant data.\n" # avoid unhelpful responses
        "Be concise and professional.\n"
        "End with a brief Sources section."
    )

    # User message: the actual question + the retrieved data
    user_message = (
        f"Question: {question}\n\n"
        f"--- DATA ---\n{context}\n\n"
        "Answer the question using the data above."
    )

    # Call Anthropic API
    api = _llm_client.Anthropic(api_key=LLM_API_KEY)
    response = api.messages.create(
        model=LLM_MODEL,       # claude-haiku-4-5-20251001
        max_tokens=1024,       # maximum length of the answer
        system=system_prompt,  # how Claude should behave
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract the text response from Claude's response object
    return response.content[0].text


# ── MAIN HYBRID ASK FUNCTION ──────────────────────────────────────────────────
def hybrid_ask(question, dataset="both", top_k=5):
    """
    The MAIN function — this is called by main.py for every user question.

    This function is the smart router. It decides which path to take:

    ROUTING ORDER (checks in this exact order):
    ─────────────────────────────────────────────────────────────
    Step 0:  Cross-dataset → looks for orgs in BOTH IRS and FEC
    Step 0b: City search   → finds nonprofits by city name
    Step 0c: Year filter   → finds nonprofits by specific tax year
    Step 1:  State search  → finds orgs by US state
    Step 2:  Specific FEC  → looks up a named committee (ActBlue etc.)
    Step 3:  FEC threshold → finds committees over $X billion
    Step 4:  IRS financial → general IRS financial rankings
    Step 5:  FEC financial → general FEC committee rankings
    Step 6:  Pinecone RAG  → document text search (fallback)
    ─────────────────────────────────────────────────────────────

    Parameters:
    - question: the user's question as a string
    - dataset: "irs", "fec", or "both" (which data to search)
    - top_k: how many results to retrieve from Pinecone (default 5)

    Returns: RAGResponse object with answer, citations, and sources_used
    """
    # Import the Pinecone RAG function and response classes from answer.py
    # These are used as the fallback when no SQL route matches
    from src.rag.answer import ask as rag_ask, RAGResponse, Citation

    # Pre-compute these flags once to avoid repeating keyword checks
    use_db = is_financial_question(question)   # True if question needs financial data
    use_fec = is_fec_question(question)         # True if question is about FEC/political data
    q_lower = question.lower()                  # lowercase version for all comparisons


    # ── STEP 0: Cross-dataset queries ────────────────────────────────────────
    # Check if question asks about connections between IRS and FEC data
    # Example: "Which nonprofits have connections to political committees?"
    cross_keywords = [
        "connections to", "linked to", "associated with", "both irs and fec",
        "nonprofit and political", "appear in both", "cross dataset", "overlap"
    ]
    if any(kw in q_lower for kw in cross_keywords):
        try:
            rows = query_cross_dataset(question)
            if rows:
                context = format_rows_as_context(rows, "IRS+FEC")
                answer = generate_answer_from_data(question, context,
                    "cross-dataset IRS nonprofit and FEC political finance")
                # Build citation objects for each result row
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
            # If cross-dataset query fails, continue to next route
            print(f"Cross-dataset query failed: {e}, falling back to RAG")


    # ── STEP 0b: City-level geographic search ─────────────────────────────────
    # Check if question mentions a specific city name
    # We only do city search if NO state was detected (to avoid ambiguity)
    # Example: "Which nonprofits are based in Boston?"
    city = detect_city(question)
    if city and not detect_state(question)[0] and dataset in ("irs", "both"):
        try:
            rows = query_irs_by_city(city)
            if rows:
                context = format_rows_as_context(rows, "IRS Financials")
                answer = generate_answer_from_data(question, context, "IRS nonprofit filings")
                citations = [Citation(
                    source="IRS",
                    file_name="irs_financials + irs_locations (PostgreSQL)",
                    org_name=r.get("org_name",""), ein="", object_id="",
                    snippet=f"City: {r.get('city','N/A')} | State: {r.get('state','N/A')} | Revenue: {r.get('total_revenue','N/A')}",
                    distance=0.0
                ) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations,
                    sources_used=["IRS Financials by City (PostgreSQL)"])
        except Exception as e:
            print(f"City query failed: {e}, falling back to RAG")


    # ── STEP 0c: Year/date filtering ──────────────────────────────────────────
    # Check if question mentions a specific year
    # Example: "Which nonprofits raised the most money in 2023?"
    year = detect_year(question)
    if year and is_financial_question(question) and dataset in ("irs", "both"):
        # Choose which metric to sort by based on the question
        metric = "total_assets" if "asset" in q_lower else "total_revenue"
        try:
            rows = query_irs_by_year(year, metric)
            if rows:
                context = format_rows_as_context(rows, "IRS Financials")
                answer = generate_answer_from_data(question, context, "IRS nonprofit filings")
                citations = [Citation(
                    source="IRS",
                    file_name="irs_financials (PostgreSQL)",
                    org_name=r.get("org_name",""), ein="", object_id="",
                    snippet=f"Year: {r.get('tax_year','N/A')} | Revenue: {r.get('total_revenue','N/A')} | State: {r.get('state','N/A')}",
                    distance=0.0
                ) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations,
                    sources_used=[f"IRS Financials {year} (PostgreSQL)"])
        except Exception as e:
            print(f"Year query failed: {e}, falling back to RAG")


    # ── STEP 1: Geographic state search ──────────────────────────────────────
    # Check if question mentions a US state name
    # Example: "Which nonprofits are based in California?"
    state_abbr, state_name = detect_state(question)
    if state_abbr:
        if use_fec and dataset in ("fec", "both"):
            # User asked about FEC committees in a state
            try:
                conn = get_db()
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("""
                    SELECT "CMTE_NM", "CMTE_TP", "CMTE_ST",
                           "TTL_RECEIPTS", "TTL_DISB", "cycle"
                    FROM fec_committees
                    WHERE UPPER("CMTE_ST") = %s          -- match state abbreviation
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
                    return RAGResponse(answer=answer, citations=citations,
                        sources_used=["FEC Committees (PostgreSQL)"])
            except Exception as e:
                print(f"FEC state query failed: {e}, falling back to RAG")

        elif dataset in ("irs", "both"):
            # User asked about IRS nonprofits in a state
            try:
                rows = query_irs_by_state(state_abbr)
                if rows:
                    context = format_rows_as_context(rows, "IRS Financials")
                    answer = generate_answer_from_data(question, context, "IRS nonprofit filings")
                    citations = [Citation(source="IRS", file_name="irs_financials (PostgreSQL)",
                        org_name=r.get("org_name",""), ein="", object_id="",
                        snippet=f"State: {r.get('state','N/A')} | Revenue: {r.get('total_revenue','N/A')}",
                        distance=0.0) for r in rows[:5]]
                    return RAGResponse(answer=answer, citations=citations,
                        sources_used=["IRS Financials (PostgreSQL)"])
            except Exception as e:
                print(f"IRS state query failed: {e}, falling back to RAG")


    # ── STEP 2: Specific FEC committee lookup ─────────────────────────────────
    # Check if question mentions a specific well-known committee name
    # Example: "How much did ActBlue raise in 2024?"
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
                return RAGResponse(answer=answer, citations=citations,
                    sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"Specific committee query failed: {e}, falling back to RAG")


    # ── STEP 3: Threshold questions ───────────────────────────────────────────
    # Check if question mentions a dollar threshold
    # Example: "Which PACs raised over 100 million dollars?"
    matched_threshold = next((v for k, v in THRESHOLD_MAP.items() if k in q_lower), None)
    if matched_threshold and use_fec and dataset in ("fec", "both"):
        try:
            # Decide whether to filter by spending or receipts
            field = "TTL_DISB" if any(w in q_lower for w in ["spent", "disbursement"]) else "TTL_RECEIPTS"
            rows = query_fec_threshold(matched_threshold, field)
            if rows:
                context = format_rows_as_context(rows, "FEC")
                answer = generate_answer_from_data(question, context, "FEC political finance")
                citations = [Citation(source="FEC", file_name="fec_committees (PostgreSQL)",
                    org_name=r.get("CMTE_NM",""), ein="", object_id="",
                    snippet=f"Receipts: {r.get('TTL_RECEIPTS','N/A')} | Disbursements: {r.get('TTL_DISB','N/A')}",
                    distance=0.0) for r in rows[:5]]
                return RAGResponse(answer=answer, citations=citations,
                    sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"Threshold query failed: {e}, falling back to RAG")


    # ── STEP 4: General IRS financial questions ───────────────────────────────
    # Handle all other financial questions about IRS nonprofit data
    # Example: "Which hospitals have the most assets?"
    # Note: We skip this if the question is clearly FEC-only
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
                return RAGResponse(answer=answer, citations=citations,
                    sources_used=["IRS Financials (PostgreSQL)"])
        except Exception as e:
            print(f"PostgreSQL query failed: {e}, falling back to RAG")


    # ── STEP 5: General FEC financial questions ───────────────────────────────
    # Handle financial questions about FEC committee data
    # Example: "Which PACs spent the most money in 2024?"
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
                return RAGResponse(answer=answer, citations=citations,
                    sources_used=["FEC Committees (PostgreSQL)"])
        except Exception as e:
            print(f"FEC PostgreSQL query failed: {e}, falling back to RAG")


    # ── STEP 5b: Fuzzy organization name search ──────────────────────────────
    # Check if question mentions a specific organization name
    # Example: "Tell me about Mass General Brigham financials"
    org_keywords = ["tell me about", "financials of", "financial status of",
                    "revenue of", "assets of", "about", "show me"]
    if any(kw in q_lower for kw in org_keywords) and dataset in ("irs", "both"):
        # Extract potential org name from question
        org_name = q_lower
        for kw in org_keywords:
            org_name = org_name.replace(kw, "").strip()
        org_name = org_name.replace("financials", "").replace("financial", "").strip()
        if len(org_name) > 3:
            try:
                rows = query_irs_fuzzy_name(org_name)
                if rows:
                    context = format_rows_as_context(rows, "IRS Financials")
                    answer = generate_answer_from_data(question, context, "IRS nonprofit finance")
                    citations = [Citation(source="IRS", file_name="irs_financials (PostgreSQL)",
                        org_name=r.get("org_name",""), ein="", object_id="",
                        snippet=f"Revenue: {r.get('total_revenue','N/A')} | State: {r.get('state','N/A')}",
                        distance=0.0) for r in rows[:5]]
                    return RAGResponse(answer=answer, citations=citations,
                        sources_used=["IRS Financials (PostgreSQL)"])
            except Exception as e:
                print(f"Fuzzy search failed: {e}")

    # ── STEP 6: Pinecone vector search (fallback) ─────────────────────────────
    # If no SQL route matched, fall back to Pinecone document text search
    # This handles questions like "What is the mission of United Way?"
    # answer.py handles: embedding the question → Pinecone search → Claude generation
    return rag_ask(question, dataset=dataset, top_k=top_k)
