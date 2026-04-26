# Capstone Group 2 — Investigative RAG

**Team:** Sai Manikanta Battula, Bhavani Danthuri, Ability Chikanya, Hanok Naidu Suravarapu
**University:** Yeshiva University
**Course:** Capstone Project

---

## What This Project Does

This system lets anyone ask plain English questions about nonprofit organizations and political finance records, and get back clear answers with citations from real government data.

For example you can ask:
- Which nonprofits raised the most money?
- Which PACs spent the most in 2024?
- Which nonprofits are based in Boston?
- How much did ActBlue raise in 2024?
- Which nonprofits have connections to political committees?

The system searches through 1.2 million IRS nonprofit records and 38,000 FEC political committee records to find the answer, then uses AI to write a clear cited response.

---

## Live Links

| What | Link |
|---|---|
| Frontend (website) | https://capstone-group2-investigative-rag.vercel.app |
| Backend API | https://capstone-group2-investigative-rag.onrender.com |
| Health check | https://capstone-group2-investigative-rag.onrender.com/health |

---

## Research Question

Can we automatically connect IRS Form 990 nonprofit filings with FEC political committee filings to answer investigative questions with cited evidence?

---

## Technology Stack

| Part | Technology | What it does |
|---|---|---|
| Frontend | React + Vite | The website users interact with |
| Backend | FastAPI + Python 3.11 | Receives questions and returns answers |
| Database | PostgreSQL on Supabase | Stores all financial data |
| Vector database | Pinecone | Stores document text for semantic search |
| Embeddings | HuggingFace API (all-MiniLM-L6-v2) | Converts text to numbers for search |
| Language model | Anthropic Claude Haiku | Writes the final cited answer |
| Evaluation | DeepEval v3.9.2 | Measures how good our answers are |
| Monitoring | UptimeRobot | Pings our server every 5 min so it never sleeps |
| Deployment | Vercel + Render + Supabase + Pinecone | All cloud, all free tier |

---

## Data We Have

### PostgreSQL Tables

| Table | Rows | Size | What it stores |
|---|---|---|---|
| irs_financials | 378,272 | 151 MB | Revenue, assets, expenses, officer pay |
| irs_locations | 1,216,026 | 298 MB | City, state, ZIP for all 1.2M orgs |
| irs_index | 100,000 | 31 MB | EIN, name, return type |
| fec_committees | 38,793 | 21 MB | Receipts, spending, cash on hand |
| Total | 1,733,091 rows | 501 MB | At Supabase free tier limit |

### Pinecone Vectors

| Namespace | Vectors | Source |
|---|---|---|
| irs | 74,529 | IRS 990 XML text chunks |
| fec | 26,306 | FEC committee descriptions |
| Total | 100,835 | Out of 1 million free tier limit |

---

## How the System Routes Questions

Every question goes through 9 steps in order until a match is found:

| Step | When it triggers | What it queries |
|---|---|---|
| Step 0 | "connections to", "linked to" | SQL JOIN between IRS and FEC tables |
| Step 0b | City name found (Boston, Chicago) | irs_financials joined with irs_locations |
| Step 0c | Year found (2023, 2024, "latest") | irs_financials filtered by tax_year |
| Step 1 | State name found (California, NY) | irs_financials or fec_committees by state |
| Step 2 | Specific committee (ActBlue, WinRed) | fec_committees WHERE name matches |
| Step 3 | Threshold phrase (over 1 billion) | fec_committees WHERE receipts >= amount |
| Step 4 | Financial keywords (revenue, assets) | irs_financials ordered by metric |
| Step 5 | FEC keywords (PAC, committee) | fec_committees ordered by receipts |
| Step 6 | Everything else | Pinecone vector search |

98.7% of questions are answered by PostgreSQL (Steps 0 to 5). Only document text questions fall through to Pinecone (Step 6).

---

## Multi-Agent Architecture

The system has 4 agents that each do one specific job:

| Agent | File | Job |
|---|---|---|
| Controller Agent | agent_controller.py | Receives question, decides which agent to call |
| Filter Agent | agent_filter.py | Runs all SQL queries against PostgreSQL |
| Retriever Agent | agent_retriever.py | Runs vector search against Pinecone |
| Writer Agent | agent_writer.py | Calls Claude to generate the cited answer |

### Reciprocal Rank Fusion

When combining IRS and FEC vector results we use RRF. The formula is:

```
score = 1 / (60 + rank)
```

A document that ranks highly in both the IRS list and the FEC list gets a higher combined score than one that only ranks well in one list. This is based on the standard RRF formula from Cormack et al. 2009.

---

## New Features Added After Ground Truth Evaluation

| Feature | What it does |
|---|---|
| City search | "Which nonprofits are in Boston?" queries irs_locations table |
| Year filtering | "Which nonprofits raised most in 2023?" filters by tax_year column |
| Fuzzy name matching | Finds organizations by partial name using word-by-word SQL LIKE queries |
| Related questions | Shows 4 clickable follow-up questions after every answer |
| Embedding cache | Caches HuggingFace vectors in memory — 493x faster on repeat queries |
| Answer cache | Caches complete answers — instant response for repeated questions |
| Request logging | Every API call logged with method, path, status code, and time taken |
| UptimeRobot | Pings health endpoint every 5 minutes so Render never goes to sleep |

---

## Performance

| What | Number |
|---|---|
| Average response time | 3.22 seconds |
| Cached response time | 0.0 seconds (493x faster) |
| Cache size | 100 entries per cache |
| Cache storage | In memory, resets on server restart |

---

## Evaluation

We use DeepEval v3.9.2 with Anthropic Claude as the LLM judge. This is true AI-based evaluation, not simple keyword matching. Claude reads every question and answer and gives a score from 0 to 1.

### Metrics

| Metric | Type | What it measures |
|---|---|---|
| Answer Relevancy | DeepEval LLM-as-judge | Is the answer relevant to the question? |
| Faithfulness | DeepEval LLM-as-judge | Is the answer grounded in retrieved data? |
| Keyword Score | Rule-based | What percentage of expected keywords appear? |
| Contains Check | Rule-based | Is the most critical expected term present? |

### Ground Truth Questions (115 total)

| Category | Count |
|---|---|
| IRS Financial Ranking | 20 |
| IRS Geographic | 15 |
| FEC Financial Ranking | 20 |
| FEC Specific Committee | 10 |
| FEC Geographic | 5 |
| Cross Dataset | 25 |
| IRS Filing Type | 5 |
| IRS City Search | 5 |
| IRS Year Filter | 5 |
| Fuzzy Name Search | 5 |
| Total | 115 |

### Results

| Metric | Score |
|---|---|
| Accuracy | 99% (99 out of 100 core questions) |
| Answer Relevancy | 0.871 out of 1.0 |
| Faithfulness | 0.919 out of 1.0 |
| Average response time | 3.22 seconds |

### Run the evaluation

```bash
DB_PASS='yourpassword' ANTHROPIC_API_KEY=yourkey python3 src/eval/evaluate.py
```

### Run the batch test (faster, no LLM cost)

```bash
DB_PASS='yourpassword' ANTHROPIC_API_KEY=yourkey python3 src/eval/batch_test.py
```

Batch test result: 98.2% pass rate on 109 questions.

---

## Project Structure

```
capstone-group2-investigative-rag/
├── frontend/
│   └── src/
│       ├── App.jsx              # Main React component
│       ├── api/client.js        # API calls to backend
│       └── components/
│           ├── SearchBar.jsx
│           ├── DatasetToggle.jsx
│           ├── AnswerPanel.jsx
│           └── CitationCard.jsx
├── src/
│   ├── api/
│   │   └── main.py             # FastAPI backend, all endpoints
│   ├── agents/
│   │   ├── agent_controller.py # Routes questions to right agent
│   │   ├── agent_filter.py     # PostgreSQL SQL queries
│   │   ├── agent_retriever.py  # Pinecone vector search with RRF
│   │   └── agent_writer.py     # Claude answer generation
│   ├── rag/
│   │   ├── hybrid.py           # 9-step hybrid router
│   │   └── answer.py           # Pinecone RAG + caching
│   ├── db/                     # Database helpers
│   ├── ingest/                 # Data loading scripts
│   └── eval/
│       ├── evaluate.py         # DeepEval evaluation
│       ├── ground_truth.py     # 115 test questions
│       ├── batch_test.py       # 109-question fast tester
│       └── anthropic_judge.py  # Anthropic wrapper for DeepEval
├── deployment/
│   └── README.md
├── docs/
│   └── use_cases.md
├── Procfile
├── requirements.txt
└── .python-version
```

---

## Local Setup

### Requirements

- Python 3.11 or higher
- Node.js 18 or higher
- PostgreSQL (local) or Supabase account
- Pinecone account
- Anthropic API key
- HuggingFace token

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/saimanikantabattula/capstone-group2-investigative-rag.git
cd capstone-group2-investigative-rag

# 2. Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Copy environment file and fill in your values
cp .env.example .env

# 4. Start the backend
DB_PASS='yourpassword' ANTHROPIC_API_KEY=yourkey PINECONE_API_KEY=yourkey HF_TOKEN=yourtoken uvicorn src.api.main:app --port 8000

# 5. Start the frontend (open a new terminal tab)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

---

## API Endpoints

| Method | Endpoint | What it does |
|---|---|---|
| GET | /health | Health check, used by UptimeRobot |
| POST | /query | Main question answering endpoint |
| POST | /suggestions | Returns 4 related questions |
| GET | /dashboard | Returns aggregated stats for charts |
| GET | /test-pinecone | Tests Pinecone connection |
| GET | /test-embedding | Tests HuggingFace embedding API |

### Example request

```bash
curl -X POST https://capstone-group2-investigative-rag.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which nonprofits raised the most money?", "dataset": "irs", "top_k": 5}'
```

### Example response

```json
{
  "question": "Which nonprofits raised the most money?",
  "answer": "Based on our dataset, the top nonprofits by revenue are: 1. Mass General Brigham (MA) — $23.5 billion [1]...",
  "citations": [{"source": "IRS", "org_name": "MASS GENERAL BRIGHAM", "snippet": "..."}],
  "sources_used": ["IRS Financials (PostgreSQL)"]
}
```

---

## Known Limitations

| Limitation | Why | Possible fix |
|---|---|---|
| Only 31% of IRS data loaded | Supabase free tier is 500MB, we used 501MB | Upgrade to Supabase Pro ($25/month) for 8GB |
| FEC data is 2024-2026 only | We only ingested recent cycles | Load older election cycles |
| Cache resets on server restart | We use in-memory cache | Add Redis for persistent cache |
| No user authentication | System is open access | Add API key middleware |
| Only 1.7% Pinecone document coverage | Storage limits | Expand Pinecone index |

---

## Team Contributions

| Member | Files |
|---|---|
| Sai Manikanta Battula | agent_controller.py, load_irs_financials.py |
| Bhavani Danthuri | README.md, agent_filter.py |
| Ability Chikanya | frontend/src/App.jsx, agent_writer.py |
| Hanok Naidu Suravarapu | agent_retriever.py, ground_truth.py |
