# Capstone Group 2 — Multi-Agent RAG / Investigative Intelligence

## What we are building
An agentic RAG system that retrieves evidence from:
- Structured data (CSV/DB)
- Unstructured documents (XML/PDF)
Then generates answers with citations.

## Datasets (stored locally, NOT committed to GitHub)
### IRS 990 (2024–2025)
- Structured: IRS index/sample CSV with OBJECT_ID
- Unstructured: XML filings named <OBJECT_ID>_public.xml

### FEC (2024 + 2025 activity)
- Structured: committee_summary_2024.csv and committee_summary_2026.csv
- Unstructured: selected filing PDFs downloaded from FEC filings page

## Repo structure
- src/ingest: ingestion scripts (IRS XML, FEC PDFs)
- src/index: embeddings + vector DB indexing
- src/query: retrieval + query pipeline
- docs/: use cases and notes

## Local setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
