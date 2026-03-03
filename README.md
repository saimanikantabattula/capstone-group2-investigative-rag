# Capstone Group 2 — Investigative RAG (IRS 990 + FEC)

## Research Question
Can we automatically connect IRS Form 990 nonprofit filings with FEC committee filings to uncover financial/organizational links, and answer investigation questions with citations?

## What we are building
A multi-agent RAG system that ingests IRS 990 (XML) + FEC filings (PDF/metadata), stores structured data in Postgres, and uses retrieval + agents to generate answers with citations.

## Architecture Diagram
```mermaid
flowchart LR
  DS[IRS 990 + FEC] --> ING[Ingestion + Cleaning]
  ING --> PG[Postgres (structured)]
  ING --> VDB[Vector Index (embeddings)]
  PG --> RAG[Multi-Agent RAG]
  VDB --> RAG
  RAG --> UI[UI / API Output]

**Then on a new empty line type:**
```text
