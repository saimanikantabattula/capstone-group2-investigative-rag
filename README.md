# Capstone Group 2 — Investigative RAG (IRS 990 + FEC)

## Research Question
Can we automatically connect IRS Form 990 nonprofit filings with FEC committee filings to uncover financial/organizational links, and answer investigation questions with citations?

## What we are building
A multi-agent RAG system that ingests IRS 990 (XML) + FEC filings (PDF/metadata), stores structured data in Postgres, and uses retrieval + agents to generate answers with citations.

```md
## Architecture Diagram
```mermaid
flowchart LR
  DS["IRS 990 + FEC"] --> ING["Ingestion + Cleaning"]
  ING --> PG["Postgres (structured)"]
  ING --> VDB["Vector Index (embeddings)"]
  PG --> RAG["Multi-Agent RAG"]
  VDB --> RAG
  RAG --> UI["UI / API Output"]

flowchart TB
  A["Raw XML/PDF/CSV"] --> B["Parse + Extract"]
  B --> C["Clean + Normalize"]
  C --> D["Write to Postgres"]
  C --> E["Chunk Text"]
  E --> F["Embeddings"]
  F --> G["Vector DB"]
  H["User Question"] --> I["Agents"]
  D --> I
  G --> I
  I --> J["Answer + Citations"]

erDiagram
  ORGANIZATION ||--o{ FILING : submits
  FILING ||--o{ DOCUMENT : contains
  DOCUMENT ||--o{ CHUNK : splits_into
  CHUNK ||--o{ EMBEDDING : has
  ORGANIZATION ||--o{ LINK : connected_to

