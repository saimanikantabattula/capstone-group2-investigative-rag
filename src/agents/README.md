# Multi-Agent Architecture

The system uses 4 specialized agents that work together to answer investigative questions.

## Agents

### agent_controller.py
Entry point. Receives the user question, classifies it using keyword detection, and routes to either the Filter Agent (PostgreSQL) or Retriever Agent (Pinecone). Then passes results to the Writer Agent.

### agent_filter.py
Handles all PostgreSQL queries. Contains SQL logic for financial rankings, geographic filters, specific committee lookups, threshold queries, and cross-dataset JOINs between IRS and FEC data.

### agent_retriever.py
Handles Pinecone vector search. Embeds the query using HuggingFace API, searches both IRS and FEC namespaces, and applies Reciprocal Rank Fusion (RRF) when combining results.

### agent_writer.py
Takes retrieved data from Filter or Retriever Agent, formats it as numbered context, calls the Anthropic language model, and generates a professional cited answer.

## Flow

User Question → Controller Agent → Filter Agent (PostgreSQL) → Writer Agent
                              OR → Retriever Agent (Pinecone) → Writer Agent
