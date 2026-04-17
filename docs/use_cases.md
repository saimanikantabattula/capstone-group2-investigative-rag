# Use Cases — Investigative RAG

## IRS 990 Queries
1. Which nonprofits raised the most money? → Returns top orgs by total revenue
2. Which hospitals have the most assets? → Filters by org name keywords
3. Which nonprofits are based in California? → Queries irs_locations by state
4. Which organizations filed 990PF returns? → Filters by return type
5. Find nonprofits with high officer compensation → Queries officer_compensation field

## FEC Political Finance Queries
6. Which PACs spent the most in 2024? → Ranks by total_disbursements
7. How much did ActBlue raise in 2024? → Specific committee lookup
8. Which committees are based in Texas? → Filters fec_committees by state

## Cross-Dataset Queries
9. Which nonprofits have connections to political committees? → SQL JOIN on org name

## Document Search (Pinecone Vector Search)
10. What programs does United Way fund? → Semantic search across IRS text chunks
