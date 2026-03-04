import { useState } from "react";
import SearchBar from "./components/SearchBar";
import DatasetToggle from "./components/DatasetToggle";
import AnswerPanel from "./components/AnswerPanel";
import CitationCard from "./components/CitationCard";
import { queryRAG } from "./api/client";
import "./App.css";

export default function App() {
  const [question, setQuestion] = useState("");
  const [dataset, setDataset] = useState("both");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSearch(q) {
    if (!q.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await queryRAG(q, dataset, 5);
      setResult(data);
    } catch (err) {
      setError("Could not reach the API. Make sure the backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo-mark">INVESTIGATIVE RAG</div>
          <p className="header-sub">
            IRS 990 &amp; FEC Filing Intelligence — Group 2
          </p>
        </div>
      </header>

      <main className="main">
        <section className="search-section">
          <DatasetToggle value={dataset} onChange={setDataset} />
          <SearchBar
            value={question}
            onChange={setQuestion}
            onSearch={handleSearch}
            loading={loading}
          />
          <div className="example-queries">
            <span className="example-label">Try:</span>
            {[
              "Which nonprofits raised the most money?",
              "Which PACs spent the most in 2024?",
              "Find nonprofits with high officer compensation",
            ].map((ex) => (
              <button
                key={ex}
                className="example-btn"
                onClick={() => {
                  setQuestion(ex);
                  handleSearch(ex);
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        </section>

        {error && <div className="error-banner">{error}</div>}

        {loading && (
          <div className="loading-state">
            <div className="loading-dots">
              <span /><span /><span />
            </div>
            <p>Searching across filings and generating answer...</p>
          </div>
        )}

        {result && !loading && (
          <section className="results-section">
            <AnswerPanel
              answer={result.answer}
              sourcesUsed={result.sources_used}
            />
            {result.citations && result.citations.length > 0 && (
              <div className="citations-section">
                <h3 className="citations-heading">Source Documents</h3>
                <div className="citations-grid">
                  {result.citations.map((citation, i) => (
                    <CitationCard key={i} citation={citation} index={i + 1} />
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Capstone Group 2 · Multi-Agent RAG · IRS 990 + FEC Filings</p>
      </footer>
    </div>
  );
}
