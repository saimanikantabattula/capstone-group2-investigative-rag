import { useState, useEffect } from "react";
import SearchBar from "./components/SearchBar";
import DatasetToggle from "./components/DatasetToggle";
import AnswerPanel from "./components/AnswerPanel";
import CitationCard from "./components/CitationCard";
import { queryRAG } from "./api/client";
import "./App.css";

const HINTS = [
  "Which nonprofits raised the most money?",
  "Which PACs spent the most in 2024?",
  "Find nonprofits with high officer compensation",
  "What did Harris for President report in 2024?",
  "Which organizations filed 990PF returns?",
];

const TEAM = [
  "Sai Manikanta Battula",
  "Bhavani Danthuri",
  "Ability Chikanya",
  "Hanok Naidu Suravarapu",
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [dataset, setDataset] = useState("both");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [history, setHistory] = useState([]);

  // Apply dark mode
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  async function handleSearch(q) {
    const query = q || question;
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    // Save to history
    setHistory((prev) => {
      const filtered = prev.filter((h) => h !== query);
      return [query, ...filtered].slice(0, 10);
    });

    try {
      const data = await queryRAG(query, dataset, 5);
      setResult(data);
    } catch (err) {
      setError("Could not reach the API. Make sure the backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  }

  function handleHistoryClick(q) {
    setQuestion(q);
    setSettingsOpen(false);
    handleSearch(q);
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="header-left">
            <div className="logo-row">
              <div className="logo-icon">IR</div>
              <div className="logo-mark">Investigative <span>RAG</span></div>
            </div>
            <p className="header-sub">IRS 990 &amp; FEC Filing Intelligence System</p>
          </div>

          {/* Settings Button */}
          <button className="settings-btn" onClick={() => setSettingsOpen(true)}>
            ⚙ Settings
          </button>
        </div>
      </header>

      {/* Settings Panel */}
      {settingsOpen && (
        <>
          <div className="settings-overlay" onClick={() => setSettingsOpen(false)} />
          <div className="settings-panel">
            <div className="settings-header">
              <span className="settings-title">Settings</span>
              <button className="settings-close" onClick={() => setSettingsOpen(false)}>✕</button>
            </div>
            <div className="settings-body">

              {/* Theme */}
              <div className="settings-section-title">Appearance</div>
              <div className="theme-toggle">
                <div className="theme-toggle-label">
                  <span className="theme-icon">{darkMode ? "🌙" : "☀️"}</span>
                  {darkMode ? "Dark Mode" : "Light Mode"}
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={darkMode}
                    onChange={(e) => setDarkMode(e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>

              {/* Search History */}
              <div className="settings-section-title">Search History</div>
              <div className="history-list">
                {history.length === 0 ? (
                  <p className="history-empty">No searches yet</p>
                ) : (
                  history.map((h, i) => (
                    <button
                      key={i}
                      className="history-item"
                      onClick={() => handleHistoryClick(h)}
                    >
                      {h}
                    </button>
                  ))
                )}
              </div>
              {history.length > 0 && (
                <button className="clear-history-btn" onClick={() => setHistory([])}>
                  Clear History
                </button>
              )}

            </div>
          </div>
        </>
      )}

      <main className="main">
        {/* Hero */}
        <div className="hero">
          <h1 className="hero-title">
            Follow the <span className="highlight">Money.</span>
          </h1>
          <p className="hero-desc">
            Ask investigative questions across 100,000+ IRS nonprofit filings
            and FEC political finance records. Get cited answers backed by real documents.
          </p>
        </div>

        {/* Search */}
        <section className="search-section">
          <DatasetToggle value={dataset} onChange={setDataset} />
          <SearchBar
            value={question}
            onChange={setQuestion}
            onSearch={handleSearch}
            loading={loading}
          />
          <div className="hints-row">
            <span className="hints-label">Try:</span>
            {HINTS.map((h) => (
              <button
                key={h}
                className="hint-btn"
                onClick={() => {
                  setQuestion(h);
                  handleSearch(h);
                }}
              >
                {h}
              </button>
            ))}
          </div>
        </section>

        {/* Stats */}
        <div className="stats-bar">
          <div className="stat-item">
            <span className="stat-value">100K+</span>
            <span className="stat-label">IRS Filings Indexed</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">75K</span>
            <span className="stat-label">Document Chunks</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">38K+</span>
            <span className="stat-label">FEC Committees</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">2024–26</span>
            <span className="stat-label">Coverage Period</span>
          </div>
        </div>

        {/* Error */}
        {error && <div className="error-banner">{error}</div>}

        {/* Loading */}
        {loading && (
          <div className="loading-state">
            <div className="loading-dots">
              <span /><span /><span />
            </div>
            <p>Retrieving documents and generating cited answer...</p>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <section className="results-section">
            <AnswerPanel answer={result.answer} sourcesUsed={result.sources_used} />
            {result.citations && result.citations.length > 0 && (
              <div className="citations-section">
                <h3 className="citations-heading">Source Documents ({result.citations.length})</h3>
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

      {/* Footer */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-left">
            <p>Capstone Group 2 — Multi-Agent RAG / Investigative Intelligence</p>
            <div className="footer-team">
              {TEAM.map((name) => (
                <span key={name} className="team-name">{name}</span>
              ))}
            </div>
          </div>
          <div className="footer-right">
            <p>IRS 990 + FEC Filings</p>
            <p>Multi-Agent RAG System</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
