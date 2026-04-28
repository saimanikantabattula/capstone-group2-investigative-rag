import { useState, useEffect, useRef, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import SearchBar from "./components/SearchBar";
import DatasetToggle from "./components/DatasetToggle";
import AnswerPanel from "./components/AnswerPanel";
import CitationCard from "./components/CitationCard";
import { queryRAG, getSuggestions } from "./api/client";
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

const LOADING_MESSAGES = [
  "Searching 1.2M+ organizations...",
  "Querying financial database...",
  "Retrieving relevant documents...",
  "Generating cited answer...",
];

function SkeletonLoader() {
  return (
    <div className="skeleton-wrapper">
      <div className="skeleton-meta">
        <div className="skeleton-pill" />
        <div className="skeleton-pill" style={{width:"120px"}} />
        <div className="skeleton-pill" style={{width:"160px"}} />
      </div>
      <div className="skeleton-answer">
        {[90,75,85,60,80,70].map((w,i) => (
          <div key={i} className="skeleton-line" style={{width:`${w}%`}} />
        ))}
      </div>
      <div className="skeleton-cards">
        {[1,2,3].map(i => (
          <div key={i} className="skeleton-card">
            <div className="skeleton-badge" />
            <div style={{flex:1}}>
              <div className="skeleton-line" style={{width:"60%",marginBottom:"8px"}} />
              <div className="skeleton-line" style={{width:"40%",marginBottom:"8px"}} />
              <div className="skeleton-line" style={{width:"80%"}} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  async function handleCopy() {
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000); } catch {}
  }
  return (
    <button className={`copy-btn ${copied ? "copy-btn--copied" : ""}`} onClick={handleCopy}>
      {copied ? (
        <><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M20 6L9 17l-5-5"/></svg>Copied!</>
      ) : (
        <><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy</>
      )}
    </button>
  );
}

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [question, setQuestion] = useState("");
  const [dataset, setDataset] = useState("both");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(0);
  const [error, setError] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    try { return localStorage.getItem("rag_dark_mode") !== "false"; } catch { return true; }
  });
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("rag_chat_history");
      return saved ? JSON.parse(saved) : [];
    } catch { return []; }
  });
  const [responseTime, setResponseTime] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [resultVisible, setResultVisible] = useState(false);
  const loadingInterval = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
    try { localStorage.setItem("rag_dark_mode", darkMode); } catch {}
  }, [darkMode]);

  useEffect(() => {
    try { localStorage.setItem("rag_chat_history", JSON.stringify(history)); } catch {}
  }, [history]);

  useEffect(() => {
    function handleKey(e) {
      if (e.key === "/" && document.activeElement.tagName !== "INPUT") {
        e.preventDefault();
        document.querySelector(".search-input")?.focus();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  const handleSearch = useCallback(async (q) => {
    const query = q || question;
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setResultVisible(false);
    setSuggestions([]);
    setResponseTime(null);
    setLoadingMsg(0);

    loadingInterval.current = setInterval(() => {
      setLoadingMsg(prev => (prev + 1) % LOADING_MESSAGES.length);
    }, 1200);

    const start = Date.now();
    try {
      const data = await queryRAG(query, dataset, 5);
      setResult(data);
      setTimeout(() => setResultVisible(true), 50);

      // Save to history with timestamp and source
      setHistory(prev => {
        const filtered = prev.filter(h => h.question !== query);
        const newEntry = {
          question: query,
          ts: Date.now(),
          dataset,
          source: data.sources_used?.[0] || "RAG",
          answer: data.answer?.slice(0, 120) || "",
        };
        return [newEntry, ...filtered].slice(0, 50);
      });

      try {
        const sugg = await getSuggestions(query, dataset);
        setSuggestions(sugg.suggestions || []);
      } catch { setSuggestions([]); }

      setResponseTime(((Date.now() - start) / 1000).toFixed(1));
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 150);
    } catch {
      setError("Could not reach the API. Make sure the backend is running on port 8000.");
    } finally {
      setLoading(false);
      clearInterval(loadingInterval.current);
    }
  }, [question, dataset]);

  function handleNewChat() {
    setQuestion("");
    setResult(null);
    setSuggestions([]);
    setError(null);
    setResponseTime(null);
    document.querySelector(".search-input")?.focus();
  }

  function handleHistorySelect(h) {
    setQuestion(h.question);
    handleSearch(h.question);
  }

  function handleExportPDF() {
    if (!result) return;
    const content = `INVESTIGATIVE RAG — ANSWER REPORT\n\nQuestion: ${result.question || question}\n\nANSWER:\n${result.answer}\n\nSOURCE DOCUMENTS (${result.citations?.length || 0}):\n${result.citations?.map((c, i) => `[${i+1}] ${c.org_name || c.file_name} | ${c.source} | EIN: ${c.ein || "N/A"}`).join("\n") || "None"}\n\nGenerated by Investigative RAG — Capstone Group 2`.trim();
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `investigative-rag-${Date.now()}.txt`; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <Sidebar
        history={history}
        currentQuestion={question}
        onSelect={handleHistorySelect}
        onClear={() => { setHistory([]); localStorage.removeItem("rag_chat_history"); }}
        onNewChat={handleNewChat}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(prev => !prev)}
      />

      {/* Main content */}
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
            <div className="header-actions">
              <button className="theme-btn" onClick={() => setDarkMode(!darkMode)}
                title={darkMode ? "Light mode" : "Dark mode"}>
                {darkMode ? (
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                  </svg>
                ) : (
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                  </svg>
                )}
              </button>
              <button className="settings-btn" onClick={() => setSettingsOpen(true)}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                </svg>
                Settings
              </button>
            </div>
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
                <div className="settings-section-title">About</div>
                <div style={{fontSize:"0.78rem", color:"var(--text-muted)", lineHeight:1.6}}>
                  <p>Investigative RAG — Capstone Group 2</p>
                  <p style={{marginTop:"0.3rem"}}>Yeshiva University · 2025-2026</p>
                  <p style={{marginTop:"0.5rem", color:"var(--green)"}}>96.8% accuracy · 125 test questions</p>
                </div>
                <div className="settings-section-title" style={{marginTop:"1.25rem"}}>Team</div>
                {TEAM.map(name => (
                  <div key={name} style={{fontSize:"0.78rem", color:"var(--text-muted)", padding:"0.3rem 0", borderBottom:"1px solid var(--border)"}}>
                    {name}
                  </div>
                ))}
                <div className="settings-section-title" style={{marginTop:"1.25rem"}}>Links</div>
                <div style={{display:"flex", flexDirection:"column", gap:"0.4rem"}}>
                  {[
                    ["GitHub", "https://github.com/saimanikantabattula/capstone-group2-investigative-rag"],
                    ["IRS.gov", "https://apps.irs.gov/app/eos/"],
                    ["FEC.gov", "https://www.fec.gov/data/"],
                  ].map(([label, url]) => (
                    <a key={label} href={url} target="_blank" rel="noopener noreferrer"
                      style={{fontSize:"0.78rem", color:"var(--green)", textDecoration:"none"}}>
                      {label} →
                    </a>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}

        <main className="main">
          {/* Hero */}
          <div className="hero">
            <h1 className="hero-title">Follow the <span className="highlight">Money.</span></h1>
            <p className="hero-desc">
              Ask investigative questions across 1.2M+ IRS nonprofit organizations
              and FEC political finance records. Get cited answers backed by real documents.
            </p>
          </div>

          {/* Search */}
          <section className="search-section">
            <DatasetToggle value={dataset} onChange={setDataset} />
            <SearchBar value={question} onChange={setQuestion} onSearch={handleSearch} loading={loading} />
            <div className="hints-row">
              <span className="hints-label">Try:</span>
              {HINTS.map(h => (
                <button key={h} className="hint-btn"
                  onClick={() => { setQuestion(h); handleSearch(h); }}>{h}</button>
              ))}
            </div>
          </section>

          {/* Stats */}
          <div className="stats-bar">
            {[["1.2M+","Organizations"],["100K+","Indexed Chunks"],["38K+","FEC Committees"],["96.8%","Accuracy"]].map(([val,lbl]) => (
              <div key={lbl} className="stat-item">
                <span className="stat-value">{val}</span>
                <span className="stat-label">{lbl}</span>
              </div>
            ))}
          </div>

          {/* Error */}
          {error && (
            <div className="error-card">
              <div className="error-card-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
              </div>
              <div className="error-card-body">
                <p className="error-card-title">Connection Error</p>
                <p className="error-card-desc">{error}</p>
                <code className="error-card-code">uvicorn src.api.main:app --port 8000</code>
              </div>
              <button className="error-card-retry" onClick={() => { setError(null); handleSearch(question); }}>Retry</button>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="loading-state">
              <div className="loading-progress"><div className="loading-progress-bar" /></div>
              <p className="loading-msg">{LOADING_MESSAGES[loadingMsg]}</p>
              <p className="loading-sub">Searching {dataset === "irs" ? "IRS" : dataset === "fec" ? "FEC" : "IRS + FEC"} data sources</p>
              <SkeletonLoader />
            </div>
          )}

          {/* Suggestions */}
          {suggestions.length > 0 && result && !loading && (
            <div className="suggestions-panel">
              <p className="suggestions-title">Related Questions</p>
              <div className="suggestions-list">
                {suggestions.map((s, i) => (
                  <button key={i} className="suggestion-btn"
                    onClick={() => { setQuestion(s); handleSearch(s); }}>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                    </svg>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Results */}
          {result && !loading && (
            <section className={`results-section ${resultVisible ? "results-section--visible" : ""}`} ref={resultsRef}>
              <div className="response-meta">
                <span className="response-meta-item response-meta-item--success">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><path d="M20 6L9 17l-5-5"/></svg>
                  Answer ready
                </span>
                {responseTime && (
                  <span className="response-meta-item">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    {responseTime}s
                  </span>
                )}
                {result.sources_used && (
                  <span className="response-meta-item">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
                    {result.sources_used.join(", ")}
                  </span>
                )}
                <div className="response-meta-actions">
                  <CopyButton text={result.answer} />
                  <button className="export-btn" onClick={handleExportPDF}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                    Export
                  </button>
                </div>
              </div>

              <AnswerPanel answer={result.answer} sourcesUsed={result.sources_used} />

              {result.citations && result.citations.length > 0 && (
                <div className="citations-section">
                  <h3 className="citations-heading">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
                    Source Documents ({result.citations.length})
                  </h3>
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
                {TEAM.map(name => <span key={name} className="team-name">{name}</span>)}
              </div>
            </div>
            <div className="footer-right">
              <a href="https://github.com/saimanikantabattula/capstone-group2-investigative-rag" target="_blank" rel="noopener noreferrer" className="footer-link">GitHub</a>
              <a href="https://apps.irs.gov/app/eos/" target="_blank" rel="noopener noreferrer" className="footer-link">IRS.gov</a>
              <a href="https://www.fec.gov/data/" target="_blank" rel="noopener noreferrer" className="footer-link">FEC.gov</a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
