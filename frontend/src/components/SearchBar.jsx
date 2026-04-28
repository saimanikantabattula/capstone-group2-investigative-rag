import { useRef, useEffect } from "react";

export default function SearchBar({ value, onChange, onSearch, loading }) {
  const inputRef = useRef(null);

  // Auto-focus on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  function handleKeyDown(e) {
    if (e.key === "Enter" && !loading) onSearch(value);
    if (e.key === "Escape") { onChange(""); inputRef.current?.focus(); }
  }

  return (
    <div className={`search-bar ${loading ? "search-bar--loading" : ""}`}>
      <div className="search-icon">
        {loading ? (
          <svg className="search-spinner" width="18" height="18" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5"
              strokeDasharray="55" strokeDashoffset="15" strokeLinecap="round"/>
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
          </svg>
        )}
      </div>
      <input
        ref={inputRef}
        className="search-input"
        type="text"
        placeholder="Ask an investigative question about IRS 990 or FEC filings..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
        autoComplete="off"
        spellCheck="false"
      />
      {value && !loading && (
        <button className="search-clear-btn"
          onClick={() => { onChange(""); inputRef.current?.focus(); }}
          title="Clear (Esc)" aria-label="Clear search">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12"/>
          </svg>
        </button>
      )}
      <button className="search-btn" onClick={() => onSearch(value)}
        disabled={loading || !value.trim()}>
        {loading ? (
          <span className="search-btn-dots">
            <span/><span/><span/>
          </span>
        ) : "Search"}
      </button>
    </div>
  );
}
