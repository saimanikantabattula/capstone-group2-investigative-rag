export default function SearchBar({ value, onChange, onSearch, loading }) {
  function handleKeyDown(e) {
    if (e.key === "Enter" && !loading) {
      onSearch(value);
    }
  }

  return (
    <div className="search-bar">
      <input
        className="search-input"
        type="text"
        placeholder="Ask an investigative question about IRS 990 or FEC filings..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
      />
      <button
        className="search-btn"
        onClick={() => onSearch(value)}
        disabled={loading || !value.trim()}
      >
        {loading ? "Searching..." : "Search"}
      </button>
    </div>
  );
}
