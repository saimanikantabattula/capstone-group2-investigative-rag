export default function CitationCard({ citation, index }) {
  const isIRS = citation.source === "IRS";

  return (
    <div className={`citation-card ${isIRS ? "irs" : "fec"}`}>
      <div className="citation-index">[{index}]</div>
      <div className="citation-content">
        <div className="citation-source-badge">{citation.source}</div>
        {citation.org_name && (
          <div className="citation-org">{citation.org_name}</div>
        )}
        {citation.ein && (
          <div className="citation-meta">EIN: {citation.ein}</div>
        )}
        <div className="citation-file">{citation.file_name}</div>
        {citation.snippet && (
          <div className="citation-snippet">
            &ldquo;{citation.snippet.slice(0, 200)}...&rdquo;
          </div>
        )}
      </div>
    </div>
  );
}
