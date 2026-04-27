export default function CitationCard({ citation, index }) {
  const isIRS = citation.source === "IRS";

  // Build the reference URL
  let refUrl = null;
  let refLabel = null;

  if (isIRS && citation.ein && citation.ein !== "N/A" && citation.ein !== "" && citation.ein !== "000000000") {
    const cleanEin = citation.ein.replace(/\D/g, "");
    refUrl = "https://apps.irs.gov/app/eos/detailsPage?ein=" + cleanEin + "&type=CHARITIES";
    refLabel = "View on IRS.gov →";
  } else if (isIRS && citation.object_id && !citation.file_name.includes("PostgreSQL")) {
    refUrl = "https://s3.amazonaws.com/irs-form-990/" + citation.object_id + "_public.xml";
    refLabel = "View Filing XML →";
  } else if (!isIRS) {
    refUrl = "https://www.fec.gov/data/committees/";
    refLabel = "Browse FEC.gov →";
  }

  // Clean up the snippet — remove long number sequences and extra spaces
  const cleanSnippet = (text) => {
    if (!text) return "";
    return text
      .replace(/\b\d{10,}\b/g, "...")   // remove very long numbers (XML artifacts)
      .replace(/\s+/g, " ")              // collapse multiple spaces
      .replace(/(\d+\s){5,}/g, "...")    // remove sequences of numbers
      .trim()
      .slice(0, 180);
  };

  const snippet = cleanSnippet(citation.snippet);

  // Parse key financial data from snippet if it's a PostgreSQL citation
  const isPostgres = citation.file_name && citation.file_name.includes("PostgreSQL");
  const parseFinancials = (snippet) => {
    const items = [];
    const parts = snippet.split("|").map(p => p.trim());
    parts.forEach(part => {
      if (part.includes(":")) {
        const [key, val] = part.split(":").map(p => p.trim());
        if (val && key && key.length < 30) {
          items.push({ key, val });
        }
      }
    });
    return items.slice(0, 4); // show max 4 key-value pairs
  };

  const financials = isPostgres ? parseFinancials(citation.snippet || "") : [];

  return (
    <div className={"citation-card " + (isIRS ? "irs" : "fec")} style={{
      borderRadius: "10px",
      border: isIRS ? "1px solid #dbeafe" : "1px solid #fde68a",
      background: isIRS ? "#f0f7ff" : "#fffbeb",
      padding: "12px 14px",
      display: "flex",
      gap: "10px",
    }}>
      {/* Index Badge */}
      <div style={{
        width: "24px", height: "24px", borderRadius: "50%",
        background: isIRS ? "#3b82f6" : "#f59e0b",
        color: "white", fontSize: "11px", fontWeight: "700",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0, marginTop: "2px",
      }}>
        {index}
      </div>

      {/* Content */}
      <div style={{flex: 1, minWidth: 0}}>
        {/* Source badge + org name */}
        <div style={{display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px", flexWrap: "wrap"}}>
          <span style={{
            fontSize: "9px", fontWeight: "700", padding: "2px 7px",
            borderRadius: "20px", letterSpacing: "0.5px",
            background: isIRS ? "#3b82f6" : "#f59e0b",
            color: "white",
          }}>
            {citation.source}
          </span>
          {citation.org_name && (
            <span style={{
              fontSize: "11px", fontWeight: "600",
              color: isIRS ? "#1e40af" : "#92400e",
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
            }}>
              {citation.org_name.length > 40 ? citation.org_name.slice(0, 40) + "..." : citation.org_name}
            </span>
          )}
        </div>

        {/* EIN */}
        {citation.ein && citation.ein !== "000000000" && citation.ein !== "N/A" && (
          <div style={{fontSize: "9px", color: "#64748b", marginBottom: "4px"}}>
            EIN: {citation.ein}
          </div>
        )}

        {/* For PostgreSQL citations — show key financial data as tags */}
        {isPostgres && financials.length > 0 ? (
          <div style={{display: "flex", flexWrap: "wrap", gap: "4px", marginBottom: "6px"}}>
            {financials.map((item, i) => (
              <span key={i} style={{
                fontSize: "9px", padding: "2px 6px",
                background: "white", borderRadius: "4px",
                border: "1px solid " + (isIRS ? "#bfdbfe" : "#fde68a"),
                color: "#374151",
              }}>
                <strong>{item.key}:</strong> {item.val.length > 20 ? item.val.slice(0, 20) + "..." : item.val}
              </span>
            ))}
          </div>
        ) : snippet ? (
          // For Pinecone citations — show cleaned text snippet
          <div style={{
            fontSize: "9px", color: "#4b5563", lineHeight: "1.5",
            background: "white", borderRadius: "6px", padding: "6px 8px",
            border: "1px solid " + (isIRS ? "#dbeafe" : "#fde68a"),
            marginBottom: "6px",
          }}>
            &ldquo;{snippet}&rdquo;
          </div>
        ) : null}

        {/* Source file name — smaller and muted */}
        <div style={{fontSize: "8px", color: "#94a3b8", marginBottom: "4px"}}>
          {citation.file_name}
        </div>

        {/* Link */}
        {refUrl && (
          <a href={refUrl} target="_blank" rel="noopener noreferrer" style={{
            fontSize: "9px", fontWeight: "600",
            color: isIRS ? "#3b82f6" : "#d97706",
            textDecoration: "none",
          }}>
            {refLabel}
          </a>
        )}
      </div>
    </div>
  );
}
