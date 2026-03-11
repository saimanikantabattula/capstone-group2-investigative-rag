export default function CitationCard({ citation, index }) {
  const isIRS = citation.source === "IRS";

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

  return (
    <div className={"citation-card " + (isIRS ? "irs" : "fec")}>
      <div className="citation-index">[{index}]</div>
      <div className="citation-content">
        <div className="citation-source-badge">{citation.source}</div>
        {citation.org_name && <div className="citation-org">{citation.org_name}</div>}
        {citation.ein && citation.ein !== "000000000" && <div className="citation-meta">EIN: {citation.ein}</div>}
        <div className="citation-file">{citation.file_name}</div>
        {citation.snippet && (
          <div className="citation-snippet">
            &ldquo;{citation.snippet.slice(0, 200)}...&rdquo;
          </div>
        )}
        {refUrl && (
          <a href={refUrl} target="_blank" rel="noopener noreferrer" className="citation-ref-link">
            {refLabel}
          </a>
        )}
      </div>
    </div>
  );
}
