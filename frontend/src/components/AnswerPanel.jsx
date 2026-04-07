import { useState } from "react";

export default function AnswerPanel({ answer, sourcesUsed }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function renderLine(line, i) {
    if (line.startsWith("# ")) return <h2 key={i} style={{fontSize:"1.1rem", fontWeight:700, marginBottom:"0.5rem", marginTop:"0.5rem"}}>{line.slice(2)}</h2>;
    if (line.startsWith("## ")) return <h3 key={i} style={{fontSize:"1rem", fontWeight:700, marginBottom:"0.4rem", marginTop:"0.4rem"}}>{line.slice(3)}</h3>;
    if (line.startsWith("- ")) return <li key={i} style={{marginLeft:"1.2rem", marginBottom:"0.2rem"}} dangerouslySetInnerHTML={{__html: formatInline(line.slice(2))}} />;
    if (/^\d+\.\s/.test(line)) return <li key={i} style={{marginLeft:"1.2rem", marginBottom:"0.3rem"}} dangerouslySetInnerHTML={{__html: formatInline(line.replace(/^\d+\.\s/, ""))}} />;
    if (line === "---") return <hr key={i} style={{border:"none", borderTop:"1px solid #e2e6ef", margin:"0.8rem 0"}} />;
    if (line.trim() === "") return <br key={i} />;
    return <p key={i} style={{marginBottom:"0.4rem"}} dangerouslySetInnerHTML={{__html: formatInline(line)}} />;
  }

  function formatInline(text) {
    return text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`(.+?)`/g, "<code style='background:#f0f2f7;padding:1px 4px;border-radius:3px;font-size:0.85em'>$1</code>");
  }

  return (
    <div className="answer-panel">
      <div className="answer-header">
        <span className="answer-label">Answer</span>
        <div style={{display:"flex", alignItems:"center", gap:"0.5rem"}}>
          {sourcesUsed && sourcesUsed.length > 0 && (
            <div className="sources-used">
              {sourcesUsed.map((s) => (
                <span key={s} className="source-tag">{s}</span>
              ))}
            </div>
          )}
          <button
            onClick={handleCopy}
            style={{
              padding:"0.25rem 0.75rem",
              fontSize:"0.72rem",
              fontWeight:600,
              border:"1px solid var(--border)",
              borderRadius:"6px",
              background: copied ? "var(--green)" : "var(--surface)",
              color: copied ? "white" : "var(--text-muted)",
              cursor:"pointer",
              transition:"all 0.2s",
              whiteSpace:"nowrap",
            }}
          >
            {copied ? "✓ Copied!" : "Copy"}
          </button>
        </div>
      </div>
      <div className="answer-body">
        {answer.split("\n").map((line, i) => renderLine(line, i))}
      </div>
    </div>
  );
}
