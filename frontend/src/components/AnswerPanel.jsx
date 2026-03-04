export default function AnswerPanel({ answer, sourcesUsed }) {
  return (
    <div className="answer-panel">
      <div className="answer-header">
        <span className="answer-label">Answer</span>
        {sourcesUsed && sourcesUsed.length > 0 && (
          <div className="sources-used">
            {sourcesUsed.map((s) => (
              <span key={s} className="source-tag">{s}</span>
            ))}
          </div>
        )}
      </div>
      <div className="answer-body">
        {answer.split("\n").map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
    </div>
  );
}
