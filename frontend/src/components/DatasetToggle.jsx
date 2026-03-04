const OPTIONS = [
  { value: "both", label: "All Sources" },
  { value: "irs", label: "IRS 990" },
  { value: "fec", label: "FEC Filings" },
];

export default function DatasetToggle({ value, onChange }) {
  return (
    <div className="dataset-toggle">
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          className={`toggle-btn ${value === opt.value ? "active" : ""}`}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
