import { useState } from "react";

export default function Sidebar({ history, onSelect, onClear, currentQuestion, onNewChat, collapsed, onToggle }) {
  const [search, setSearch] = useState("");

  const filtered = history.filter(h =>
    h.question.toLowerCase().includes(search.toLowerCase())
  );

  const now = Date.now();
  const DAY = 86400000;
  const groups = [
    { label: "Today",     items: filtered.filter(h => now - h.ts < DAY) },
    { label: "Yesterday", items: filtered.filter(h => now - h.ts >= DAY && now - h.ts < 2 * DAY) },
    { label: "Older",     items: filtered.filter(h => now - h.ts >= 2 * DAY) },
  ].filter(g => g.items.length > 0);

  return (
    <aside className={`sidebar ${collapsed ? "sidebar--collapsed" : ""}`}>

      {/* Header row — collapse button + new chat */}
      <div className="sidebar-header">
        <button className="sidebar-toggle" onClick={onToggle} title={collapsed ? "Open sidebar" : "Close sidebar"}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6"/>
            <line x1="3" y1="12" x2="21" y2="12"/>
            <line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
        {!collapsed && (
          <button className="sidebar-new" onClick={onNewChat} title="New Search">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M12 5v14M5 12h14"/>
            </svg>
            New Search
          </button>
        )}
      </div>

      {/* Expanded content */}
      {!collapsed && (
        <div className="sidebar-body">
          {/* Search */}
          {history.length > 4 && (
            <div className="sidebar-search-box">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
              </svg>
              <input
                className="sidebar-search-input"
                type="text"
                placeholder="Search..."
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
          )}

          {/* List */}
          <div className="sidebar-list">
            {history.length === 0 ? (
              <div className="sidebar-empty">
                <p>No searches yet</p>
              </div>
            ) : groups.length > 0 ? (
              groups.map(group => (
                <div key={group.label} className="sidebar-group">
                  <p className="sidebar-group-label">{group.label}</p>
                  {group.items.map((h, i) => (
                    <button
                      key={i}
                      className={`sidebar-item ${currentQuestion === h.question ? "sidebar-item--active" : ""}`}
                      onClick={() => onSelect(h)}
                      title={h.question}
                    >
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={{flexShrink:0, marginTop:"2px"}}>
                        <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                      </svg>
                      <span className="sidebar-item-text">
                        {h.question.length > 34 ? h.question.slice(0,34)+"..." : h.question}
                      </span>
                    </button>
                  ))}
                </div>
              ))
            ) : (
              <div className="sidebar-empty"><p>No results</p></div>
            )}
          </div>

          {/* Clear */}
          {history.length > 0 && (
            <div className="sidebar-footer">
              <button className="sidebar-clear" onClick={onClear}>
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                </svg>
                Clear all
              </button>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
