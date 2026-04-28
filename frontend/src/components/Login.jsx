import { useState } from "react";

const USERS = {
  "sai":     { password: "sai2026",     name: "Sai Manikanta Battula",  role: "System Architect" },
  "bhavani": { password: "bhavani2026", name: "Bhavani Danthuri",       role: "Documentation Lead" },
  "ability": { password: "ability2026", name: "Ability Chikanya",       role: "Frontend Engineer" },
  "hanok":   { password: "hanok2026",   name: "Hanok Naidu Suravarapu", role: "ML Engineer" },
  "demo":    { password: "demo2026",    name: "Guest",                  role: "Demo Access" },
};

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [showPass, setShowPass] = useState(false);
  const [focused, setFocused]   = useState(null);

  function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    setTimeout(() => {
      const user = USERS[username.toLowerCase().trim()];
      if (user && user.password === password) {
        onLogin({ username: username.toLowerCase().trim(), name: user.name, role: user.role });
      } else {
        setError("Wrong credentials. Try again.");
        setLoading(false);
      }
    }, 800);
  }

  return (
    <div className="lp-root">
      {/* Animated grid background */}
      <div className="lp-grid" />

      {/* Floating orbs */}
      <div className="lp-orb lp-orb--1" />
      <div className="lp-orb lp-orb--2" />
      <div className="lp-orb lp-orb--3" />

      {/* Center card */}
      <div className="lp-card">

        {/* Top — logo only */}
        <div className="lp-top">
          <div className="lp-logo">
            <span className="lp-logo-ir">IR</span>
          </div>
          <div className="lp-title-block">
            <h1 className="lp-title">Investigative<br/><span>RAG</span></h1>
          </div>
        </div>

        {/* Divider with label */}
        <div className="lp-divider">
          <span>sign in to continue</span>
        </div>

        {/* Form */}
        <form className="lp-form" onSubmit={handleSubmit}>
          {/* Username field */}
          <div className={`lp-field ${focused === "user" ? "lp-field--focused" : ""}`}>
            <label className="lp-field-label">USERNAME</label>
            <input
              className="lp-field-input"
              type="text"
              placeholder="your username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              onFocus={() => setFocused("user")}
              onBlur={() => setFocused(null)}
              autoFocus
              autoComplete="username"
              required
            />
            <div className="lp-field-line" />
          </div>

          {/* Password field */}
          <div className={`lp-field ${focused === "pass" ? "lp-field--focused" : ""}`}>
            <label className="lp-field-label">PASSWORD</label>
            <div className="lp-field-row">
              <input
                className="lp-field-input"
                type={showPass ? "text" : "password"}
                placeholder="your password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                onFocus={() => setFocused("pass")}
                onBlur={() => setFocused(null)}
                autoComplete="current-password"
                required
              />
              <button type="button" className="lp-eye" onClick={() => setShowPass(!showPass)} tabIndex={-1}>
                {showPass ? (
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
                    <line x1="1" y1="1" x2="23" y2="23"/>
                  </svg>
                ) : (
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
                    <circle cx="12" cy="12" r="3"/>
                  </svg>
                )}
              </button>
            </div>
            <div className="lp-field-line" />
          </div>

          {/* Error */}
          {error && (
            <p className="lp-error">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              {error}
            </p>
          )}

          {/* Submit */}
          <button className="lp-btn" type="submit" disabled={loading || !username || !password}>
            {loading ? (
              <span className="lp-btn-dots"><span/><span/><span/></span>
            ) : (
              <>
                <span>Enter</span>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
              </>
            )}
          </button>
        </form>


      </div>
    </div>
  );
}
