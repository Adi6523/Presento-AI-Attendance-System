import { useState } from 'react';
import { login, AuthUser } from '../api';

interface LoginPageProps {
  onLogin: (user: AuthUser) => void;
}

export default function LoginPage({ onLogin }: LoginPageProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const user = await login(username, password);
      onLogin(user);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-root">
      <div className="login-card">
        <div className="login-brand">
          presento<span>.</span>
        </div>
        <p className="login-subtitle">Smart Biometric Attendance &amp; Academic Management</p>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="input-group">
            <label htmlFor="username">Username or Student ID</label>
            <input
              id="username"
              type="text"
              placeholder="e.g. teacher1 or 23UAM090"
              value={username}
              onChange={e => setUsername(e.target.value)}
              required
              autoFocus
            />
          </div>
          <div className="input-group">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              required
            />
          </div>

          {error && <div className="login-error">⚠️ {error}</div>}

          <button type="submit" className="btn-login" disabled={loading}>
            {loading ? 'Signing In…' : 'Sign In →'}
          </button>
        </form>

        <div className="login-roles">
          <span className="role-chip admin">Admin</span>
          <span className="role-chip teacher">Teacher</span>
          <span className="role-chip student">Student</span>
        </div>

        <div className="login-hint">
          <p style={{ marginBottom: 10, fontWeight: 700, fontSize: 11, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '1px' }}>
            Quick-Fill Demo Accounts
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              ['admin', 'admin123', 'admin', 'Full Administrator'],
              ['teacher1', 'teacher123', 'teacher', 'Faculty Advisor (TY CSE-A)'],
            ].map(([u, p, r, label]) => (
              <div
                key={u}
                className="demo-account-pill"
                onClick={() => { setUsername(u); setPassword(p); }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '6px 10px',
                  borderRadius: 8,
                  fontSize: 12,
                  border: '1px solid rgba(255,255,255,0.06)',
                  background: 'rgba(255,255,255,0.025)'
                }}
              >
                <div>
                  <code style={{ color: '#ffffff', fontWeight: 600 }}>{u}</code>
                  <span style={{ color: '#64748b', fontSize: 11, marginLeft: 8 }}>{label}</span>
                </div>
                <span className={`role-chip ${r}`} style={{ fontSize: 10, padding: '1px 8px' }}>
                  {r}
                </span>
              </div>
            ))}
          </div>
          <p style={{ marginTop: 10, fontSize: 11, color: '#64748b', textAlign: 'center' }}>
            💡 Click <strong>Admin</strong> or <strong>Teacher</strong> to auto-fill credentials. Student accounts log in with their <strong>Student ID</strong> once registered.
          </p>
        </div>
      </div>
    </div>
  );
}
