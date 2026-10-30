import { useEffect, useState } from 'react';
import {
  AuthUser, logout, getStudentAttendance, changePassword, listSessions, downloadStudentAttendanceCSV
} from '../api';
import Modal from '../components/Modal';

interface Props { user: AuthUser; onLogout: () => void; }
type View = 'My Attendance' | 'Class Schedule' | 'My Profile';

const STATUS_LABELS: Record<string, string> = {
  present: 'Present', late: 'Late', absent: 'Absent',
  nc: 'NC', present_nc: 'Present-NC',
};

export default function StudentPortal({ user, onLogout }: Props) {
  const [view, setView]               = useState<View>('My Attendance');
  const [data, setData]               = useState<any>(null);
  const [sessions, setSessions]       = useState<any[]>([]);
  const [statusFilter, setStatusFilter] = useState('all');
  const [error, setError]             = useState('');
  const [loading, setLoading]         = useState(true);
  const [modal, setModal]             = useState<{
    title: string; message: string;
    type: 'success' | 'error' | 'warning' | 'confirm';
    onConfirm?: () => void;
  } | null>(null);

  const showModal = (m: typeof modal) => setModal(m);
  const closeModal = () => setModal(null);

  const loadSessions = () => {
    listSessions().then(setSessions).catch(() => {});
  };

  const loadAttendance = () => {
    loadSessions();
    if (!user.student_id) { setLoading(false); return; }
    setLoading(true);
    getStudentAttendance(user.student_id)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { loadAttendance(); }, []);

  const rate = data?.attendance_rate;
  const ratePercent = rate != null ? Math.round(rate * 100) : null;
  const isAtRisk = ratePercent != null && ratePercent < 75;

  // Filter records
  const records = (data?.records || []).filter((r: any) => {
    if (statusFilter === 'all') return true;
    return r.status === statusFilter;
  });

  const handleExportCSV = async () => {
    if (!user.student_id) return;
    try {
      await downloadStudentAttendanceCSV(user.student_id);
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <main className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="brand">presento<span>.</span></div>
        <nav>
          {(['My Attendance', 'Class Schedule', 'My Profile'] as View[]).map(v => (
            <button
              key={v}
              className={`nav-btn ${view === v ? 'active' : ''}`}
              onClick={() => setView(v)}
            >
              {v === 'My Attendance' && '📊 '}
              {v === 'Class Schedule' && '📅 '}
              {v === 'My Profile' && '👤 '}
              {v}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="edge-indicator">
            <strong>Connected Class</strong>
            <small>{data?.profile?.class_label ?? 'TY CSE-AIML A'}</small>
          </div>
          <div className="user-info">
            <span className="role-badge student">Student</span>
            <span style={{ fontWeight: 600, color: '#ffffff' }}>{user.username}</span>
            {user.student_id && (
              <small style={{ color: '#94a3b8' }}>ID: <code>{user.student_id}</code></small>
            )}
            <button
              className="btn-logout"
              onClick={() => showModal({
                title: 'Sign out',
                message: 'Are you sure you want to sign out?',
                type: 'confirm',
                onConfirm: () => { logout(); onLogout(); },
              })}
            >
              Sign out
            </button>
          </div>
        </div>
      </aside>

      {/* ── Content ── */}
      <section className="content">
        <header className="page-header">
          <div>
            <p className="eyebrow">STUDENT PORTAL</p>
            <h1>{view}</h1>
            {data?.profile && (
              <div style={{ display: 'flex', gap: 12, marginTop: 6, flexWrap: 'wrap', fontSize: 13, color: '#94a3b8' }}>
                <span><strong>Class:</strong> <code style={{ color: '#34d6b5' }}>{data.profile.class_label}</code></span>
                <span>•</span>
                <span><strong>Dept:</strong> {data.profile.department_name}</span>
                {data.profile.roll_number && (
                  <>
                    <span>•</span>
                    <span><strong>Roll No:</strong> <code style={{ color: '#ffffff' }}>{data.profile.roll_number}</code></span>
                  </>
                )}
              </div>
            )}
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            {view === 'My Attendance' && (
              <>
                <button className="btn-export" onClick={handleExportCSV}>
                  📥 Export CSV
                </button>
                <button className="btn-secondary" onClick={loadAttendance}>
                  ↻ Refresh
                </button>
              </>
            )}
            {view === 'Class Schedule' && (
              <button className="btn-secondary" onClick={loadSessions}>
                ↻ Refresh
              </button>
            )}
          </div>
        </header>

        {error && (
          <div className="alert-error">
            {error}
            <button className="alert-close" onClick={() => setError('')}>×</button>
          </div>
        )}

        {/* ── MY ATTENDANCE ── */}
        {view === 'My Attendance' && (
          <>
            {!user.student_id && (
              <div className="alert-error" style={{ marginBottom: 16 }}>
                ⚠️ Your account is not linked to a student ID yet. Please ask your administrator to link it.
              </div>
            )}

            {isAtRisk && (
              <div className="alert-risk">
                ⚠️ Your overall attendance is <strong>{ratePercent}%</strong>, which is below the 75% minimum requirement.
                Please ensure you attend all upcoming scheduled lectures.
              </div>
            )}

            {data && (
              <div className="cards-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
                <div className={`card ${isAtRisk ? 'danger' : ''}`}>
                  <p>Overall Attendance Rate</p>
                  <strong style={{ color: isAtRisk ? '#f87171' : '#34d6b5' }}>
                    {ratePercent != null ? `${ratePercent}%` : '—'}
                  </strong>
                </div>
                <div className="card">
                  <p>Lectures Attended</p>
                  <strong>{data.attended_sessions}</strong>
                </div>
                <div className="card">
                  <p>Total Class Lectures</p>
                  <strong>{data.total_sessions}</strong>
                </div>
              </div>
            )}

            {/* Subject-wise breakdown */}
            {data?.subjects && data.subjects.length > 0 && (
              <div className="panel" style={{ marginTop: 20 }}>
                <h2>Subject-wise Attendance</h2>
                <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
                  Attendance percentage breakdown across all registered subjects for your division.
                </p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
                  {data.subjects.map((sub: any) => {
                    const subPercent = sub.attendance_rate != null ? Math.round(sub.attendance_rate * 100) : null;
                    const subRisk = sub.is_at_risk;
                    return (
                      <div
                        key={sub.subject_id}
                        style={{
                          background: 'rgba(255, 255, 255, 0.035)',
                          border: `1px solid ${subRisk ? 'rgba(239, 68, 68, 0.35)' : 'rgba(255, 255, 255, 0.08)'}`,
                          borderRadius: 14,
                          padding: 18,
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                          <div>
                            <strong style={{ fontSize: 15, display: 'block', color: '#ffffff' }}>{sub.subject_name}</strong>
                            <small style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>{sub.subject_code}</small>
                          </div>
                          <span className={`status-chip ${subRisk ? 'absent' : 'present'}`} style={{ fontSize: 12 }}>
                            {subPercent != null ? `${subPercent}%` : 'N/A'}
                          </span>
                        </div>

                        <div className="progress-bar-wrap" style={{ marginTop: 14, height: 6 }}>
                          <div
                            className={`progress-bar-fill ${subRisk ? 'danger' : 'safe'}`}
                            style={{ width: `${Math.min(subPercent ?? 0, 100)}%` }}
                          />
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginTop: 8 }}>
                          <span>Attended: {sub.attended_sessions} / {sub.total_sessions}</span>
                          {subRisk ? (
                            <span style={{ color: '#f87171', fontWeight: 600 }}>At Risk (&lt;75%)</span>
                          ) : (
                            <span style={{ color: '#34d6b5' }}>Good Standing</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Attendance History */}
            <div className="panel" style={{ marginTop: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
                <h2>Attendance Records Log</h2>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>Filter:</span>
                  <select
                    value={statusFilter}
                    onChange={e => setStatusFilter(e.target.value)}
                    style={{ padding: '6px 12px', background: 'rgba(8,12,24,0.8)', border: '1px solid var(--border-subtle)', color: '#ffffff', borderRadius: 8, fontSize: 13 }}
                  >
                    <option value="all">All Records</option>
                    <option value="present">Present Only</option>
                    <option value="late">Late Only</option>
                    <option value="absent">Absent Only</option>
                  </select>
                </div>
              </div>

              {loading ? (
                <p style={{ color: '#94a3b8', padding: 16 }}>Loading attendance logs…</p>
              ) : (
                <div className="table-responsive">
                  <table>
                    <thead>
                      <tr>
                        <th>Date &amp; Time</th>
                        <th>Subject / Lecture</th>
                        <th>Teacher</th>
                        <th>Room</th>
                        <th>Status</th>
                        <th>Confidence</th>
                        <th>Reason / Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {records.length === 0 ? (
                        <tr>
                          <td colSpan={7} style={{ textAlign: 'center', color: '#94a3b8', padding: 28 }}>
                            No attendance records match your criteria.
                          </td>
                        </tr>
                      ) : (
                        records.map((r: any, i: number) => (
                          <tr key={i}>
                            <td style={{ fontSize: 13 }}>{new Date(r.observed_at + (r.observed_at.endsWith('Z') ? '' : 'Z')).toLocaleString()}</td>
                            <td>
                              <strong style={{ color: '#ffffff' }}>{r.subject_name ?? r.course_code ?? '—'}</strong>
                              {r.subject_code && <small style={{ display: 'block', color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>{r.subject_code}</small>}
                            </td>
                            <td>{r.teacher_name ?? '—'}</td>
                            <td>{r.room ?? '—'}</td>
                            <td>
                              <span className={`status-chip ${r.status}`}>
                                {STATUS_LABELS[r.status] ?? r.status}
                              </span>
                              {r.manually_added ? <span style={{ fontSize: 11, marginLeft: 6, color: '#94a3b8' }} title="Manual adjustment by faculty">✏️</span> : null}
                            </td>
                            <td>{r.manually_added ? 'Manual' : `${Math.round(r.confidence * 100)}%`}</td>
                            <td>
                              {r.reason ? (
                                <span title={r.reason} style={{ fontSize: 12, color: '#94a3b8' }}>
                                  {r.reason.length > 35 ? r.reason.slice(0, 35) + '…' : r.reason}
                                </span>
                              ) : '—'}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}

        {/* ── CLASS SCHEDULE ── */}
        {view === 'Class Schedule' && (
          <div className="panel">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div>
                <h2>Enrolled Lecture Timetable</h2>
                <p style={{ color: '#94a3b8', margin: '4px 0 0' }}>
                  All scheduled lectures and laboratory sessions for your class division.
                </p>
              </div>
              <button className="btn-secondary" onClick={loadSessions}>↻ Refresh</button>
            </div>
            <div className="table-responsive">
              <table>
                <thead>
                  <tr>
                    <th>Subject</th>
                    <th>Class</th>
                    <th>Teacher</th>
                    <th>Room</th>
                    <th>Starts At</th>
                    <th>Ends At</th>
                    <th>Session Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.length === 0 ? (
                    <tr>
                      <td colSpan={7} style={{ textAlign: 'center', color: '#94a3b8', padding: 28 }}>
                        No sessions scheduled yet.
                      </td>
                    </tr>
                  ) : (
                    sessions.map((s: any) => {
                      const start = new Date(s.starts_at + (s.starts_at.endsWith('Z') ? '' : 'Z'));
                      const end   = new Date(s.ends_at + (s.ends_at.endsWith('Z') ? '' : 'Z'));
                      const isOpen = s.status === 'open';
                      return (
                        <tr key={s.id}>
                          <td>
                            <strong style={{ color: '#ffffff' }}>{s.subject_name ?? s.course_code}</strong>
                            {s.subject_code && <small style={{ display: 'block', color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>{s.subject_code}</small>}
                          </td>
                          <td><span className="status-chip present">{s.class_label ?? '—'}</span></td>
                          <td>{s.teacher_username ?? '—'}</td>
                          <td><code style={{ color: '#ffffff' }}>{s.room}</code></td>
                          <td style={{ fontSize: 13 }}>{start.toLocaleString()}</td>
                          <td style={{ fontSize: 13 }}>{end.toLocaleString()}</td>
                          <td>
                            {isOpen ? (
                              <span className="status-chip present">● Active Session</span>
                            ) : (
                              <span className="status-chip nc">Completed</span>
                            )}
                          </td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── MY PROFILE ── */}
        {view === 'My Profile' && (
          <ProfileView user={user} showModal={showModal} setError={setError} data={data} />
        )}

        {modal && (
          <Modal
            title={modal.title}
            message={modal.message}
            type={modal.type}
            onConfirm={modal.onConfirm ?? (modal.type === 'confirm' ? () => { logout(); onLogout(); } : undefined)}
            onClose={closeModal}
            confirmLabel={modal.type === 'confirm' ? 'Yes, Sign Out' : 'OK'}
          />
        )}
      </section>
    </main>
  );
}

// ── Profile & Password Management ─────────────────────────────────────────────
function ProfileView({ user, showModal, setError, data }: any) {
  const [cur, setCur]   = useState('');
  const [nw,  setNw]    = useState('');
  const [conf, setConf] = useState('');
  const [busy, setBusy] = useState(false);

  const handlePw = async (e: React.FormEvent) => {
    e.preventDefault();
    if (nw !== conf) { setError('New passwords do not match.'); return; }
    setBusy(true);
    try {
      await changePassword(cur, nw);
      setCur(''); setNw(''); setConf('');
      showModal({ title: 'Password Updated', message: 'Your account password was updated successfully.', type: 'success' });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="panel">
        <h2>Student Academic Profile</h2>
        <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
          Official academic enrollment information linked to your student credentials.
        </p>
        <div className="table-responsive">
          <table>
            <tbody>
              <tr><td style={{ width: 180, color: '#94a3b8' }}>Account Username</td><td><strong style={{ color: '#ffffff' }}>{user.username}</strong></td></tr>
              <tr>
                <td style={{ color: '#94a3b8' }}>Student ID</td>
                <td>
                  {user.student_id ? (
                    <code style={{ color: '#34d6b5', fontWeight: 600 }}>{user.student_id}</code>
                  ) : (
                    <span style={{ color: '#f87171' }}>Not linked — contact administration</span>
                  )}
                </td>
              </tr>
              <tr><td style={{ color: '#94a3b8' }}>Full Legal Name</td><td style={{ color: '#ffffff' }}>{data?.student?.name ?? '—'}</td></tr>
              <tr><td style={{ color: '#94a3b8' }}>Registered Email</td><td>{data?.student?.email ?? '—'}</td></tr>
              <tr><td style={{ color: '#94a3b8' }}>Enrolled Class Division</td><td><span className="status-chip present">{data?.profile?.class_label ?? '—'}</span></td></tr>
              <tr><td style={{ color: '#94a3b8' }}>Department</td><td>{data?.profile?.department_name ?? '—'}</td></tr>
              <tr><td style={{ color: '#94a3b8' }}>Assigned Roll Number</td><td><code>{data?.profile?.roll_number ?? '—'}</code></td></tr>
              <tr>
                <td style={{ color: '#94a3b8' }}>System Role</td>
                <td><span className="role-badge student">Student</span></td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="info-note" style={{ marginTop: 18 }}>
          ℹ️ To correct your division, roll number, or linked email, please consult your department academic coordinator.
        </p>
      </div>

      <div className="panel form-panel">
        <h2>Security &amp; Password Update</h2>
        <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
          Update your student portal login password regularly for account security.
        </p>
        <form onSubmit={handlePw} className="form-grid" style={{ maxWidth: 460 }}>
          <div className="input-group">
            <label>Current Password</label>
            <input
              type="password"
              placeholder="Enter current password"
              value={cur}
              onChange={e => setCur(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <label>New Password (min 6 characters)</label>
            <input
              type="password"
              placeholder="Enter new password"
              value={nw}
              onChange={e => setNw(e.target.value)}
              minLength={6}
              required
            />
          </div>
          <div className="input-group">
            <label>Confirm New Password</label>
            <input
              type="password"
              placeholder="Confirm new password"
              value={conf}
              onChange={e => setConf(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="btn-primary" disabled={busy} style={{ marginTop: 6 }}>
            {busy ? 'Updating…' : 'Update Password'}
          </button>
        </form>
      </div>
    </>
  );
}
