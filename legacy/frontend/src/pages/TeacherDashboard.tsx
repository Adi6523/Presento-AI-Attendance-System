import { useEffect, useState } from 'react';
import {
  AuthUser, logout,
  getDashboard, listStudents, listSessions, listUsers, getAtRisk, getStudentAttendance,
  listAllAttendance, addAttendanceManual, editAttendance, deleteAttendance,
  updateStudent, deactivateStudent, deleteSession, createSession, closeSession,
  updateUser, createUser, createStudentRecord, enrollImages, changePassword,
  listDepartments, createDepartment, deleteDepartment,
  listAcademicYears, createAcademicYear, deleteAcademicYear,
  listClasses, createClass, deleteClass,
  listSubjects, createSubject, deleteSubject,
  listTeacherProfiles, listTeacherAssignments, createTeacherAssignment, deleteTeacherAssignment,
  listStudentProfiles, assignStudentToClass, removeStudentFromClass,
  getMyAssignments, getMyClasses, getClassStudents, getAuditLog,
  finalizeAbsent, downloadAttendanceCSV,
} from '../api';
import Camera from '../components/Camera';
import Modal from '../components/Modal';

interface Props { user: AuthUser; onLogout: () => void; }

type AdminView =
  | 'Overview'
  | 'Academic Setup'
  | 'Assignments & Enrollment'
  | 'Students'
  | 'Attendance'
  | 'Sessions'
  | 'Manage Users'
  | 'Audit Log';

type TeacherView =
  | 'Overview'
  | 'My Classes'
  | 'Students'
  | 'Attendance'
  | 'Take Attendance';

type View = AdminView | TeacherView;

interface ModalState {
  title: string; message: string;
  type: 'success' | 'error' | 'warning' | 'info' | 'confirm';
  onConfirm?: () => void;
}

const STATUS_LABELS: Record<string, string> = {
  present: 'Present', late: 'Late', absent: 'Absent',
  nc: 'NC (Not Considered)', present_nc: 'Present-NC',
};
const STATUS_VALUES = Object.keys(STATUS_LABELS);
const REASON_REQUIRED = ['nc', 'present_nc'];

// ── Helpers ───────────────────────────────────────────────────────────────────
function StatCard({ label, value, danger }: { label: string; value: any; danger?: boolean }) {
  return (
    <div className={`card ${danger ? 'danger' : ''}`}>
      <p>{label}</p><strong>{value ?? '—'}</strong>
    </div>
  );
}

function FormModal({ title, onClose, children }: { title: string; onClose: () => void; children: React.ReactNode }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);
  return (
    <div className="form-modal-backdrop" onClick={onClose}>
      <div className="form-modal-card" onClick={e => e.stopPropagation()}>
        <button className="form-modal-close" onClick={onClose}>×</button>
        <h3>{title}</h3>
        {children}
      </div>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function TeacherDashboard({ user, onLogout }: Props) {
  const isAdmin = user.role === 'admin';

  const [view, setView]               = useState<View>('Overview');
  const [dash, setDash]               = useState<any>(null);
  const [students, setStudents]       = useState<any[]>([]);
  const [sessions, setSessions]       = useState<any[]>([]);
  const [classes, setClasses]         = useState<any[]>([]);
  const [assignments, setAssignments] = useState<any[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<number | undefined>(undefined);
  const [modal, setModal]             = useState<ModalState | null>(null);
  const [error, setError]             = useState('');

  // Selected student drilldown
  const [selectedStudent, setSelectedStudent] = useState<any>(null);
  const [studentDetails, setStudentDetails]   = useState<any>(null);

  const showModal = (m: ModalState) => setModal(m);
  const closeModal = () => setModal(null);

  const loadDash = () => getDashboard().then(setDash).catch(e => setError(e.message));
  const loadStudents = () => listStudents().then(setStudents).catch(e => setError(e.message));
  const loadSessions = () => listSessions().then(setSessions).catch(e => setError(e.message));
  const loadAssignments = () => {
    if (isAdmin) {
      listTeacherAssignments().then(setAssignments).catch(() => {});
    } else {
      getMyAssignments().then(setAssignments).catch(() => {});
    }
  };
  const loadClasses = () => {
    if (isAdmin) {
      listClasses().then(setClasses).catch(() => {});
    } else {
      getMyClasses().then(setClasses).catch(() => {});
    }
  };

  useEffect(() => {
    loadDash();
    loadSessions();
    loadStudents();
    loadAssignments();
    loadClasses();
    const t = setInterval(loadDash, 12000);
    return () => clearInterval(t);
  }, []);

  const handleViewChange = (v: View) => {
    setView(v); setError('');
    setSelectedStudent(null); setStudentDetails(null);
    if (v === 'Students') loadStudents();
    if (v === 'Attendance') { loadStudents(); loadSessions(); loadClasses(); }
    if (v === 'My Classes' || v === 'Sessions') { loadSessions(); loadAssignments(); loadClasses(); }
  };

  const navItems: View[] = isAdmin
    ? ['Overview', 'Academic Setup', 'Assignments & Enrollment', 'Students', 'Attendance', 'Sessions', 'Manage Users', 'Audit Log']
    : ['Overview', 'My Classes', 'Students', 'Attendance', 'Take Attendance'];

  return (
    <main className="app-layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="brand">presento<span>.</span></div>
        <nav>
          {navItems.map(item => (
            <button
              key={item}
              className={`nav-btn ${view === item ? 'active' : ''}`}
              onClick={() => handleViewChange(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="user-info">
            <span className={`role-badge ${user.role}`}>{user.role.toUpperCase()}</span>
            <span>{user.username}</span>
            <button className="btn-logout" onClick={() => showModal({
              title: 'Sign out', message: 'Are you sure you want to sign out?',
              type: 'confirm', onConfirm: () => { logout(); onLogout(); },
            })}>Sign out</button>
          </div>
        </div>
      </aside>

      {/* ── Main Content Area ── */}
      <section className="content">
        <header className="page-header">
          <div>
            <p className="eyebrow">{isAdmin ? 'ADMINISTRATOR CONSOLE' : 'FACULTY DASHBOARD'}</p>
            <h1>{view}</h1>
          </div>
          {view === 'Overview' && (
            <button className="btn-secondary" onClick={loadDash}>↻ Refresh</button>
          )}
        </header>

        {error && (
          <div className="alert-error">{error}
            <button className="alert-close" onClick={() => setError('')}>×</button>
          </div>
        )}

        {/* 1. OVERVIEW VIEW */}
        {view === 'Overview' && (
          <OverviewView
            dash={dash}
            isAdmin={isAdmin}
            sessions={sessions}
            assignments={assignments}
            onNavigate={(v: View) => handleViewChange(v)}
          />
        )}

        {/* 2. ACADEMIC SETUP VIEW (Admin Only) */}
        {view === 'Academic Setup' && isAdmin && (
          <AcademicSetupView showModal={showModal} setError={setError} />
        )}

        {/* 3. ASSIGNMENTS & ENROLLMENT VIEW (Admin Only) */}
        {view === 'Assignments & Enrollment' && isAdmin && (
          <AssignmentsEnrollmentView showModal={showModal} setError={setError} />
        )}

        {/* 4. MY CLASSES & SESSIONS VIEW (Teacher / Admin) */}
        {(view === 'My Classes' || view === 'Sessions') && (
          <MyClassesView
            isAdmin={isAdmin}
            sessions={sessions}
            assignments={assignments}
            classes={classes}
            onRefresh={() => { loadSessions(); loadAssignments(); }}
            onStartSessionCamera={(sessionId: number) => {
              setActiveSessionId(sessionId);
              setView('Take Attendance');
            }}
            showModal={showModal}
            setError={setError}
          />
        )}

        {/* 5. STUDENTS VIEW */}
        {view === 'Students' && (
          <StudentsView
            userRole={user.role}
            students={students}
            classes={classes}
            selectedStudent={selectedStudent}
            studentDetails={studentDetails}
            onSelectStudent={(s: any) => {
              setSelectedStudent(s);
              getStudentAttendance(s.id).then(setStudentDetails).catch(e => setError(e.message));
            }}
            onBack={() => { setSelectedStudent(null); setStudentDetails(null); }}
            onRefresh={loadStudents}
            showModal={showModal}
            setError={setError}
          />
        )}

        {/* 6. ATTENDANCE CRUD & ROSTER VIEW */}
        {view === 'Attendance' && (
          <AttendanceView
            userRole={user.role}
            students={students}
            sessions={sessions}
            classes={classes}
            showModal={showModal}
            setError={setError}
          />
        )}

        {/* 7. TAKE ATTENDANCE (LIVE WEBCAM CHECK-IN) */}
        {view === 'Take Attendance' && (
          <TakeAttendanceView
            sessions={sessions}
            preselectedSessionId={activeSessionId}
            showModal={showModal}
            setError={setError}
          />
        )}

        {/* 8. MANAGE USERS (Admin Only) */}
        {view === 'Manage Users' && isAdmin && (
          <ManageUsersView showModal={showModal} setError={setError} />
        )}

        {/* 9. AUDIT LOG (Admin Only) */}
        {view === 'Audit Log' && isAdmin && (
          <AuditLogView setError={setError} />
        )}

        {modal && (
          <Modal
            title={modal.title} message={modal.message} type={modal.type}
            onConfirm={modal.onConfirm ?? (modal.type === 'confirm' ? () => { logout(); onLogout(); } : undefined)}
            onClose={closeModal}
            confirmLabel={modal.type === 'confirm' ? 'Yes, proceed' : 'OK'}
          />
        )}
      </section>
    </main>
  );
}

// ── Sub-component 1: Overview ─────────────────────────────────────────────────
function OverviewView({ dash, isAdmin, sessions, assignments, onNavigate }: any) {
  const activeSessions = sessions.filter((s: any) => s.status === 'open');

  return (
    <>
      <div className="cards-grid">
        <StatCard label="Total Enrolled Students" value={dash?.total_students} />
        <StatCard label="Present Today" value={dash?.present_today} />
        <StatCard label="Late Today" value={dash?.late_today} />
        <StatCard label="At-Risk Students (<75%)" value={dash?.at_risk} danger={dash?.at_risk > 0} />
      </div>

      {activeSessions.length > 0 && (
        <div className="panel" style={{ background: '#11172f', border: '1px solid #3b82f640', marginTop: 18 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h2 style={{ color: '#60a5fa', margin: 0 }}>Active Attendance Session(s) Open Now</h2>
              <p style={{ color: '#8a9bb8', margin: '4px 0 0' }}>
                {activeSessions.length} session(s) currently open for face recognition & roster check-in.
              </p>
            </div>
            <button className="btn-primary" onClick={() => onNavigate(isAdmin ? 'Sessions' : 'My Classes')}>
              View Active Sessions →
            </button>
          </div>
        </div>
      )}

      <div className="panel" style={{ marginTop: 20 }}>
        <h2>Recent Check-in Events</h2>
        <table>
          <thead>
            <tr>
              <th>Student</th>
              <th>Subject / Class</th>
              <th>Status</th>
              <th>Time</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {(!dash?.recent || dash.recent.length === 0) && (
              <tr><td colSpan={5} style={{ textAlign: 'center', color: '#8a9bb8', padding: 24 }}>No events recorded today.</td></tr>
            )}
            {dash?.recent?.map((r: any, i: number) => (
              <tr key={i}>
                <td><strong>{r.student_name}</strong></td>
                <td>{r.subject_name ?? '—'}</td>
                <td><span className={`status-chip ${r.status}`}>{STATUS_LABELS[r.status] ?? r.status}</span></td>
                <td style={{ fontSize: 13 }}>{new Date(r.observed_at + 'Z').toLocaleTimeString()}</td>
                <td>{Math.round(r.confidence * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ── Sub-component 2: Academic Setup (Admin Only) ──────────────────────────────
function AcademicSetupView({ showModal, setError }: any) {
  const [tab, setTab] = useState<'Departments' | 'Years' | 'Classes' | 'Subjects'>('Classes');
  const [depts, setDepts] = useState<any[]>([]);
  const [years, setYears] = useState<any[]>([]);
  const [classes, setClasses] = useState<any[]>([]);
  const [subjects, setSubjects] = useState<any[]>([]);

  // Forms
  const [deptForm, setDeptForm] = useState({ name: '', code: '' });
  const [yearForm, setYearForm] = useState({ label: '', is_current: true });
  const [classForm, setClassForm] = useState({ department_id: 0, academic_year_id: 0, year: 'TY', semester: 5, division: 'A', label: '' });
  const [subjForm, setSubjForm] = useState({ department_id: 0, name: '', code: '', semester: 5 });

  const loadAll = () => {
    listDepartments().then(setDepts).catch(e => setError(e.message));
    listAcademicYears().then(setYears).catch(e => setError(e.message));
    listClasses().then(setClasses).catch(e => setError(e.message));
    listSubjects().then(setSubjects).catch(e => setError(e.message));
  };

  useEffect(() => { loadAll(); }, []);

  // Update default dropdown selections when depts or years load
  useEffect(() => {
    if (depts.length && !classForm.department_id) {
      setClassForm(prev => ({ ...prev, department_id: depts[0].id }));
      setSubjForm(prev => ({ ...prev, department_id: depts[0].id }));
    }
  }, [depts]);

  useEffect(() => {
    if (years.length && !classForm.academic_year_id) {
      setClassForm(prev => ({ ...prev, academic_year_id: years[0].id }));
    }
  }, [years]);

  return (
    <div className="panel">
      <div className="sub-tabs">
        {(['Classes', 'Subjects', 'Departments', 'Years'] as const).map(t => (
          <button key={t} className={`sub-tab-btn ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {/* ── CLASSES ── */}
      {tab === 'Classes' && (
        <>
          <form
            onSubmit={async e => {
              e.preventDefault();
              try {
                const autoLabel = classForm.label || `${classForm.year} ${depts.find(d => d.id === Number(classForm.department_id))?.code ?? ''}-${classForm.division}`;
                await createClass({ ...classForm, department_id: Number(classForm.department_id), academic_year_id: Number(classForm.academic_year_id), label: autoLabel });
                loadAll();
                showModal({ title: 'Class Created', message: `Class "${autoLabel}" added.`, type: 'success' });
              } catch (err: any) { setError(err.message); }
            }}
            className="form-grid form-grid-2"
            style={{ marginBottom: 24, padding: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12 }}
          >
            <select value={classForm.department_id} onChange={e => setClassForm({ ...classForm, department_id: Number(e.target.value) })} required>
              {depts.map(d => <option key={d.id} value={d.id}>{d.name} ({d.code})</option>)}
            </select>
            <select value={classForm.academic_year_id} onChange={e => setClassForm({ ...classForm, academic_year_id: Number(e.target.value) })} required>
              {years.map(y => <option key={y.id} value={y.id}>{y.label} {y.is_current ? '(Current)' : ''}</option>)}
            </select>
            <input placeholder="Year (e.g. FY, SY, TY)" value={classForm.year} onChange={e => setClassForm({ ...classForm, year: e.target.value })} required />
            <input placeholder="Division (e.g. A, B)" value={classForm.division} onChange={e => setClassForm({ ...classForm, division: e.target.value })} required />
            <input placeholder="Custom Label (optional, e.g. TY CSE-A)" value={classForm.label} onChange={e => setClassForm({ ...classForm, label: e.target.value })} />
            <button type="submit" className="btn-primary">+ Add Class Division</button>
          </form>

          <table>
            <thead>
              <tr><th>Label</th><th>Department</th><th>Academic Year</th><th>Year &amp; Div</th><th>Students</th><th>Action</th></tr>
            </thead>
            <tbody>
              {classes.length === 0 && <tr><td colSpan={6} style={{ textAlign: 'center', color: '#8a9bb8' }}>No classes created yet.</td></tr>}
              {classes.map(c => (
                <tr key={c.id}>
                  <td><strong>{c.label}</strong></td>
                  <td>{c.department_name} ({c.department_code})</td>
                  <td>{c.academic_year_label}</td>
                  <td>{c.year} - {c.division}</td>
                  <td><span className="status-chip present">{c.student_count ?? 0} enrolled</span></td>
                  <td>
                    <button className="btn-delete" onClick={() => {
                      showModal({
                        title: 'Delete Class', type: 'confirm', message: `Delete class ${c.label}?`,
                        onConfirm: async () => { await deleteClass(c.id); loadAll(); },
                      });
                    }}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {/* ── SUBJECTS ── */}
      {tab === 'Subjects' && (
        <>
          <form
            onSubmit={async e => {
              e.preventDefault();
              try {
                await createSubject({ ...subjForm, department_id: Number(subjForm.department_id) });
                setSubjForm({ ...subjForm, name: '', code: '' });
                loadAll();
                showModal({ title: 'Subject Created', message: `Subject "${subjForm.name}" created.`, type: 'success' });
              } catch (err: any) { setError(err.message); }
            }}
            className="form-grid form-grid-2"
            style={{ marginBottom: 24, padding: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12 }}
          >
            <select value={subjForm.department_id} onChange={e => setSubjForm({ ...subjForm, department_id: Number(e.target.value) })} required>
              {depts.map(d => <option key={d.id} value={d.id}>{d.name} ({d.code})</option>)}
            </select>
            <input placeholder="Subject Name (e.g. Machine Learning)" value={subjForm.name} onChange={e => setSubjForm({ ...subjForm, name: e.target.value })} required />
            <input placeholder="Subject Code (e.g. CS-401)" value={subjForm.code} onChange={e => setSubjForm({ ...subjForm, code: e.target.value })} required />
            <input type="number" placeholder="Semester (e.g. 5)" value={subjForm.semester} onChange={e => setSubjForm({ ...subjForm, semester: Number(e.target.value) })} />
            <button type="submit" className="btn-primary">+ Add Subject</button>
          </form>

          <table>
            <thead>
              <tr><th>Subject Code</th><th>Name</th><th>Department</th><th>Semester</th><th>Action</th></tr>
            </thead>
            <tbody>
              {subjects.length === 0 && <tr><td colSpan={5} style={{ textAlign: 'center', color: '#8a9bb8' }}>No subjects added yet.</td></tr>}
              {subjects.map(s => (
                <tr key={s.id}>
                  <td><code>{s.code}</code></td>
                  <td><strong>{s.name}</strong></td>
                  <td>{s.department_name}</td>
                  <td>Sem {s.semester ?? '—'}</td>
                  <td>
                    <button className="btn-delete" onClick={() => {
                      showModal({
                        title: 'Delete Subject', type: 'confirm', message: `Delete subject ${s.name} (${s.code})?`,
                        onConfirm: async () => { await deleteSubject(s.id); loadAll(); },
                      });
                    }}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {/* ── DEPARTMENTS ── */}
      {tab === 'Departments' && (
        <>
          <form
            onSubmit={async e => {
              e.preventDefault();
              try {
                await createDepartment(deptForm);
                setDeptForm({ name: '', code: '' });
                loadAll();
                showModal({ title: 'Department Created', message: `Department "${deptForm.name}" created.`, type: 'success' });
              } catch (err: any) { setError(err.message); }
            }}
            className="form-grid form-grid-2"
            style={{ marginBottom: 24, padding: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12 }}
          >
            <input placeholder="Department Name (e.g. Computer Science)" value={deptForm.name} onChange={e => setDeptForm({ ...deptForm, name: e.target.value })} required />
            <input placeholder="Department Code (e.g. CSE)" value={deptForm.code} onChange={e => setDeptForm({ ...deptForm, code: e.target.value })} required />
            <button type="submit" className="btn-primary">+ Add Department</button>
          </form>

          <table>
            <thead>
              <tr><th>Code</th><th>Name</th><th>Action</th></tr>
            </thead>
            <tbody>
              {depts.map(d => (
                <tr key={d.id}>
                  <td><code>{d.code}</code></td>
                  <td><strong>{d.name}</strong></td>
                  <td>
                    <button className="btn-delete" onClick={() => {
                      showModal({
                        title: 'Delete Department', type: 'confirm', message: `Delete department ${d.name}?`,
                        onConfirm: async () => { await deleteDepartment(d.id); loadAll(); },
                      });
                    }}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {/* ── YEARS ── */}
      {tab === 'Years' && (
        <>
          <form
            onSubmit={async e => {
              e.preventDefault();
              try {
                await createAcademicYear(yearForm);
                setYearForm({ label: '', is_current: true });
                loadAll();
                showModal({ title: 'Academic Year Added', message: `Academic year "${yearForm.label}" added.`, type: 'success' });
              } catch (err: any) { setError(err.message); }
            }}
            className="form-grid form-grid-2"
            style={{ marginBottom: 24, padding: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12 }}
          >
            <input placeholder="Label (e.g. 2026-2027)" value={yearForm.label} onChange={e => setYearForm({ ...yearForm, label: e.target.value })} required />
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#8a9bb8' }}>
              <input type="checkbox" checked={yearForm.is_current} onChange={e => setYearForm({ ...yearForm, is_current: e.target.checked })} />
              Mark as Current Academic Year
            </label>
            <button type="submit" className="btn-primary">+ Add Academic Year</button>
          </form>

          <table>
            <thead>
              <tr><th>Label</th><th>Current Year</th><th>Action</th></tr>
            </thead>
            <tbody>
              {years.map(y => (
                <tr key={y.id}>
                  <td><strong>{y.label}</strong></td>
                  <td>{y.is_current ? <span className="status-chip present">Current</span> : '—'}</td>
                  <td>
                    <button className="btn-delete" onClick={() => {
                      showModal({
                        title: 'Delete Academic Year', type: 'confirm', message: `Delete academic year ${y.label}?`,
                        onConfirm: async () => { await deleteAcademicYear(y.id); loadAll(); },
                      });
                    }}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

// ── Sub-component 3: Assignments & Enrollment (Admin Only) ────────────────────
function AssignmentsEnrollmentView({ showModal, setError }: any) {
  const [subTab, setSubTab] = useState<'Teacher Assignments' | 'Student Class Enrollment'>('Teacher Assignments');
  const [teachers, setTeachers]       = useState<any[]>([]);
  const [assignments, setAssignments] = useState<any[]>([]);
  const [classes, setClasses]         = useState<any[]>([]);
  const [subjects, setSubjects]       = useState<any[]>([]);
  const [students, setStudents]       = useState<any[]>([]);
  const [enrollments, setEnrollments] = useState<any[]>([]);

  // Forms
  const [taForm, setTaForm] = useState({ teacher_id: 0, subject_id: 0, class_id: 0 });
  const [spForm, setSpForm] = useState({ student_id: '', class_id: 0, roll_number: '' });

  const loadAll = () => {
    listTeacherProfiles().then(setTeachers).catch(e => setError(e.message));
    listTeacherAssignments().then(setAssignments).catch(e => setError(e.message));
    listClasses().then(setClasses).catch(e => setError(e.message));
    listSubjects().then(setSubjects).catch(e => setError(e.message));
    listStudents().then(setStudents).catch(e => setError(e.message));
    listStudentProfiles().then(setEnrollments).catch(e => setError(e.message));
  };

  useEffect(() => { loadAll(); }, []);

  useEffect(() => {
    if (teachers.length && !taForm.teacher_id) setTaForm(prev => ({ ...prev, teacher_id: teachers[0].id }));
    if (subjects.length && !taForm.subject_id) setTaForm(prev => ({ ...prev, subject_id: subjects[0].id }));
    if (classes.length && !taForm.class_id) setTaForm(prev => ({ ...prev, class_id: classes[0].id }));
    if (students.length && !spForm.student_id) setSpForm(prev => ({ ...prev, student_id: students[0].id }));
    if (classes.length && !spForm.class_id) setSpForm(prev => ({ ...prev, class_id: classes[0].id }));
  }, [teachers, subjects, classes, students]);

  return (
    <div className="panel">
      <div className="sub-tabs">
        {(['Teacher Assignments', 'Student Class Enrollment'] as const).map(t => (
          <button key={t} className={`sub-tab-btn ${subTab === t ? 'active' : ''}`} onClick={() => setSubTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {subTab === 'Teacher Assignments' && (
        <>
          <form
            onSubmit={async e => {
              e.preventDefault();
              try {
                await createTeacherAssignment({
                  teacher_id: Number(taForm.teacher_id),
                  subject_id: Number(taForm.subject_id),
                  class_id: Number(taForm.class_id),
                });
                loadAll();
                showModal({ title: 'Assigned', message: 'Teacher assigned to subject and class.', type: 'success' });
              } catch (err: any) { setError(err.message); }
            }}
            className="form-grid form-grid-2"
            style={{ marginBottom: 24, padding: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12 }}
          >
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Teacher Profile:</label>
              <select value={taForm.teacher_id} onChange={e => setTaForm({ ...taForm, teacher_id: Number(e.target.value) })} required>
                {teachers.map(t => <option key={t.id} value={t.id}>{t.username} ({t.designation ?? 'Teacher'})</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Subject:</label>
              <select value={taForm.subject_id} onChange={e => setTaForm({ ...taForm, subject_id: Number(e.target.value) })} required>
                {subjects.map(s => <option key={s.id} value={s.id}>{s.name} ({s.code})</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Class Division:</label>
              <select value={taForm.class_id} onChange={e => setTaForm({ ...taForm, class_id: Number(e.target.value) })} required>
                {classes.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
              </select>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button type="submit" className="btn-primary" style={{ width: '100%' }}>+ Assign Teacher</button>
            </div>
          </form>

          <table>
            <thead>
              <tr><th>Teacher</th><th>Subject</th><th>Class Division</th><th>Action</th></tr>
            </thead>
            <tbody>
              {assignments.length === 0 && <tr><td colSpan={4} style={{ textAlign: 'center', color: '#8a9bb8' }}>No assignments made yet.</td></tr>}
              {assignments.map(a => (
                <tr key={a.id}>
                  <td><strong>{a.teacher_name}</strong></td>
                  <td>{a.subject_name} (<code>{a.subject_code}</code>)</td>
                  <td><span className="status-chip present">{a.class_label}</span></td>
                  <td>
                    <button className="btn-delete" onClick={() => {
                      showModal({
                        title: 'Remove Assignment', type: 'confirm', message: `Remove assignment of ${a.teacher_name} for ${a.subject_name}?`,
                        onConfirm: async () => { await deleteTeacherAssignment(a.id); loadAll(); },
                      });
                    }}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {subTab === 'Student Class Enrollment' && (
        <>
          <form
            onSubmit={async e => {
              e.preventDefault();
              try {
                await assignStudentToClass({
                  student_id: spForm.student_id,
                  class_id: Number(spForm.class_id),
                  roll_number: spForm.roll_number || undefined,
                });
                loadAll();
                showModal({ title: 'Student Enrolled', message: `Enrolled student into class.`, type: 'success' });
              } catch (err: any) { setError(err.message); }
            }}
            className="form-grid form-grid-2"
            style={{ marginBottom: 24, padding: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12 }}
          >
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Student:</label>
              <select value={spForm.student_id} onChange={e => setSpForm({ ...spForm, student_id: e.target.value })} required>
                {students.map(s => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Assign to Class:</label>
              <select value={spForm.class_id} onChange={e => setSpForm({ ...spForm, class_id: Number(e.target.value) })} required>
                {classes.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Roll Number (Optional):</label>
              <input placeholder="Roll Number (e.g. 090)" value={spForm.roll_number} onChange={e => setSpForm({ ...spForm, roll_number: e.target.value })} />
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button type="submit" className="btn-primary" style={{ width: '100%' }}>+ Enroll Student into Class</button>
            </div>
          </form>

          <table>
            <thead>
              <tr><th>Roll No</th><th>Student Name</th><th>Student ID</th><th>Class</th><th>Action</th></tr>
            </thead>
            <tbody>
              {enrollments.length === 0 && <tr><td colSpan={5} style={{ textAlign: 'center', color: '#8a9bb8' }}>No students enrolled into classes yet.</td></tr>}
              {enrollments.map(ep => (
                <tr key={ep.id}>
                  <td><strong>{ep.roll_number ?? '—'}</strong></td>
                  <td>{ep.student_name}</td>
                  <td><code>{ep.student_id}</code></td>
                  <td><span className="status-chip present">{ep.class_label}</span></td>
                  <td>
                    <button className="btn-delete" onClick={() => {
                      showModal({
                        title: 'Remove from Class', type: 'confirm', message: `Remove ${ep.student_name} from ${ep.class_label}?`,
                        onConfirm: async () => { await removeStudentFromClass(ep.id); loadAll(); },
                      });
                    }}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

// ── Sub-component 4: My Classes & Sessions (Teacher / Admin) ─────────────────
function MyClassesView({ isAdmin, sessions, assignments, classes, onRefresh, onStartSessionCamera, showModal, setError }: any) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedRosterSession, setSelectedRosterSession] = useState<any>(null);
  const [rosterStudents, setRosterStudents]   = useState<any[]>([]);
  const [rosterLoading, setRosterLoading]     = useState(false);

  // Form for starting a new session
  const now = new Date();
  const localStart = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
  const localEnd   = new Date(now.getTime() + 60 * 60000 - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);

  const [form, setForm] = useState({
    assignment_id: 0,
    room: 'A101',
    starts_at: localStart,
    ends_at: localEnd,
    late_after_minutes: 10,
  });

  useEffect(() => {
    if (assignments.length && !form.assignment_id) {
      setForm(prev => ({ ...prev, assignment_id: assignments[0].assignment_id ?? assignments[0].id }));
    }
  }, [assignments]);

  const handleStartSession = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const selected = assignments.find((a: any) => (a.assignment_id ?? a.id) === Number(form.assignment_id));
      if (!selected) {
        setError('Please select a class and subject assignment.');
        return;
      }
      const res = await createSession({
        subject_id: selected.subject_id,
        class_id: selected.class_id,
        room: form.room,
        starts_at: new Date(form.starts_at).toISOString(),
        ends_at: new Date(form.ends_at).toISOString(),
        late_after_minutes: form.late_after_minutes,
      });
      setShowCreateModal(false);
      onRefresh();
      showModal({
        title: 'Session Started!',
        message: `Attendance session opened for ${selected.subject_name} (${selected.class_label}) in ${form.room}. You can now launch camera check-in or mark the roster.`,
        type: 'success',
      });
    } catch (err: any) { setError(err.message); }
  };

  const handleCloseSession = (s: any) => {
    showModal({
      title: 'Close Session',
      type: 'confirm',
      message: `Close session for ${s.subject_name ?? s.course_code}? Once closed, students cannot check in via webcam.`,
      onConfirm: async () => {
        try {
          await closeSession(s.id);
          onRefresh();
        } catch (err: any) { setError(err.message); }
      },
    });
  };

  const openRoster = async (s: any) => {
    setSelectedRosterSession(s);
    setRosterLoading(true);
    try {
      if (s.class_id) {
        const list = await getClassStudents(s.class_id);
        setRosterStudents(list);
      } else {
        const all = await listStudents();
        setRosterStudents(all);
      }
    } catch (err: any) { setError(err.message); }
    finally { setRosterLoading(false); }
  };

  const markStudentQuick = async (studentId: string, status: string) => {
    try {
      await addAttendanceManual({
        student_id: studentId,
        session_id: selectedRosterSession.id,
        status,
        observed_at: new Date().toISOString(),
        reason: status === 'nc' ? 'Faculty marked NC' : undefined,
      });
      showModal({ title: 'Marked', message: `Student marked as ${status.toUpperCase()}.`, type: 'success' });
    } catch (err: any) { setError(err.message); }
  };

  return (
    <>
      {/* Assigned Classes Cards */}
      <div className="panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <h2 style={{ margin: 0 }}>My Assigned Classes &amp; Subjects</h2>
            <p style={{ color: '#8a9bb8', margin: '4px 0 0' }}>Classes you are scheduled to teach.</p>
          </div>
          <button className="btn-add-manual" onClick={() => setShowCreateModal(true)}>
            + Start Attendance Session
          </button>
        </div>

        {assignments.length === 0 ? (
          <p style={{ color: '#8a9bb8', padding: '16px 0' }}>
            No class assignments found. {isAdmin ? 'Use Assignments & Enrollment tab to assign classes.' : 'Contact admin to assign subjects.'}
          </p>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14 }}>
            {assignments.map((a: any) => (
              <div key={a.assignment_id ?? a.id} style={{
                background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: 12, padding: 16
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span className="status-chip present">{a.class_label}</span>
                    <strong style={{ fontSize: 16, display: 'block', marginTop: 8 }}>{a.subject_name}</strong>
                    <small style={{ color: '#8a9bb8' }}>{a.subject_code}</small>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: 13, color: '#8a9bb8' }}>{a.student_count ?? 0} students</span>
                  </div>
                </div>
                <div style={{ marginTop: 14 }}>
                  <button
                    className="btn-primary"
                    style={{ width: '100%', padding: '7px 12px', fontSize: 13 }}
                    onClick={() => {
                      setForm(prev => ({ ...prev, assignment_id: a.assignment_id ?? a.id }));
                      setShowCreateModal(true);
                    }}
                  >
                    Start Session for this Class →
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Attendance Sessions List */}
      <div className="panel" style={{ marginTop: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <h2 style={{ margin: 0 }}>Attendance Sessions</h2>
            <p style={{ color: '#8a9bb8', margin: '4px 0 0' }}>Active and past sessions with student check-in access.</p>
          </div>
          <button className="btn-secondary" onClick={onRefresh}>↻ Refresh</button>
        </div>

        <table>
          <thead>
            <tr>
              <th>Subject</th>
              <th>Class</th>
              <th>Room</th>
              <th>Time</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {sessions.length === 0 && (
              <tr><td colSpan={6} style={{ textAlign: 'center', color: '#8a9bb8', padding: 24 }}>No sessions yet. Click "Start Attendance Session" above.</td></tr>
            )}
            {sessions.map((s: any) => {
              const isOpen = s.status === 'open';
              return (
                <tr key={s.id}>
                  <td>
                    <strong>{s.subject_name ?? s.course_code}</strong>
                    {s.subject_code && <small style={{ display: 'block', color: '#8a9bb8' }}>{s.subject_code}</small>}
                  </td>
                  <td>{s.class_label ?? '—'}</td>
                  <td>{s.room}</td>
                  <td style={{ fontSize: 12 }}>
                    <div>Starts: {new Date(s.starts_at + 'Z').toLocaleString()}</div>
                    <div>Ends: {new Date(s.ends_at + 'Z').toLocaleString()}</div>
                  </td>
                  <td>
                    {isOpen ? (
                      <span className="status-chip present">● Open</span>
                    ) : (
                      <span className="status-chip nc">Closed</span>
                    )}
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      {isOpen && (
                        <>
                          <button
                            className="btn-edit"
                            style={{ background: '#10b981', color: '#fff', border: 'none' }}
                            onClick={() => onStartSessionCamera(s.id)}
                          >
                            📷 Face Scan
                          </button>
                          <button
                            className="btn-edit"
                            onClick={() => openRoster(s)}
                          >
                            📋 Class Roster
                          </button>
                          <button
                            className="btn-deactivate"
                            onClick={() => handleCloseSession(s)}
                          >
                            Close
                          </button>
                        </>
                      )}
                      <button
                        className="btn-delete"
                        onClick={() => {
                          showModal({
                            title: 'Delete Session', type: 'confirm',
                            message: `Delete session ${s.subject_name ?? s.course_code}?`,
                            onConfirm: async () => { await deleteSession(s.id); onRefresh(); },
                          });
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Start Session Modal */}
      {showCreateModal && (
        <FormModal title="Start Attendance Session" onClose={() => setShowCreateModal(false)}>
          <form onSubmit={handleStartSession} className="form-grid">
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Select Class &amp; Subject:</label>
              <select value={form.assignment_id} onChange={e => setForm({ ...form, assignment_id: Number(e.target.value) })} required>
                {assignments.map((a: any) => (
                  <option key={a.assignment_id ?? a.id} value={a.assignment_id ?? a.id}>
                    {a.class_label} — {a.subject_name} ({a.subject_code})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Classroom / Lab:</label>
              <input placeholder="Room (e.g. A101)" value={form.room} onChange={e => setForm({ ...form, room: e.target.value })} required />
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Session Starts At:</label>
              <input type="datetime-local" value={form.starts_at} onChange={e => setForm({ ...form, starts_at: e.target.value })} required />
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Session Ends At:</label>
              <input type="datetime-local" value={form.ends_at} onChange={e => setForm({ ...form, ends_at: e.target.value })} required />
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#8a9bb8' }}>Late Grace Period (Minutes):</label>
              <input type="number" min={0} max={60} value={form.late_after_minutes} onChange={e => setForm({ ...form, late_after_minutes: Number(e.target.value) })} required />
            </div>
            <div className="form-modal-actions">
              <button type="submit" className="btn-primary">Open Attendance Session</button>
              <button type="button" className="btn-secondary" onClick={() => setShowCreateModal(false)}>Cancel</button>
            </div>
          </form>
        </FormModal>
      )}

      {/* Interactive Class Roster Quick-Mark Modal */}
      {selectedRosterSession && (
        <FormModal title={`Quick Class Roster — ${selectedRosterSession.subject_name ?? selectedRosterSession.course_code}`} onClose={() => setSelectedRosterSession(null)}>
          <p style={{ color: '#8a9bb8', fontSize: 13, marginBottom: 12 }}>
            Class: <strong>{selectedRosterSession.class_label ?? 'General'}</strong> | Room: <strong>{selectedRosterSession.room}</strong>
          </p>
          {rosterLoading ? <p style={{ color: '#8a9bb8' }}>Loading class roster…</p> : (
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              {rosterStudents.length === 0 && <p style={{ color: '#8a9bb8' }}>No students enrolled in this class.</p>}
              <div className="roster-grid">
                {rosterStudents.map((st: any) => (
                  <div key={st.id} className="roster-card">
                    <div>
                      <strong>{st.name}</strong>
                      <div style={{ fontSize: 12, color: '#8a9bb8' }}>
                        Roll: {st.roll_number ?? '—'} | ID: <code>{st.id}</code>
                      </div>
                    </div>
                    <div className="roster-actions">
                      <button className="roster-btn present" onClick={() => markStudentQuick(st.id, 'present')}>Present</button>
                      <button className="roster-btn late" onClick={() => markStudentQuick(st.id, 'late')}>Late</button>
                      <button className="roster-btn absent" onClick={() => markStudentQuick(st.id, 'absent')}>Absent</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="form-modal-actions" style={{ marginTop: 16 }}>
            <button type="button" className="btn-secondary" onClick={() => setSelectedRosterSession(null)}>Done</button>
          </div>
        </FormModal>
      )}
    </>
  );
}

// ── Sub-component 5: Students View ───────────────────────────────────────────
function StudentsView({
  userRole, students, classes, selectedStudent, studentDetails,
  onSelectStudent, onBack, onRefresh, showModal, setError,
}: any) {
  const isAdmin = userRole === 'admin';
  const [addForm, setAddForm] = useState<any>(null);
  const [editForm, setEditForm] = useState<any>(null);
  const [enrollStudentId, setEnrollStudentId] = useState<string | null>(null);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await createStudentRecord({
        id: addForm.id,
        name: addForm.name,
        email: addForm.email || undefined,
        class_id: addForm.class_id ? Number(addForm.class_id) : undefined,
        roll_number: addForm.roll_number || undefined,
      });
      setAddForm(null);
      onRefresh();
      showModal({ title: 'Student Created', message: `Student "${addForm.name}" added.`, type: 'success' });
    } catch (err: any) { setError(err.message); }
  };

  const handleEdit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await updateStudent(editForm.id, {
        name: editForm.name,
        email: editForm.email || undefined,
        class_id: editForm.class_id ? Number(editForm.class_id) : undefined,
        roll_number: editForm.roll_number || undefined,
      });
      setEditForm(null);
      onRefresh();
      showModal({ title: 'Student Updated', message: 'Student record updated successfully.', type: 'success' });
    } catch (err: any) { setError(err.message); }
  };

  const handleDeactivate = (s: any) => {
    showModal({
      title: 'Deactivate Student', type: 'confirm',
      message: `Deactivate ${s.name} (${s.id})?`,
      onConfirm: async () => {
        try { await deactivateStudent(s.id); onRefresh(); }
        catch (err: any) { setError(err.message); }
      },
    });
  };

  return (
    <>
      {!selectedStudent ? (
        <div className="panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div>
              <h2 style={{ margin: 0 }}>Student Directory</h2>
              <p style={{ color: '#8a9bb8', margin: '4px 0 0' }}>
                {isAdmin ? 'All registered students across departments.' : 'Students enrolled in your assigned classes.'}
              </p>
            </div>
            {isAdmin && (
              <button className="btn-add-manual" onClick={() => setAddForm({ id: '', name: '', email: '', class_id: '', roll_number: '' })}>
                + Add Student Record
              </button>
            )}
          </div>

          <table>
            <thead>
              <tr>
                <th>Roll No</th>
                <th>Name</th>
                <th>Student ID / Login</th>
                <th>Class</th>
                <th>Face Enrolled</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {students.length === 0 && (
                <tr><td colSpan={7} style={{ textAlign: 'center', color: '#8a9bb8', padding: 24 }}>No students found.</td></tr>
              )}
              {students.map((s: any) => (
                <tr key={s.id} style={{ cursor: 'pointer' }} onClick={() => onSelectStudent(s)}>
                  <td><strong>{s.roll_number ?? '—'}</strong></td>
                  <td><strong>{s.name}</strong></td>
                  <td><code>{s.id}</code></td>
                  <td>{s.class_label ? <span className="status-chip present">{s.class_label}</span> : '—'}</td>
                  <td>
                    {s.enrolled
                      ? <span className="status-chip present">Enrolled</span>
                      : <span className="status-chip absent">Not enrolled</span>}
                  </td>
                  <td>
                    {s.active
                      ? <span className="status-chip present">Active</span>
                      : <span className="status-chip nc">Inactive</span>}
                  </td>
                  <td onClick={e => e.stopPropagation()}>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button className="btn-edit" onClick={() => setEnrollStudentId(s.id)}>📷 Enroll Face</button>
                      {isAdmin && (
                        <>
                          <button className="btn-edit" onClick={() => setEditForm({ ...s, class_id: s.class_id ?? '' })}>Edit</button>
                          <button className="btn-delete" onClick={() => handleDeactivate(s)}>Deactivate</button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        /* Student Drilldown Detail */
        <div className="panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <button className="btn-secondary" onClick={onBack}>← Back to Student List</button>
            <button className="btn-primary" onClick={() => setEnrollStudentId(selectedStudent.id)}>📷 Enroll Face</button>
          </div>

          <h2>{selectedStudent.name} (<code>{selectedStudent.id}</code>)</h2>
          <div style={{ display: 'flex', gap: 20, margin: '12px 0 20px', color: '#8a9bb8', flexWrap: 'wrap' }}>
            <span>Class: <strong>{studentDetails?.profile?.class_label ?? selectedStudent.class_label ?? '—'}</strong></span>
            <span>Roll No: <strong>{studentDetails?.profile?.roll_number ?? selectedStudent.roll_number ?? '—'}</strong></span>
            <span>Email: <strong>{selectedStudent.email ?? '—'}</strong></span>
            <span>Overall Attendance: <strong>{studentDetails?.attendance_rate != null ? `${Math.round(studentDetails.attendance_rate * 100)}%` : '—'}</strong></span>
          </div>

          {studentDetails?.subjects && studentDetails.subjects.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3>Subject Attendance Breakdown</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, marginTop: 10 }}>
                {studentDetails.subjects.map((sb: any) => (
                  <div key={sb.subject_id} style={{ background: 'rgba(255, 255, 255, 0.03)', padding: 14, borderRadius: 10, border: '1px solid rgba(255, 255, 255, 0.08)' }}>
                    <strong>{sb.subject_name}</strong>
                    <div style={{ fontSize: 12, color: '#8a9bb8', marginTop: 4 }}>
                      Attended: {sb.attended_sessions} / {sb.total_sessions}
                    </div>
                    <div className="progress-bar-wrap" style={{ marginTop: 8, height: 5 }}>
                      <div className={`progress-bar-fill ${sb.is_at_risk ? 'danger' : 'safe'}`}
                        style={{ width: `${Math.min(Math.round((sb.attendance_rate ?? 0) * 100), 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <h3>Attendance History</h3>
          <table>
            <thead>
              <tr><th>Date &amp; Time</th><th>Subject</th><th>Room</th><th>Status</th><th>Confidence</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {(!studentDetails?.records || studentDetails.records.length === 0) && (
                <tr><td colSpan={6} style={{ textAlign: 'center', color: '#8a9bb8' }}>No attendance records found.</td></tr>
              )}
              {studentDetails?.records?.map((r: any, i: number) => (
                <tr key={i}>
                  <td>{new Date(r.observed_at + 'Z').toLocaleString()}</td>
                  <td>{r.subject_name ?? r.course_code ?? '—'}</td>
                  <td>{r.room ?? '—'}</td>
                  <td><span className={`status-chip ${r.status}`}>{STATUS_LABELS[r.status] ?? r.status}</span></td>
                  <td>{r.manually_added ? 'Manual' : `${Math.round(r.confidence * 100)}%`}</td>
                  <td>{r.reason ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Add Student Modal */}
      {addForm && (
        <FormModal title="Add Student Record" onClose={() => setAddForm(null)}>
          <form onSubmit={handleAdd} className="form-grid">
            <input placeholder="Student ID (e.g. 23UAM095)" value={addForm.id} onChange={e => setAddForm({ ...addForm, id: e.target.value })} required />
            <input placeholder="Full Name" value={addForm.name} onChange={e => setAddForm({ ...addForm, name: e.target.value })} required />
            <input type="email" placeholder="Email (optional)" value={addForm.email} onChange={e => setAddForm({ ...addForm, email: e.target.value })} />
            <select value={addForm.class_id} onChange={e => setAddForm({ ...addForm, class_id: e.target.value })}>
              <option value="">Select Class Division (optional)</option>
              {classes.map((c: any) => <option key={c.id} value={c.id}>{c.label}</option>)}
            </select>
            <input placeholder="Roll Number (optional)" value={addForm.roll_number} onChange={e => setAddForm({ ...addForm, roll_number: e.target.value })} />
            <p style={{ color: '#8a9bb8', fontSize: 12, margin: '2px 0 6px', gridColumn: 'span 2' }}>
              💡 A standardized login account (Username = Student ID, Password = <code>student123</code>) will be created automatically.
            </p>
            <div className="form-modal-actions">
              <button type="submit" className="btn-primary">Create Record</button>
              <button type="button" className="btn-secondary" onClick={() => setAddForm(null)}>Cancel</button>
            </div>
          </form>
        </FormModal>
      )}

      {/* Edit Student Modal */}
      {editForm && (
        <FormModal title="Edit Student Record" onClose={() => setEditForm(null)}>
          <form onSubmit={handleEdit} className="form-grid">
            <input placeholder="Full Name" value={editForm.name} onChange={e => setEditForm({ ...editForm, name: e.target.value })} required />
            <input type="email" placeholder="Email" value={editForm.email ?? ''} onChange={e => setEditForm({ ...editForm, email: e.target.value })} />
            <select value={editForm.class_id} onChange={e => setEditForm({ ...editForm, class_id: e.target.value })}>
              <option value="">Select Class Division</option>
              {classes.map((c: any) => <option key={c.id} value={c.id}>{c.label}</option>)}
            </select>
            <input placeholder="Roll Number" value={editForm.roll_number ?? ''} onChange={e => setEditForm({ ...editForm, roll_number: e.target.value })} />
            <div className="form-modal-actions">
              <button type="submit" className="btn-primary">Save Changes</button>
              <button type="button" className="btn-secondary" onClick={() => setEditForm(null)}>Cancel</button>
            </div>
          </form>
        </FormModal>
      )}

      {/* Face Enrollment Modal */}
      {enrollStudentId && (
        <FormModal title={`Multi-Angle Face Enrollment — ${enrollStudentId}`} onClose={() => setEnrollStudentId(null)}>
          <Camera
            mode="enroll"
            studentId={enrollStudentId}
            onDone={() => {
              setEnrollStudentId(null);
              onRefresh();
              showModal({ title: 'Enrolled Successfully', message: 'Face biometric embedding created.', type: 'success' });
            }}
          />
        </FormModal>
      )}
    </>
  );
}

// ── Sub-component 6: Attendance View ──────────────────────────────────────────
function AttendanceView({ userRole, students, sessions, classes, showModal, setError }: any) {
  const [records, setRecords] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({ student_id: '', session_id: '', class_id: '', date: '' });
  const [editRec, setEditRec] = useState<any>(null);
  const [addForm, setAddForm] = useState<any>(null);

  const load = async () => {
    setLoading(true);
    try {
      const params: any = {};
      if (filters.student_id) params.student_id = filters.student_id;
      if (filters.session_id) params.session_id = Number(filters.session_id);
      if (filters.class_id) params.class_id = Number(filters.class_id);
      if (filters.date) params.date = filters.date;
      const data = await listAllAttendance(params);
      setRecords(data);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);

  const handleEdit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await editAttendance(editRec.id, { status: editRec.status, reason: editRec.reason || undefined });
      setEditRec(null); load();
    } catch (err: any) { setError(err.message); }
  };

  const handleDelete = (r: any) => {
    showModal({
      title: 'Delete record', type: 'confirm',
      message: `Delete ${r.student_name}'s ${r.status} record on ${new Date(r.observed_at + 'Z').toLocaleDateString()}?`,
      onConfirm: async () => {
        try { await deleteAttendance(r.id); load(); }
        catch (err: any) { setError(err.message); }
      },
    });
  };

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await addAttendanceManual({
        student_id: addForm.student_id,
        session_id: addForm.session_id ? Number(addForm.session_id) : null,
        status: addForm.status,
        observed_at: new Date(addForm.observed_at).toISOString(),
        reason: addForm.reason || undefined,
      });
      setAddForm(null); load();
      showModal({ title: 'Record added', message: 'Manual attendance record created.', type: 'success' });
    } catch (err: any) { setError(err.message); }
  };

  const now = new Date();
  const localNow = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16);

  return (
    <>
      <div className="panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
          <h2>Attendance Records</h2>
          <button className="btn-add-manual" onClick={() => setAddForm({
            student_id: '', session_id: '', status: 'present',
            observed_at: localNow, reason: '',
          })}>+ Add manual record</button>
        </div>
        <p style={{ fontSize: 13, color: '#8a9bb8', marginBottom: 16 }}>
          Filter records by student, class, session or date. Status modifications are tracked in the audit trail.
        </p>

        {/* Filter row */}
        <div className="filter-row">
          <select value={filters.class_id} onChange={e => setFilters({ ...filters, class_id: e.target.value })}>
            <option value="">All Classes</option>
            {classes.map((c: any) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </select>
          <select value={filters.student_id} onChange={e => setFilters({ ...filters, student_id: e.target.value })}>
            <option value="">All Students</option>
            {students.map((s: any) => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
          </select>
          <select value={filters.session_id} onChange={e => setFilters({ ...filters, session_id: e.target.value })}>
            <option value="">All Sessions</option>
            {sessions.map((s: any) => (
              <option key={s.id} value={s.id}>{s.subject_name ?? s.course_code} — {s.room} ({new Date(s.starts_at + 'Z').toLocaleDateString()})</option>
            ))}
          </select>
          <input type="date" value={filters.date} onChange={e => setFilters({ ...filters, date: e.target.value })} />
          <button className="btn-primary" onClick={load} disabled={loading} style={{ flex: 'none', padding: '9px 20px' }}>
            {loading ? 'Loading…' : 'Apply'}
          </button>
          <button
            type="button"
            className="btn-export"
            onClick={() => downloadAttendanceCSV({
              class_id: filters.class_id ? Number(filters.class_id) : undefined,
              student_id: filters.student_id || undefined,
              session_id: filters.session_id ? Number(filters.session_id) : undefined,
              date: filters.date || undefined,
            })}
            style={{ flex: 'none', padding: '9px 18px', background: 'rgba(99, 102, 241, 0.15)', border: '1px solid rgba(99, 102, 241, 0.4)', color: '#c7d2fe', borderRadius: 8, cursor: 'pointer', fontWeight: 600 }}
          >
            📥 Export CSV
          </button>
        </div>

        <table>
          <thead>
            <tr>
              <th>Date &amp; Time</th>
              <th>Student</th>
              <th>Class</th>
              <th>Subject</th>
              <th>Room</th>
              <th>Status</th>
              <th>Confidence</th>
              <th>Modified By / Reason</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {records.length === 0 && (
              <tr><td colSpan={9} style={{ textAlign: 'center', color: '#8a9bb8', padding: 24 }}>No attendance records found.</td></tr>
            )}
            {records.map(r => (
              <tr key={r.id}>
                <td style={{ fontSize: 13 }}>{new Date(r.observed_at + 'Z').toLocaleString()}</td>
                <td><strong>{r.student_name}</strong> <small style={{ color: '#8a9bb8' }}>({r.student_id})</small></td>
                <td>{r.class_label ?? '—'}</td>
                <td>{r.subject_name ?? r.course_code ?? '—'}</td>
                <td>{r.room ?? '—'}</td>
                <td>
                  <span className={`status-chip ${r.status}`}>{STATUS_LABELS[r.status] ?? r.status}</span>
                  {r.original_status && (
                    <small style={{ display: 'block', color: '#8a9bb8', fontSize: 10 }}>was {r.original_status}</small>
                  )}
                </td>
                <td>{r.manually_added ? 'Manual' : `${Math.round(r.confidence * 100)}%`}</td>
                <td>
                  {r.modified_by && <span style={{ fontSize: 11, color: '#60a5fa', display: 'block' }}>by {r.modified_by}</span>}
                  <span className="reason-cell" title={r.reason ?? ''}>{r.reason ?? '—'}</span>
                </td>
                <td>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button className="btn-edit" onClick={() => setEditRec({ ...r, reason: r.reason ?? '' })}>Edit</button>
                    <button className="btn-delete" onClick={() => handleDelete(r)}>Delete</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {editRec && (
        <FormModal title="Edit Attendance Record" onClose={() => setEditRec(null)}>
          <form onSubmit={handleEdit} className="form-grid">
            <select value={editRec.status} onChange={e => setEditRec({ ...editRec, status: e.target.value })}>
              {STATUS_VALUES.map(s => <option key={s} value={s}>{STATUS_LABELS[s]}</option>)}
            </select>
            {(REASON_REQUIRED.includes(editRec.status) || editRec.status === 'absent') && (
              <textarea
                placeholder={REASON_REQUIRED.includes(editRec.status) ? 'Reason (required)' : 'Reason (optional)'}
                value={editRec.reason}
                onChange={e => setEditRec({ ...editRec, reason: e.target.value })}
                required={REASON_REQUIRED.includes(editRec.status)}
              />
            )}
            <div className="form-modal-actions">
              <button type="submit" className="btn-primary"
                disabled={REASON_REQUIRED.includes(editRec.status) && !editRec.reason?.trim()}>
                Save Record
              </button>
              <button type="button" className="btn-secondary" onClick={() => setEditRec(null)}>Cancel</button>
            </div>
          </form>
        </FormModal>
      )}

      {addForm && (
        <FormModal title="Add Manual Attendance Record" onClose={() => setAddForm(null)}>
          <form onSubmit={handleAdd} className="form-grid">
            <select value={addForm.student_id} onChange={e => setAddForm({ ...addForm, student_id: e.target.value })} required>
              <option value="">Select Student</option>
              {students.map((s: any) => <option key={s.id} value={s.id}>{s.name} ({s.id})</option>)}
            </select>
            <select value={addForm.session_id} onChange={e => setAddForm({ ...addForm, session_id: e.target.value })}>
              <option value="">Select Session (optional)</option>
              {sessions.map((s: any) => (
                <option key={s.id} value={s.id}>{s.subject_name ?? s.course_code} — {s.room}</option>
              ))}
            </select>
            <select value={addForm.status} onChange={e => setAddForm({ ...addForm, status: e.target.value })}>
              {STATUS_VALUES.map(s => <option key={s} value={s}>{STATUS_LABELS[s]}</option>)}
            </select>
            <input type="datetime-local" value={addForm.observed_at} onChange={e => setAddForm({ ...addForm, observed_at: e.target.value })} required />
            <textarea
              placeholder={REASON_REQUIRED.includes(addForm.status) ? 'Reason (required)' : 'Reason (optional)'}
              value={addForm.reason}
              onChange={e => setAddForm({ ...addForm, reason: e.target.value })}
              required={REASON_REQUIRED.includes(addForm.status)}
            />
            <div className="form-modal-actions">
              <button type="submit" className="btn-primary"
                disabled={REASON_REQUIRED.includes(addForm.status) && !addForm.reason?.trim()}>
                Save Record
              </button>
              <button type="button" className="btn-secondary" onClick={() => setAddForm(null)}>Cancel</button>
            </div>
          </form>
        </FormModal>
      )}
    </>
  );
}

// ── Sub-component 7: Take Attendance (Webcam Check-in & Live Class Roster) ───
function TakeAttendanceView({ sessions, preselectedSessionId, showModal, setError }: any) {
  const [selectedSessionId, setSelectedSessionId] = useState<number | undefined>(preselectedSessionId);
  const [roster, setRoster]                       = useState<any[]>([]);
  const [records, setRecords]                     = useState<any[]>([]);
  const [loading, setLoading]                     = useState(false);
  const [lastCheckinMsg, setLastCheckinMsg]       = useState<{ text: string; type: 'success' | 'info' | 'warning' } | null>(null);

  const selectedSession = sessions.find((s: any) => s.id === selectedSessionId) || null;

  useEffect(() => {
    if (preselectedSessionId) setSelectedSessionId(preselectedSessionId);
    else {
      const open = sessions.find((s: any) => s.status === 'open');
      if (open) setSelectedSessionId(open.id);
      else if (sessions.length > 0) setSelectedSessionId(sessions[0].id);
    }
  }, [preselectedSessionId, sessions]);

  const loadSessionData = async () => {
    if (!selectedSessionId || !selectedSession) return;
    setLoading(true);
    try {
      const attPromise = listAllAttendance({ session_id: selectedSessionId });
      const studentsPromise = selectedSession.class_id
        ? getClassStudents(selectedSession.class_id)
        : Promise.resolve([]);
      const [att, stList] = await Promise.all([attPromise, studentsPromise]);
      setRecords(att);
      setRoster(stList);
    } catch (e: any) {
      setError?.(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSessionData();
  }, [selectedSessionId]);

  const attendanceMap = new Map<string, any>();
  records.forEach((r: any) => {
    attendanceMap.set(r.student_id, r);
  });

  const presentCount = records.filter(r => r.status === 'present').length;
  const lateCount    = records.filter(r => r.status === 'late').length;
  const absentCount  = records.filter(r => r.status === 'absent').length;
  const totalCount   = roster.length;

  const handleCameraMatch = (r: any) => {
    loadSessionData();
    if (r.already_present) {
      setLastCheckinMsg({
        text: `ℹ️ ${r.student_name ?? 'Student'} is already marked ${r.status?.toUpperCase() || 'PRESENT'}.`,
        type: 'info'
      });
    } else {
      setLastCheckinMsg({
        text: `✅ ${r.student_name ?? 'Student'} recognized and marked ${r.status?.toUpperCase() || 'PRESENT'}!`,
        type: 'success'
      });
    }
  };

  const handleQuickMark = async (studentId: string, status: string) => {
    if (!selectedSessionId) return;
    try {
      await addAttendanceManual({
        student_id: studentId,
        session_id: selectedSessionId,
        status,
        observed_at: new Date().toISOString(),
        reason: 'Manual lecture roster adjustment'
      });
      loadSessionData();
    } catch (e: any) {
      setError?.(e.message);
    }
  };

  const handleFinalizeAbsent = async () => {
    if (!selectedSessionId) return;
    try {
      const res = await finalizeAbsent(selectedSessionId);
      showModal({
        title: 'Attendance Finalized',
        message: `Marked ${res.marked_absent_count} absent student(s) in the database for this session.`,
        type: 'success'
      });
      loadSessionData();
    } catch (e: any) {
      setError?.(e.message);
    }
  };

  return (
    <div className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h2>Lecture Live Attendance Center</h2>
          <p style={{ color: '#94a3b8', fontSize: 13, margin: '4px 0 0' }}>
            Multi-student face recognition &amp; live class roster. Unmarked students will be recorded as Absent upon finalization.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {selectedSessionId && (
            <>
              <button
                className="btn-export"
                onClick={() => downloadAttendanceCSV({ session_id: selectedSessionId })}
              >
                📥 Export Session CSV
              </button>
              <button
                className="btn-primary"
                style={{ background: 'linear-gradient(135deg, #ef4444, #dc2626)' }}
                onClick={handleFinalizeAbsent}
              >
                ✅ Finalize &amp; Mark Remaining Absent
              </button>
            </>
          )}
        </div>
      </div>

      {/* Session selector */}
      <div style={{ margin: '18px 0', display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 280, maxWidth: 520 }}>
          <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 6 }}>
            Active Lecture Session
          </label>
          <select
            value={selectedSessionId ?? ''}
            onChange={e => setSelectedSessionId(e.target.value ? Number(e.target.value) : undefined)}
            style={{ width: '100%', padding: '11px 14px', background: 'rgba(8,12,24,0.85)', border: '1.5px solid var(--border-subtle)', color: '#ffffff', borderRadius: 10 }}
          >
            <option value="">-- Select a lecture session --</option>
            {sessions.map((s: any) => (
              <option key={s.id} value={s.id}>
                {s.subject_name ?? s.course_code} — {s.class_label ?? s.room} ({s.status.toUpperCase()})
              </option>
            ))}
          </select>
        </div>

        {selectedSession && (
          <div style={{ display: 'flex', gap: 16, marginTop: 18, fontSize: 13, color: '#94a3b8' }}>
            <span>Room: <strong style={{ color: '#ffffff' }}>{selectedSession.room}</strong></span>
            <span>•</span>
            <span>Class: <span className="status-chip present">{selectedSession.class_label ?? 'General'}</span></span>
            <span>•</span>
            <span>Status: <strong style={{ color: selectedSession.status === 'open' ? '#34d6b5' : '#94a3b8' }}>{selectedSession.status?.toUpperCase()}</strong></span>
          </div>
        )}
      </div>

      {/* Live metrics tally */}
      {selectedSession && (
        <div className="cards-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', margin: '20px 0' }}>
          <div className="card">
            <p>Class Enrolled</p>
            <strong>{totalCount}</strong>
          </div>
          <div className="card">
            <p>Marked Present</p>
            <strong style={{ color: '#34d6b5' }}>{presentCount}</strong>
          </div>
          <div className="card">
            <p>Late Check-ins</p>
            <strong style={{ color: '#fbbf24' }}>{lateCount}</strong>
          </div>
          <div className="card">
            <p>Unmarked / Absent</p>
            <strong style={{ color: '#f87171' }}>{totalCount - presentCount - lateCount}</strong>
          </div>
        </div>
      )}

      {/* Notification toast */}
      {lastCheckinMsg && (
        <div
          style={{
            padding: '12px 18px',
            borderRadius: 10,
            marginBottom: 18,
            fontSize: 14,
            fontWeight: 600,
            background: lastCheckinMsg.type === 'success' ? 'rgba(16,185,129,0.15)' : 'rgba(56,189,248,0.15)',
            border: `1px solid ${lastCheckinMsg.type === 'success' ? 'rgba(16,185,129,0.4)' : 'rgba(56,189,248,0.4)'}`,
            color: lastCheckinMsg.type === 'success' ? '#34d399' : '#7dd3fc',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>{lastCheckinMsg.text}</span>
          <button
            onClick={() => setLastCheckinMsg(null)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', fontSize: 16 }}
          >
            ×
          </button>
        </div>
      )}

      {/* Workstation 2-column layout: Camera on Left, Live Roster on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 460px) 1fr', gap: 24, alignItems: 'start' }}>
        <div>
          <h3 style={{ fontSize: 16, color: '#ffffff', marginBottom: 8 }}>📷 Biometric Camera Check-in</h3>
          <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 12 }}>
            Keep camera running as students face the sensor. Recognitions are logged continuously.
          </p>
          <Camera
            mode="checkin"
            sessionId={selectedSessionId}
            keepStreamOnDone={true}
            onDone={handleCameraMatch}
          />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={{ fontSize: 16, color: '#ffffff' }}>📋 Enrolled Class Roster ({roster.length})</h3>
            <button className="btn-secondary" style={{ padding: '4px 12px', fontSize: 12 }} onClick={loadSessionData}>
              ↻ Refresh
            </button>
          </div>

          {loading ? (
            <p style={{ color: '#94a3b8', padding: 18 }}>Updating class roster…</p>
          ) : roster.length === 0 ? (
            <p style={{ color: '#94a3b8', padding: 18 }}>
              {selectedSession?.class_id ? 'No students enrolled into this class division yet.' : 'Please select a session with an assigned class.'}
            </p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 520, overflowY: 'auto', paddingRight: 4 }}>
              {roster.map((st: any) => {
                const att = attendanceMap.get(st.id);
                const status = att?.status;
                const isPresent = status === 'present';
                const isLate = status === 'late';
                const isAbsent = status === 'absent';
                return (
                  <div
                    key={st.id}
                    className={`roster-card ${isPresent ? 'marked-present' : ''}`}
                    style={{ padding: '12px 16px' }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <strong style={{ color: '#ffffff', fontSize: 14 }}>{st.name}</strong>
                        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>
                          Roll: <code style={{ color: '#cbd5e1' }}>{st.roll_number ?? '—'}</code> | ID: <code style={{ color: '#34d6b5' }}>{st.id}</code>
                        </div>
                      </div>
                      <div>
                        {isPresent && <span className="status-chip present">● Present</span>}
                        {isLate && <span className="status-chip late">● Late</span>}
                        {isAbsent && <span className="status-chip absent">● Absent</span>}
                        {!status && <span className="status-chip nc">○ Unmarked</span>}
                      </div>
                    </div>

                    <div className="roster-actions" style={{ marginTop: 8 }}>
                      <button
                        className="roster-btn present"
                        style={{ opacity: isPresent ? 1 : 0.65 }}
                        onClick={() => handleQuickMark(st.id, 'present')}
                      >
                        ✓ Mark Present
                      </button>
                      <button
                        className="roster-btn late"
                        style={{ opacity: isLate ? 1 : 0.65 }}
                        onClick={() => handleQuickMark(st.id, 'late')}
                      >
                        ⏰ Late
                      </button>
                      <button
                        className="roster-btn absent"
                        style={{ opacity: isAbsent ? 1 : 0.65 }}
                        onClick={() => handleQuickMark(st.id, 'absent')}
                      >
                        ✗ Absent
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Sub-component 8: Manage Users (Admin Only) ────────────────────────────────
function ManageUsersView({ showModal, setError }: any) {
  const [users, setUsers] = useState<any[]>([]);
  const [form, setForm] = useState({ username: '', password: '', email: '', role: 'student', student_id: '', full_name: '' });
  const [loading, setLoading] = useState(false);
  const [editUser, setEditUser] = useState<any>(null);
  const [resetPw, setResetPw] = useState<any>(null);

  const load = () => listUsers().then(setUsers).catch(e => setError(e.message));
  useEffect(() => { load(); }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true);
    try {
      if (form.role === 'student') {
        const sid = (form.student_id || form.username).trim();
        const sname = (form.full_name || form.username).trim();
        await createUser({
          ...form,
          username: sid,
          student_id: sid,
          name: sname,
          email: form.email || undefined,
        });
      } else {
        await createUser(form);
      }
      setForm({ username: '', password: '', email: '', role: 'student', student_id: '', full_name: '' });
      load();
      showModal({ title: 'Account created', message: `User "${form.username || form.student_id}" (${form.role}) created successfully.`, type: 'success' });
    } catch (err: any) { setError(err.message); }
    finally { setLoading(false); }
  };

  const handleUpdateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await updateUser(editUser.id, { email: editUser.email || undefined, active: editUser.active });
      setEditUser(null); load();
      showModal({ title: 'User updated', message: 'User account has been updated.', type: 'success' });
    } catch (err: any) { setError(err.message); }
  };

  const handleResetPw = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await updateUser(resetPw.id, { new_password: resetPw.new_password });
      setResetPw(null);
      showModal({ title: 'Password reset', message: `Password for "${resetPw.username}" has been reset.`, type: 'success' });
    } catch (err: any) { setError(err.message); }
  };

  return (
    <>
      <div className="panel form-panel">
        <h2>Create user account</h2>
        <form onSubmit={submit} className="form-grid form-grid-2">
          {form.role === 'student' ? (
            <>
              <input placeholder="Student ID / Username (e.g. 23UAM090)" value={form.student_id}
                onChange={e => setForm({ ...form, student_id: e.target.value, username: e.target.value })} required />
              <input placeholder="Student Full Name (e.g. Aditya Sharma)" value={form.full_name}
                onChange={e => setForm({ ...form, full_name: e.target.value })} required />
            </>
          ) : (
            <input placeholder="Username" value={form.username} onChange={e => setForm({ ...form, username: e.target.value })} required />
          )}
          <input type="password" placeholder="Password (min 6 chars)" value={form.password}
            onChange={e => setForm({ ...form, password: e.target.value })} minLength={6} required />
          <input type="email" placeholder="Email (optional)" value={form.email}
            onChange={e => setForm({ ...form, email: e.target.value })} />
          <select value={form.role} onChange={e => setForm({ ...form, role: e.target.value })}>
            <option value="student">Student</option>
            <option value="teacher">Teacher</option>
            <option value="admin">Admin</option>
          </select>
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Creating…' : 'Create account'}
          </button>
        </form>
      </div>

      <div className="panel" style={{ marginTop: 20 }}>
        <h2>All Users</h2>
        <table>
          <thead>
            <tr><th>Username / ID</th><th>Full Name</th><th>Role</th><th>Email</th><th>Linked Student ID</th><th>Last Login</th><th>Status</th><th>Actions</th></tr>
          </thead>
          <tbody>
            {users.map(u => (
              <tr key={u.id}>
                <td><strong>{u.username}</strong></td>
                <td>{u.student_name ? <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{u.student_name}</span> : <span style={{ color: '#64748b' }}>—</span>}</td>
                <td><span className={`role-badge ${u.role}`}>{u.role}</span></td>
                <td>{u.email ?? '—'}</td>
                <td>{u.student_id ? <code>{u.student_id}</code> : '—'}</td>
                <td style={{ fontSize: 12 }}>{u.last_login ? new Date(u.last_login).toLocaleString() : 'Never'}</td>
                <td>{u.active ? <span className="status-chip present">Active</span> : <span className="status-chip nc">Inactive</span>}</td>
                <td>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button className="btn-edit" onClick={() => setEditUser({ id: u.id, email: u.email ?? '', active: u.active })}>Edit</button>
                    <button className="btn-reset-pw" onClick={() => setResetPw({ id: u.id, username: u.username, new_password: '' })}>🔑 Reset PW</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {editUser && (
        <FormModal title="Edit user account" onClose={() => setEditUser(null)}>
          <form onSubmit={handleUpdateUser} className="form-grid">
            <input type="email" placeholder="Email address" value={editUser.email}
              onChange={e => setEditUser({ ...editUser, email: e.target.value })} />
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#8a9bb8' }}>
              <input type="checkbox" checked={editUser.active} onChange={e => setEditUser({ ...editUser, active: e.target.checked })} />
              Account is active
            </label>
            <div className="form-modal-actions">
              <button type="submit" className="btn-primary">Save changes</button>
              <button type="button" className="btn-secondary" onClick={() => setEditUser(null)}>Cancel</button>
            </div>
          </form>
        </FormModal>
      )}

      {resetPw && (
        <FormModal title={`🔑 Reset Password — ${resetPw.username}`} onClose={() => setResetPw(null)}>
          <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
            Set a new login password for user <strong>{resetPw.username}</strong>. The user will be required to use this new credential on next sign-in.
          </p>
          <form onSubmit={handleResetPw} className="form-grid">
            <div>
              <label style={{ color: '#cbd5e1', fontSize: 12, fontWeight: 600, display: 'block', marginBottom: 6 }}>
                New Password (minimum 6 characters)
              </label>
              <input
                type="password"
                placeholder="Enter new password"
                value={resetPw.new_password}
                onChange={e => setResetPw({ ...resetPw, new_password: e.target.value })}
                minLength={6}
                required
                autoFocus
              />
            </div>
            <div className="form-modal-actions" style={{ marginTop: 12 }}>
              <button type="submit" className="btn-primary">Update Password</button>
              <button type="button" className="btn-secondary" onClick={() => setResetPw(null)}>Cancel</button>
            </div>
          </form>
        </FormModal>
      )}
    </>
  );
}

// ── Sub-component 9: Audit Log (Admin Only) ───────────────────────────────────
function AuditLogView({ setError }: any) {
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    getAuditLog()
      .then(setLogs)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <div>
          <h2 style={{ margin: 0 }}>System Audit Trail</h2>
          <p style={{ color: '#8a9bb8', margin: '4px 0 0' }}>Security and operational event history.</p>
        </div>
        <button className="btn-secondary" onClick={load} disabled={loading}>↻ Refresh</button>
      </div>

      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Actor</th>
            <th>Action</th>
            <th>Target Type</th>
            <th>Target ID</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {logs.length === 0 && (
            <tr><td colSpan={6} style={{ textAlign: 'center', color: '#8a9bb8', padding: 24 }}>No audit log entries found.</td></tr>
          )}
          {logs.map(lg => (
            <tr key={lg.id}>
              <td style={{ fontSize: 12 }}>{new Date(lg.created_at + 'Z').toLocaleString()}</td>
              <td><strong>{lg.actor_username}</strong></td>
              <td><span className="status-chip present">{lg.action}</span></td>
              <td>{lg.target_type ?? '—'}</td>
              <td><code>{lg.target_id ?? '—'}</code></td>
              <td style={{ fontSize: 13 }}>{lg.details ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
