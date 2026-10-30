const BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export type Role = 'admin' | 'teacher' | 'student';

export interface AuthUser {
  username: string;
  role: Role;
  student_id: string | null;
  access_token: string;
}

// ── Token storage (sessionStorage ensures no auto-login on new browser session) ─
export const getToken = (): string | null => sessionStorage.getItem('presento_token');
export const getUser = (): AuthUser | null => {
  const raw = sessionStorage.getItem('presento_user');
  return raw ? JSON.parse(raw) : null;
};
export const saveSession = (user: AuthUser) => {
  sessionStorage.setItem('presento_token', user.access_token);
  sessionStorage.setItem('presento_user', JSON.stringify(user));
};
export const clearSession = () => {
  sessionStorage.removeItem('presento_token');
  sessionStorage.removeItem('presento_user');
  localStorage.removeItem('presento_token');
  localStorage.removeItem('presento_user');
};

// ── Core fetch wrapper ────────────────────────────────────────────────────────
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const res = await fetch(BASE + path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init.headers,
    },
  });
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => ({ detail: 'Request failed' }));
  if (!res.ok) {
    const msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    throw new Error(msg);
  }
  return body as T;
}

// ── Auth ──────────────────────────────────────────────────────────────────────
export async function login(username: string, password: string): Promise<AuthUser> {
  const form = new URLSearchParams({ username, password });
  const res = await fetch(BASE + '/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: form.toString(),
  });
  const body = await res.json().catch(() => ({ detail: 'Login failed' }));
  if (!res.ok) throw new Error(body.detail ?? 'Login failed');
  const user: AuthUser = {
    username: body.username,
    role: body.role,
    student_id: body.student_id,
    access_token: body.access_token,
  };
  saveSession(user);
  return user;
}

export const logout = () => clearSession();

export const changePassword = (currentPassword: string, newPassword: string) =>
  request('/auth/change-password', {
    method: 'POST',
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });

// ── Users (admin) ─────────────────────────────────────────────────────────────
export const listUsers = () => request<any[]>('/auth/users');
export const createUser = (body: object) =>
  request('/auth/register', { method: 'POST', body: JSON.stringify(body) });
export const updateUser = (id: number, body: { email?: string; active?: boolean; new_password?: string }) =>
  request(`/auth/users/${id}`, { method: 'PATCH', body: JSON.stringify(body) });

// ── Academic Structure ────────────────────────────────────────────────────────
export const listDepartments = () => request<any[]>('/v1/academic/departments');
export const createDepartment = (body: { name: string; code: string }) =>
  request('/v1/academic/departments', { method: 'POST', body: JSON.stringify(body) });
export const deleteDepartment = (id: number) =>
  request(`/v1/academic/departments/${id}`, { method: 'DELETE' });

export const listAcademicYears = () => request<any[]>('/v1/academic/academic-years');
export const createAcademicYear = (body: { label: string; is_current?: boolean }) =>
  request('/v1/academic/academic-years', { method: 'POST', body: JSON.stringify(body) });
export const deleteAcademicYear = (id: number) =>
  request(`/v1/academic/academic-years/${id}`, { method: 'DELETE' });

export const listClasses = () => request<any[]>('/v1/academic/classes');
export const createClass = (body: {
  department_id: number;
  academic_year_id: number;
  year: string;
  semester?: number;
  division: string;
  label: string;
}) => request('/v1/academic/classes', { method: 'POST', body: JSON.stringify(body) });
export const deleteClass = (id: number) =>
  request(`/v1/academic/classes/${id}`, { method: 'DELETE' });

export const listSubjects = (departmentId?: number) =>
  request<any[]>(`/v1/academic/subjects${departmentId ? `?department_id=${departmentId}` : ''}`);
export const createSubject = (body: {
  department_id: number;
  name: string;
  code: string;
  semester?: number;
}) => request('/v1/academic/subjects', { method: 'POST', body: JSON.stringify(body) });
export const deleteSubject = (id: number) =>
  request(`/v1/academic/subjects/${id}`, { method: 'DELETE' });

export const listTeacherProfiles = () => request<any[]>('/v1/academic/teacher-profiles');
export const listTeacherAssignments = () => request<any[]>('/v1/academic/teacher-assignments');
export const createTeacherAssignment = (body: {
  teacher_id: number;
  subject_id: number;
  class_id: number;
}) => request('/v1/academic/teacher-assignments', { method: 'POST', body: JSON.stringify(body) });
export const deleteTeacherAssignment = (id: number) =>
  request(`/v1/academic/teacher-assignments/${id}`, { method: 'DELETE' });

export const listStudentProfiles = (classId?: number) =>
  request<any[]>(`/v1/academic/student-profiles${classId ? `?class_id=${classId}` : ''}`);
export const assignStudentToClass = (body: {
  student_id: string;
  class_id: number;
  roll_number?: string;
}) => request('/v1/academic/student-profiles', { method: 'POST', body: JSON.stringify(body) });
export const removeStudentFromClass = (id: number) =>
  request(`/v1/academic/student-profiles/${id}`, { method: 'DELETE' });

// ── Teacher Scoped Endpoints ──────────────────────────────────────────────────
export const getMyAssignments = () => request<any[]>('/v1/teacher/my-assignments');
export const getMyClasses = () => request<any[]>('/v1/teacher/my-classes');
export const getClassStudents = (classId: number) =>
  request<any[]>(`/v1/teacher/classes/${classId}/students`);

// ── Students ──────────────────────────────────────────────────────────────────
export const listStudents = () => request<any[]>('/v1/students');
export const createStudentRecord = (body: {
  id: string;
  name: string;
  email?: string;
  class_id?: number;
  roll_number?: string;
}) => request('/v1/students/manual', { method: 'POST', body: JSON.stringify(body) });
export const updateStudent = (id: string, body: { name?: string; email?: string; class_id?: number; roll_number?: string }) =>
  request(`/v1/students/${id}`, { method: 'PATCH', body: JSON.stringify(body) });
export const deactivateStudent = (id: string) =>
  request(`/v1/students/${id}`, { method: 'DELETE' });
export const enrollImages = (studentId: string, images: string[]) =>
  request(`/v1/students/${studentId}/enroll-images`, { method: 'POST', body: JSON.stringify({ images }) });
export const getStudentAttendance = (studentId: string) =>
  request<any>(`/v1/students/${studentId}/attendance`);

// ── Sessions ──────────────────────────────────────────────────────────────────
export const listSessions = () => request<any[]>('/v1/sessions');
export const createSession = (body: {
  course_code?: string;
  room: string;
  starts_at: string;
  ends_at: string;
  late_after_minutes?: number;
  teacher_id?: number;
  subject_id?: number;
  class_id?: number;
}) => request('/v1/sessions', { method: 'POST', body: JSON.stringify(body) });
export const closeSession = (id: number) =>
  request(`/v1/sessions/${id}/close`, { method: 'PATCH' });
export const finalizeAbsent = (id: number) =>
  request<{ session_id: number; marked_absent_count: number }>(`/v1/sessions/${id}/finalize-absent`, { method: 'POST' });
export const deleteSession = (id: number) =>
  request(`/v1/sessions/${id}`, { method: 'DELETE' });

// ── Attendance ────────────────────────────────────────────────────────────────
export const webCheckin = (images: string[], sessionId?: number) =>
  request('/v1/attendance/web-checkin', {
    method: 'POST',
    body: JSON.stringify({ images, session_id: sessionId ?? null }),
  });

export const listAllAttendance = (params?: {
  student_id?: string;
  session_id?: number;
  date?: string;
  class_id?: number;
}) => {
  const qs = new URLSearchParams();
  if (params?.student_id) qs.set('student_id', params.student_id);
  if (params?.session_id !== undefined) qs.set('session_id', String(params.session_id));
  if (params?.date) qs.set('date', params.date);
  if (params?.class_id !== undefined) qs.set('class_id', String(params.class_id));
  const q = qs.toString();
  return request<any[]>(`/v1/attendance${q ? '?' + q : ''}`);
};

export const addAttendanceManual = (body: {
  student_id: string;
  session_id?: number | null;
  status: string;
  observed_at: string;
  reason?: string;
}) => request('/v1/attendance/manual', { method: 'POST', body: JSON.stringify(body) });

export const editAttendance = (id: number, body: { status: string; reason?: string }) =>
  request(`/v1/attendance/${id}`, { method: 'PATCH', body: JSON.stringify(body) });

export const deleteAttendance = (id: number) =>
  request(`/v1/attendance/${id}`, { method: 'DELETE' });

// ── Analytics, Dashboard & Audit ──────────────────────────────────────────────
export const getDashboard = () => request<any>('/v1/dashboard/overview');
export const getAtRisk = (threshold = 0.75) =>
  request<any[]>(`/v1/analytics/at-risk?threshold=${threshold}`);
export const getAuditLog = (limit = 100) =>
  request<any[]>(`/v1/admin/audit-log?limit=${limit}`);

// ── CSV Export Helpers ────────────────────────────────────────────────────────
export async function downloadAttendanceCSV(params?: { session_id?: number; class_id?: number; student_id?: string; date?: string }) {
  const token = getToken();
  const qs = new URLSearchParams();
  if (params?.session_id !== undefined) qs.set('session_id', String(params.session_id));
  if (params?.class_id !== undefined) qs.set('class_id', String(params.class_id));
  if (params?.student_id) qs.set('student_id', params.student_id);
  if (params?.date) qs.set('date', params.date);
  const q = qs.toString();
  const res = await fetch(`${BASE}/v1/attendance/export${q ? '?' + q : ''}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error('Failed to download CSV export');
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `attendance_export_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

export async function downloadStudentAttendanceCSV(studentId: string) {
  const token = getToken();
  const res = await fetch(`${BASE}/v1/students/${studentId}/attendance/export`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error('Failed to download attendance history');
  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `attendance_${studentId}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
}

// ── On-Demand Pi Camera Control API ──────────────────────────────────────────
export const startCameraSession = () => request<{ status: string; session_type: string }>('/v1/camera/session/start', { method: 'POST' });
export const stopCameraSession = () => request<{ status: string; active: boolean }>('/v1/camera/session/stop', { method: 'POST' });
export const getCameraSessionStatus = () => request<{ backend_session_active: boolean; pi_status: any }>('/v1/camera/session/status');

