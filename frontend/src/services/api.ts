import type {
  User,
  LoginResponse,
  Student,
  StudentCreate,
  StudentUpdate,
  ClassItem,
  ClassCreate,
  ClassUpdate,
  SessionItem,
  SessionCreate,
  AttendanceRecord,
  DashboardSummary,
  RecentAttendanceResponse,
} from '../types';

const TOKEN_KEY = 'presento_auth_token';

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');
  }

  public getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  public setToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
  }

  public clearToken(): void {
    localStorage.removeItem(TOKEN_KEY);
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...(options.headers as Record<string, string> || {}),
    };

    const token = this.getToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    let response: Response;
    try {
      response = await fetch(url, {
        ...options,
        headers,
      });
    } catch (networkError) {
      throw new Error('Backend service is unreachable. Please verify the FastAPI server is running.');
    }

    if (response.status === 401) {
      this.clearToken();
      // Notify application of auth failure
      window.dispatchEvent(new CustomEvent('presento:unauthorized'));
      throw new Error('Your session has expired. Please log in again.');
    }

    if (response.status === 204) {
      return {} as T;
    }

    let data: any;
    try {
      data = await response.json();
    } catch {
      if (!response.ok) {
        throw new Error(`Server returned error HTTP ${response.status}`);
      }
      return {} as T;
    }

    if (!response.ok) {
      const errorMsg = data?.message || data?.detail || data?.error || `Request failed with status ${response.status}`;
      throw new Error(typeof errorMsg === 'string' ? errorMsg : JSON.stringify(errorMsg));
    }

    return data as T;
  }

  // ── Authentication ─────────────────────────────────────────────────────────

  public async login(username: string, password: string): Promise<LoginResponse> {
    const data = await this.request<LoginResponse>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    if (data.access_token) {
      this.setToken(data.access_token);
    }
    return data;
  }

  public async getMe(): Promise<User> {
    return this.request<User>('/api/auth/me', { method: 'GET' });
  }

  public async logout(): Promise<void> {
    try {
      await this.request('/api/auth/logout', { method: 'POST' });
    } catch {
      // Ignore logout backend errors; clear token locally regardless
    } finally {
      this.clearToken();
    }
  }

  // ── Students ───────────────────────────────────────────────────────────────

  public async getStudents(): Promise<Student[]> {
    return this.request<Student[]>('/api/students', { method: 'GET' });
  }

  public async getStudent(id: number): Promise<Student> {
    return this.request<Student>(`/api/students/${id}`, { method: 'GET' });
  }

  public async createStudent(data: StudentCreate): Promise<Student> {
    return this.request<Student>('/api/students', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  public async updateStudent(id: number, data: StudentUpdate): Promise<Student> {
    return this.request<Student>(`/api/students/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  public async deleteStudent(id: number): Promise<void> {
    await this.request(`/api/students/${id}`, { method: 'DELETE' });
  }

  // ── Classes ────────────────────────────────────────────────────────────────

  public async getClasses(): Promise<ClassItem[]> {
    return this.request<ClassItem[]>('/api/classes', { method: 'GET' });
  }

  public async getClass(id: number): Promise<ClassItem> {
    return this.request<ClassItem>(`/api/classes/${id}`, { method: 'GET' });
  }

  public async createClass(data: ClassCreate): Promise<ClassItem> {
    return this.request<ClassItem>('/api/classes', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  public async updateClass(id: number, data: ClassUpdate): Promise<ClassItem> {
    return this.request<ClassItem>(`/api/classes/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  public async enrollStudent(classId: number, studentId: string): Promise<any> {
    return this.request(`/api/classes/${classId}/students/${encodeURIComponent(studentId)}`, {
      method: 'POST',
    });
  }

  public async removeStudent(classId: number, studentId: string): Promise<any> {
    return this.request(`/api/classes/${classId}/students/${encodeURIComponent(studentId)}`, {
      method: 'DELETE',
    });
  }

  // ── Sessions ───────────────────────────────────────────────────────────────

  public async getSessions(params?: { class_id?: number; status?: string; date?: string }): Promise<SessionItem[]> {
    const query = new URLSearchParams();
    if (params?.class_id) query.append('class_id', params.class_id.toString());
    if (params?.status) query.append('status', params.status);
    if (params?.date) query.append('date', params.date);

    const qs = query.toString();
    return this.request<SessionItem[]>(`/api/sessions${qs ? `?${qs}` : ''}`, { method: 'GET' });
  }

  public async getSession(id: number): Promise<SessionItem> {
    return this.request<SessionItem>(`/api/sessions/${id}`, { method: 'GET' });
  }

  public async createSession(data: SessionCreate): Promise<SessionItem> {
    return this.request<SessionItem>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  public async startSession(id: number): Promise<SessionItem> {
    return this.request<SessionItem>(`/api/sessions/${id}/start`, { method: 'POST' });
  }

  public async closeSession(id: number): Promise<SessionItem> {
    return this.request<SessionItem>(`/api/sessions/${id}/close`, { method: 'POST' });
  }

  public async cancelSession(id: number): Promise<SessionItem> {
    return this.request<SessionItem>(`/api/sessions/${id}/cancel`, { method: 'POST' });
  }

  // ── Attendance ─────────────────────────────────────────────────────────────

  public async getSessionAttendance(sessionId: number): Promise<AttendanceRecord[]> {
    return this.request<AttendanceRecord[]>(`/api/sessions/${sessionId}/attendance`, { method: 'GET' });
  }

  public async getAttendanceHistory(params?: {
    date?: string;
    class_id?: number;
    student_id?: string;
    status?: string;
  }): Promise<AttendanceRecord[]> {
    const query = new URLSearchParams();
    if (params?.date) query.append('date', params.date);
    if (params?.class_id) query.append('class_id', params.class_id.toString());
    if (params?.student_id) query.append('student_id', params.student_id);
    if (params?.status) query.append('status', params.status);

    const qs = query.toString();
    return this.request<AttendanceRecord[]>(`/api/attendance/history${qs ? `?${qs}` : ''}`, { method: 'GET' });
  }

  // ── Dashboard ──────────────────────────────────────────────────────────────

  public async getDashboardSummary(): Promise<DashboardSummary> {
    return this.request<DashboardSummary>('/api/dashboard/summary', { method: 'GET' });
  }

  public async getRecentAttendance(): Promise<RecentAttendanceResponse> {
    return this.request<RecentAttendanceResponse>('/api/dashboard/recent-attendance', { method: 'GET' });
  }
}

export const api = new ApiService();
export default api;
