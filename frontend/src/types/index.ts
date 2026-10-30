export interface User {
  id: number;
  username: string;
  role: 'ADMIN' | 'TEACHER';
  is_active: boolean;
  created_at: string;
  teacher?: TeacherBrief | null;
}

export interface TeacherBrief {
  id: number;
  user_id: number;
  name: string;
  email: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Student {
  id: number;
  student_id: string;
  name: string;
  email?: string | null;
  department?: string | null;
  year?: string | null;
  is_active: boolean;
  created_at: string;
}

export interface StudentCreate {
  student_id: string;
  name: string;
  email?: string;
  department?: string;
  year?: string;
}

export interface StudentUpdate {
  name?: string;
  email?: string;
  department?: string;
  year?: string;
  is_active?: boolean;
}

export interface ClassItem {
  id: number;
  name: string;
  subject: string;
  teacher_id?: number | null;
  academic_year: string;
  semester: number;
  created_at: string;
  enrolled_count: number;
  students?: Student[];
}

export interface ClassCreate {
  name: string;
  subject: string;
  teacher_id?: number;
  academic_year: string;
  semester: number;
}

export interface ClassUpdate {
  name?: string;
  subject?: string;
  teacher_id?: number;
  academic_year?: string;
  semester?: number;
}

export interface SessionItem {
  id: number;
  class_id: number;
  teacher_id?: number | null;
  date: string;
  start_time?: string | null;
  end_time?: string | null;
  status: 'SCHEDULED' | 'ACTIVE' | 'COMPLETED' | 'CANCELLED';
  created_at: string;
  class_name?: string | null;
  subject?: string | null;
}

export interface SessionCreate {
  class_id: number;
  teacher_id?: number;
  date: string;
  start_time?: string;
  end_time?: string;
}

export interface AttendanceRecord {
  id: number;
  session_id: number;
  student_id: string;
  student_name: string;
  timestamp: string;
  status: 'PRESENT' | 'ABSENT' | 'BLOCKED' | 'REJECTED';
  confidence?: number | null;
  liveness_score?: number | null;
  liveness_state?: string | null;
  source: string;
}

export interface DashboardSummary {
  total_students: number;
  total_classes: number;
  today_sessions: number;
  present_today: number;
  absent_today: number;
  attendance_percentage: number;
}

export interface RecentAttendanceResponse {
  records: AttendanceRecord[];
}

export interface ApiError {
  success: boolean;
  error?: string;
  message: string;
}
