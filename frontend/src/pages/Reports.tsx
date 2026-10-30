import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import type { DashboardSummary, ClassItem, AttendanceRecord } from '../types';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

export const Reports: React.FC = () => {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [history, setHistory] = useState<AttendanceRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadReportData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [sum, clsList, hist] = await Promise.all([
        api.getDashboardSummary(),
        api.getClasses(),
        api.getAttendanceHistory(),
      ]);
      setSummary(sum);
      setClasses(clsList);
      setHistory(hist);
    } catch (err: any) {
      setError(err?.message || 'Failed to generate academic reports.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadReportData();
  }, [loadReportData]);

  // Aggregate student attendance totals from recorded history
  const studentMetrics = React.useMemo(() => {
    const studentMap = new Map<string, { id: string; name: string; present: number; absent: number }>();

    history.forEach((rec) => {
      if (!studentMap.has(rec.student_id)) {
        studentMap.set(rec.student_id, {
          id: rec.student_id,
          name: rec.student_name,
          present: 0,
          absent: 0,
        });
      }
      const item = studentMap.get(rec.student_id)!;
      if (rec.status === 'PRESENT') {
        item.present += 1;
      } else if (rec.status === 'ABSENT') {
        item.absent += 1;
      }
    });

    return Array.from(studentMap.values()).map((s) => {
      const total = s.present + s.absent;
      const rate = total > 0 ? (s.present / total) * 100 : 0;
      return {
        ...s,
        total,
        rate,
      };
    });
  }, [history]);

  // Students below 75% threshold
  const lowAttendanceStudents = studentMetrics.filter((s) => s.total >= 1 && s.rate < 75.0);

  if (isLoading && !summary) {
    return <LoadingSpinner message="Generating academic attendance report..." size="lg" />;
  }

  return (
    <div className="page-container reports-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Academic Attendance Reports</h1>
          <p className="page-subtitle">Departmental summaries, compliance thresholds, and course statistics</p>
        </div>
        <div className="header-actions">
          <button type="button" onClick={() => window.print()} className="btn btn-outline">
            Print / Export Report
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

      {/* KPI Overview Summary */}
      <section className="stats-grid" aria-label="Academic Attendance KPI Summary">
        <div className="stat-card">
          <div className="stat-label">Total Course Offerings</div>
          <div className="stat-value">{summary?.total_classes ?? classes.length}</div>
          <div className="stat-desc">Active classes monitored</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Total Verified Attendances</div>
          <div className="stat-value text-success">{summary?.present_today ?? 0}</div>
          <div className="stat-desc">Verified today</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Total Recorded Absences</div>
          <div className="stat-value text-danger">{summary?.absent_today ?? 0}</div>
          <div className="stat-desc">Recorded today</div>
        </div>

        <div className="stat-card stat-card-highlight">
          <div className="stat-label">Institutional Attendance Rate</div>
          <div className="stat-value">
            {summary ? `${summary.attendance_percentage.toFixed(1)}%` : '0%'}
          </div>
          <div className="stat-desc">Calculated across today's sessions</div>
        </div>
      </section>

      {/* Course Enrollment Breakdown */}
      <section className="section-container">
        <h2 className="section-title">Course Enrollment & Roster Breakdown</h2>
        <div className="table-responsive-wrapper card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Subject Code</th>
                <th>Course Name</th>
                <th>Academic Year</th>
                <th>Semester</th>
                <th>Enrolled Students</th>
              </tr>
            </thead>
            <tbody>
              {classes.length === 0 ? (
                <tr>
                  <td colSpan={5} className="text-center py-6 text-muted">
                    No classes available for reporting.
                  </td>
                </tr>
              ) : (
                classes.map((c) => (
                  <tr key={c.id}>
                    <td className="font-mono font-medium">{c.subject}</td>
                    <td className="font-semibold">{c.name}</td>
                    <td className="text-muted">{c.academic_year}</td>
                    <td className="text-muted">Semester {c.semester}</td>
                    <td className="font-semibold">{c.enrolled_count} Students</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Student Attendance Compliance Breakdown */}
      <section className="section-container">
        <div className="section-header">
          <div>
            <h2 className="section-title">Student Attendance Roster Rates</h2>
            <p className="section-subtitle">Aggregated attendance percentages based on audited session logs</p>
          </div>
        </div>

        <div className="table-responsive-wrapper card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Student ID</th>
                <th>Student Name</th>
                <th>Total Sessions</th>
                <th>Present</th>
                <th>Absent</th>
                <th>Attendance Rate</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {studentMetrics.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-6 text-muted">
                    No session logs recorded yet to calculate student metrics.
                  </td>
                </tr>
              ) : (
                studentMetrics.map((sm) => (
                  <tr key={sm.id}>
                    <td className="font-mono font-medium">{sm.id}</td>
                    <td className="font-semibold">{sm.name}</td>
                    <td className="text-muted">{sm.total}</td>
                    <td className="text-success font-medium">{sm.present}</td>
                    <td className="text-danger font-medium">{sm.absent}</td>
                    <td className="font-semibold">{sm.rate.toFixed(1)}%</td>
                    <td>
                      {sm.rate >= 75.0 ? (
                        <span className="status-pill active-pill">Satisfactory</span>
                      ) : (
                        <span className="status-pill badge-blocked">Low Attendance (&lt;75%)</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Low Attendance Warning */}
      {lowAttendanceStudents.length > 0 && (
        <section className="section-container">
          <div className="warning-callout card">
            <h3 className="callout-title">&#9888; Low Attendance Advisory (&lt; 75% Threshold)</h3>
            <p className="callout-desc">
              The following {lowAttendanceStudents.length} student(s) currently fail to satisfy the mandatory 75% attendance threshold:
            </p>
            <ul className="callout-list">
              {lowAttendanceStudents.map((st) => (
                <li key={st.id}>
                  <strong>{st.name}</strong> ({st.id}) &mdash; {st.rate.toFixed(1)}% attendance ({st.present}/{st.total} sessions)
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}
    </div>
  );
};

export default Reports;
