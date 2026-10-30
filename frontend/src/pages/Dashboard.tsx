import React, { useState, useEffect, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../services/api';
import type { DashboardSummary, AttendanceRecord } from '../types';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';
import { formatTime, formatConfidence, formatLiveness, formatSource } from '../utils/format';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [recentAttendance, setRecentAttendance] = useState<AttendanceRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [sumData, recentData] = await Promise.all([
        api.getDashboardSummary(),
        api.getRecentAttendance(),
      ]);
      setSummary(sumData);
      setRecentAttendance(recentData.records || []);
    } catch (err: any) {
      setError(err?.message || 'Failed to load dashboard metrics.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  if (isLoading && !summary) {
    return <LoadingSpinner message="Loading dashboard analytics..." size="lg" />;
  }

  return (
    <div className="page-container dashboard-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Attendance Overview</h1>
          <p className="page-subtitle">Real-time attendance metrics across college lectures and sessions</p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            onClick={fetchDashboardData}
            className="btn btn-outline"
            title="Refresh dashboard metrics"
          >
            Refresh
          </button>
          <button
            type="button"
            onClick={() => navigate('/classes')}
            className="btn btn-primary"
          >
            Start New Session
          </button>
        </div>
      </div>

      {error && (
        <ErrorMessage
          message={error}
          onDismiss={() => setError(null)}
          className="dashboard-error"
        />
      )}

      {/* KPI Stats Grid */}
      <section className="stats-grid" aria-label="Dashboard Summary Statistics">
        <div className="stat-card">
          <div className="stat-label">Total Enrolled Students</div>
          <div className="stat-value">{summary?.total_students ?? 0}</div>
          <div className="stat-desc">Across all active batches</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Total Classes</div>
          <div className="stat-value">{summary?.total_classes ?? 0}</div>
          <div className="stat-desc">Active academic courses</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Today's Sessions</div>
          <div className="stat-value">{summary?.today_sessions ?? 0}</div>
          <div className="stat-desc">Scheduled & active lectures</div>
        </div>

        <div className="stat-card stat-card-success">
          <div className="stat-label">Present Today</div>
          <div className="stat-value text-success">{summary?.present_today ?? 0}</div>
          <div className="stat-desc">Verified attendances</div>
        </div>

        <div className="stat-card stat-card-danger">
          <div className="stat-label">Absent Today</div>
          <div className="stat-value text-danger">{summary?.absent_today ?? 0}</div>
          <div className="stat-desc">Unrecorded students</div>
        </div>

        <div className="stat-card stat-card-highlight">
          <div className="stat-label">Overall Attendance</div>
          <div className="stat-value">
            {summary ? `${summary.attendance_percentage.toFixed(1)}%` : '0%'}
          </div>
          <div className="stat-desc">Today's completion rate</div>
        </div>
      </section>

      {/* Recent Attendance Section */}
      <section className="dashboard-section" aria-label="Recent Attendance Stream">
        <div className="section-header">
          <h2 className="section-title">Recent Attendance Activity</h2>
          <Link to="/attendance" className="link-subtle">
            View Full History &rarr;
          </Link>
        </div>

        {recentAttendance.length === 0 ? (
          <EmptyState
            title="No Attendance Activity Yet Today"
            description="Start a lecture session from the Classes page to record attendance automatically via AI Camera."
            actionLabel="View Classes"
            onAction={() => navigate('/classes')}
          />
        ) : (
          <div className="table-responsive-wrapper card">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Student ID</th>
                  <th>Student Name</th>
                  <th>Time Marked</th>
                  <th>Status</th>
                  <th>Confidence</th>
                  <th>Liveness</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {recentAttendance.map((rec) => (
                  <tr key={rec.id}>
                    <td className="font-mono font-medium">{rec.student_id}</td>
                    <td className="font-semibold">{rec.student_name}</td>
                    <td className="text-muted">{formatTime(rec.timestamp)}</td>
                    <td>
                      <StatusBadge status={rec.status} />
                    </td>
                    <td className="text-muted">{formatConfidence(rec.confidence)}</td>
                    <td>
                      <span className="liveness-pill">
                        {formatLiveness(rec.liveness_state, rec.liveness_score)}
                      </span>
                    </td>
                    <td className="text-muted">{formatSource(rec.source)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
};

export default Dashboard;
