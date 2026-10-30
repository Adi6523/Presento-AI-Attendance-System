import React, { useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import type { AttendanceRecord, ClassItem } from '../types';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import EmptyState from '../components/EmptyState';
import { formatDate, formatTime, formatConfidence, formatLiveness, formatSource } from '../utils/format';

export const AttendanceHistory: React.FC = () => {
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters State
  const [selectedClassId, setSelectedClassId] = useState<string>('');
  const [studentIdFilter, setStudentIdFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [dateFilter, setDateFilter] = useState<string>('');

  const fetchRecords = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [history, classList] = await Promise.all([
        api.getAttendanceHistory({
          class_id: selectedClassId ? parseInt(selectedClassId, 10) : undefined,
          student_id: studentIdFilter.trim() || undefined,
          status: statusFilter || undefined,
          date: dateFilter || undefined,
        }),
        api.getClasses(),
      ]);
      setRecords(history);
      setClasses(classList);
    } catch (err: any) {
      setError(err?.message || 'Failed to retrieve attendance logs.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedClassId, studentIdFilter, statusFilter, dateFilter]);

  useEffect(() => {
    fetchRecords();
  }, [fetchRecords]);

  const handleResetFilters = () => {
    setSelectedClassId('');
    setStudentIdFilter('');
    setStatusFilter('');
    setDateFilter('');
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title">Attendance History Logs</h1>
          <p className="page-subtitle">Audited record of all biometric face recognitions and attendance entries</p>
        </div>
        <div className="header-actions">
          <button type="button" onClick={fetchRecords} className="btn btn-outline">
            Refresh Logs
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

      {/* Filter Control Strip */}
      <div className="filter-card card">
        <div className="filters-grid">
          <div className="filter-item">
            <label htmlFor="filterClass" className="filter-label">Filter by Course</label>
            <select
              id="filterClass"
              value={selectedClassId}
              onChange={(e) => setSelectedClassId(e.target.value)}
              className="form-input form-select"
            >
              <option value="">All Courses</option>
              {classes.map((c) => (
                <option key={c.id} value={c.id.toString()}>
                  {c.name} ({c.subject})
                </option>
              ))}
            </select>
          </div>

          <div className="filter-item">
            <label htmlFor="filterStudent" className="filter-label">Student ID (Roll No)</label>
            <input
              id="filterStudent"
              type="text"
              value={studentIdFilter}
              onChange={(e) => setStudentIdFilter(e.target.value)}
              placeholder="e.g. 23UAM090"
              className="form-input font-mono"
            />
          </div>

          <div className="filter-item">
            <label htmlFor="filterStatus" className="filter-label">Attendance Status</label>
            <select
              id="filterStatus"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="form-input form-select"
            >
              <option value="">All Statuses</option>
              <option value="PRESENT">PRESENT</option>
              <option value="ABSENT">ABSENT</option>
              <option value="BLOCKED">BLOCKED</option>
              <option value="REJECTED">REJECTED</option>
            </select>
          </div>

          <div className="filter-item">
            <label htmlFor="filterDate" className="filter-label">Date</label>
            <input
              id="filterDate"
              type="date"
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
              className="form-input"
            />
          </div>
        </div>

        {(selectedClassId || studentIdFilter || statusFilter || dateFilter) && (
          <div className="filter-actions-row">
            <button
              type="button"
              onClick={handleResetFilters}
              className="btn btn-outline btn-sm"
            >
              Reset All Filters
            </button>
            <span className="filter-count">
              Showing <strong>{records.length}</strong> matching entries
            </span>
          </div>
        )}
      </div>

      {isLoading && records.length === 0 ? (
        <LoadingSpinner message="Searching attendance records..." size="lg" />
      ) : records.length === 0 ? (
        <EmptyState
          title="No Attendance Logs Found"
          description="No records match the current filter criteria, or no attendance has been recorded yet."
          actionLabel="Clear Filters"
          onAction={handleResetFilters}
        />
      ) : (
        <div className="table-responsive-wrapper card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Session ID</th>
                <th>Student ID</th>
                <th>Student Name</th>
                <th>Status</th>
                <th>Time</th>
                <th>Confidence</th>
                <th>Liveness</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {records.map((r) => (
                <tr key={r.id}>
                  <td className="font-medium">{formatDate(r.timestamp)}</td>
                  <td className="font-mono text-muted">#{r.session_id}</td>
                  <td className="font-mono font-medium">{r.student_id}</td>
                  <td className="font-semibold">{r.student_name}</td>
                  <td>
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="text-muted">{formatTime(r.timestamp)}</td>
                  <td className="text-muted">{formatConfidence(r.confidence)}</td>
                  <td>
                    <span className="liveness-pill">
                      {formatLiveness(r.liveness_state, r.liveness_score)}
                    </span>
                  </td>
                  <td className="text-muted">{formatSource(r.source)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AttendanceHistory;
