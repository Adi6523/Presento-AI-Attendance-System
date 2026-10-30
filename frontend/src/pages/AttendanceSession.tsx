import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../services/api';
import type { SessionItem, AttendanceRecord, ClassItem } from '../types';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import StatusBadge from '../components/StatusBadge';
import { usePolling } from '../hooks/usePolling';
import { formatTime, formatConfidence, formatLiveness, formatSource, formatDate } from '../utils/format';

export const AttendanceSession: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const sid = parseInt(sessionId || '', 10);

  const [session, setSession] = useState<SessionItem | null>(null);
  const [classInfo, setClassInfo] = useState<ClassItem | null>(null);
  const [records, setRecords] = useState<AttendanceRecord[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isClosing, setIsClosing] = useState<boolean>(false);
  const [lastSyncedTime, setLastSyncedTime] = useState<string>('');

  // Initial load
  const loadInitialData = useCallback(async () => {
    if (isNaN(sid)) return;
    setIsLoading(true);
    setError(null);
    try {
      const sess = await api.getSession(sid);
      setSession(sess);

      const [cls, att] = await Promise.all([
        api.getClass(sess.class_id),
        api.getSessionAttendance(sid),
      ]);
      setClassInfo(cls);
      setRecords(att);
      setLastSyncedTime(new Date().toLocaleTimeString());
    } catch (err: any) {
      setError(err?.message || 'Failed to load attendance session.');
    } finally {
      setIsLoading(false);
    }
  }, [sid]);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  // Polling function: only queries attendance records while session is ACTIVE
  const isSessionActive = session?.status === 'ACTIVE';

  const pollAttendance = useCallback(async () => {
    if (!isSessionActive || isNaN(sid)) return;
    try {
      const att = await api.getSessionAttendance(sid);
      setRecords(att);
      setLastSyncedTime(new Date().toLocaleTimeString());
    } catch (err) {
      // Polling network blips handled quietly
    }
  }, [isSessionActive, sid]);

  // Poll every 2.5 seconds while ACTIVE
  usePolling(pollAttendance, 2500, isSessionActive);

  // Close Session Handler
  const handleCloseSession = async () => {
    if (!window.confirm('Are you sure you want to finalize and close this attendance session? The backend will mark all remaining unverified students as ABSENT.')) {
      return;
    }

    setIsClosing(true);
    setError(null);
    try {
      // 1. Call backend close session endpoint
      const closedSession = await api.closeSession(sid);
      setSession(closedSession);

      // 2. Refresh attendance records immediately to pull backend-generated ABSENT rows
      const finalRecords = await api.getSessionAttendance(sid);
      setRecords(finalRecords);
      setLastSyncedTime(new Date().toLocaleTimeString());
    } catch (err: any) {
      setError(err?.message || 'Failed to close attendance session.');
    } finally {
      setIsClosing(false);
    }
  };

  // Build the complete roster view:
  // If session is ACTIVE, enrolled students who haven't yet been recognized show as PENDING / NOT YET MARKED.
  // If session is COMPLETED, records array contains both PRESENT and backend-generated ABSENT rows!
  const rosterRows = useMemo(() => {
    const enrolledStudents = classInfo?.students || [];
    const recordedMap = new Map<string, AttendanceRecord>();
    records.forEach((r) => recordedMap.set(r.student_id, r));

    if (session?.status === 'COMPLETED') {
      // Session finalized: backend is authoritative
      return records;
    }

    // While session is ACTIVE:
    // Combine recorded students with remaining enrolled students
    const rows: AttendanceRecord[] = [...records];
    enrolledStudents.forEach((st) => {
      if (!recordedMap.has(st.student_id)) {
        rows.push({
          id: -st.id,
          session_id: sid,
          student_id: st.student_id,
          student_name: st.name,
          timestamp: '',
          status: 'ABSENT', // Pending recognition
          confidence: null,
          liveness_score: null,
          liveness_state: null,
          source: '—',
        });
      }
    });

    return rows;
  }, [records, classInfo, session, sid]);

  // Metrics Counters
  const totalRoster = classInfo?.enrolled_count || (classInfo?.students ? classInfo.students.length : records.length);
  const presentCount = records.filter((r) => r.status === 'PRESENT').length;
  const absentCount = session?.status === 'COMPLETED'
    ? records.filter((r) => r.status === 'ABSENT').length
    : Math.max(0, totalRoster - presentCount);
  const attendanceRate = totalRoster > 0 ? (presentCount / totalRoster) * 100 : 0;

  if (isLoading && !session) {
    return <LoadingSpinner message="Connecting to live session feed..." size="lg" />;
  }

  if (!session) {
    return (
      <div className="page-container">
        <ErrorMessage message={error || 'Attendance session not found'} />
        <button type="button" onClick={() => navigate('/classes')} className="btn btn-outline">
          Back to Classes
        </button>
      </div>
    );
  }

  return (
    <div className="page-container active-session-page">
      {/* Breadcrumb Navigation */}
      <div className="breadcrumb-nav">
        <button
          type="button"
          onClick={() => navigate(`/classes/${session.class_id}`)}
          className="breadcrumb-link"
        >
          &larr; Back to Class: {classInfo?.name || 'Class'}
        </button>
      </div>

      {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

      {/* Session Hero Banner */}
      <div className="session-banner card">
        <div className="session-banner-main">
          <div className="session-tag-row">
            <span className="session-subject-tag">{classInfo?.subject || session.subject || 'Lecture'}</span>
            <StatusBadge status={session.status} />
            {isSessionActive && (
              <span className="live-polling-badge" title="Live background polling every 2.5s">
                <span className="live-dot-pulse" />
                Live Camera Feed Active
              </span>
            )}
          </div>

          <h1 className="session-title">{classInfo?.name || session.class_name || 'Class Session'}</h1>

          <div className="session-meta-row">
            <span>Date: <strong>{formatDate(session.date)}</strong></span>
            <span className="dot-sep">&bull;</span>
            <span>Started: <strong>{session.start_time || '—'}</strong></span>
            {session.end_time && (
              <>
                <span className="dot-sep">&bull;</span>
                <span>Closed: <strong>{session.end_time}</strong></span>
              </>
            )}
            {lastSyncedTime && (
              <>
                <span className="dot-sep">&bull;</span>
                <span className="text-muted">Last Synced: {lastSyncedTime}</span>
              </>
            )}
          </div>
        </div>

        <div className="session-banner-actions">
          {isSessionActive ? (
            <button
              type="button"
              onClick={handleCloseSession}
              disabled={isClosing}
              className="btn btn-danger btn-lg close-session-btn"
            >
              {isClosing ? 'Closing Session...' : 'Close Attendance Session'}
            </button>
          ) : (
            <div className="session-completed-notice">
              <span className="notice-icon">&#10003;</span>
              <div>
                <strong>Session Completed</strong>
                <p>Absence records finalized by backend</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Attendance Stats Counter Strip */}
      <div className="session-stats-strip">
        <div className="session-stat-box card">
          <span className="stat-label">Total Roster</span>
          <span className="stat-num">{totalRoster}</span>
          <span className="stat-sub">Enrolled students</span>
        </div>

        <div className="session-stat-box card stat-box-present">
          <span className="stat-label">Verified Present</span>
          <span className="stat-num text-success">{presentCount}</span>
          <span className="stat-sub">Confirmed via AI Vision</span>
        </div>

        <div className="session-stat-box card stat-box-absent">
          <span className="stat-label">Absent / Pending</span>
          <span className="stat-num text-danger">{absentCount}</span>
          <span className="stat-sub">
            {session.status === 'COMPLETED' ? 'Marked absent by backend' : 'Awaiting recognition'}
          </span>
        </div>

        <div className="session-stat-box card stat-box-rate">
          <span className="stat-label">Attendance Rate</span>
          <span className="stat-num">{attendanceRate.toFixed(1)}%</span>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill" style={{ width: `${Math.min(100, attendanceRate)}%` }} />
          </div>
        </div>
      </div>

      {/* Real-time Attendance Roster Table */}
      <div className="section-container">
        <div className="section-header">
          <div>
            <h2 className="section-title">Classroom Attendance Roster</h2>
            <p className="section-subtitle">
              {isSessionActive
                ? 'Updates in real-time as the AI Vision camera identifies students with verified liveness.'
                : 'Finalized attendance record for this completed session.'}
            </p>
          </div>
          {isSessionActive && (
            <button
              type="button"
              onClick={pollAttendance}
              className="btn btn-outline btn-sm"
              title="Manually trigger update"
            >
              Sync Now
            </button>
          )}
        </div>

        <div className="table-responsive-wrapper card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Student ID</th>
                <th>Student Name</th>
                <th>Status</th>
                <th>Time Marked</th>
                <th>Recognition Confidence</th>
                <th>Liveness Status</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {rosterRows.length === 0 ? (
                <tr>
                  <td colSpan={7} className="text-center py-6 text-muted">
                    No students currently in this session roster.
                  </td>
                </tr>
              ) : (
                rosterRows.map((row) => (
                  <tr key={`${row.student_id}-${row.id}`} className={row.status === 'PRESENT' ? 'row-highlight-present' : ''}>
                    <td className="font-mono font-medium">{row.student_id}</td>
                    <td className="font-semibold">{row.student_name}</td>
                    <td>
                      <StatusBadge status={row.status} />
                    </td>
                    <td className="text-muted">{row.timestamp ? formatTime(row.timestamp) : '—'}</td>
                    <td className="text-muted">{formatConfidence(row.confidence)}</td>
                    <td>
                      {row.status === 'PRESENT' ? (
                        <span className="liveness-pill verified">
                          {formatLiveness(row.liveness_state, row.liveness_score)}
                        </span>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                    </td>
                    <td className="text-muted">{formatSource(row.source)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default AttendanceSession;
