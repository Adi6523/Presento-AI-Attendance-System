import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../services/api';
import type { ClassItem, Student } from '../types';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import Modal from '../components/Modal';
import EmptyState from '../components/EmptyState';

export const ClassDetails: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const classId = parseInt(id || '', 10);

  const [classData, setClassData] = useState<ClassItem | null>(null);
  const [allStudents, setAllStudents] = useState<Student[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<boolean>(false);

  // Enroll Student Modal State
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState<boolean>(false);
  const [selectedStudentId, setSelectedStudentId] = useState<string>('');
  const [enrollError, setEnrollError] = useState<string | null>(null);

  const fetchClassDetails = useCallback(async () => {
    if (isNaN(classId)) return;
    setIsLoading(true);
    setError(null);
    try {
      const [cls, students] = await Promise.all([
        api.getClass(classId),
        api.getStudents(),
      ]);
      setClassData(cls);
      setAllStudents(students);
    } catch (err: any) {
      setError(err?.message || 'Failed to load class details.');
    } finally {
      setIsLoading(false);
    }
  }, [classId]);

  useEffect(() => {
    fetchClassDetails();
  }, [fetchClassDetails]);

  // Start Attendance Handler
  const handleStartAttendance = async () => {
    if (!classData) return;
    setActionLoading(true);
    setError(null);
    try {
      const today = new Date().toISOString().split('T')[0];
      // 1. Create Session
      const session = await api.createSession({
        class_id: classData.id,
        date: today,
      });

      // 2. Start Session (transitions to ACTIVE)
      await api.startSession(session.id);

      // 3. Navigate to Live Active Attendance Session
      navigate(`/attendance/${session.id}`);
    } catch (err: any) {
      setError(err?.message || 'Failed to launch live attendance session.');
      setActionLoading(false);
    }
  };

  // Enroll Student Handler
  const handleEnrollStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedStudentId.trim()) {
      setEnrollError('Please select or specify a student ID.');
      return;
    }

    setEnrollError(null);
    try {
      await api.enrollStudent(classId, selectedStudentId.trim());
      setIsEnrollModalOpen(false);
      setSelectedStudentId('');
      fetchClassDetails();
    } catch (err: any) {
      setEnrollError(err?.message || 'Failed to enroll student in class.');
    }
  };

  // Remove Student Handler
  const handleRemoveStudent = async (studentId: string, studentName: string) => {
    if (!window.confirm(`Are you sure you want to remove ${studentName} (${studentId}) from this class roster?`)) {
      return;
    }

    try {
      await api.removeStudent(classId, studentId);
      fetchClassDetails();
    } catch (err: any) {
      setError(err?.message || 'Failed to remove student from class.');
    }
  };

  if (isLoading && !classData) {
    return <LoadingSpinner message="Loading course roster and details..." size="lg" />;
  }

  if (!classData) {
    return (
      <div className="page-container">
        <EmptyState
          title="Class Not Found"
          description="The requested class record could not be found or has been deleted."
          actionLabel="Back to Classes"
          onAction={() => navigate('/classes')}
        />
      </div>
    );
  }

  // Determine which enrolled students exist
  const enrolledStudents = classData.students || [];
  const enrolledIds = new Set(enrolledStudents.map((s) => s.student_id));
  const availableStudents = allStudents.filter((s) => !enrolledIds.has(s.student_id));

  return (
    <div className="page-container">
      {/* Top Breadcrumb */}
      <div className="breadcrumb-nav">
        <button type="button" onClick={() => navigate('/classes')} className="breadcrumb-link">
          &larr; All Classes
        </button>
      </div>

      {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

      {/* Class Hero Banner */}
      <div className="class-hero-card card">
        <div className="hero-content-left">
          <div className="hero-subject-tag">{classData.subject}</div>
          <h1 className="hero-title">{classData.name}</h1>
          <div className="hero-meta-row">
            <span>Academic Year: <strong>{classData.academic_year}</strong></span>
            <span className="dot-sep">&bull;</span>
            <span>Semester: <strong>{classData.semester}</strong></span>
            <span className="dot-sep">&bull;</span>
            <span>Enrolled Roster: <strong>{enrolledStudents.length} Students</strong></span>
          </div>
        </div>

        <div className="hero-content-right">
          <button
            type="button"
            onClick={handleStartAttendance}
            disabled={actionLoading}
            className="btn btn-success btn-lg start-attendance-hero-btn"
          >
            {actionLoading ? (
              'Initializing Session...'
            ) : (
              <>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                <span>START ATTENDANCE</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Roster Section */}
      <div className="section-container">
        <div className="section-header">
          <div>
            <h2 className="section-title">Enrolled Student Roster</h2>
            <p className="section-subtitle">
              Students eligible for biometric face recognition during this class lecture
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              setEnrollError(null);
              setSelectedStudentId('');
              setIsEnrollModalOpen(true);
            }}
            className="btn btn-outline"
          >
            + Add Student to Class
          </button>
        </div>

        {enrolledStudents.length === 0 ? (
          <EmptyState
            title="No Students Enrolled in this Class"
            description="Add enrolled students so the AI camera system can verify their identity."
            actionLabel="Add Student Now"
            onAction={() => setIsEnrollModalOpen(true)}
          />
        ) : (
          <div className="table-responsive-wrapper card">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Roll / Student ID</th>
                  <th>Student Name</th>
                  <th>Department</th>
                  <th>Academic Year</th>
                  <th className="text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {enrolledStudents.map((s) => (
                  <tr key={s.id}>
                    <td className="font-mono font-medium">{s.student_id}</td>
                    <td className="font-semibold">{s.name}</td>
                    <td className="text-muted">{s.department || '—'}</td>
                    <td className="text-muted">{s.year || '—'}</td>
                    <td className="text-right">
                      <button
                        type="button"
                        onClick={() => handleRemoveStudent(s.student_id, s.name)}
                        className="btn btn-outline-danger btn-sm"
                        title="Remove student from this class"
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Enroll Student Modal */}
      <Modal
        isOpen={isEnrollModalOpen}
        onClose={() => setIsEnrollModalOpen(false)}
        title="Add Student to Class Roster"
      >
        {enrollError && <ErrorMessage message={enrollError} onDismiss={() => setEnrollError(null)} />}

        <form onSubmit={handleEnrollStudent} className="modal-form">
          <div className="form-group">
            <label htmlFor="studentSelect" className="form-label">
              Select Enrolled Student
            </label>
            {availableStudents.length > 0 ? (
              <select
                id="studentSelect"
                value={selectedStudentId}
                onChange={(e) => setSelectedStudentId(e.target.value)}
                className="form-input form-select"
                required
              >
                <option value="">-- Choose from registered students --</option>
                {availableStudents.map((st) => (
                  <option key={st.id} value={st.student_id}>
                    {st.student_id} — {st.name} ({st.department || 'General'})
                  </option>
                ))}
              </select>
            ) : (
              <p className="form-hint-text">
                All registered students are currently enrolled in this class. You can also manually enter a Student ID below:
              </p>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="manualStudentId" className="form-label">
              Or Type Student ID (Roll No)
            </label>
            <input
              id="manualStudentId"
              type="text"
              value={selectedStudentId}
              onChange={(e) => setSelectedStudentId(e.target.value)}
              placeholder="e.g. 23UAM090"
              className="form-input font-mono"
            />
          </div>

          <div className="modal-actions">
            <button
              type="button"
              onClick={() => setIsEnrollModalOpen(false)}
              className="btn btn-outline"
            >
              Cancel
            </button>
            <button type="submit" className="btn btn-primary">
              Enroll Student
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
};

export default ClassDetails;
