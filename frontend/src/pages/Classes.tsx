import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import type { ClassItem, ClassCreate } from '../types';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import Modal from '../components/Modal';
import EmptyState from '../components/EmptyState';

export const Classes: React.FC = () => {
  const navigate = useNavigate();
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [formError, setFormError] = useState<string | null>(null);

  const [formData, setFormData] = useState<ClassCreate>({
    name: '',
    subject: '',
    academic_year: new Date().getFullYear().toString(),
    semester: 1,
  });

  const fetchClasses = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getClasses();
      setClasses(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load classes.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchClasses();
  }, [fetchClasses]);

  const handleOpenModal = () => {
    setFormData({
      name: '',
      subject: '',
      academic_year: new Date().getFullYear().toString(),
      semester: 1,
    });
    setFormError(null);
    setIsModalOpen(true);
  };

  const handleCreateClass = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name.trim() || !formData.subject.trim()) {
      setFormError('Please enter both Class Name and Subject.');
      return;
    }

    setIsSubmitting(true);
    setFormError(null);
    try {
      await api.createClass({
        name: formData.name.trim(),
        subject: formData.subject.trim(),
        academic_year: formData.academic_year.trim(),
        semester: Number(formData.semester),
      });
      setIsModalOpen(false);
      fetchClasses();
    } catch (err: any) {
      setFormError(err?.message || 'Failed to create class.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoading && classes.length === 0) {
    return <LoadingSpinner message="Loading course directory..." size="lg" />;
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title">Classes & Courses</h1>
          <p className="page-subtitle">Manage academic classes, student enrollments, and attendance sessions</p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            onClick={handleOpenModal}
            className="btn btn-primary"
          >
            + Create Class
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

      {classes.length === 0 ? (
        <EmptyState
          title="No Classes Created Yet"
          description="Create your first class to enroll students and begin taking attendance."
          actionLabel="Create Class Now"
          onAction={handleOpenModal}
        />
      ) : (
        <div className="classes-grid">
          {classes.map((c) => (
            <div key={c.id} className="class-card card" onClick={() => navigate(`/classes/${c.id}`)}>
              <div className="class-card-header">
                <div>
                  <h3 className="class-title">{c.name}</h3>
                  <span className="class-subject-pill">{c.subject}</span>
                </div>
                <span className="class-sem-badge">Sem {c.semester}</span>
              </div>

              <div className="class-card-meta">
                <div className="meta-item">
                  <span className="meta-label">Academic Year</span>
                  <span className="meta-value">{c.academic_year}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Enrolled Roster</span>
                  <span className="meta-value font-semibold">{c.enrolled_count} Students</span>
                </div>
              </div>

              <div className="class-card-footer">
                <button
                  type="button"
                  className="btn btn-primary btn-sm btn-block"
                  onClick={(e) => {
                    e.stopPropagation();
                    navigate(`/classes/${c.id}`);
                  }}
                >
                  Manage & Start Attendance &rarr;
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Class Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title="Create New Academic Class"
      >
        {formError && <ErrorMessage message={formError} onDismiss={() => setFormError(null)} />}

        <form onSubmit={handleCreateClass} className="modal-form">
          <div className="form-group">
            <label htmlFor="className" className="form-label">
              Class Name
            </label>
            <input
              id="className"
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g. Computer Science - Year 4"
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label htmlFor="classSubject" className="form-label">
              Subject Code / Title
            </label>
            <input
              id="classSubject"
              type="text"
              required
              value={formData.subject}
              onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
              placeholder="e.g. CS401 - Machine Learning"
              className="form-input"
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="classYear" className="form-label">
                Academic Year
              </label>
              <input
                id="classYear"
                type="text"
                required
                value={formData.academic_year}
                onChange={(e) => setFormData({ ...formData, academic_year: e.target.value })}
                placeholder="2026"
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="classSemester" className="form-label">
                Semester
              </label>
              <input
                id="classSemester"
                type="number"
                min={1}
                max={12}
                required
                value={formData.semester}
                onChange={(e) => setFormData({ ...formData, semester: parseInt(e.target.value, 10) || 1 })}
                className="form-input"
              />
            </div>
          </div>

          <div className="modal-actions">
            <button
              type="button"
              onClick={() => setIsModalOpen(false)}
              className="btn btn-outline"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Creating...' : 'Create Class'}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
};

export default Classes;
