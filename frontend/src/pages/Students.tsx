import React, { useState, useEffect, useCallback, useMemo } from 'react';
import api from '../services/api';
import type { Student, StudentCreate, StudentUpdate } from '../types';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import Modal from '../components/Modal';
import EmptyState from '../components/EmptyState';

export const Students: React.FC = () => {
  const [students, setStudents] = useState<Student[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Modal State for Create / Edit
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [editingStudent, setEditingStudent] = useState<Student | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [modalError, setModalError] = useState<string | null>(null);

  const [formData, setFormData] = useState<StudentCreate>({
    student_id: '',
    name: '',
    email: '',
    department: '',
    year: 'Year 4',
  });

  const fetchStudents = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getStudents();
      setStudents(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load students.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStudents();
  }, [fetchStudents]);

  // Open Create Modal
  const handleOpenCreateModal = () => {
    setEditingStudent(null);
    setFormData({
      student_id: '',
      name: '',
      email: '',
      department: 'Computer Science',
      year: 'Year 4',
    });
    setModalError(null);
    setIsModalOpen(true);
  };

  // Open Edit Modal
  const handleOpenEditModal = (student: Student) => {
    setEditingStudent(student);
    setFormData({
      student_id: student.student_id,
      name: student.name,
      email: student.email || '',
      department: student.department || '',
      year: student.year || '',
    });
    setModalError(null);
    setIsModalOpen(true);
  };

  // Submit Create or Edit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.name.trim()) {
      setModalError('Student Name is required.');
      return;
    }

    setIsSubmitting(true);
    setModalError(null);

    try {
      if (editingStudent) {
        // Update existing
        const updatePayload: StudentUpdate = {
          name: formData.name.trim(),
          email: formData.email?.trim() || undefined,
          department: formData.department?.trim() || undefined,
          year: formData.year?.trim() || undefined,
        };
        await api.updateStudent(editingStudent.id, updatePayload);
      } else {
        // Create new
        if (!formData.student_id.trim()) {
          setModalError('Student ID / Roll No is required.');
          setIsSubmitting(false);
          return;
        }
        await api.createStudent({
          student_id: formData.student_id.trim(),
          name: formData.name.trim(),
          email: formData.email?.trim() || undefined,
          department: formData.department?.trim() || undefined,
          year: formData.year?.trim() || undefined,
        });
      }
      setIsModalOpen(false);
      fetchStudents();
    } catch (err: any) {
      setModalError(err?.message || 'Failed to save student record.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Delete Student
  const handleDelete = async (student: Student) => {
    if (!window.confirm(`Are you sure you want to deactivate student ${student.name} (${student.student_id})?`)) {
      return;
    }

    try {
      await api.deleteStudent(student.id);
      fetchStudents();
    } catch (err: any) {
      setError(err?.message || 'Failed to delete student.');
    }
  };

  // Filter students by query
  const filteredStudents = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return students;
    return students.filter(
      (s) =>
        s.student_id.toLowerCase().includes(q) ||
        s.name.toLowerCase().includes(q) ||
        (s.department && s.department.toLowerCase().includes(q))
    );
  }, [students, searchQuery]);

  if (isLoading && students.length === 0) {
    return <LoadingSpinner message="Loading student roster directory..." size="lg" />;
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title">Registered Students</h1>
          <p className="page-subtitle">Directory of students enrolled in the Presento biometric attendance database</p>
        </div>
        <div className="header-actions">
          <button
            type="button"
            onClick={handleOpenCreateModal}
            className="btn btn-primary"
          >
            + Register Student
          </button>
        </div>
      </div>

      {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

      {/* Search and Filter Bar */}
      <div className="filter-card card">
        <div className="search-input-wrapper">
          <span className="search-icon">&sub;</span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Student ID (e.g. 23UAM090), Name, or Department..."
            className="search-input"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              className="search-clear-btn"
            >
              Clear
            </button>
          )}
        </div>
        <div className="filter-count">
          Showing <strong>{filteredStudents.length}</strong> of {students.length} students
        </div>
      </div>

      {filteredStudents.length === 0 ? (
        <EmptyState
          title={searchQuery ? 'No Matching Students Found' : 'No Students Registered Yet'}
          description={
            searchQuery
              ? `No student matching "${searchQuery}". Try searching with a different roll number or name.`
              : 'Add your first student to the database to begin enrolling them in classes.'
          }
          actionLabel={searchQuery ? 'Clear Search' : 'Register Student Now'}
          onAction={searchQuery ? () => setSearchQuery('') : handleOpenCreateModal}
        />
      ) : (
        <div className="table-responsive-wrapper card">
          <table className="data-table">
            <thead>
              <tr>
                <th>Student ID</th>
                <th>Full Name</th>
                <th>Department</th>
                <th>Academic Year</th>
                <th>Status</th>
                <th className="text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredStudents.map((st) => (
                <tr key={st.id} className={!st.is_active ? 'row-inactive' : ''}>
                  <td className="font-mono font-medium">{st.student_id}</td>
                  <td>
                    <div className="student-name-block">
                      <span className="font-semibold">{st.name}</span>
                      {st.email && <span className="student-email-sub">{st.email}</span>}
                    </div>
                  </td>
                  <td className="text-muted">{st.department || '—'}</td>
                  <td className="text-muted">{st.year || '—'}</td>
                  <td>
                    <span className={`status-pill ${st.is_active ? 'active-pill' : 'inactive-pill'}`}>
                      {st.is_active ? 'Active' : 'Inactive'}
                    </span>
                  </td>
                  <td className="text-right">
                    <div className="action-buttons-group">
                      <button
                        type="button"
                        onClick={() => handleOpenEditModal(st)}
                        className="btn btn-outline btn-sm"
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDelete(st)}
                        className="btn btn-outline-danger btn-sm"
                        title="Deactivate student"
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Add / Edit Student Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        title={editingStudent ? `Edit Student: ${editingStudent.student_id}` : 'Register New Student'}
      >
        {modalError && <ErrorMessage message={modalError} onDismiss={() => setModalError(null)} />}

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="form-group">
            <label htmlFor="studentIdInput" className="form-label">
              Student ID / Roll Number
            </label>
            <input
              id="studentIdInput"
              type="text"
              required
              disabled={!!editingStudent}
              value={formData.student_id}
              onChange={(e) => setFormData({ ...formData, student_id: e.target.value })}
              placeholder="e.g. 23UAM090"
              className="form-input font-mono"
            />
            {editingStudent && <span className="form-hint-text">Student ID cannot be modified after registration.</span>}
          </div>

          <div className="form-group">
            <label htmlFor="studentNameInput" className="form-label">
              Full Legal Name
            </label>
            <input
              id="studentNameInput"
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g. Aditya Patil"
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label htmlFor="studentEmailInput" className="form-label">
              Email Address (Optional)
            </label>
            <input
              id="studentEmailInput"
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              placeholder="student@college.edu"
              className="form-input"
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="studentDeptInput" className="form-label">
                Department
              </label>
              <input
                id="studentDeptInput"
                type="text"
                value={formData.department}
                onChange={(e) => setFormData({ ...formData, department: e.target.value })}
                placeholder="Computer Science"
                className="form-input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="studentYearInput" className="form-label">
                Year / Batch
              </label>
              <input
                id="studentYearInput"
                type="text"
                value={formData.year}
                onChange={(e) => setFormData({ ...formData, year: e.target.value })}
                placeholder="Year 4"
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
              {isSubmitting ? 'Saving...' : editingStudent ? 'Update Student' : 'Register Student'}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
};

export default Students;
