import React from 'react';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const norm = (status || '').toUpperCase();

  let badgeClass = 'badge-default';
  let label = norm;

  switch (norm) {
    case 'PRESENT':
      badgeClass = 'badge-present';
      label = 'Present';
      break;
    case 'ABSENT':
      badgeClass = 'badge-absent';
      label = 'Absent';
      break;
    case 'BLOCKED':
      badgeClass = 'badge-blocked';
      label = 'Blocked';
      break;
    case 'REJECTED':
      badgeClass = 'badge-rejected';
      label = 'Rejected';
      break;
    case 'ACTIVE':
      badgeClass = 'badge-active';
      label = 'Active';
      break;
    case 'COMPLETED':
      badgeClass = 'badge-completed';
      label = 'Completed';
      break;
    case 'SCHEDULED':
      badgeClass = 'badge-scheduled';
      label = 'Scheduled';
      break;
    case 'CANCELLED':
      badgeClass = 'badge-cancelled';
      label = 'Cancelled';
      break;
    default:
      badgeClass = 'badge-default';
      label = norm;
  }

  return (
    <span className={`status-badge ${badgeClass} ${className}`}>
      <span className="badge-dot" />
      {label}
    </span>
  );
};

export default StatusBadge;
