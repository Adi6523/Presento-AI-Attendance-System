import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading data...',
  size = 'md',
}) => {
  return (
    <div className={`loading-container size-${size}`} role="status" aria-live="polite">
      <div className="spinner-ring" />
      {message && <p className="loading-message">{message}</p>}
      <span className="sr-only">Loading</span>
    </div>
  );
};

export default LoadingSpinner;
