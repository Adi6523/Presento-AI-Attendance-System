import React from 'react';
import { useAuth } from '../hooks/useAuth';

interface NavbarProps {
  onToggleSidebar?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onToggleSidebar }) => {
  const { user, logout } = useAuth();

  const displayName = user?.teacher?.name || user?.username || 'Teacher';

  return (
    <header className="top-navbar" role="banner">
      <div className="navbar-left">
        {onToggleSidebar && (
          <button
            type="button"
            className="navbar-toggle-btn"
            onClick={onToggleSidebar}
            aria-label="Toggle navigation menu"
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="3" y1="12" x2="21" y2="12" />
              <line x1="3" y1="6" x2="21" y2="6" />
              <line x1="3" y1="18" x2="21" y2="18" />
            </svg>
          </button>
        )}
        <div className="brand-lockup">
          <span className="brand-logo-mark">P</span>
          <div className="brand-text">
            <span className="brand-title">Presento</span>
            <span className="brand-subtitle">Smart Attendance</span>
          </div>
        </div>
      </div>

      <div className="navbar-right">
        <div className="user-profile-badge">
          <div className="user-avatar" aria-hidden="true">
            {displayName.charAt(0).toUpperCase()}
          </div>
          <div className="user-details">
            <span className="user-name">{displayName}</span>
            <span className="user-role">{user?.role || 'Teacher'}</span>
          </div>
        </div>

        <button
          type="button"
          onClick={() => logout()}
          className="btn btn-outline btn-logout"
          title="Sign out of Presento"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0 0 13.5 3h-6a2.25 2.25 0 0 0-2.25 2.25v13.5A2.25 2.25 0 0 0 7.5 21h6a2.25 2.25 0 0 0 2.25-2.25V15M12 9l-3 3m0 0 3 3m-3-3h12.75" />
          </svg>
          <span>Logout</span>
        </button>
      </div>
    </header>
  );
};

export default Navbar;
