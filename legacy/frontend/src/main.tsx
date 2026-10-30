import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import { AuthUser, clearSession } from './api';
import LoginPage from './pages/LoginPage';
import TeacherDashboard from './pages/TeacherDashboard';
import StudentPortal from './pages/StudentPortal';

function App() {
  // Always require login on fresh open (no auto-login)
  const [user, setUser] = useState<AuthUser | null>(null);

  const handleLogin = (u: AuthUser) => setUser(u);
  const handleLogout = () => {
    clearSession();
    setUser(null);
  };

  if (!user) return <LoginPage onLogin={handleLogin} />;
  if (user.role === 'student') return <StudentPortal user={user} onLogout={handleLogout} />;
  return <TeacherDashboard user={user} onLogout={handleLogout} />;
}

createRoot(document.getElementById('root')!).render(<App />);
