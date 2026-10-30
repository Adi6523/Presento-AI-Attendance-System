import { useEffect } from 'react';

interface ModalProps {
  title: string;
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info' | 'confirm';
  confirmLabel?: string;
  cancelLabel?: string;
  onConfirm?: () => void;
  onClose: () => void;
}

const ICONS = {
  success: '✨',
  error:   '⚠️',
  warning: '🔔',
  info:    'ℹ️',
  confirm: '❓',
};

const COLORS = {
  success: { border: '#10b981', btn: 'linear-gradient(135deg, #10b981, #059669)', glow: 'rgba(16, 185, 129, 0.25)' },
  error:   { border: '#ef4444', btn: 'linear-gradient(135deg, #ef4444, #dc2626)', glow: 'rgba(239, 68, 68, 0.25)' },
  warning: { border: '#f59e0b', btn: 'linear-gradient(135deg, #f59e0b, #d97706)', glow: 'rgba(245, 158, 11, 0.25)' },
  info:    { border: '#6366f1', btn: 'linear-gradient(135deg, #6366f1, #4f46e5)', glow: 'rgba(99, 102, 241, 0.25)' },
  confirm: { border: '#6366f1', btn: 'linear-gradient(135deg, #6366f1, #4f46e5)', glow: 'rgba(99, 102, 241, 0.25)' },
};

export default function Modal({
  title, message, type = 'info',
  confirmLabel = 'OK', cancelLabel = 'Cancel',
  onConfirm, onClose,
}: ModalProps) {
  const c = COLORS[type];
  const isConfirm = type === 'confirm';

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card"
        style={{ borderTop: `3px solid ${c.border}`, boxShadow: `0 24px 70px rgba(0,0,0,0.6), 0 0 25px ${c.glow}` }}
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-icon">{ICONS[type]}</div>
        <h3 className="modal-title">{title}</h3>
        <p className="modal-message">{message}</p>
        <div className="modal-actions">
          {isConfirm && (
            <button className="modal-btn cancel" onClick={onClose}>{cancelLabel}</button>
          )}
          <button
            className="modal-btn confirm"
            style={{ background: c.btn }}
            onClick={() => { onConfirm?.(); onClose(); }}
            autoFocus
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
