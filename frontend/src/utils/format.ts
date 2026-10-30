/**
 * Formatting helpers for Presento academic attendance UI.
 */

export function formatDate(isoDate?: string | null): string {
  if (!isoDate) return '—';
  try {
    const d = new Date(isoDate);
    if (isNaN(d.getTime())) {
      // If it's already YYYY-MM-DD
      return isoDate;
    }
    return d.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return isoDate;
  }
}

export function formatTime(isoDate?: string | null): string {
  if (!isoDate) return '—';
  try {
    const d = new Date(isoDate);
    if (isNaN(d.getTime())) {
      // If it's HH:MM:SS
      return isoDate;
    }
    return d.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    });
  } catch {
    return isoDate;
  }
}

export function formatDateTime(isoDate?: string | null): string {
  if (!isoDate) return '—';
  try {
    const d = new Date(isoDate);
    if (isNaN(d.getTime())) return isoDate;
    return `${formatDate(isoDate)} ${formatTime(isoDate)}`;
  } catch {
    return isoDate;
  }
}

export function formatConfidence(score?: number | null): string {
  if (score === null || score === undefined) return '—';
  return `${Math.round(score * 100)}%`;
}

export function formatLiveness(state?: string | null, score?: number | null): string {
  if (!state && score === null) return '—';
  if (state === 'REAL_LOCKED') return 'Verified Real';
  if (state === 'SPOOF') return 'Spoof Blocked';
  if (state === 'CHECKING') return 'Verifying...';
  if (state === 'HOLD') return 'Holding';
  if (state === 'RESET' || state === 'NEW') return 'Analyzing';
  if (score !== null && score !== undefined) {
    return score >= 0.5 ? 'Verified Real' : 'Suspicious';
  }
  return state || '—';
}

export function formatSource(src?: string | null): string {
  if (!src) return '—';
  switch (src.toUpperCase()) {
    case 'AI_CAMERA':
      return 'AI Vision Camera';
    case 'LAPTOP_CAMERA':
      return 'Laptop Camera';
    case 'MANUAL':
      return 'Manual Entry';
    default:
      return src;
  }
}
