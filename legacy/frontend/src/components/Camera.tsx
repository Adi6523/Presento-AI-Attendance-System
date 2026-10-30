import { useRef, useState, useEffect, useCallback } from 'react';
import { enrollImages, webCheckin } from '../api';

const CHECKIN_FRAMES   = 8;
const CHECKIN_INTERVAL = 200;   // ms — 8 frames × 200 ms = 1.6 s capture
const ENROLL_FRAMES    = 15;
const ENROLL_INTERVAL  = 420;   // ms — 15 frames × 420 ms ≈ 6.3 s total

const ENROLL_GUIDES = [
  '👁️ Look straight ahead — hold still',
  '👁️ Stay centred — hold still',
  '↙️ Slowly turn head slightly to the LEFT',
  '↙️ Hold angle on the LEFT',
  '👁️ Return to centre',
  '↘️ Slowly turn head slightly to the RIGHT',
  '↘️ Hold angle on the RIGHT',
  '👁️ Return to centre',
  '⬇️ Tilt chin DOWN slightly',
  '👁️ Return to centre',
  '⬆️ Tilt chin UP slightly',
  '👁️ Return to centre',
  '↙️ Slight diagonal angle',
  '↘️ Slight diagonal angle',
  '✅ Final capture — looking straight',
];

interface CameraProps {
  mode: 'enroll' | 'checkin';
  studentId?: string;
  sessionId?: number;
  keepStreamOnDone?: boolean;
  onDone: (result: any) => void;
}

export default function Camera({ mode, studentId, sessionId, keepStreamOnDone, onDone }: CameraProps) {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [active,       setActive]       = useState(false);
  const [msg,          setMsg]          = useState('Camera is offline.');
  const [busy,         setBusy]         = useState(false);
  const [progress,     setProgress]     = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [totalFrames,  setTotalFrames]  = useState(0);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => { t.stop(); t.enabled = false; });
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
    setBusy(false);
    setProgress(0);
    setCurrentFrame(0);
    setTotalFrames(0);
    setMsg('Camera is offline.');
  }, []);

  useEffect(() => {
    const handleBeforeUnload = () => stopStream();
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      stopStream();
    };
  }, [stopStream]);

  const enable = async () => {
    stopStream();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(() => {});
      }
      setActive(true);
      setMsg(
        mode === 'enroll'
          ? '📷 Camera active. Position face in the reticle and click Start Enrollment.'
          : '📷 Camera active. Click Scan Face to recognize student for this lecture.'
      );
    } catch {
      setMsg('❌ Camera access denied. Allow camera permissions in browser settings.');
    }
  };

  const capture = async () => {
    if (!videoRef.current?.videoWidth) return setMsg('Enable camera first.');
    setBusy(true);

    const count    = mode === 'enroll' ? ENROLL_FRAMES : CHECKIN_FRAMES;
    const interval = mode === 'enroll' ? ENROLL_INTERVAL : CHECKIN_INTERVAL;
    setTotalFrames(count);

    const canvas = document.createElement('canvas');
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d')!;
    const images: string[] = [];

    for (let i = 0; i < count; i++) {
      setCurrentFrame(i + 1);
      setProgress(Math.round(((i + 1) / count) * 70));

      if (mode === 'enroll') {
        setMsg(ENROLL_GUIDES[i] ?? ENROLL_GUIDES[ENROLL_GUIDES.length - 1]);
      } else {
        setMsg(`🔍 Biometric scan in progress… frame ${i + 1} / ${count}`);
      }

      ctx.drawImage(videoRef.current, 0, 0);
      images.push(canvas.toDataURL('image/jpeg', 0.88));

      if (i < count - 1) await new Promise(r => setTimeout(r, interval));
    }

    setProgress(85);
    setMsg(mode === 'enroll' ? '⚙️ Computing facial embeddings…' : '⚙️ Matching face with enrolled class roster…');

    try {
      const result: any = mode === 'enroll'
        ? await enrollImages(studentId!, images)
        : await webCheckin(images, sessionId);
      setProgress(100);

      const shouldKeep = keepStreamOnDone ?? (mode === 'checkin');
      if (!shouldKeep) {
        stopStream();
      } else {
        setBusy(false);
        setProgress(0);
        setCurrentFrame(0);
        setTotalFrames(0);
      }

      const successMsg = result.message || (mode === 'enroll' ? '✅ Face enrolled successfully!' : '✅ Student recognized!');
      setMsg(successMsg);
      onDone(result);
    } catch (e: any) {
      setMsg(`❌ ${e.message}`);
      setProgress(0);
      setBusy(false);
    }
  };

  const isEnroll = mode === 'enroll';
  const msgColor = msg.startsWith('✅') ? '#34d6b5' : msg.startsWith('❌') ? '#f87171' : '#94a3b8';

  return (
    <div className="capture">
      {/* ── Video + scanner overlay ── */}
      <div className="capture-video-wrap" style={{ maxWidth: 480 }}>
        <video
          ref={videoRef}
          autoPlay playsInline muted
          style={{ width: '100%', display: 'block', background: '#050711', minHeight: 280 }}
        />

        {/* Reticle guide */}
        {active && (
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{
              width: 180, height: 230, borderRadius: '50%',
              border: `2px dashed ${busy ? '#34d6b5' : 'rgba(99,102,241,0.6)'}`,
              boxShadow: `0 0 0 4px ${busy ? 'rgba(52,214,181,0.15)' : 'rgba(99,102,241,0.12)'}`,
              transition: 'all 0.3s ease',
            }} />
          </div>
        )}

        {/* Check-in: animated scanline */}
        {!isEnroll && active && busy && (
          <div className="scan-line" />
        )}

        {/* Live telemetry badge */}
        {active && (
          <div style={{
            position: 'absolute', top: 12, left: 14,
            background: 'rgba(10, 14, 28, 0.75)',
            border: '1px solid rgba(255,255,255,0.1)',
            padding: '3px 10px', borderRadius: 20,
            fontSize: 11, fontWeight: 700, letterSpacing: '0.5px',
            color: busy ? '#34d6b5' : '#a5b4fc',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: busy ? '#34d6b5' : '#10b981',
              boxShadow: `0 0 8px ${busy ? '#34d6b5' : '#10b981'}`,
            }} />
            {busy ? 'SCANNING' : 'FEED READY'}
          </div>
        )}

        {/* Frame counter badge */}
        {busy && totalFrames > 0 && (
          <div style={{
            position: 'absolute', top: 12, right: 14,
            background: 'rgba(0, 0, 0, 0.75)',
            border: '1px solid rgba(255,255,255,0.1)',
            color: '#ffffff',
            fontSize: 12, fontWeight: 700, padding: '3px 10px', borderRadius: 20,
            fontFamily: 'JetBrains Mono',
          }}>
            {currentFrame} / {totalFrames}
          </div>
        )}
      </div>

      {/* ── Progress bar ── */}
      {busy && (
        <div style={{
          width: '100%', maxWidth: 480, background: 'rgba(255,255,255,0.08)',
          borderRadius: 8, height: 6, overflow: 'hidden',
        }}>
          <div style={{
            width: `${progress}%`, height: '100%',
            background: 'linear-gradient(90deg, #6366f1, #34d6b5)',
            borderRadius: 8, transition: 'width 0.3s ease',
          }} />
        </div>
      )}

      {/* ── Controls ── */}
      <div className="capture-controls">
        {!active ? (
          <button className="btn-primary" onClick={enable} disabled={busy}>
            📷 Enable Camera
          </button>
        ) : (
          <>
            <button className="btn-primary" onClick={capture} disabled={busy} id="btn-start-capture">
              {busy
                ? '⏳ Recognizing…'
                : isEnroll
                  ? '📸 Start Multi-Angle Enrollment (~6 s)'
                  : '🔍 Scan Student Face'}
            </button>
            <button className="btn-stop" onClick={stopStream} disabled={busy}>
              🔴 Stop Camera
            </button>
          </>
        )}
      </div>

      <p style={{ color: msgColor, fontSize: 13, minHeight: 20, fontWeight: 500 }}>
        {msg}
      </p>
    </div>
  );
}
