# Presento — AI Vision Engine & Backend Integration Contract (Phase 2)

## 1. System Architecture

```
Webcam / Raspberry Pi Camera (30 FPS)
            ↓
SCRFD Face Detection & 5-Landmark Alignment
            ↓
Frozen ArcFace (w600k_mbf.onnx) -> 512-D Normalized Embedding
            ↓
Presento Identity Engine (Cosine Similarity >= 0.48, Stability Gate)
            ↓
Asynchronous Dual Anti-Spoof Worker (MN3 + MiniFASNet V2 SE)
            ↓
Temporal Liveness Session State Machine (30-frame window, EMA, Lease)
            ↓
[REAL_LOCKED State Achieved]
            ↓
Asynchronous AttendanceIOWorker / AsyncBackendClient (Non-blocking queue)
            ↓  (Bearer Auth: PRESENTO_AI_API_KEY)
FastAPI Backend (POST /api/attendance/events)
            ↓
9-Step Attendance Validation Pipeline
            ↓
SQLite Database (data/presento.db) -> Attendance Record (PRESENT / DUPLICATE / BLOCKED)
```

---

## 2. Attendance Event JSON Schema

The edge client sends an HTTP `POST` request to `/api/attendance/events`:

```json
{
  "student_id": "23UAM090",
  "session_id": 1,
  "confidence": 0.91,
  "liveness_state": "REAL_LOCKED",
  "liveness_score": 0.95,
  "source": "AI_CAMERA",
  "timestamp": "2026-10-06T10:30:00Z"
}
```

### Field Definitions

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | `string` | **Yes** | Enrolled college student identifier (e.g., `"23UAM090"`). |
| `session_id` | `integer` | **Yes** | Database primary key of the target attendance session. |
| `confidence` | `float` | **Yes** | Cosine similarity from ArcFace matching ($\ge 0.48$). |
| `liveness_state` | `string` | **Yes** | Explicit state from `LivenessSession`. Must be `"REAL_LOCKED"`. |
| `source` | `string` | **Yes** | Hardware source: `"AI_CAMERA"`, `"LAPTOP_CAMERA"`, or `"MANUAL"`. |
| `liveness_score` | `float` | Optional | Raw fused anti-spoof probability ($[0.0, 1.0]$). Informational only. |
| `timestamp` | `string` | Optional | ISO-8601 UTC timestamp of observation. Defaults to current time. |

---

## 3. Liveness States Specification

The backend treats the AI engine as the **exclusive authority** for liveness decisions. The backend does **not** convert raw probability scores into "REAL".

| State | Status in Backend | Attendance Decision | Description |
| :--- | :--- | :--- | :--- |
| `REAL_LOCKED` | **Accepted** | **Admitted to validation** | Biometric liveness verified by multi-model agreement & temporal stability. |
| `HOLD` | **Blocked** | `BLOCKED` (HTTP 200, success=false) | Face temporarily untracked or brief drop; grace period active. |
| `SPOOF` | **Blocked** | `BLOCKED` (HTTP 200, success=false) | Presentation attack confirmed (photo, video replay, mask). |
| `CHECKING` | **Blocked** | `BLOCKED` (HTTP 200, success=false) | Evidence accumulator filling (needs $\ge 15$ frames / 80% consensus). |
| `NEW` | **Blocked** | `BLOCKED` (HTTP 200, success=false) | New face track initialized, no anti-spoof evidence collected yet. |
| `RESET` | **Blocked** | `BLOCKED` (HTTP 200, success=false) | Session reset due to prolonged absence ($>1.5\text{s}$) or contradictory frames. |

> **Crucial Rule**: `REAL_LOCKED` is the **only** liveness state eligible for attendance recording. All other states are blocked without modifying attendance tables.

---

## 4. 9-Step Attendance Validation Pipeline

When an event arrives at `POST /api/attendance/events`, the backend runs this strict ordered validation:

1. **Student Exists**: Does `student_id` exist in the database?
   - *Failure*: Status `REJECTED` ("Student ID not found").
2. **Student Active**: Is `student.is_active == True`?
   - *Failure*: Status `REJECTED` ("Student is marked inactive").
3. **Session Exists**: Does `session_id` exist?
   - *Failure*: Status `REJECTED` ("Session not found").
4. **Session ACTIVE**: Is `session.status == "ACTIVE"`?
   - *Failure*: Status `REJECTED` ("Session is in state 'COMPLETED'. Only ACTIVE sessions accept attendance.").
5. **Class Enrollment**: Is the student enrolled in `session.class_id`?
   - *Failure*: Status `REJECTED` ("Student is not enrolled in this session's class").
6. **Recognition Confidence**: Is `confidence >= settings.attendance_min_confidence` (default 0.48)?
   - *Failure*: Status `BLOCKED` ("Confidence below operating threshold").
7. **Liveness State Gate**: Is `liveness_state == "REAL_LOCKED"`?
   - *Failure*: Status `BLOCKED` ("Attendance blocked: liveness state is not REAL_LOCKED").
8. **Duplicate Protection**: Is there already an `Attendance` record for this student and session?
   - *Match*: Status `DUPLICATE` (success=true, existing record returned, no duplicate inserted).
9. **Record Creation**: Insert new row into `attendances` table with status `PRESENT`, device source, confidence, and timestamp.
   - *Success*: Status `PRESENT` (success=true).

---

## 5. Authentication & Security

- **AI Endpoint**: Protected by dedicated `PRESENTO_AI_API_KEY` via Bearer token:
  ```http
  Authorization: Bearer <PRESENTO_AI_API_KEY>
  ```
- **Architectural Separation**: The AI Vision engine key is completely decoupled from the teacher/admin JWT credentials used by the React web application.
- **Pi Camera Control Token**: Distinct from the backend API key; camera hardware session tokens remain isolated to stream negotiation.
- **Configuration**: Configurable via `.env` (`PRESENTO_AI_API_KEY=...`) and documented in `.env.example`.

---

## 6. Dynamic Active Session Discovery

The AI edge client never hardcodes session IDs.

1. Teacher starts an attendance session in the dashboard.
2. The session transitions to state `ACTIVE`.
3. AI client calls `GET /api/sessions?status=ACTIVE`.
4. AI client extracts the active `session_id` and targets all recognition events to that session.
5. If no active session exists, the camera loop continues displaying the HUD but suppresses attendance submissions.

---

## 7. Non-Blocking Async Network I/O & Offline Queue

1. **Decoupled Architecture**: `AttendanceIOWorker` and `AsyncBackendClient` run in a dedicated daemon thread. Calls to `submit_event()` take `<0.01 ms` and never delay camera capture or OpenCV rendering.
2. **Offline Resilience**: When FastAPI is offline or network fails:
   - Events are buffered in memory and appended to `data/offline-events.jsonl`.
   - The camera display continues running at full 30 FPS.
3. **Automatic Synchronization**:
   - `flush_queue()` re-attempts delivery when network is restored.
   - Synced events are removed from `data/offline-events.jsonl`.
   - Idempotency guarantees that previously recorded events receive `DUPLICATE` rather than creating phantom rows.

---

## 8. Response Statuses

### Successful Attendance (`PRESENT`)
```json
{
  "success": true,
  "status": "PRESENT",
  "message": "Attendance marked successfully",
  "attendance_id": 14,
  "student_id": "23UAM090",
  "student_name": "Aditya Patil",
  "session_id": 1,
  "marked_at": "2026-10-06T10:30:00Z"
}
```

### Duplicate Attempt (`DUPLICATE`)
```json
{
  "success": true,
  "status": "DUPLICATE",
  "message": "Student '23UAM090' attendance already recorded for this session",
  "attendance_id": 14,
  "student_id": "23UAM090",
  "student_name": "Aditya Patil",
  "session_id": 1,
  "marked_at": "2026-10-06T10:30:00Z"
}
```

### Liveness Blocked (`BLOCKED`)
```json
{
  "success": false,
  "status": "BLOCKED",
  "message": "Presentation attack detected (SPOOF state). Attendance blocked.",
  "student_id": "23UAM090",
  "student_name": "Aditya Patil"
}
```

### Student or Session Validation Failure (`REJECTED`)
```json
{
  "success": false,
  "status": "REJECTED",
  "message": "Session is in state 'COMPLETED'. Only ACTIVE sessions accept attendance.",
  "student_id": "23UAM090",
  "student_name": "Aditya Patil"
}
```

---

## 9. Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/api/attendance/events" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me-before-production-ai-key" \
  -d '{
    "student_id": "23UAM090",
    "session_id": 1,
    "confidence": 0.912,
    "liveness_state": "REAL_LOCKED",
    "liveness_score": 0.954,
    "source": "AI_CAMERA",
    "timestamp": "2026-10-06T10:30:00Z"
  }'
```
