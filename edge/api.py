from __future__ import annotations
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import requests
from .config import settings

logger = logging.getLogger("presento.edge")

# Resolve queue path relative to the project root (where .env lives),
# not the current working directory — safe under systemd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = _PROJECT_ROOT / "data" / "offline-events.jsonl"


def headers() -> dict[str, str]:
    return {
        "X-API-Token": settings.api_token,
        "Authorization": f"Bearer {settings.api_token}",
    }


def get_people() -> list[dict]:
    # Try new FastAPI students endpoint first, then legacy
    try:
        r = requests.get(f"{settings.api_url}/api/students", headers=headers(), timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    response = requests.get(f"{settings.api_url}/v1/students/embeddings", headers=headers(), timeout=10)
    response.raise_for_status()
    return response.json()


def get_people_for_class(class_id: str) -> list[dict]:
    try:
        r = requests.get(f"{settings.api_url}/api/classes/{class_id}/students", headers=headers(), timeout=5)
        if r.ok:
            return r.json()
    except Exception:
        pass
    response = requests.get(f"{settings.api_url}/v1/classes/{class_id}/students/embeddings", headers=headers(), timeout=10)
    response.raise_for_status()
    return response.json()


def get_active_session() -> dict | None:
    """Fetch the currently active session from new FastAPI backend, with legacy fallback."""
    # 1. Try new FastAPI endpoint (/api/sessions?status=ACTIVE)
    try:
        r = requests.get(f"{settings.api_url}/api/sessions?status=ACTIVE", headers=headers(), timeout=5)
        if r.ok:
            sessions = r.json()
            if isinstance(sessions, list) and len(sessions) > 0:
                return sessions[0]
    except requests.RequestException:
        pass

    # 2. Legacy fallback
    try:
        response = requests.get(
            f"{settings.api_url}/v1/sessions",
            headers=headers(),
            timeout=5,
        )
        if response.ok:
            sessions = response.json()
            now = datetime.now(timezone.utc).isoformat()
            for sess in sessions:
                if (sess.get("status") == "open" and
                    sess.get("room") == settings.room and
                    sess.get("starts_at", "") <= now <= sess.get("ends_at", "")):
                    return sess
    except requests.RequestException:
        logger.debug("Could not fetch active session — will send event without session_id")
    return None


def post_event(payload: dict) -> bool:
    """
    Sends attendance event to FastAPI backend (POST /api/attendance/events)
    with backward-compatible fallback to legacy /v1/attendance/events.
    """
    body = dict(payload)
    # Ensure liveness contract compliance for new FastAPI backend
    if "liveness_state" not in body:
        score = body.get("liveness_score", 0.0)
        body["liveness_state"] = "REAL_LOCKED" if score >= 0.5 else "CHECKING"
    if "source" not in body:
        body["source"] = "AI_CAMERA"

    try:
        # Try new FastAPI backend
        url = f"{settings.api_url}/api/attendance/events"
        response = requests.post(url, headers=headers(), json=body, timeout=10)
        if response.status_code == 404:
            # Fall back to legacy backend endpoint
            response = requests.post(f"{settings.api_url}/v1/attendance/events", headers=headers(), json=payload, timeout=10)

        response.raise_for_status()
        result = response.json()
        if result.get("status") == "DUPLICATE" or result.get("duplicate"):
            logger.info("Event %s was a duplicate — already recorded", payload.get("event_id", payload.get("student_id", "?")))
        elif result.get("status") == "BLOCKED":
            logger.warning("Event %s was blocked by backend: %s", payload.get("event_id", payload.get("student_id", "?")), result.get("message"))
        return True
    except requests.RequestException as exc:
        logger.warning("API unreachable (%s) — queuing event %s offline", exc, payload.get("event_id", payload.get("student_id", "?")))
        QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with QUEUE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return False


def flush_queue() -> int:
    """Flushes offline events from local disk to the backend."""
    if not QUEUE_PATH.exists():
        return 0
    pending = [json.loads(line) for line in QUEUE_PATH.read_text(encoding="utf-8").splitlines() if line]
    if not pending:
        return 0
    remaining, sent = [], 0
    for event in pending:
        body = dict(event)
        if "liveness_state" not in body:
            score = body.get("liveness_score", 0.0)
            body["liveness_state"] = "REAL_LOCKED" if score >= 0.5 else "CHECKING"
        if "source" not in body:
            body["source"] = "AI_CAMERA"

        try:
            r = requests.post(f"{settings.api_url}/api/attendance/events", headers=headers(), json=body, timeout=10)
            if r.status_code == 404:
                r = requests.post(f"{settings.api_url}/v1/attendance/events", headers=headers(), json=event, timeout=10)
            if r.ok:
                sent += 1
                logger.info("Synced offline event %s", event.get("event_id", event.get("student_id", "?")))
            else:
                remaining.append(event)
        except requests.RequestException:
            remaining.append(event)
    if remaining:
        QUEUE_PATH.write_text("".join(json.dumps(e) + "\n" for e in remaining), encoding="utf-8")
    else:
        QUEUE_PATH.unlink(missing_ok=True)
    if sent:
        logger.info("Flushed %d offline events (%d remaining)", sent, len(remaining))
    return sent
