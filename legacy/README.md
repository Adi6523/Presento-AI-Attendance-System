# LEGACY SYSTEM — REFERENCE ONLY

> [!WARNING]
> This folder contains the legacy backend implementation for Presento (reference only).
> The NEW Presento backend is implemented under `backend/` using SQLAlchemy 2.x, SQLite, and modular services.
> 
> DO NOT USE THIS LEGACY BACKEND OR ITS DATABASE AS THE SOURCE OF TRUTH.

## Legacy Components Preserved Here:
- `legacy/backend/main.py`: Old monolithic FastAPI routes (with old attendance logic and direct vision bindings).
- `legacy/backend/database.py`: Old raw SQLite3 script.
- `legacy/backend/auth.py`: Old authentication helpers.
- `legacy/backend/integrations.py`: Old third-party hooks.
- `legacy/backend/seed_demo.py`: Old demo seeding script.
- `legacy/backend/web_vision.py`: Old browser-based vision endpoints.

Please refer to `backend/` for the current, active production backend.
