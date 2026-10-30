"""Optional managed-service adapters.

Set SUPABASE_URL/SUPABASE_KEY to mirror verified records to Supabase and
TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM_NUMBER to send approved alerts.
Credentials never belong in the React application or source control.
"""
import os

def supabase_enabled() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))

def twilio_enabled() -> bool:
    return bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN") and os.getenv("TWILIO_FROM_NUMBER"))
