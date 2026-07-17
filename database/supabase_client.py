import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

url = os.getenv("SUPABASE_URL", "").strip()
key = os.getenv("SUPABASE_KEY", "").strip()

supabase = create_client(url, key) if url and key else None


def create_authenticated_client(access_token: str) -> Client:
    """Supabase client scoped to the signed-in user (respects RLS)."""
    if not url or not key:
        raise RuntimeError("Supabase is not configured (missing SUPABASE_URL or SUPABASE_KEY)")
    client = create_client(url, key)
    client.postgrest.auth(access_token)
    return client