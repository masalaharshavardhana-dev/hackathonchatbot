from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database.supabase_client import supabase, create_authenticated_client
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()


def _extract_buyer_name(user: object, buyer_profile: dict) -> str | None:
    user_metadata = getattr(user, "user_metadata", None) or {}
    if isinstance(user_metadata, dict):
        for field in ("full_name", "name", "display_name"):
            value = user_metadata.get(field)
            if value:
                return value

    for field in ("full_name", "name", "business_name"):
        value = buyer_profile.get(field)
        if value:
            return value

    return None

def get_current_buyer(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify the JWT token and return the buyer_id and profile from logged-in user.
    Requires valid authentication token.
    """
    token = credentials.credentials

    try:
        if supabase is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Supabase is not configured (missing SUPABASE_URL or SUPABASE_KEY)",
            )

        user_response = supabase.auth.get_user(token)

        if not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = user_response.user.id

        # buyers.id is the auth.users id (no separate auth_id column)
        authed = create_authenticated_client(token)
        buyer_response = (
            authed.table("buyers").select("*").eq("id", user_id).maybe_single().execute()
        )

        if not buyer_response or not buyer_response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Buyer profile not found. Sign in with a buyer account.",
            )

        return {
            "buyer_id": buyer_response.data["id"],
            "buyer_name": _extract_buyer_name(user_response.user, buyer_response.data),
            "profile": buyer_response.data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Authentication failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
