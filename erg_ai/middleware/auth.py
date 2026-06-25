"""Supabase JWT → FastAPI user_id dependency."""

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

_security = HTTPBearer(auto_error=False)


def _supabase_auth_config() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    api_key = os.environ.get("SUPABASE_ANON_KEY", "")
    if not url or not api_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY env vars must be set")
    return url, api_key


def _get_user_id_from_token(token: str) -> str:
    """Validate the access token with Supabase Auth (works for HS256 and ES256)."""
    supabase_url, api_key = _supabase_auth_config()
    request = urllib.request.Request(
        f"{supabase_url}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": api_key,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        raise JWTError(f"Supabase rejected token ({exc.code})") from exc

    user_id = data.get("id")
    if not user_id:
        raise JWTError("Supabase user response missing id")
    return str(user_id)


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        return _get_user_id_from_token(credentials.credentials)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> str:
    """Returns user_id from JWT if present, 'local' otherwise (backward-compat)."""
    if credentials is None:
        return "local"
    try:
        return _get_user_id_from_token(credentials.credentials)
    except (JWTError, RuntimeError):
        return "local"
