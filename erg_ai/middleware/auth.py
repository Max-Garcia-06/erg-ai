"""Supabase JWT → FastAPI user_id dependency."""

import json
import os
import urllib.request
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwk, jwt

_security = HTTPBearer(auto_error=False)

_jwks_cache: Optional[Dict[str, Any]] = None


def _get_jwt_secret() -> str:
    secret = os.environ.get("SUPABASE_JWT_SECRET", "")
    if not secret:
        raise RuntimeError("SUPABASE_JWT_SECRET env var not set")
    return secret


def _get_supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("SUPABASE_URL env var not set")
    return url


def _fetch_jwks() -> Dict[str, Any]:
    global _jwks_cache
    if _jwks_cache is None:
        supabase_url = _get_supabase_url()
        with urllib.request.urlopen(f"{supabase_url}/auth/v1/.well-known/jwks.json") as resp:
            _jwks_cache = json.load(resp)
    return _jwks_cache


def _decode_access_token(token: str) -> Dict[str, Any]:
    """Verify Supabase access tokens (legacy HS256 or asymmetric ES256/RS256)."""
    header = jwt.get_unverified_header(token)
    alg = header.get("alg", "HS256")

    if alg == "HS256":
        return jwt.decode(
            token,
            _get_jwt_secret(),
            algorithms=["HS256"],
            audience="authenticated",
        )

    supabase_url = _get_supabase_url()
    kid = header.get("kid")
    key_data = next((k for k in _fetch_jwks().get("keys", []) if k.get("kid") == kid), None)
    if key_data is None:
        raise JWTError("Signing key not found")

    return jwt.decode(
        token,
        jwk.construct(key_data),
        algorithms=[alg],
        audience="authenticated",
        issuer=f"{supabase_url}/auth/v1",
    )


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security),
) -> str:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = _decode_access_token(credentials.credentials)
        user_id: Optional[str] = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return user_id
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
        payload = _decode_access_token(credentials.credentials)
        return payload.get("sub") or "local"
    except (JWTError, RuntimeError):
        return "local"
