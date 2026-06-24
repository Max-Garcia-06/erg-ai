import pytest
from unittest.mock import patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from jose import jwt

SECRET = "test-secret-at-least-32-chars-long!!"


def _make_token(secret: str = SECRET, sub: str = "user-123", aud: str = "authenticated") -> str:
    return jwt.encode({"sub": sub, "aud": aud}, secret, algorithm="HS256")


def test_get_current_user_id_valid():
    from erg_ai.middleware.auth import get_current_user_id
    token = _make_token()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with patch("erg_ai.middleware.auth._get_jwt_secret", return_value=SECRET):
        user_id = get_current_user_id(creds)
    assert user_id == "user-123"


def test_get_current_user_id_no_token():
    from erg_ai.middleware.auth import get_current_user_id
    with pytest.raises(HTTPException) as exc:
        get_current_user_id(None)
    assert exc.value.status_code == 401


def test_get_current_user_id_bad_token():
    from erg_ai.middleware.auth import get_current_user_id
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="not.a.token")
    with patch("erg_ai.middleware.auth._get_jwt_secret", return_value=SECRET):
        with pytest.raises(HTTPException) as exc:
            get_current_user_id(creds)
    assert exc.value.status_code == 401


def test_get_optional_user_id_no_token():
    from erg_ai.middleware.auth import get_optional_user_id
    result = get_optional_user_id(None)
    assert result == "local"


def test_get_optional_user_id_valid():
    from erg_ai.middleware.auth import get_optional_user_id
    token = _make_token()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    with patch("erg_ai.middleware.auth._get_jwt_secret", return_value=SECRET):
        result = get_optional_user_id(creds)
    assert result == "user-123"
