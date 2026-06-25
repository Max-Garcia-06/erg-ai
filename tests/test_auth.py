import io
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import urllib.error

USER_ID = "11111111-2222-3333-4444-555555555555"


def _mock_urlopen(response_data: dict, status: int = 200):
    body = json.dumps(response_data).encode()

    if status != 200:
        err = urllib.error.HTTPError(
            url="https://example.supabase.co/auth/v1/user",
            code=status,
            msg="error",
            hdrs=None,
            fp=io.BytesIO(body),
        )
        return MagicMock(side_effect=err)

    bio = io.BytesIO(body)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=bio)
    cm.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=cm)


def test_get_current_user_id_valid():
    from erg_ai.middleware.auth import get_current_user_id

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")
    with patch("erg_ai.middleware.auth._supabase_auth_config", return_value=("https://x.supabase.co", "anon-key")):
        with patch("urllib.request.urlopen", _mock_urlopen({"id": USER_ID})):
            user_id = get_current_user_id(creds)
    assert user_id == USER_ID


def test_get_current_user_id_no_token():
    from erg_ai.middleware.auth import get_current_user_id

    with pytest.raises(HTTPException) as exc:
        get_current_user_id(None)
    assert exc.value.status_code == 401


def test_get_current_user_id_bad_token():
    from erg_ai.middleware.auth import get_current_user_id

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-token")
    with patch("erg_ai.middleware.auth._supabase_auth_config", return_value=("https://x.supabase.co", "anon-key")):
        with patch("urllib.request.urlopen", _mock_urlopen({}, status=401)):
            with pytest.raises(HTTPException) as exc:
                get_current_user_id(creds)
    assert exc.value.status_code == 401


def test_get_optional_user_id_no_token():
    from erg_ai.middleware.auth import get_optional_user_id

    result = get_optional_user_id(None)
    assert result == "local"


def test_get_optional_user_id_valid():
    from erg_ai.middleware.auth import get_optional_user_id

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")
    with patch("erg_ai.middleware.auth._supabase_auth_config", return_value=("https://x.supabase.co", "anon-key")):
        with patch("urllib.request.urlopen", _mock_urlopen({"id": USER_ID})):
            result = get_optional_user_id(creds)
    assert result == USER_ID
