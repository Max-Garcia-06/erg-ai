import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy.engine import make_url
from erg_ai.db.session import reset_engine, get_engine


def test_sqlite_connect_args():
    reset_engine()
    with patch.dict("os.environ", {"DATABASE_URL": "sqlite:///test.db"}):
        engine = get_engine()
        assert engine.url.drivername == "sqlite"
    reset_engine()


def test_postgres_url_normalized():
    reset_engine()
    captured = {}

    def fake_create_engine(url, **kwargs):
        captured["url"] = url
        mock_engine = MagicMock()
        mock_engine.url = make_url(url)
        return mock_engine

    with patch.dict("os.environ", {"DATABASE_URL": "postgres://user:pass@host/db"}), \
         patch("erg_ai.db.session.create_engine", side_effect=fake_create_engine):
        engine = get_engine()
        assert str(engine.url).startswith("postgresql")
    reset_engine()
