"""Backward-compatible entry point — use `uvicorn erg_ai.main:app` instead."""

from erg_ai.main import app

if __name__ == "__main__":
    import uvicorn
    from erg_ai.config import get_config

    server_config = get_config().get("server", {})
    uvicorn.run(
        "erg_ai.main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=server_config.get("reload", False),
    )
