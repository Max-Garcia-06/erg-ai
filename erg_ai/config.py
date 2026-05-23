"""Application configuration loader."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

_config: Optional[Dict[str, Any]] = None


def get_config(reload: bool = False) -> Dict[str, Any]:
    global _config
    if _config is None or reload:
        with open(_CONFIG_PATH, "r") as f:
            _config = yaml.safe_load(f) or {}
    return _config


def project_root() -> Path:
    return _ROOT
