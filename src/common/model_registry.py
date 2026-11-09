
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def load_model_registry(path: str|Path|None) -> dict[str,Any]:
    if not path or not Path(path).exists(): return {}
    with open(path,encoding="utf-8") as f: data=json.load(f)
    return data.get("models",data)

def require_model_path(registry: dict[str,Any], key: str) -> str:
    item=registry.get(key)
    if not item: raise FileNotFoundError(f"Model key '{key}' is not configured.")
    value=item.get("path") if isinstance(item,dict) else item
    if not Path(value).exists(): raise FileNotFoundError(f"Model file not found: {value}")
    return str(value)
