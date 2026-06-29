import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_FILE = BASE_DIR / "mock_documents.json"
EVENTS_FILE = BASE_DIR / "mock_events.json"


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path.name}")
    return data


def load_documents() -> List[Dict[str, Any]]:
    return _load_json(DOCUMENTS_FILE)


def load_events() -> List[Dict[str, Any]]:
    return _load_json(EVENTS_FILE)


def dataset_hash(items: Iterable[Dict[str, Any]]) -> str:
    raw = json.dumps(list(items), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
