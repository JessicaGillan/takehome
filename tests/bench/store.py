"""Append-only JSONL checkpoint. A killed run resumes instead of repeating.

Writes are blocking rather than async: one short line per request is not worth
an async file layer, and the append keeps the record durable if a run is killed.
"""

import json
import os

from .results import RequestResult

class CheckpointStore:
    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.completed_ids = self._load_completed_ids()

    def _load_completed_ids(self) -> set[str]:
        if not os.path.exists(self._path):
            return set()
        done: set[str] = set()
        with open(self._path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["request_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a torn final line from a hard kill
        return done

    def append(self, result: RequestResult) -> None:
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(result.to_jsonl() + "\n")
            handle.flush()
        self.completed_ids.add(result.request_id)

    @property
    def path(self) -> str:
        return self._path
