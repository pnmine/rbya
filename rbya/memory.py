"""Procedure store: knowledge compilation of fixes that were verified to work."""
from __future__ import annotations

from .actions import Action


class ProcedureStore:
    def __init__(self):
        self._store: dict[str, Action] = {}
        self.hits = 0

    def lookup(self, key: str | None) -> Action | None:
        if key and key in self._store:
            self.hits += 1
            return self._store[key]
        return None

    def compile(self, key: str | None, action: Action) -> None:
        if key:
            self._store[key] = action

    def forget(self, key: str | None) -> None:
        self._store.pop(key, None)

    def __len__(self):
        return len(self._store)
