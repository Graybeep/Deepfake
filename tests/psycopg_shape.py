"""A psycopg3-shaped connection, so `Db`'s own method bodies can be executed.

Why this exists
---------------
`FakeDb` replaces `df.db.Db` wholesale, which means no test in this suite has
ever executed a single line of `Db`. That is exactly how `insert_items` shipped
calling `executemany` on a `Connection` -- a method psycopg3 only puts on
`Cursor` -- and dead-lettered every job against a real database while pytest
stayed green (CLAUDE.md, "Testing status").

`FakeDb` cannot close that gap: it stands in for the layer above the one that
broke. So this module stands in for the layer *below* it. Patch
`df.db.psycopg.connect` to hand back a `RecordingConnection` and the real `Db`
methods run their real bodies, against an object that raises `AttributeError`
for anything psycopg would not provide.

The permissive-fake problem, and the guard against it
-----------------------------------------------------
CLAUDE.md records three times the fake and the real query diverged, and every
time the fake was the *more permissive* of the two, so the fake-only check
stayed green. A hand-written list of "methods psycopg has" would be a fourth:
it encodes a belief about psycopg, and a belief is what state 3 provenance
looks like.

So the allowed surface is not written down here. It is read off the installed
`psycopg.Connection` / `psycopg.Cursor` at runtime by
`tests/test_db_api_shape.py`, which fails if this module exposes any public
name the real class does not. If psycopg ever moves `executemany` onto
`Connection`, the guard relaxes on its own -- correctly -- rather than
asserting a stale fact about a library that has moved on.

What this CANNOT see -- stated inline per CLAUDE.md
---------------------------------------------------
No SQL is executed. This proves the psycopg API *shape* every `Db` method
touches is real; it proves nothing about whether the statement is valid, whether
the columns exist, or whether a predicate selects the right rows. Only the live
probes do that: `scripts/verify_attribution.py`, `scripts/verify_retention.py`,
`scripts/verify_queue.py`, `scripts/smoke_compose.py`. Do not read a green run
here as "the DB layer works".
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


class RecordingCursor:
    """Stands in for psycopg.Cursor.

    Every public name here must also exist on the real `psycopg.Cursor`; the
    conformance test enforces that rather than trusting this comment.
    """

    def __init__(self, log: list[tuple[str, str, Any]], rows: list[list[dict]]) -> None:
        self._log = log
        self._rows = rows
        self._result: list[dict] = []

    # --- psycopg.Cursor surface ---
    def execute(self, sql: str, params: Any = None) -> "RecordingCursor":
        self._log.append(("execute", _norm(sql), params))
        self._result = self._rows.pop(0) if self._rows else []
        return self

    def executemany(self, sql: str, seq: Any) -> None:
        # Cursor-only in psycopg3. A Connection reaching this is the bug.
        self._log.append(("executemany", _norm(sql), list(seq)))

    def fetchone(self) -> dict | None:
        return self._result[0] if self._result else None

    def fetchall(self) -> list[dict]:
        return list(self._result)

    def close(self) -> None:
        pass

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class RecordingConnection:
    """Stands in for psycopg.Connection.

    Deliberately has NO `executemany` and NO `fetchone`. psycopg3 puts both on
    `Cursor` only, and `Db.insert_items` once assumed otherwise. `execute()`
    returns a cursor, which is what makes `c.execute(...).fetchone()` legal on a
    real connection while `c.executemany(...)` is not.
    """

    def __init__(self, rows: list[list[dict]] | None = None) -> None:
        self.calls: list[tuple[str, str, Any]] = []
        self._rows = rows if rows is not None else []

    # --- psycopg.Connection surface ---
    def execute(self, sql: str, params: Any = None) -> RecordingCursor:
        return RecordingCursor(self.calls, self._rows).execute(sql, params)

    def cursor(self) -> RecordingCursor:
        return RecordingCursor(self.calls, self._rows)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.calls.append(("transaction", "BEGIN", None))
        yield

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "RecordingConnection":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    # --- assertions for tests (underscore-prefixed: not part of the shape) ---
    def _statements(self) -> list[str]:
        return [sql for kind, sql, _ in self.calls if kind != "transaction"]

    def _kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.calls]


def _norm(sql: str) -> str:
    return " ".join(sql.split())
