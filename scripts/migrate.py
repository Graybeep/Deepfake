"""Apply SQL migrations in order. Deliberately not Alembic.

The 5-day budget doesn't have room for a migration framework's own setup, and
the schema is small enough that plain ordered .sql files are readable by anyone
reviewing the retention logic. Revisit if the schema starts churning.

Usage:  python scripts/migrate.py
"""
from __future__ import annotations

import pathlib
import sys

import psycopg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from df.config import settings  # noqa: E402

MIGRATIONS = pathlib.Path(__file__).resolve().parents[1] / "migrations"


def main() -> int:
    files = sorted(MIGRATIONS.glob("*.sql"))
    if not files:
        print("no migrations found")
        return 1

    with psycopg.connect(settings.pg_dsn, autocommit=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        applied = {
            r[0] for r in conn.execute("SELECT filename FROM schema_migrations").fetchall()
        }

        for path in files:
            if path.name in applied:
                print(f"skip  {path.name}")
                continue
            print(f"apply {path.name}")
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,)
                )

    print("migrations up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
