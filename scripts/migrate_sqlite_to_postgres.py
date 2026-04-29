"""
One-time migration tool: copy Geo-Watch data from local SQLite to PostgreSQL.

Usage:
  python scripts/migrate_sqlite_to_postgres.py --postgres-url "postgresql+psycopg://USER:PASS@HOST:5432/geowatch"

Optional:
  python scripts/migrate_sqlite_to_postgres.py --sqlite-path data/geowatch.db --postgres-url "..." --truncate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import MetaData, create_engine, select, text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate SQLite data to PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        default="data/geowatch.db",
        help="Path to SQLite file (default: data/geowatch.db)",
    )
    parser.add_argument(
        "--postgres-url",
        required=True,
        help="Target PostgreSQL SQLAlchemy URL (postgresql+psycopg://...)",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate target tables before importing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")

    sqlite_url = f"sqlite:///{sqlite_path.resolve()}"
    pg_url = args.postgres_url

    source_engine = create_engine(sqlite_url)
    target_engine = create_engine(pg_url, pool_pre_ping=True)

    # Import model metadata so target schema can be created from ORM definitions.
    from backend.database import Base
    import backend.alerts  # noqa: F401 - ensures monitoring_alerts model is registered

    Base.metadata.create_all(bind=target_engine)

    source_meta = MetaData()
    source_meta.reflect(bind=source_engine)

    target_meta = MetaData()
    target_meta.reflect(bind=target_engine)

    source_tables = [
        name
        for name in source_meta.tables.keys()
        if name in target_meta.tables and not name.startswith("sqlite_")
    ]

    if not source_tables:
        print("No matching tables found to migrate.")
        return 0

    with source_engine.connect() as src_conn, target_engine.begin() as tgt_conn:
        if args.truncate:
            for table_name in reversed(source_tables):
                tgt_conn.execute(text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE'))
            print("Truncated target tables.")

        for table_name in source_tables:
            src_table = source_meta.tables[table_name]
            tgt_table = target_meta.tables[table_name]

            rows = [dict(row._mapping) for row in src_conn.execute(select(src_table))]
            if not rows:
                print(f"{table_name}: 0 rows (skipped)")
                continue

            tgt_conn.execute(tgt_table.insert(), rows)
            print(f"{table_name}: migrated {len(rows)} rows")

    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
