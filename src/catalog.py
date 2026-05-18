"""Relational product catalog backed by SQLite.

Schema:
    products(sku PRIMARY KEY, class_id UNIQUE, name, category, price, description)
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku         TEXT PRIMARY KEY,
    class_id    INTEGER NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    price       REAL NOT NULL,
    description TEXT
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
"""


@contextmanager
def get_conn(db_path=DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a sqlite3 connection with row factory set."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_product(
    sku: str,
    class_id: int,
    name: str,
    category: str,
    price: float,
    description: str = "",
    db_path=DB_PATH,
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO products (sku, class_id, name, category, price, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET
                class_id = excluded.class_id,
                name = excluded.name,
                category = excluded.category,
                price = excluded.price,
                description = excluded.description
            """,
            (sku, class_id, name, category, price, description),
        )


def get_by_class_id(class_id: int, db_path=DB_PATH) -> Optional[dict]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE class_id = ?", (class_id,)
        ).fetchone()
    return dict(row) if row else None


def get_by_sku(sku: str, db_path=DB_PATH) -> Optional[dict]:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE sku = ?", (sku,)
        ).fetchone()
    return dict(row) if row else None


def list_products(db_path=DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM products ORDER BY class_id"
        ).fetchall()
    return [dict(r) for r in rows]
