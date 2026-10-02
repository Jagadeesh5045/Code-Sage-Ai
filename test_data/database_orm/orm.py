"""Custom lightweight ORM for SQLite databases."""

import sqlite3
from contextlib import contextmanager

class Database:
    """SQLite database connection manager."""

    def __init__(self, db_path):
        self.db_path = db_path

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

class Model:
    """Base model class for ORM entities."""
    _table_name = ""
    _fields = {}

    def __init__(self, db, **kwargs):
        self._db = db
        self.id = kwargs.get("id")
        for field, default in self._fields.items():
            setattr(self, field, kwargs.get(field, default))

    def save(self):
        """Insert or update the record."""
        if self.id:
            self._update()
        else:
            self._insert()

    def _insert(self):
        fields = list(self._fields.keys())
        values = [getattr(self, f) for f in fields]
        placeholders = ", ".join("?" for _ in fields)
        sql = f"INSERT INTO {self._table_name} ({', '.join(fields)}) VALUES ({placeholders})"
        with self._db.connection() as conn:
            cursor = conn.execute(sql, values)
            self.id = cursor.lastrowid

    def _update(self):
        fields = list(self._fields.keys())
        values = [getattr(self, f) for f in fields]
        set_clause = ", ".join(f"{f} = ?" for f in fields)
        sql = f"UPDATE {self._table_name} SET {set_clause} WHERE id = ?"
        with self._db.connection() as conn:
            conn.execute(sql, values + [self.id])

    def delete(self):
        """Delete this record from the database."""
        sql = f"DELETE FROM {self._table_name} WHERE id = ?"
        with self._db.connection() as conn:
            conn.execute(sql, [self.id])

    @classmethod
    def find(cls, db, record_id):
        """Find a record by ID."""
        sql = f"SELECT * FROM {cls._table_name} WHERE id = ?"
        with db.connection() as conn:
            row = conn.execute(sql, [record_id]).fetchone()
            if row:
                return cls(db, **dict(row))
        return None

    @classmethod
    def all(cls, db):
        """Get all records."""
        sql = f"SELECT * FROM {cls._table_name}"
        with db.connection() as conn:
            rows = conn.execute(sql).fetchall()
            return [cls(db, **dict(r)) for r in rows]

    @classmethod
    def where(cls, db, **conditions):
        """Find records matching conditions."""
        clauses = " AND ".join(f"{k} = ?" for k in conditions)
        sql = f"SELECT * FROM {cls._table_name} WHERE {clauses}"
        with db.connection() as conn:
            rows = conn.execute(sql, list(conditions.values())).fetchall()
            return [cls(db, **dict(r)) for r in rows]

    @classmethod
    def create_table(cls, db):
        """Create the table for this model."""
        field_defs = ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
        type_map = {str: "TEXT", int: "INTEGER", float: "REAL", bool: "INTEGER"}
        for name, default in cls._fields.items():
            sql_type = type_map.get(type(default), "TEXT")
            field_defs.append(f"{name} {sql_type}")
        sql = f"CREATE TABLE IF NOT EXISTS {cls._table_name} ({', '.join(field_defs)})"
        with db.connection() as conn:
            conn.execute(sql)
