# utils1.py
import os
import sqlite3
import pandas as pd
from datetime import datetime

# ======================================================
# 📌 Audit DB Setup
# ======================================================
AUDIT_DB = ".data/audit.sqlite"
os.makedirs(".data", exist_ok=True)

# ======================================================
# 📝 Audit Logging
# ======================================================
def init_audit():
    """Initialize the audit DB and table."""
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS query_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        user TEXT,
        role TEXT,
        user_query TEXT,
        sql TEXT,
        status TEXT,
        rows INTEGER,
        duration REAL
    )
    """)
    conn.commit()
    conn.close()


def log_query(user: str, role: str, user_q: str, sql: str, status: str, rows: int, duration: float):
    """Insert a query log record into the audit DB."""
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    cur = conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO query_audit (ts, user, role, user_query, sql, status, rows, duration) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, user, role, user_q, sql, status, rows, duration)
    )
    conn.commit()
    conn.close()


def fetch_audit(limit: int = 200):
    """Fetch recent audit logs as a pandas DataFrame."""
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    df = pd.read_sql_query(f"SELECT * FROM query_audit ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df


def clear_audit():
    """Clear audit table (destructive — for demo/admin only)."""
    conn = sqlite3.connect(AUDIT_DB, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("DELETE FROM query_audit")
    conn.commit()
    conn.close()


def audit_to_csv_bytes(limit: int = 10000) -> bytes:
    """Return audit log CSV bytes for download."""
    df = fetch_audit(limit=limit)
    return df.to_csv(index=False).encode("utf-8")


# ======================================================
# 🏗️ Schema Formatter
# ======================================================
class SchemaFormatter:
    """
    Pretty printer for the existing schema dict from db.get_schema(engine).
    Does not change how schema is built; only formats it nicely for display/LLM.
    """

    def __init__(self, schema: dict):
        self.schema = schema or {}

    @staticmethod
    def _fmt_int(n):
        try:
            return f"{int(n):,}"
        except Exception:
            return str(n) if n is not None else "unknown"

    def _build_fk_lookup(self, fk_defs):
        """
        Convert inspector FK list into {column: [(ref_table, [ref_cols])...]}.
        """
        fk_map = {}
        for fk in fk_defs or []:
            constrained = fk.get("constrained_columns") or []
            ref_table = fk.get("referred_table")
            ref_cols = fk.get("referred_columns") or []
            for c in constrained:
                fk_map.setdefault(c, []).append((ref_table, ref_cols))
        return fk_map

    def _format_column_line(self, col: dict, pk_cols: set, fk_lookup: dict) -> str:
        """
        Produce lines like:
          - TrackId (INTEGER, PK, AUTOINCREMENT, NOT NULL)
          - AlbumId (INTEGER, NULL, FK→ Album(AlbumId))
          - UnitPrice (NUMERIC(10,2), NOT NULL, default=0.99)
        """
        name = col.get("name", "UNKNOWN")
        ctype = col.get("type")
        ctype_str = str(ctype) if ctype is not None else "UNKNOWN"

        # inspector provides 'nullable' on many dialects; default to True when missing
        nullable = bool(col.get("nullable", True))

        # prefer 'default' or 'server_default' if present
        default = col.get("default", None)
        if default is None:
            default = col.get("server_default", None)

        is_pk = name in pk_cols
        autoinc = bool(col.get("autoincrement", False))

        parts = [ctype_str]
        if is_pk:
            parts.append("PK")
        if autoinc:
            parts.append("AUTOINCREMENT")
        parts.append("NOT NULL" if not nullable else "NULL")
        if default not in (None, "NULL"):
            parts.append(f"default={default}")

        # FK annotation (first target for brevity)
        fk_str = None
        if name in fk_lookup:
            rt, rc = fk_lookup[name][0]
            if rt:
                fk_str = f"FK→ {rt}({', '.join(rc)})" if rc else f"FK→ {rt}"
        if fk_str:
            parts.append(fk_str)

        # Two leading spaces before hyphen to match your sample
        return f"  - {name} ({', '.join(parts)})"

    def formatted(self) -> str:
        """
        Create a human-friendly multiline schema string for all tables.
        """
        lines = []
        for table in sorted(self.schema.keys(), key=str.lower):
            meta = self.schema.get(table) or {}
            cols = meta.get("columns", [])
            fk_defs = meta.get("foreign_keys", [])
            pk_cols = set(meta.get("primary_key") or [])
            row_count = meta.get("row_count", None)

            fk_lookup = self._build_fk_lookup(fk_defs)

            # Header with row count (if available)
            suffix = f" (~{self._fmt_int(row_count)} rows)" if row_count is not None else ""
            lines.append(f"📂 Table `{table}`{suffix}")
            lines.append("Columns")

            # Columns
            for c in cols:
                lines.append(self._format_column_line(c, pk_cols, fk_lookup))

            # Summary blocks
            if pk_cols:
                lines.append(f"🔑 Primary Key: {sorted(pk_cols)}")
            if fk_defs:
                for fk in fk_defs:
                    lines.append(
                        f"🔗 FK {fk.get('constrained_columns')} → "
                        f"{fk.get('referred_table')}({fk.get('referred_columns')})"
                    )
            lines.append("")  # spacer between tables
        return "\n".join(lines)


# ======================================================
# (Optional) Legacy helper retained for compatibility
# ======================================================
def get_role(userinfo: dict) -> str:
    """Legacy helper retained for backward compatibility. Not used by Streamlit auth.
    Returns a default role of 'admin' to avoid breaking older imports."""
    return "admin"
