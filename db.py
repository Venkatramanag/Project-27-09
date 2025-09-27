# db.py
import os
import re
import requests
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
import sqlparse
from typing import Dict, Any, List

# ==============================
# 📂 Database URLs
# ==============================
DATABASE_URLS = {
    "chinook": "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
    "northwind": "https://raw.githubusercontent.com/harryho/db-samples/master/sqlite/northwind.db"
}

def ensure_db_file(db_name: str, path: str = None) -> str:
    """Ensure database file exists locally. If not, download or create fallback."""
    if path is None:
        path = f".data/{db_name}.sqlite"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if os.path.exists(path):
        return path

    try:
        url = DATABASE_URLS.get(db_name.lower())
        if not url:
            raise ValueError(f"No URL defined for database: {db_name}")
        
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        
        # Verify content is a SQLite file (not a .sql dump)
        if r.headers.get('content-type', '').startswith('text'):
            raise ValueError(f"Expected SQLite binary file, got text for {db_name}")
        
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"Download failed for {db_name}: {e}")
        # Fallback database creation
        from sqlite3 import connect
        conn = connect(path)
        cur = conn.cursor()
        
        if db_name.lower() == "chinook":
            cur.executescript(
                """
                DROP TABLE IF EXISTS Customers;
                DROP TABLE IF EXISTS Invoices;
                CREATE TABLE Customers(
                    CustomerId INTEGER PRIMARY KEY,
                    FirstName TEXT,
                    LastName TEXT,
                    Country TEXT
                );
                CREATE TABLE Invoices(
                    InvoiceId INTEGER PRIMARY KEY,
                    CustomerId INTEGER,
                    InvoiceDate TEXT,
                    BillingCountry TEXT,
                    Total REAL
                );
                INSERT INTO Customers VALUES
                    (1,'Jane','Doe','USA'),
                    (2,'Arun','K','India'),
                    (3,'Maria','Santos','Brazil');
                INSERT INTO Invoices VALUES
                    (1,1,'2024-11-10','USA',25.0),
                    (2,1,'2025-01-12','USA',18.0),
                    (3,2,'2025-02-05','India',30.0);
                """
            )
        elif db_name.lower() == "northwind":
            cur.executescript(
                """
                DROP TABLE IF EXISTS Customers;
                DROP TABLE IF EXISTS Orders;
                CREATE TABLE Customers(
                    CustomerID TEXT PRIMARY KEY,
                    CompanyName TEXT,
                    ContactName TEXT,
                    Country TEXT
                );
                CREATE TABLE Orders(
                    OrderID INTEGER PRIMARY KEY,
                    CustomerID TEXT,
                    OrderDate TEXT,
                    Total REAL
                );
                INSERT INTO Customers VALUES
                    ('CUST1','Acme Corp','John Doe','USA'),
                    ('CUST2','Global Traders','Arun K','India');
                INSERT INTO Orders VALUES
                    (1001,'CUST1','2024-11-10',150.0),
                    (1002,'CUST2','2025-01-15',200.0);
                """
            )
        else:
            raise ValueError(f"No fallback schema defined for {db_name}")
        
        conn.commit()
        conn.close()
        return path

# ==============================
# 🔗 Engine + Schema
# ==============================
def create_engine_from_uri(uri: str) -> Engine:
    """Create SQLAlchemy engine from DB name or path."""
    if uri.strip().lower() in DATABASE_URLS:
        dbfile = ensure_db_file(uri.strip().lower())
        uri = f"sqlite:///{dbfile}"
    elif not uri.startswith("sqlite:///") and uri.endswith((".db", ".sqlite")):
        uri = f"sqlite:///{uri}"
    
    return create_engine(uri, future=True)

# ==============================
# 🔍 Schema Introspection
# ==============================
def get_schema(engine: Engine) -> Dict[str, Any]:
    """
    Introspect schema: tables, columns, PKs, FKs + row counts (non‑breaking enhancement).
    """
    insp = inspect(engine)
    schema: Dict[str, Any] = {}

    # Reuse a single connection for faster counts
    with engine.connect() as conn:
        for t in insp.get_table_names():
            cols = insp.get_columns(t)
            fks = insp.get_foreign_keys(t)
            pk = insp.get_pk_constraint(t)

            # Safe best‑effort row count (quoted table name)
            row_count = None
            try:
                # Double‑quotes okay for SQLite/Postgres (your demo DBs)
                rc = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
                row_count = int(rc) if rc is not None else None
            except Exception:
                row_count = None

            schema[t] = {
                "columns": cols,
                "foreign_keys": fks,
                "primary_key": pk.get("constrained_columns", []),
                "row_count": row_count,
            }

    return schema

def list_tables(engine: Engine) -> List[str]:
    insp = inspect(engine)
    return insp.get_table_names()

# ==============================
# ✅ Safe SQL Execution
# ==============================
DESTRUCTIVE = [
    "drop", "delete", "update", "insert", "alter", "truncate",
    "attach", "detach", "vacuum", "reindex", "pragma"
]

def is_safe_sql(sql: str) -> bool:
    """Validate that SQL is a single SELECT, not destructive."""
    if not sql or not isinstance(sql, str):
        return False
    try:
        parsed = sqlparse.parse(sql)
        if len(parsed) != 1:
            return False
        token0 = parsed[0].token_first(skip_cm=True)
        if token0 is None:
            return False
        if token0.normalized.upper() != "SELECT":
            return False
    except Exception:
        return False

    lower = sql.lower()
    for bw in DESTRUCTIVE:
        if re.search(r"\b" + re.escape(bw) + r"\b", lower):
            return False

    if ";" in lower.strip()[1:]:
        return False

    return True

def _maybe_add_limit(sql: str, engine: Engine, limit: int = 500) -> str:
    """Ensure SELECT has a LIMIT to prevent huge outputs."""
    if "limit" in sql.lower():
        return sql
    dialect = engine.dialect.name
    if dialect in ("sqlite", "postgresql", "mysql", "mariadb"):
        return sql.strip() + f" LIMIT {limit}"
    return sql

def run_sql(engine: Engine, sql: str, limit: int = 500) -> pd.DataFrame:
    """Execute safe SELECT query and return DataFrame."""
    if not is_safe_sql(sql):
        raise ValueError("Only single SELECT queries allowed. Query rejected as unsafe.")

    sql_to_run = _maybe_add_limit(sql, engine, limit=limit)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql_to_run), conn)
    return df
