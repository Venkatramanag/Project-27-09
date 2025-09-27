import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
import os
import hashlib


def _hash_password(password: str) -> str:
    # Compatibility with different streamlit-authenticator versions
    try:
        return stauth.Hasher().hash(password)  # newer API
    except Exception:
        return stauth.Hasher([password]).generate()[0]  # older API

def load_credentials() -> Dict[str, Any]:
    """Build the credentials dict expected by streamlit-authenticator from our DB."""
    creds = {"usernames": {}}
    with _auth_connect() as conn:
        cur = conn.execute("SELECT username, name, password_hash FROM users")
        for username, name, pw_hash in cur.fetchall():
            creds["usernames"][username] = {"name": name, "password": pw_hash}
    return creds

def create_user(username: str, name: str, email: str, role: str, password: str) -> Tuple[bool, str]:
    try:
        if username_exists(username):
            return False, "Username already exists (case-insensitive)."
        if email and email_exists(email):
            return False, "Email already registered (case-insensitive)."

        pw_hash = _hash_password(password)
        with _auth_connect() as conn:
            conn.execute(
                "INSERT INTO users (username, name, email, role, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username.strip(), name.strip(), email.strip() or None, role.strip().lower(), pw_hash, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
        return True, "Account created successfully. You can sign in now."
    except sqlite3.IntegrityError as e:
        # Covers unique index collisions
        msg = str(e).lower()
        if "username" in msg:
            return False, "Username already exists (case-insensitive)."
        if "email" in msg:
            return False, "Email already registered (case-insensitive)."
        return False, f"Registration failed due to a constraint: {e}"
    except Exception as e:
        return False, f"Registration failed: {e}"

def username_exists(username: str) -> bool:
    with _auth_connect() as conn:
        cur = conn.execute(
            "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),)
        )
        return cur.fetchone() is not None

def email_exists(email: str) -> bool:
    if not email:
        return False
    with _auth_connect() as conn:
        cur = conn.execute(
            "SELECT 1 FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip(),)
        )
        return cur.fetchone() is not None

def get_user_count() -> int:
    with _auth_connect() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM users")
        return int(cur.fetchone()[0])

def count_managers() -> int:
    with _auth_connect() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM users WHERE role='manager'")
        return int(cur.fetchone()[0])