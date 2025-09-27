# app.py
import os
import html
import sqlite3
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional, List

import streamlit as st
import pandas as pd
import plotly.express as px  # used by VisualizationAgent

# Additional utils for unique chart keys
import hashlib, json

# Auth
import streamlit_authenticator as stauth  # pip install streamlit-authenticator

# Your modules (unchanged)
from db import create_engine_from_uri, get_schema
from agents2 import NL2SQLAgent, AccuracyAgent, SQLExecutorAgent, VisualizationAgent, create_crew
from utils1 import (
    init_audit, log_query, fetch_audit, clear_audit, audit_to_csv_bytes, SchemaFormatter
)

# --- Stylish scrollable schema card (always inject CSS) ----------------------
def render_schema_card(schema_text: str, title: str = "🔎 Schema (auto‑detected)"):
    st.markdown(
        """
        <style>
          :root{
            /* Dark theme defaults */
            --schema-bg: #0b1220;
            --schema-text: #eaf2ff;
            --schema-muted: #a9b4c9;
            --schema-accent: #7c4dff;   /* purple */
            --schema-accent2:#00e5ff;   /* aqua  */
            --schema-shadow: rgba(0,0,0,.35);
            --schema-border: rgba(124,77,255,.5);
          }
          @media (prefers-color-scheme: light) {
            :root{
              /* Light theme overrides */
              --schema-bg: #ffffff;
              --schema-text: #0f172a;
              --schema-muted: #475569;
              --schema-accent: #6d28d9;
              --schema-accent2:#06b6d4;
              --schema-shadow: rgba(16,24,40,.18);
              --schema-border: rgba(109,40,217,.35);
            }
          }

          .schema-card{
            border-radius: 16px;
            padding: 1px; /* gradient border width */
            background: linear-gradient(135deg,var(--schema-accent),var(--schema-accent2));
            box-shadow: 0 10px 28px var(--schema-shadow);
            margin-bottom: 1rem;
          }
          .schema-card__inner{
            border-radius: 15px;
            background: var(--schema-bg);
            overflow: hidden;
          }
          .schema-card__header{
            position: sticky;
            top: 0;
            z-index: 2;
            display:flex; align-items:center; justify-content:space-between;
            padding: 10px 14px;
            color: #fff;
            font-weight: 600;
            letter-spacing:.2px;
            background: linear-gradient(90deg,var(--schema-accent) 0%, rgba(124,77,255,.85) 22%, rgba(0,229,255,.35) 100%);
            border-bottom: 1px solid rgba(255,255,255,.12);
            text-shadow: 0 1px 0 rgba(0,0,0,.25);
          }
          .schema-card__body{
            max-height: 460px;            /* 👈 scroll area height */
            overflow: auto;                /* 👈 scrolling */
            padding: 14px 16px;
            background:
              radial-gradient(ellipse at top left, rgba(124,77,255,.08), transparent 45%),
              radial-gradient(ellipse at bottom right, rgba(0,229,255,.08), transparent 45%);
          }
          .schema-pre{
            margin:0;
            color: var(--schema-text);
            white-space: pre-wrap;         /* keep bullets & wrapping */
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            font-size: 13.5px;
            line-height: 1.5;
          }
          /* Fancy minimal scrollbars */
          .schema-card__body::-webkit-scrollbar{ width: 10px; height:10px; }
          .schema-card__body::-webkit-scrollbar-track{ background: transparent; }
          .schema-card__body::-webkit-scrollbar-thumb{
            border-radius: 10px;
            background: linear-gradient(180deg,var(--schema-accent),var(--schema-accent2));
          }
          .schema-card__body{ scrollbar-width: thin; scrollbar-color: var(--schema-accent) transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    from html import escape as _esc
    safe = _esc(schema_text or "")
    st.markdown(
        f"""
        <div class="schema-card">
          <div class="schema-card__inner">
            <div class="schema-card__header">{_esc(title)}</div>
            <div class="schema-card__body">
              <pre class="schema-pre">{safe}</pre>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# -----------------------------------------------------------------------------

# --- Format schema text for display (drop types, tidy spacing) ---------------
import re

def _drop_type_token(inside: str) -> str:
    """
    inside: e.g., 'NUMERIC(10, 2), NOT NULL, FK→ X(Y)'
    Return everything AFTER the first top-level comma (i.e., the flags),
    respecting nested parentheses in the type.
    If there is no top-level comma -> return "" (means only type existed).
    """
    depth = 0
    for i, ch in enumerate(inside):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            return inside[i+1:].strip().strip(',')
    return ""  # only type present

def format_schema_for_display(raw: str) -> str:
    if not raw:
        return ""
    out = []
    prev_blank = True
    first_table_seen = False

    lines = raw.splitlines()

    for s in map(str.rstrip, lines):
        # compress multiple blank lines
        if not s.strip():
            if not prev_blank:
                out.append("")      # single blank
                prev_blank = True
            continue

        # ensure exactly one blank line between tables
        if s.startswith("📂 Table "):
            if first_table_seen and not (out and out[-1] == ""):
                out.append("")
            first_table_seen = True
            out.append(s)
            prev_blank = False
            continue

        # drop types from bullet lines: "  - Name (TYPE, flags...)" -> "  - Name (flags...)"
        if s.startswith("  - ") and "(" in s and s.endswith(")"):
            l = s.find("(")
            r = len(s) - 1  # last ')'
            if l != -1 and s[r] == ')':
                prefix = s[:l].rstrip()        # "  - ColumnName"
                inside = s[l+1:r].strip()
                flags = _drop_type_token(inside)
                s = f"{prefix} ({flags})" if flags else prefix

        out.append(s)
        prev_blank = False

    # remove possible leading blank
    while out and out[0] == "":
        out.pop(0)

    return "\n".join(out)
# -----------------------------------------------------------------------------


# ======================================================
# App Config
# ======================================================
st.set_page_config(page_title="Multi-Agent Data Query System", layout="wide")

st.markdown("## 🚀 Cognitive Multi‑Agent System for Natural Language → Insights")
st.write("Interact with databases using **natural language** — secure, role‑aware, and auditable.")

# ======================================================
# Simple local auth DB (SQLite) for users/roles
# ======================================================
AUTH_DB = ".data/auth.sqlite"
os.makedirs(os.path.dirname(AUTH_DB), exist_ok=True)
AUTH_DB_ABS = os.path.abspath(AUTH_DB)

def _auth_connect():
    return sqlite3.connect(AUTH_DB, check_same_thread=False)

def init_auth_db():
    with _auth_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                role TEXT NOT NULL CHECK (role IN ('manager','admin','guest')),
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Prevent duplicates differing only by case
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username_nocase ON users(username COLLATE NOCASE)")
        # Optional: also ensure email uniqueness case-insensitively
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email_nocase ON users(email COLLATE NOCASE)")
        conn.commit()

def get_user_count() -> int:
    with _auth_connect() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM users")
        return int(cur.fetchone()[0])

def get_user_role(username: str) -> Optional[str]:
    with _auth_connect() as conn:
        cur = conn.execute(
            "SELECT role FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),)
        )
        row = cur.fetchone()
        return row[0] if row else None

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

def _hash_password(password: str) -> str:
    # Compatibility with different streamlit-authenticator versions
    try:
        return stauth.Hasher().hash(password)  # newer API
    except Exception:
        return stauth.Hasher([password]).generate()[0]  # older API

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

def load_credentials() -> Dict[str, Any]:
    """Build the credentials dict expected by streamlit-authenticator from our DB."""
    creds = {"usernames": {}}
    with _auth_connect() as conn:
        cur = conn.execute("SELECT username, name, password_hash FROM users")
        for username, name, pw_hash in cur.fetchall():
            creds["usernames"][username] = {"name": name, "password": pw_hash}
    return creds

# ---- User Management helpers

def fetch_users(search: str = "") -> pd.DataFrame:
    q = (
        "SELECT username, name, COALESCE(email,'') AS email, role, created_at "
        "FROM users ORDER BY CASE role WHEN 'manager' THEN 1 WHEN 'admin' THEN 2 ELSE 3 END, lower(username)"
    )
    with _auth_connect() as conn:
        df = pd.read_sql_query(q, conn)
    if search:
        s = search.lower().strip()
        mask = (
            df["username"].str.lower().str.contains(s)
            | df["name"].str.lower().str.contains(s)
            | df["email"].str.lower().str.contains(s)
            | df["role"].str.lower().str.contains(s)
        )
        df = df[mask].reset_index(drop=True)
    return df

def count_managers() -> int:
    with _auth_connect() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM users WHERE role='manager'")
        return int(cur.fetchone()[0])

def update_user_role(target_username: str, new_role: str) -> Tuple[bool, str]:
    new_role = (new_role or "").strip().lower()
    if new_role not in {"manager", "admin", "guest"}:
        return False, "Invalid role."

    caller = st.session_state.get("user")
    caller_role = (st.session_state.get("role") or "guest").strip().lower()

    # Only Manager can change roles
    if caller_role != "manager":
        return False, "Only Managers can change user roles."

    # Prevent removing last Manager
    if new_role != "manager":
        # If target is the only manager, block demotion
        with _auth_connect() as conn:
            cur = conn.execute(
                "SELECT role FROM users WHERE username = ? COLLATE NOCASE",
                (target_username.strip(),)
            )
            row = cur.fetchone()
        if row and row[0] == "manager" and count_managers() == 1:
            return False, "Cannot demote the last remaining Manager."

    try:
        with _auth_connect() as conn:
            cur = conn.execute(
                "UPDATE users SET role=? WHERE username=? COLLATE NOCASE",
                (new_role, target_username.strip())
            )
            conn.commit()
        if cur.rowcount == 0:
            return False, "User not found."
        # Audit trail
        try:
            log_query(
                user=caller or "unknown",
                role=caller_role,
                user_q=f"role_change: {target_username}->{new_role}",
                sql="N/A",
                status="success",
                rows=0,
                duration=0.0,
            )
        except Exception:
            pass
        return True, f"Role updated: {target_username} → {new_role}"
    except Exception as e:
        return False, f"Failed to update role: {e}"

# ======================================================
# RBAC helpers
# ======================================================

def current_role() -> str:
    role = st.session_state.get("role") or "guest"
    role = str(role).strip().lower()
    if role not in {"manager","admin","guest"}:
        role = "guest"
    return role

def is_manager() -> bool: return current_role() == "manager"

def is_admin() -> bool:   return current_role() == "admin"

def is_guest() -> bool:   return current_role() == "guest"

def allow_generate_and_execute() -> bool:
    return is_manager() or is_admin()

def allow_audit_read() -> bool:
    return is_manager() or is_admin()

def allow_audit_clear() -> bool:
    return is_manager()

# ======================================================
# Helper: Unique keys for Plotly charts
# ======================================================

def _plotly_key(prefix: str, idx: int, fig) -> str:
    """Create a stable, unique key for st.plotly_chart to avoid duplicate element IDs."""
    try:
        payload = fig.to_plotly_json() if hasattr(fig, "to_plotly_json") else fig.to_dict()
        dump = json.dumps(payload, sort_keys=True, default=str)
        h = hashlib.md5(dump.encode("utf-8")).hexdigest()[:10]
    except Exception:
        # Fallback: memory address hash
        h = f"{id(fig):x}"
    return f"{prefix}_{idx}_{h}"

# ======================================================
# Initialize subsystems (audit + auth DB)
# ======================================================
init_audit()
init_auth_db()

# ======================================================
# Authentication UI: Sign In / Sign Up
# ======================================================
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

if not st.session_state["auth_ok"]:
    tab_signin, tab_signup = st.tabs(["🔐 Sign in", "🆕 Sign up"])

    with tab_signin:
        credentials = load_credentials()
        if not credentials["usernames"]:
            st.info("No users found. Go to **Sign up** to create the first Manager account.")

        authenticator = stauth.Authenticate(
            credentials=credentials,
            cookie_name=os.getenv("AUTH_COOKIE_NAME", "mai_app_auth"),
            cookie_key=os.getenv("AUTH_SIGNATURE_KEY", "replace-this-with-a-long-random-string"),
            cookie_expiry_days=int(os.getenv("AUTH_COOKIE_DAYS", "7")),
        )

        login_out = authenticator.login(
            location="main",
            fields={
                "Form name": "Sign in",
                "Username": "Username",
                "Password": "Password",
                "Login": "Sign in",
            },
            key="signin",
        )
        if login_out:
            name, auth_status, username = login_out
        else:
            name = st.session_state.get("name")
            username = st.session_state.get("username")
            auth_status = st.session_state.get("authentication_status")

        if auth_status is False:
            st.error("❌ Invalid username or password.")
        elif auth_status is None:
            st.info("Please enter your credentials.")
        elif auth_status:
            # Fetch canonical username + role case-insensitively
            with _auth_connect() as conn:
                cur = conn.execute(
                    "SELECT username, role FROM users WHERE username = ? COLLATE NOCASE",
                    (username.strip(),)
                )
                row = cur.fetchone()

            if row:
                canonical_username, role = row[0], (row[1] or "guest")
            else:
                canonical_username, role = username.strip(), "guest"

            role = role.strip().lower()
            if role not in {"manager","admin","guest"}:
                role = "guest"

            # Bootstrap safety: if DB has 0 managers, promote current user
            try:
                if count_managers() == 0:
                    with _auth_connect() as c:
                        c.execute(
                            "UPDATE users SET role='manager' WHERE username = ? COLLATE NOCASE",
                            (canonical_username,)
                        )
                        c.commit()
                    role = "manager"
            except Exception:
                pass

            st.session_state["user"] = canonical_username
            st.session_state["name"] = name
            st.session_state["role"] = role
            st.session_state["auth_ok"] = True
            st.rerun()

    with tab_signup:
        c1, c2 = st.columns([1, 2])
        with c2:
            user_count = get_user_count()
            st.markdown("#### Create an account")
            with st.form("signup_form", clear_on_submit=False):
                username = st.text_input("Username").strip()
                name = st.text_input("Full name").strip()
                email = st.text_input("Email (optional)").strip()
                if user_count == 0:
                    st.info("First user becomes a **Manager** by default (bootstrap).")
                    role = "manager"
                else:
                    st.info("New registrations get the **Guest** role. Managers can upgrade roles later.")
                    role = "guest"
                pw1 = st.text_input("Password", type="password")
                pw2 = st.text_input("Confirm password", type="password")
                agree = st.checkbox("I agree to the Terms & Conditions")
                submitted = st.form_submit_button("Create account")

            if submitted:
                if not (username and name and pw1 and pw2):
                    st.warning("Please fill all required fields.")
                elif pw1 != pw2:
                    st.warning("Passwords do not match.")
                elif not agree:
                    st.warning("Please accept the Terms & Conditions.")
                elif username_exists(username):
                    st.error("🚫 Username already exists (case-insensitive). Please choose a different username.")
                elif email and email_exists(email):
                    st.error("🚫 Email already registered (case-insensitive). Try signing in or use a different email.")
                else:
                    ok, msg = create_user(username, name, email, role, pw1)
                    if ok:
                        st.success("✅ " + msg)
                        st.info("Go to **Sign in** to log in.")
                    else:
                        st.error("❌ " + msg)

    st.stop()  # Do not show the app until authenticated

# ======================================================
# Authenticated area
# ======================================================

# Sidebar: Logout (renderless to avoid widget key conflicts)
st.sidebar.markdown("---")
if st.sidebar.button("Logout", key="btn_logout"):
    try:
        stauth.Authenticate(
            credentials=load_credentials(),
            cookie_name=os.getenv("AUTH_COOKIE_NAME", "mai_app_auth"),
            cookie_key=os.getenv("AUTH_SIGNATURE_KEY", "replace-this-with-a-long-random-string"),
            cookie_expiry_days=int(os.getenv("AUTH_COOKIE_DAYS", "7")),
        ).logout(location="unrendered")
    finally:
        # Clear app session keys
        for k in (
            "auth_ok","user","name","role","GOOGLE_API_KEY",
            "engine","schema","schema_text","last_df","last_fig","last_figs","active_view","nav"
        ):
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

role_title = st.session_state.get("role", "guest").capitalize()
st.success(f"✅ Logged in as **{st.session_state.get('user')}** (*{role_title}*)")

with st.expander("Auth debug (local)"):
    st.caption(f"Auth DB: `{AUTH_DB_ABS}`")
    st.caption(f"Logged in user: `{st.session_state.get('user')}`; role: `{st.session_state.get('role')}`")

# ---- Sidebar Navigation
if is_manager() or is_admin():
    st.sidebar.markdown("---")
    st.sidebar.header("🧭 Navigation")
    nav = st.sidebar.radio("Select page", ["Workspace", "User Management"], label_visibility="collapsed")
else:
    nav = "Workspace"

st.session_state["nav"] = nav

# ======================================================
# WORKSPACE PAGE (original features)
# ======================================================
if st.session_state["nav"] == "Workspace":

    # Sidebar: DB + Audit
    with st.sidebar:
        st.header("🗂 Database")
        db_choice = st.text_input("DB name/path", "chinook")
        if st.button("Connect"):
            try:
                engine = create_engine_from_uri(db_choice)
                schema = get_schema(engine)
                st.session_state["engine"] = engine
                st.session_state["schema"] = schema
                st.session_state["schema_text"] = SchemaFormatter(schema).formatted()
                st.success("✅ Connected successfully!")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

        st.markdown("---")
        st.header("📜 Audit")
        if allow_audit_read():
            if st.button("Clear history", disabled=not allow_audit_clear()):
                if allow_audit_clear():
                    clear_audit()
                    st.success("✅ Audit cleared")
            csv_bytes = audit_to_csv_bytes()
            st.download_button("Download audit (CSV)",
                               data=csv_bytes,
                               file_name="audit_log.csv",
                               mime="text/csv")
        else:
            if st.button("Open Audit"):
                st.warning("You need Manager/Admin access to view Audit.")

    # Main layout
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("💡 Suggested Queries")
        if "schema" in st.session_state:
            show_suggestions = st.checkbox("Show AI suggestions (uses LLM quota)", value=False)
            if show_suggestions:
                try:
                    nl2sql_sugg = NL2SQLAgent(st.session_state["schema"])
                    suggestions = []
                    try:
                        suggestions = nl2sql_sugg.suggest_queries()
                    except Exception:
                        suggestions = []
                except Exception:
                    suggestions = []
                for q in suggestions:
                    st.markdown(f"- {q}")

        st.subheader("📝 Ask a Query")
        question = st.text_area("Type your natural language query", height=120)

        # Resolve Gemini API key (UI > session_state > secrets > env)
        llm_api_key = st.text_input(
            "Gemini API key (optional)",
            type="password",
            help="If empty, uses st.secrets['GOOGLE_API_KEY'] or env GOOGLE_API_KEY / GEMINI_API_KEY."
        )

        run_clicked = st.button("Generate & Run SQL", disabled=not allow_generate_and_execute())

        if not allow_generate_and_execute():
            st.info("Guests cannot generate/execute queries. Please contact a Manager/Admin for access.")

    with right_col:
        #st.subheader("🔎 Schema (auto‑detected)")
        #if "schema_text" in st.session_state:
        #    schema_html = f"<div class='schema-box'>{html.escape(st.session_state['schema_text'])}</div>"
        #    st.markdown(schema_html, unsafe_allow_html=True)
        #else:
        #    st.info("Connect a database to see schema.")
    
        st.subheader("")
        if "schema_text" in st.session_state:
            #st.code(st.session_state["schema_text"], language="")
            display_schema = format_schema_for_display(st.session_state["schema_text"])
            render_schema_card(display_schema)
        else:
            st.info("Connect a database to see schema.")

        st.markdown("---")
        st.subheader("🧪 SQL & Results")

        if "active_view" not in st.session_state:
            st.session_state["active_view"] = "Results"
        view = st.radio(
            "View", ["Results", "Visualizations"],
            horizontal=True,
            index=(0 if st.session_state["active_view"] == "Results" else 1),
            label_visibility="collapsed",
        )
        st.session_state["active_view"] = view

        sql_display = st.empty()
        result_display = st.empty()
        chart_display = st.empty()

    # ---- Lifecycle helpers
    def _mk_lifecycle_container():
        exp = st.expander("🛠️ Agent Lifecycle (live)", expanded=True)
        lc = exp.container()
        stages = {
            "received": lc.empty(),
            "nl2sql": lc.empty(),
            "validate": lc.empty(),
            "execute": lc.empty(),
            "viz": lc.empty(),
            "summary": lc.empty(),
        }
        _titles = { "received": "Received", "nl2sql": "NL→SQL", "validate": "Validation", "execute": "Execution", "viz": "Visualization", "summary": "Summary" }
        _icons = { "running": "⏳", "success": "✅", "error": "❌", "info": "ℹ️" }
        def set_stage(stage: str, status: str, message: str):
            icon = _icons.get(status, "•")
            title = _titles.get(stage, stage.capitalize())
            stages[stage].markdown(f"{icon} **{title}** — {message}")
        def emit(msg: str):
            m = (msg or "").strip()
            if "NL2SQL started" in m: set_stage("nl2sql", "running", "Generating SQL…")
            elif "Validation started" in m: set_stage("validate", "running", "Validating SQL…")
            elif "Executing SQL" in m: set_stage("execute", "running", "Running query…")
            elif "Building visualizations" in m or "Building visualization" in m: set_stage("viz", "running", "Creating charts…")
            elif m.startswith("✅ Execution done"): set_stage("execute", "success", m.replace("✅ ", ""))
            elif "Visualization step complete" in m: set_stage("viz", "success", "Charts generated")
            elif m.startswith("❌ Execution error"): set_stage("execute", "error", m.replace("❌ ", ""))
            elif m.startswith("ℹ️ Visualization skipped"): set_stage("viz", "info", "No suitable data")
        return set_stage, emit

    set_stage, emit = _mk_lifecycle_container()

    # ---- Main Action
    if run_clicked:
        if "engine" not in st.session_state or "schema" not in st.session_state:
            st.error("Please connect a database first.")
        elif not question.strip():
            st.warning("Please type a natural language query.")
        elif not allow_generate_and_execute():
            st.warning("You need Manager/Admin rights to run queries.")
            try:
                log_query(
                    user=st.session_state.get("user", "unknown"),
                    role=current_role(),
                    user_q=question,
                    sql="N/A",
                    status="denied",
                    rows=0,
                    duration=0.0,
                )
            except Exception:
                pass
        else:
            engine = st.session_state["engine"]
            schema = st.session_state["schema"]

            # Resolve API key
            resolved_key = None
            if llm_api_key and llm_api_key.strip():
                resolved_key = llm_api_key.strip()
                st.session_state["GOOGLE_API_KEY"] = resolved_key
            elif "GOOGLE_API_KEY" in st.session_state:
                resolved_key = st.session_state["GOOGLE_API_KEY"]
            else:
                resolved_key = (st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None)
            if not resolved_key:
                resolved_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not resolved_key:
                st.error("Gemini API key not found. Provide it in the input field, set st.secrets['GOOGLE_API_KEY'], or export GOOGLE_API_KEY.")
                st.stop()

            # Initialize agents
            nl2sql = NL2SQLAgent(schema, llm_api_key=resolved_key, emit=lambda m: emit(f"NL2SQL: {m}"))
            accuracy = AccuracyAgent(schema, llm_api_key=resolved_key, emit=lambda m: emit(f"Validator: {m}"))
            executor = SQLExecutorAgent(engine)
            vis = VisualizationAgent()

            crew = create_crew(nl2sql, accuracy, executor, vis, question, emit=emit)

            # Initialize stages
            set_stage("received", "info", question)
            for s in ("nl2sql","validate","execute","viz"):
                set_stage(s, "info", "Pending")

            # Run workflow
            try:
                result = crew.kickoff()
                sql = result.get("sql")
                df = result.get("df")
                fig = result.get("fig")
                figs = result.get("figs", [])
                rationale = result.get("rationale", [])
                duration = float(result.get("duration", 0.0))

                # Stage updates
                if sql: set_stage("nl2sql", "success", "SQL generated")
                else:   set_stage("nl2sql", "error", "Could not generate SQL")

                val_line = next((r for r in rationale if "Validation:" in r), "")
                if "VALID" in val_line.upper(): set_stage("validate", "success", "SQL validated")
                elif "INVALID" in val_line.upper(): set_stage("validate", "error", "Invalid SQL")
                else:
                    if df is not None: set_stage("validate", "success", "SQL validated")
                    else: set_stage("validate", "info", "Unknown (see logs)")

                if df is not None and isinstance(df, pd.DataFrame):
                    set_stage("execute", "success", f"{len(df)} rows in {duration:.3f}s")
                else:
                    set_stage("execute", "error", "Failed")

                if figs: set_stage("viz", "success", f"{len(figs)} chart(s) available")
                elif fig is not None: set_stage("viz", "success", "1 chart available")
                else: set_stage("viz", "info", "No charts")

                # Render SQL
                if sql:
                    sql_display.code(sql, language="sql")
                else:
                    st.error("❌ No valid SQL generated.")
                    try:
                        log_query(
                            user=st.session_state["user"], role=current_role(),
                            user_q=question, sql="N/A", status="failed", rows=0, duration=duration
                        )
                    except Exception:
                        pass
                    result_display.empty(); chart_display.empty(); st.stop()

                # Persist outputs for view switching
                st.session_state["last_df"] = df if isinstance(df, pd.DataFrame) else None
                st.session_state["last_figs"] = figs if figs else None
                st.session_state["last_fig"] = fig if fig is not None else None

                # Render based on active view
                if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                    if st.session_state["active_view"] == "Results":
                        result_display.dataframe(df, use_container_width=True)
                        try:
                            log_query(
                                user=st.session_state["user"], role=current_role(),
                                user_q=question, sql=sql, status="success", rows=len(df), duration=duration
                            )
                        except Exception:
                            pass

                        has_charts = bool(figs) or (fig is not None)
                        if has_charts:
                            if st.button("📊 Click to view visualizations", key="btn_view_viz"):
                                st.session_state["active_view"] = "Visualizations"
                                st.rerun()
                    else:
                        # Visualizations view
                        chart_display.empty()
                        if st.session_state.get("last_figs"):
                            charts = st.session_state["last_figs"]
                            cols = st.columns(min(3, len(charts)))
                            for i, (f, caption) in enumerate(charts):
                                with cols[i % len(cols)]:
                                    st.plotly_chart(f, use_container_width=True, key=_plotly_key("viz_run", i, f))
                                    if caption: st.caption(caption)
                        elif st.session_state.get("last_fig") is not None:
                            st.plotly_chart(
                                st.session_state["last_fig"],
                                use_container_width=True,
                                key=_plotly_key("viz_run_single", 0, st.session_state["last_fig"]),
                            )
                        else:
                            st.info("No suitable data for visualization.")
                else:
                    st.error("No data returned from query.")
                    try:
                        log_query(
                            user=st.session_state["user"], role=current_role(),
                            user_q=question, sql=sql if sql else "N/A", status="failed", rows=0, duration=duration
                        )
                    except Exception:
                        pass

                rows = len(df) if isinstance(df, pd.DataFrame) else 0
                charts = len(figs) or (1 if fig is not None else 0)
                set_stage("summary", "success", f"Done — rows: {rows}, time: {duration:.3f}s, charts: {charts}")

            except Exception as e:
                msg = str(e)
                set_stage("summary", "error", f"Workflow failed: {msg}")
                st.error(f"Workflow failed: {msg}")
                try:
                    log_query(
                        user=st.session_state.get("user", "unknown"),
                        role=current_role(),
                        user_q=question,
                        sql="N/A",
                        status="failed",
                        rows=0,
                        duration=0.0,
                    )
                except Exception:
                    pass

    # Persisted view rendering when not running (toggle views)
    if "last_df" in st.session_state:
        df = st.session_state.get("last_df")
        if st.session_state.get("active_view") == "Results":
            if isinstance(df, pd.DataFrame) and not df.empty:
                result_display = st.empty()
                result_display.dataframe(df, use_container_width=True)
        elif st.session_state.get("active_view") == "Visualizations":
            chart_display = st.empty()
            if st.session_state.get("last_figs"):
                charts = st.session_state["last_figs"]
                cols = st.columns(min(3, len(charts)))
                for i, (f, caption) in enumerate(charts):
                    with cols[i % len(cols)]:
                        st.plotly_chart(f, use_container_width=True, key=_plotly_key("viz_persist", i, f))
                        if caption: st.caption(caption)
            elif st.session_state.get("last_fig") is not None:
                st.plotly_chart(
                    st.session_state["last_fig"],
                    use_container_width=True,
                    key=_plotly_key("viz_persist_single", 0, st.session_state["last_fig"]),
                )

    # ---- Audit (bottom)
    st.markdown("---")
    st.subheader("📜 Audit Log (recent)")
    if allow_audit_read():
        audit_df = fetch_audit(limit=100)
        st.dataframe(audit_df, use_container_width=True)
    else:
        st.info("Guests cannot view audit history.")

# ======================================================
# USER MANAGEMENT PAGE
# ======================================================
else:
    st.header("👥 User Management")

    if is_admin() and not is_manager():
        st.info("You have read-only access to user list. Only Managers can change roles.")

    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        search = st.text_input("Search users", placeholder="Search by username, name, email, or role")
    with colB:
        st.markdown("<div class='muted'>Role filter</div>", unsafe_allow_html=True)
        role_filter = st.selectbox("Role filter", ["All", "manager", "admin", "guest"], index=0, label_visibility="collapsed")
    with colC:
        users_df = fetch_users()
        st.metric("Total users", len(users_df))

    # Filtered view
    df = fetch_users(search)
    if role_filter != "All":
        df = df[df["role"] == role_filter].reset_index(drop=True)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Managers", int((df["role"] == "manager").sum()))
    with m2:
        st.metric("Admins", int((df["role"] == "admin").sum()))
    with m3:
        st.metric("Guests", int((df["role"] == "guest").sum()))

    # Users table
    st.subheader("Users")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download CSV
    st.download_button(
        "Download users (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="users.csv",
        mime="text/csv",
        key="dl_users_csv"
    )

    st.markdown("---")
    st.subheader("Promote / Demote")

    # Manager-only controls
    if is_manager():
        with st.form("form_role_change", clear_on_submit=False):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                all_users_df = fetch_users()
                user_list = all_users_df["username"].tolist()
                target_user = st.selectbox("Select user", user_list, index=0 if user_list else None)
            with c2:
                new_role = st.selectbox("Select new role", ["manager", "admin", "guest"], index=1)
            with c3:
                submitted = st.form_submit_button("Apply")

        if submitted:
            ok, msg = update_user_role(target_user, new_role)
            if ok:
                st.success("✅ " + msg)
                # If we changed our own role, update session state
                if target_user == st.session_state.get("user"):
                    st.session_state["role"] = new_role
                st.rerun()
            else:
                st.error("❌ " + msg)
    else:
        st.info("Only Managers can change user roles.")


