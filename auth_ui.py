import streamlit as st
import streamlit_authenticator as stauth
import os
import sqlite3
from auth_helpers import (
    load_credentials, create_user, username_exists,
    email_exists, get_user_count, count_managers
)

def _auth_connect():
    return sqlite3.connect(".data/auth.sqlite", check_same_thread=False)

def apply_custom_css(css_file="style.css"):
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_auth_ui():
    apply_custom_css()

    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False

    if not st.session_state["auth_ok"]:
        tab_signin, tab_signup = st.tabs(["🔐 Sign in", "🆕 Sign up"])

        with tab_signin:
            st.caption(
                "Roles: **Manager** (owner-level, can change roles & clear audit) > **Admin** (operators, can run queries & read audit) > **Guest** (view-only)."
            )
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
                if role not in {"manager", "admin", "guest"}:
                    role = "guest"

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
                    pw2 = st.text_input("Confirm Password", type="password")
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

        st.stop()
