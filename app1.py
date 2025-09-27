# app1.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import html  # for safe schema rendering with scroll

# Fallback to Gemini if OpenAI key is missing
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', 'AIzaSyD1zr0PY8QclTlrhsIURsBZoPmkMlybvz4')

from db import create_engine_from_uri, get_schema
from agents2 import NL2SQLAgent, AccuracyAgent, SQLExecutorAgent, VisualizationAgent, create_crew
from utils1 import (
    verify_user,
    init_audit,
    log_query,
    fetch_audit,
    clear_audit,
    audit_to_csv_bytes,
    SchemaFormatter,
)
from crewai import Crew

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(page_title="Multi-Agent Data Query System", layout="wide")

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    .big-title {font-size: 32px; font-weight: 700; color: #1F77B4; text-align:center; animation: fadeIn 1s;}
    .muted {color: #6b6b6b;}
    .sidebar .stButton button {background-color:#1F77B4;color:white;border-radius:8px; transition: 0.3s;}
    .sidebar .stButton button:hover {background-color:#155a8a;}
    .st-expanderHeader {font-weight:600; color:#1F77B4;}
    .schema-scroll {max-height: 260px; overflow:auto; padding:8px; background:#0e1117; border-radius:6px; border:1px solid #263238;}
    .schema-scroll pre {margin:0; color:#d1d5db; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    @keyframes fadeIn { from {opacity:0;} to {opacity:1;} }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🚀 Cognitive Multi-Agent System for Natural Language to Insights Automation </div>', unsafe_allow_html=True)
st.write("Interact with databases using **natural language** — powered by agentic AI frameworks. Secure and auditable.")

# Initialize audit DB
init_audit()

# -------------------------
# Sidebar: Login, DB, Audit
# -------------------------
with st.sidebar:
    st.header("🔐 Login")
    if "user" not in st.session_state:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            auth = verify_user(username, password)
            if auth:
                st.session_state["user"] = auth["userinfo"]["preferred_username"]
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        st.stop()
    else:
        st.success(f"✅ Logged in as {st.session_state['user']}")

    st.markdown("---")
    st.header("📂 Database Connection")
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
    if st.button("Clear history"):
        clear_audit()
        st.success("✅ Audit cleared")
    csv_bytes = audit_to_csv_bytes()
    st.download_button("Download audit (CSV)", data=csv_bytes, file_name="audit_log.csv", mime="text/csv")

# -------------------------
# Main layout: Left (controls) / Right (results)
# -------------------------
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("💡 Suggested Queries")
    if "schema" in st.session_state:
        # Avoid burning LLM quota unnecessarily
        show_suggestions = st.checkbox("Show AI suggestions (uses LLM quota)", value=False)
        if show_suggestions:
            nl2sql_sugg = NL2SQLAgent(st.session_state["schema"])
            try:
                suggestions = nl2sql_sugg.suggest_queries()
            except Exception:
                suggestions = []
            for q in suggestions:
                st.markdown(f"- {q}")

    st.subheader("📝 Ask a Query")
    question = st.text_area("Type your natural language query", height=120)

    # NOTE: The key resolution below looks at this field first;
    # leave it empty if you want to rely on st.secrets/env.
    llm_api_key = st.text_input(
        "Gemini API key (optional)",
        type="password",
        help="If empty, uses st.secrets['GOOGLE_API_KEY'] or env GOOGLE_API_KEY / GEMINI_API_KEY."
    )
    run_clicked = st.button("Generate & Run SQL")

    # -------------------------
    # Agent lifecycle (structured)
    # -------------------------
    lifecycle_exp = st.expander("🛠️ Agent Lifecycle (live)", expanded=True)
    lc = lifecycle_exp.container()

    # Create one placeholder per stage to keep the panel tidy
    stages = {
        "received": lc.empty(),
        "nl2sql":   lc.empty(),
        "validate": lc.empty(),
        "execute":  lc.empty(),
        "viz":      lc.empty(),
        "summary":  lc.empty(),
    }

    _stage_titles = {
        "received": "Received",
        "nl2sql":   "NL→SQL",
        "validate": "Validation",
        "execute":  "Execution",
        "viz":      "Visualization",
        "summary":  "Summary",
    }

    _status_icon = {
        "running": "⏳",
        "success": "✅",
        "error":   "❌",
        "info":    "ℹ️",
    }

    def set_stage(stage: str, status: str, message: str):
        icon = _status_icon.get(status, "•")
        title = _stage_titles.get(stage, stage.capitalize())
        stages[stage].markdown(f"{icon} **{title}** — {message}")

    # Structured emitter mapping verbose signals to compact stage lines
    def emit(msg: str):
        m = (msg or "").strip()

        # Start events → set to running
        if "NL2SQL started" in m:
            set_stage("nl2sql", "running", "Generating SQL…")
        elif "Validation started" in m:
            set_stage("validate", "running", "Validating SQL…")
        elif "Executing SQL" in m:
            set_stage("execute", "running", "Running query…")
        elif "Building visualization" in m or "Building visualizations" in m:
            set_stage("viz", "running", "Creating charts…")

        # Success checkpoints
        elif m.startswith("✅ Execution done"):
            set_stage("execute", "success", m.replace("✅ ", ""))
        elif "Visualization step complete" in m:
            set_stage("viz", "success", "Charts generated")

        # Errors / info
        elif m.startswith("❌ Execution error"):
            set_stage("execute", "error", m.replace("❌ ", ""))
        elif m.startswith("ℹ️ Visualization skipped"):
            set_stage("viz", "info", "No suitable data")
        # All other debug noise is ignored to keep the panel clean.

with right_col:
    # ---------- Schema with scrollbar (space saver) ----------
    st.subheader("🔎 Schema (auto-detected)")
    if "schema_text" in st.session_state:
        #schema_html = f"<div class='schema-scroll'><pre>{html.escape(st.session_state['schema_text'])}</pre></div>"
        #st.markdown(schema_html, unsafe_allow_html=True)
        st.text(st.session_state['schema_text'])
    else:
        st.info("Connect a database to see schema.")

    st.markdown("---")
    st.subheader("🧾 SQL & Results")

    # ---- View switch (acts like tabs) + persistence ----
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Results"
    view = st.radio(
        "View",
        ["Results", "Visualizations"],
        horizontal=True,
        index=(0 if st.session_state["active_view"] == "Results" else 1),
        label_visibility="collapsed",
    )
    st.session_state["active_view"] = view

    sql_display = st.empty()
    result_display = st.empty()
    chart_display = st.empty()

# -------------------------
# Main action: Generate & Run
# -------------------------
if run_clicked:
    if "engine" not in st.session_state or "schema" not in st.session_state:
        st.error("Please connect a database first.")
    elif not question.strip():
        st.warning("Please type a natural language query.")
    else:
        engine = st.session_state["engine"]
        schema = st.session_state["schema"]

        # Resolve API key priority: UI > st.session_state > st.secrets > env
        resolved_key = None
        if llm_api_key and llm_api_key.strip():
            resolved_key = llm_api_key.strip()
            st.session_state["GOOGLE_API_KEY"] = resolved_key
        elif "GOOGLE_API_KEY" in st.session_state:
            resolved_key = st.session_state["GOOGLE_API_KEY"]
        else:
            resolved_key = st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None
            if not resolved_key:
                resolved_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not resolved_key:
            st.error("Gemini API key not found. Provide it in the input field, set st.secrets['GOOGLE_API_KEY'], or export GOOGLE_API_KEY.")
            st.stop()

        # Initialize agents with the resolved key and lifecycle emitters
        nl2sql = NL2SQLAgent(schema, llm_api_key=resolved_key, emit=lambda m: emit(f"NL2SQL: {m}"))
        accuracy = AccuracyAgent(schema, llm_api_key=resolved_key, emit=lambda m: emit(f"Validator: {m}"))
        executor = SQLExecutorAgent(engine)
        vis = VisualizationAgent()

        # Create Crew-like orchestrator with lifecycle emitter
        crew = create_crew(nl2sql, accuracy, executor, vis, question, emit=emit)

        # Initialize stages
        set_stage("received", "info", question)
        set_stage("nl2sql",   "info", "Pending")
        set_stage("validate", "info", "Pending")
        set_stage("execute",  "info", "Pending")
        set_stage("viz",      "info", "Pending")

        try:
            # Run the crew (returns dict with sql, df, fig, rationale, duration, and optionally 'figs')
            result = crew.kickoff()

            # Extract results
            sql = result.get("sql")
            df = result.get("df")
            fig = result.get("fig")
            figs = result.get("figs", [])
            rationale = result.get("rationale", [])
            duration = float(result.get("duration", 0.0))

            # Compact final statuses based on result
            # 1) NL→SQL
            if sql:
                set_stage("nl2sql", "success", "SQL generated")
            else:
                set_stage("nl2sql", "error", "Could not generate SQL")

            # 2) Validation (infer from rationale if available; else infer from execution)
            val_line = next((r for r in rationale if "Validation:" in r), "")
            if "VALID" in val_line.upper():
                set_stage("validate", "success", "SQL validated")
            elif "INVALID" in val_line.upper():
                set_stage("validate", "error", "Invalid SQL")
            else:
                if df is not None:
                    set_stage("validate", "success", "SQL validated")
                else:
                    set_stage("validate", "info", "Unknown (see logs)")

            # 3) Execution
            if df is not None and isinstance(df, pd.DataFrame):
                set_stage("execute", "success", f"{len(df)} rows in {duration:.3f}s")
            else:
                set_stage("execute", "error", "Failed")

            # 4) Visualization status only (render controlled by view switch below)
            if figs:
                set_stage("viz", "success", f"{len(figs)} chart(s) available")
            elif fig is not None:
                set_stage("viz", "success", "1 chart available")
            else:
                set_stage("viz", "info", "No charts")

            # Render SQL
            if sql:
                sql_display.code(sql, language="sql")
            else:
                st.error("❌ No valid SQL generated.")
                try:
                    log_query(
                        user=st.session_state["user"],
                        role="basic",
                        user_q=question,
                        sql="N/A",
                        status="failed",
                        rows=0,
                        duration=duration,
                    )
                except Exception:
                    pass
                result_display.empty()
                chart_display.empty()
                st.stop()

            # Persist outputs so user can switch view after rerun
            st.session_state["last_df"] = df if isinstance(df, pd.DataFrame) else None
            st.session_state["last_figs"] = figs if figs else None
            st.session_state["last_fig"] = fig if fig is not None else None

            # Render depending on active view
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                if st.session_state["active_view"] == "Results":
                    result_display.dataframe(df, use_container_width=True)
                    try:
                        log_query(
                            user=st.session_state["user"],
                            role="basic",
                            user_q=question,
                            sql=sql,
                            status="success",
                            rows=len(df),
                            duration=duration,
                        )
                    except Exception:
                        pass

                    # If charts exist, show a button to jump to the Visualizations view
                    has_charts = bool(figs) or (fig is not None)
                    if has_charts:
                        if st.button("📊 Click to view visualizations", key="btn_view_viz"):
                            st.session_state["active_view"] = "Visualizations"
                            st.experimental_rerun()
                else:
                    # Visualizations view
                    chart_display.empty()  # clear old
                    if st.session_state.get("last_figs"):
                        charts = st.session_state["last_figs"]
                        cols = st.columns(min(3, len(charts)))
                        for i, (f, caption) in enumerate(charts):
                            with cols[i % len(cols)]:
                                st.plotly_chart(f, use_container_width=True)
                                if caption:
                                    st.caption(caption)
                    elif st.session_state.get("last_fig") is not None:
                        st.plotly_chart(st.session_state["last_fig"], use_container_width=True)
                    else:
                        st.info("No suitable data for visualization.")
            else:
                st.error("No data returned from query.")
                try:
                    log_query(
                        user=st.session_state["user"],
                        role="basic",
                        user_q=question,
                        sql=sql if sql else "N/A",
                        status="failed",
                        rows=0,
                        duration=duration,
                    )
                except Exception:
                    pass

            # 5) Summary
            rows = len(df) if isinstance(df, pd.DataFrame) else 0
            charts = len(figs) or (1 if fig is not None else 0)
            set_stage("summary", "success", f"Done — rows: {rows}, time: {duration:.3f}s, charts: {charts}")

        except Exception as e:
            msg = str(e)
            if "ResourceExhausted" in msg or "429" in msg:
                set_stage("nl2sql", "error", "Gemini quota exhausted")
                st.warning(
                    "Gemini free-tier quota exhausted. Try again later, switch model (e.g., "
                    "`gemini-1.5-flash-8b`), or add billing to your API key."
                )
            else:
                set_stage("summary", "error", f"Workflow failed: {msg}")
                st.error(f"Workflow failed: {msg}")
            # Robust logging even on error
            try:
                log_query(
                    user=st.session_state.get("user", "unknown"),
                    role="basic",
                    user_q=question,
                    sql="N/A",  # fallback if sql is undefined
                    status="failed",
                    rows=0,
                    duration=0.0,
                )
            except Exception:
                pass

# -------------------------
# Persisted view rendering when not running (user toggled the view)
# -------------------------
if "last_df" in st.session_state:
    df = st.session_state.get("last_df")
    if st.session_state.get("active_view") == "Results":
        if isinstance(df, pd.DataFrame) and not df.empty:
            result_display.dataframe(df, use_container_width=True)
    elif st.session_state.get("active_view") == "Visualizations":
        if st.session_state.get("last_figs"):
            charts = st.session_state["last_figs"]
            cols = st.columns(min(3, len(charts)))
            for i, (f, caption) in enumerate(charts):
                with cols[i % len(cols)]:
                    st.plotly_chart(f, use_container_width=True)
                    if caption:
                        st.caption(caption)
        elif st.session_state.get("last_fig") is not None:
            st.plotly_chart(st.session_state["last_fig"], use_container_width=True)

# -------------------------
# Audit log (bottom)
# -------------------------
st.markdown("---")
st.subheader("📜 Audit Log (recent)")
audit_df = fetch_audit(limit=100)
st.dataframe(audit_df, use_container_width=True)
