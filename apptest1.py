# app1.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
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
        # Use NL2SQLAgent to suggest queries (leverages LangChain reasoning)
        nl2sql = NL2SQLAgent(st.session_state["schema"])
        suggestions = nl2sql.suggest_queries()
        for q in suggestions:
            st.markdown(f"- {q}")

    st.subheader("📝 Ask a Query")
    question = st.text_area("Type your natural language query", height=120)
    llm_api_key = st.text_input("LLM API key (optional)", type="password")
    run_clicked = st.button("Generate & Run SQL")

    # Agent lifecycle expander
    lifecycle_exp = st.expander("🛠️ Agent Lifecycle (live)")
    lifecycle_placeholder = lifecycle_exp.empty()

with right_col:
    st.subheader("🔎 Schema (auto-detected)")
    if "schema_text" in st.session_state:
        st.code(st.session_state["schema_text"], language="markdown")
    else:
        st.info("Connect a database to see schema.")

    st.markdown("---")
    st.subheader("🧾 SQL & Results")
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

        # Initialize agents
        nl2sql = NL2SQLAgent(schema, llm_api_key=llm_api_key or None)
        accuracy = AccuracyAgent(schema)
        executor = SQLExecutorAgent(engine)
        vis = VisualizationAgent()

        # Create CrewAI crew
        crew = create_crew(nl2sql, accuracy, executor, vis, question)

        # Lifecycle steps display
        steps = []
        def emit(step):
            steps.append(step)
            lifecycle_placeholder.write("\n".join(steps))

        emit("➡️ Received query: " + question)
        emit("🧭 Starting multi-agent workflow...")

        try:
            # Run the crew
            result = crew.kickoff()

            # Extract results from crew output
            sql = result.get("sql")
            df = result.get("df")
            fig = result.get("fig")
            rationale = result.get("rationale", [])

            # Display lifecycle steps from crew
            for step in rationale:
                emit(step)

            if sql:
                sql_display.code(sql, language="sql")
            else:
                st.error("❌ No valid SQL generated.")
                log_query(user=st.session_state["user"], role="basic", user_q=question,
                          sql="N/A", status="failed", rows=0, duration=0.0)
                result_display.empty()
                chart_display.empty()
                st.stop()

            if df is not None and not df.empty:
                emit(f"✅ Execution successful ({len(df)} rows)")
                result_display.dataframe(df, use_container_width=True)
                log_query(user=st.session_state["user"], role="basic", user_q=question,
                          sql=sql, status="success", rows=len(df), duration=0.0)
                st.session_state["last_df"] = df

                if fig:
                    emit(f"📊 Visualization generated")
                    chart_display.plotly_chart(fig, use_container_width=True)
                else:
                    emit("ℹ️ No visualization generated")
                    chart_display.info("No suitable data for visualization.")
            else:
                emit("❌ No data returned")
                st.error("No data returned from query.")
                log_query(user=st.session_state["user"], role="basic", user_q=question,
                          sql=sql, status="failed", rows=0, duration=0.0)

        except Exception as e:
            emit(f"❌ Error in workflow: {str(e)}")
            st.error(f"Workflow failed: {str(e)}")
            log_query(user=st.session_state["user"], role="basic", user_q=question,
                      sql=sql or "N/A", status="failed", rows=0, duration=0.0)

# -------------------------
# Audit log (bottom)
# -------------------------
st.markdown("---")
st.subheader("📜 Audit Log (recent)")
audit_df = fetch_audit(limit=100)
st.dataframe(audit_df, use_container_width=True)