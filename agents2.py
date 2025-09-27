# agents2.py
import os
import re
import time
from typing import Any, Dict, Optional, Tuple, List, Callable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import inspect, text

from db import get_schema, is_safe_sql  # keep the import names you used
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType, AgentExecutor

# ✅ Gemini via LangChain (no OpenAI)
from langchain_google_genai import ChatGoogleGenerativeAI

# CrewAI
from crewai import Agent as CrewAgent, Task, Crew


# -------------------------
# Configuration knobs
# -------------------------
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change via env if needed
GEMINI_MODEL = DEFAULT_GEMINI_MODEL
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "0"))  # avoid long 33s backoff loops
ENABLE_SQL_REPAIR = os.getenv("ENABLE_SQL_REPAIR", "false").lower() == "true"  # default OFF for speed/quota


def _strip_code_fences(s: str) -> str:
    """Strip ```sql ...``` or ``` ...``` blocks safely."""
    s = (s or "").strip()
    if s.startswith("```"):
        # remove leading fence (and optional 'sql')
        s = s[3:]
        s = s.lstrip("`").lstrip()
        if s.lower().startswith("sql"):
            s = s[3:]
        s = s.strip()
        # remove trailing fence
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _resolve_gemini_key(llm_api_key: Optional[str] = None) -> str:
    """Priority: explicit argument > GOOGLE_API_KEY > GEMINI_API_KEY; else raise."""
    key = llm_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required for Gemini.")
    return key


# ------------------------------------------------------
# SchemaAgent (unchanged)
# ------------------------------------------------------
class SchemaAgent:
    def __init__(self, engine, schema=None):
        self.engine = engine
        if schema:
            self.schema = schema
        else:
            self.schema = self._introspect()

    def _introspect(self):
        insp = inspect(self.engine)
        schema: Dict[str, Any] = {}
        for table in insp.get_table_names():
            cols = insp.get_columns(table)
            fks = insp.get_foreign_keys(table)
            pk = insp.get_pk_constraint(table)
            schema[table] = {
                "columns": cols,
                "foreign_keys": fks,
                "primary_key": pk.get("constrained_columns", []),
            }
        return schema


# ------------------------------------------------------
# NL2SQLAgent — Gemini + strong prompt + (optional) self-repair + live emits
# ------------------------------------------------------
class NL2SQLAgent:
    def __init__(
        self,
        schema: Dict[str, Any],
        llm_api_key: Optional[str] = None,
        emit: Optional[Callable[[str], None]] = None,
    ):
        self.schema = schema
        self.schema_text = self._format_schema()
        api_key = _resolve_gemini_key(llm_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
            max_retries=GEMINI_MAX_RETRIES,  # stop long backoff
        )
        # We keep a ReAct agent object for “framework parity”, but we’ll use explicit prompts for reliability.
        self.agent = self._create_agent()
        self._emit = emit or (lambda *_: None)

    def _format_schema(self) -> str:
        lines = []
        for table, meta in self.schema.items():
            cols = [f"{c['name']} ({c.get('type')})" for c in meta.get("columns", [])]
            lines.append(f"{table}: {', '.join(cols)}")
        return "\n".join(lines)

    def _create_agent(self) -> AgentExecutor:
        tools = [
            Tool(
                name="Schema Lookup",
                func=lambda _: self.schema_text,
                description="Retrieve the database schema.",
            )
        ]
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )

    # ---- Zero‑LLM rule‑based fast path for trivial prompts ----
    def _try_rule_based_sql(self, user_query: str) -> Optional[str]:
        """
        Fast path for simple prompts, avoids calling the LLM:
          - "show col1, col2 from Table"
          - "show * from Table" / "show all from Table"
          - "count rows in Table" / "how many rows in Table"
        """
        q = (user_query or "").strip().lower()

        # 1) show X, Y from T
        m = re.match(r"^show\s+(.+?)\s+from\s+([a-zA-Z0-9_]+)$", q)
        if m:
            cols_part = m.group(1).strip()
            tbl_lc = m.group(2).strip().lower()

            # resolve table casing from schema
            table = next((t for t in self.schema.keys() if t.lower() == tbl_lc), None)
            if not table:
                return None

            if cols_part in ("*", "all", "all columns"):
                cols = "*"
            else:
                req_cols = [c.strip(" ,") for c in cols_part.split(",")]
                schema_cols = {c["name"]: c for c in self.schema[table].get("columns", [])}
                out_cols = []
                for rc in req_cols:
                    match = next((name for name in schema_cols if name.lower() == rc.lower()), None)
                    if not match:
                        return None
                    out_cols.append(match)
                cols = ", ".join(out_cols)

            sql = f"SELECT {cols} FROM {table}"
            return sql if is_safe_sql(sql) else None

        # 2) count rows in T
        m = re.match(r"^(count|how many)\s+rows\s+in\s+([a-zA-Z0-9_]+)$", q)
        if m:
            tbl_lc = m.group(2).strip().lower()
            table = next((t for t in self.schema.keys() if t.lower() == tbl_lc), None)
            if not table:
                return None
            sql = f"SELECT COUNT(*) AS RowCount FROM {table}"
            return sql if is_safe_sql(sql) else None

        return None

    def _sql_prompt(self, user_query: str) -> str:
        return f"""
You are an expert SQLite SQL generator.
Return exactly ONE complete, syntactically correct SELECT query that answers the user's question.
STRICT RULES:
- Use ONLY the following schema (table/column names exactly as-is).
- DO NOT use CTEs (no WITH), subqueries only if necessary.
- Always include a FROM clause.
- If you use aggregate functions (COUNT, SUM, AVG, MIN, MAX), include a proper GROUP BY over all non-aggregated selected columns.
- For questions about "most", "top", "highest", "largest", include ORDER BY <metric> DESC and LIMIT 1 (or LIMIT 10 if the question says "top N").
- Prefer including human-friendly fields (e.g., names) from related tables via JOINs when appropriate.
- Do NOT include backticks, comments, explanations, or extra lines. Return ONLY the SQL on one line with NO trailing semicolon.

SCHEMA:
{self.schema_text}

USER QUESTION:
{user_query}

OUTPUT:
<SQL only, one line, no trailing semicolon>
""".strip()

    def _refine_prompt(self, user_query: str, feedback: str) -> str:
        return f"""
Your previous SQL draft was invalid because:
{feedback}

Using the same STRICT RULES and the schema below, regenerate a single valid, complete SELECT query.
Return ONLY the SQL on one line with NO trailing semicolon.

SCHEMA:
{self.schema_text}

USER QUESTION:
{user_query}

OUTPUT:
<SQL only, one line, no trailing semicolon>
""".strip()

    def _postcheck_sql(self, sql: Optional[str]) -> Optional[str]:
        if not sql:
            return None
        s = sql.strip()
        if not s.lower().startswith("select"):
            return None
        if " from " not in f" {s.lower()} ":
            return None
        return s

    def to_sql(self, user_query: str) -> Tuple[Optional[str], str, str]:
        # 🚀 Fast path: avoid LLM for trivial prompts
        rb = self._try_rule_based_sql(user_query)
        if rb:
            self._emit("rule-based SQL generated (no LLM)")
            return rb, "Rule-based fallback", "rule"

        try:
            self._emit("prompting Gemini for SQL…")
            prompt = self._sql_prompt(user_query)
            msg = self.llm.invoke(prompt)
            raw = (getattr(msg, "content", None) or str(msg)).strip()
            out = _strip_code_fences(raw)
            # Take the first line starting with SELECT
            sql = None
            for ln in out.splitlines():
                ln = ln.strip().rstrip(";")
                if ln.lower().startswith("select"):
                    sql = ln
                    break

            sql = self._postcheck_sql(sql)
            if not sql:
                self._emit("no valid SELECT (missing FROM or malformed)")
                return None, "Failed to generate valid SELECT statement.", "gemini"

            self._emit(f"candidate SQL: {sql[:160]}{'…' if len(sql) > 160 else ''}")

            if not is_safe_sql(sql):
                self._emit("SQL rejected by safety checks")
                return None, "Failed safety checks (single SELECT only).", "gemini"

            self._emit("SQL passed safety checks")
            return sql, "Generated by Gemini (guided prompt).", "gemini"

        except Exception as e:
            self._emit(f"error: {e}")
            return None, f"Gemini error: {e}", "gemini"

    def refine_sql(self, user_query: str, feedback: str) -> Tuple[Optional[str], str]:
        """Refinement pass after validator feedback."""
        try:
            self._emit("refining SQL based on validator feedback…")
            prompt = self._refine_prompt(user_query, feedback)
            msg = self.llm.invoke(prompt)
            raw = (getattr(msg, "content", None) or str(msg)).strip()
            out = _strip_code_fences(raw)
            sql = None
            for ln in out.splitlines():
                ln = ln.strip().rstrip(";")
                if ln.lower().startswith("select"):
                    sql = ln
                    break
            sql = self._postcheck_sql(sql)
            if not sql or not is_safe_sql(sql):
                return None, "Refinement failed to produce a safe, valid SELECT."
            return sql, "Refined SQL generated by Gemini."
        except Exception as e:
            return None, f"Refinement error: {e}"


# ------------------------------------------------------
# AccuracyAgent — Gemini validator with live emits
# ------------------------------------------------------
class AccuracyAgent:
    def __init__(
        self,
        schema: Dict[str, Any],
        llm_api_key: Optional[str] = None,
        emit: Optional[Callable[[str], None]] = None,
    ):
        self.schema = schema
        self.schema_text = self._format_schema()
        api_key = _resolve_gemini_key(llm_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
            max_retries=GEMINI_MAX_RETRIES,
        )
        self._emit = emit or (lambda *_: None)

    def _format_schema(self) -> str:
        lines = []
        for table, meta in self.schema.items():
            cols = [f"{c['name']} ({c.get('type')})" for c in meta.get("columns", [])]
            lines.append(f"{table}: {', '.join(cols)}")
        return "\n".join(lines)

    def validate(self, user_query: str, sql: str) -> Tuple[bool, str]:
        prompt = f"""
You are validating a SQL query for SQLite.

SCHEMA:
{self.schema_text}

USER QUERY:
{user_query}

SQL:
{sql}

Return exactly one of:
- VALID
- INVALID: <one-sentence reason>

Rules:
- Must be a single SELECT (no multiple statements, no DML/DDL).
- Must use only tables/columns in schema and match user intent.
- If aggregates are used, GROUP BY must be present for non-aggregated selected columns.
- Prefer JOINs to include human-friendly columns when appropriate.
""".strip()
        try:
            self._emit("validating SQL with Gemini…")
            msg = self.llm.invoke(prompt)
            out = (getattr(msg, "content", None) or str(msg)).strip()
            self._emit(f"validator output: {out}")
            if out.upper().startswith("VALID"):
                return True, "SQL validated by Gemini."
            return False, f"INVALID: {out[7:].strip()}" if out.upper().startswith("INVALID") else f"Validation failed: {out}"
        except Exception as e:
            self._emit(f"error: {e}")
            return False, f"Validation error: {e}"


# ------------------------------------------------------
# SQLExecutorAgent
# ------------------------------------------------------
class SQLExecutorAgent:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, sql: str) -> Tuple[pd.DataFrame, float]:
        if not sql:
            raise ValueError("No SQL was provided.")
        if not is_safe_sql(sql):
            raise ValueError("Query rejected as unsafe by is_safe_sql.")
        if not sql.lower().startswith("select"):
            raise ValueError("Only SELECT queries allowed.")

        start = time.perf_counter()
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        dur = time.perf_counter() - start
        return df, dur


# ------------------------------------------------------
# VisualizationAgent — return up to 3 relevant Plotly figures
# ------------------------------------------------------
class VisualizationAgent:
    def generate_chart(self, df: pd.DataFrame) -> Tuple[Optional[Any], str]:
        """Backward-compatible: returns a single best-effort chart."""
        charts = self.generate_charts(df, max_charts=1)
        if charts:
            return charts[0]
        return None, "No suitable data for visualization."

    def generate_charts(self, df: pd.DataFrame, max_charts: int = 3) -> List[Tuple[Any, str]]:
        figs: List[Tuple[Any, str]] = []
        if df is None or df.empty:
            return figs

        # Basic typing
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(exclude=["number"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

        # Try to parse any date-like columns if not already parsed
        if not datetime_cols:
            for col in df.columns:
                if df[col].dtype == object:
                    try:
                        parsed = pd.to_datetime(df[col], errors="raise", utc=False, infer_datetime_format=True)
                        if parsed.notna().mean() > 0.8:
                            df[col] = parsed
                            datetime_cols.append(col)
                    except Exception:
                        pass
        # 1) If categorical + numeric → bar
        if categorical and numeric:
            x = categorical[0]
            y = numeric[0]
            # If too many categories, show top 10 by y if present
            ddf = df.sort_values(by=y, ascending=False).head(10) if y in df.columns else df
            fig1 = px.bar(ddf, x=x, y=y, title=f"{y} by {x}")
            figs.append((fig1, f"Bar chart of {y} by {x} (top 10 if applicable)"))

            # 2) Pie (only if enough categories)
            if len(ddf) >= 3 and y in ddf.columns:
                fig2 = px.pie(ddf, names=x, values=y, title=f"Share of {y} by {x}")
                figs.append((fig2, f"Pie: Share of {y} by {x}"))

        # 3) If datetime present + numeric → line over time (choose first)
        if datetime_cols and numeric:
            t = datetime_cols[0]
            y = numeric[0]
            try:
                ts = df[[t, y]].dropna().copy()
                ts = ts.groupby(ts[t].dt.to_period("M"))[y].sum().reset_index()
                ts[t] = ts[t].dt.to_timestamp()
                fig3 = px.line(ts, x=t, y=y, title=f"{y} over time ({t})")
                figs.append((fig3, f"Trend of {y} over time"))
            except Exception:
                pass

        # 4) If only numeric → histogram
        if not categorical and len(numeric) >= 1 and len(figs) < max_charts:
            y = numeric[0]
            fig4 = px.histogram(df, x=y, title=f"Distribution of {y}")
            figs.append((fig4, f"Histogram of {y}"))

        # 5) Always add a table figure as a last resort (helps when df has 1–2 rows)
        if len(figs) < max_charts:
            fig5 = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns), fill_color='lightgrey', align='left'),
                cells=dict(values=[df[c] for c in df.columns], align='left')
            )])
            fig5.update_layout(title="Result Table")
            figs.append((fig5, "Result table"))

        return figs[:max_charts]


# ------------------------------------------------------
# CrewAI Orchestration — keep roles/tasks; optional self-repair
# ------------------------------------------------------
class _SimpleCrew:
    def __init__(
        self,
        nl2sql: NL2SQLAgent,
        accuracy: AccuracyAgent,
        executor: SQLExecutorAgent,
        vis: VisualizationAgent,
        user_query: str,
        emit: Optional[Callable[[str], None]] = None,
    ):
        self.nl2sql = nl2sql
        self.accuracy = accuracy
        self.executor = executor
        self.vis = vis
        self.user_query = user_query
        self._emit = emit or (lambda *_: None)

        # CrewAI agents (for explainability / future autonomous use)
        self.nl2sql_agent = CrewAgent(
            role="NL-to-SQL Converter",
            goal="Convert natural language queries to valid SQL SELECT statements.",
            backstory="Expert in translating business questions into precise SQL queries using Gemini.",
            verbose=True,
            allow_delegation=False
        )
        self.validator_agent = CrewAgent(
            role="SQL Validator",
            goal="Ensure generated SQL is accurate, safe, and aligned with user intent.",
            backstory="Specialist in validating SQL queries for correctness and alignment with user intent.",
            verbose=True,
            allow_delegation=False
        )
        self.executor_agent = CrewAgent(
            role="SQL Executor",
            goal="Execute SQL queries safely and return results.",
            backstory="Expert in secure database interactions, ensuring only safe SELECT queries are executed.",
            verbose=True,
            allow_delegation=False
        )
        self.viz_agent = CrewAgent(
            role="Visualization Generator",
            goal="Create interactive visualizations from query results.",
            backstory="Transforms data into insightful Plotly charts.",
            verbose=True,
            allow_delegation=False
        )

        self.tasks = [
            Task(
                description=f"Convert the user query '{user_query}' into a valid single SELECT.",
                agent=self.nl2sql_agent,
                expected_output="A valid SQL SELECT or 'None'.",
            ),
            Task(
                description="Validate SQL against schema and user intent. Output VALID or INVALID:<reason>.",
                agent=self.validator_agent,
                expected_output="VALID or INVALID:<reason>",
            ),
            Task(
                description="Execute the SQL and return rows as a table.",
                agent=self.executor_agent,
                expected_output="Tabular result",
            ),
            Task(
                description="Generate 2–3 relevant Plotly visualizations if possible.",
                agent=self.viz_agent,
                expected_output="Multiple charts or explanation.",
            ),
        ]

        self.crew = Crew(
            agents=[self.nl2sql_agent, self.validator_agent, self.executor_agent, self.viz_agent],
            tasks=self.tasks,
            verbose=True,
        )

    def kickoff(self) -> Dict[str, Any]:
        rationale: List[str] = []
        sql = None
        df = None
        figs: List[Tuple[Any, str]] = []
        primary_fig = None
        total_duration = 0.0

        # Step 1: NL → SQL
        self._emit("🧠 NL2SQL started")
        sql, r1, _ = self.nl2sql.to_sql(self.user_query)
        rationale.append(f"🧠 NL2SQL: {r1}")
        if not sql:
            self._emit("🛑 NL2SQL failed (no SQL)")
            return {"sql": None, "df": None, "fig": None, "figs": [], "rationale": rationale, "duration": total_duration}

        # Step 2: Validate
        self._emit("🔎 Validation started")
        is_valid, r2 = self.accuracy.validate(self.user_query, sql)
        rationale.append(f"✅ Validation: {r2}")
        if not is_valid:
            if ENABLE_SQL_REPAIR:
                # 🔁 Self-repair attempt using validator feedback
                self._emit("🔁 Attempting SQL repair…")
                sql2, refine_msg = self.nl2sql.refine_sql(self.user_query, r2)
                rationale.append(f"🔁 Repair: {refine_msg}")
                if sql2:
                    sql = sql2
                    self._emit("🔁 Re-validating repaired SQL…")
                    is_valid2, r3 = self.accuracy.validate(self.user_query, sql)
                    rationale.append(f"🔁 Re-Validation: {r3}")
                    if not is_valid2:
                        self._emit("🛑 Repair failed")
                        return {"sql": sql, "df": None, "fig": None, "figs": [], "rationale": rationale, "duration": total_duration}
                else:
                    self._emit("🛑 Repair failed (no SQL)")
                    return {"sql": sql, "df": None, "fig": None, "figs": [], "rationale": rationale, "duration": total_duration}
            else:
                self._emit("🛑 Validation failed — repair disabled to save quota")
                return {"sql": sql, "df": None, "fig": None, "figs": [], "rationale": rationale, "duration": total_duration}

        # Step 3: Execute
        try:
            self._emit("🏃 Executing SQL…")
            df, dur = self.executor.execute(sql)
            total_duration += dur
            self._emit(f"✅ Execution done: {0 if df is None else len(df)} rows in {dur:.3f}s")
            rationale.append(f"🗄️ Execution: returned {0 if df is None else len(df)} rows in {dur:.3f}s")
        except Exception as e:
            self._emit(f"❌ Execution error: {e}")
            rationale.append(f"❌ Execution error: {e}")
            return {"sql": sql, "df": None, "fig": None, "figs": [], "rationale": rationale, "duration": total_duration}

        # Step 4: Visualization — up to 3 charts
        try:
            self._emit("📊 Building visualizations…")
            viz_list = self.vis.generate_charts(df, max_charts=3)
            if viz_list:
                primary_fig = viz_list[0][0]
                figs = viz_list  # list of (fig, caption)
                self._emit("✅ Visualization step complete")
                for _, cap in viz_list:
                    rationale.append(f"📊 Visualization: {cap}")
            else:
                rationale.append("ℹ️ No suitable data for visualization.")
                self._emit("ℹ️ No visualization produced")
        except Exception as e:
            self._emit(f"ℹ️ Visualization skipped: {e}")
            rationale.append(f"ℹ️ Visualization skipped: {e}")

        return {"sql": sql, "df": df, "fig": primary_fig, "figs": figs, "rationale": rationale, "duration": total_duration}


def create_crew(
    nl2sql: NL2SQLAgent,
    accuracy: AccuracyAgent,
    executor: SQLExecutorAgent,
    vis: VisualizationAgent,
    user_query: str,
    emit: Optional[Callable[[str], None]] = None,
):
    """Create a CrewAI-backed wrapper that returns a dict result for the UI."""
    return _SimpleCrew(nl2sql, accuracy, executor, vis, user_query, emit=emit)
