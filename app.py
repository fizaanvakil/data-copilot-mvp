# app.py
import re
import io
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Data Copilot MVP", page_icon="📊", layout="wide")

st.title("📊 Data Copilot MVP")
st.caption("Upload a CSV/Excel, ask a question in plain English or SQL, get an answer + chart. Runs locally in this session.")

# --------- Sidebar ----------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "- **Step 1:** Upload a CSV or Excel file\n"
        "- **Step 2:** Type a question (e.g., 'average price by city, top 10')\n"
        "- **Step 3:** See results + chart\n"
        "- Toggle **SQL mode** to write SQL directly\n"
    )
    st.divider()
    sql_mode = st.toggle("Advanced: SQL mode (write SQL yourself)", value=False)

# --------- File upload ----------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload a data file to begin.", icon="🗂️")
    st.stop()

# Read file
try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# --- CLEANUP PATCH FOR STREAMLIT + PYARROW ---
# Fix column names (remove line breaks, extra spaces)
df.columns = [str(c).strip().replace("\n", " ").replace("\r", " ") for c in df.columns]

# Convert object/mixed columns to string (PyArrow bug workaround)
obj_cols = df.select_dtypes(include=["object"]).columns
if len(obj_cols) > 0:
    df[obj_cols] = df[obj_cols].astype(str)

# Replace NaN with None so Arrow/Streamlit can handle them
df = df.where(pd.notna(df), None)
# --- END PATCH ---

if df.empty:
    st.warning("Your file seems empty.")
    st.stop()


# Clean column names (spaces -> underscores)
df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

# Show preview + schema
with st.expander("Preview & schema", expanded=False):
    st.write(df.head(20))
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes]
    })
    st.caption("Detected schema:")
    st.dataframe(schema, use_container_width=True)

# Register dataframe in DuckDB
con = duckdb.connect()
con.register("t", df)

# --------- Helper: simple NL -> SQL ---------
def nl_to_sql(question: str, columns: list[str]) -> str | None:
    """
    Very simple keyword-based translator for common questions.
    Examples it can handle:
      - 'average price by city top 10'
      - 'sum of sales by month'
      - 'count rows'
      - 'max revenue where region = EMEA'
      - 'min age'
      - 'count by category'
    Supports:
      agg: sum/average/mean/max/min/count
      group by: 'by <col>'
      limit: 'top N'
      filter equals: 'where <col> = <value>'  (value without spaces or quoted "multi word")
    """
    q = question.lower().strip()

    # detect aggregation
    agg = None
    for word, fn in [
        ("average", "avg"), ("mean", "avg"), ("avg", "avg"),
        ("sum", "sum"), ("total", "sum"),
        ("max", "max"), ("min", "min"),
        ("count", "count")
    ]:
        if re.search(rf"\b{word}\b", q):
            agg = fn
            break

    # detect measure (first numeric-ish column if none explicitly hinted)
    measure = None
    # try to find a column name in the text
    for c in columns:
        if re.search(rf"\b{re.escape(c.lower())}\b", q):
            measure = c
            break

    # if we saw 'count' without a measure, use *
    if agg == "count" and measure is None:
        measure = "*"

    # fallback: choose first numeric column if agg needs one
    if measure is None and agg in {"avg", "sum", "max", "min"}:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            measure = numeric_cols[0]

    # group by
    groupby = None
    m = re.search(r"\bby\s+([a-zA-Z0-9_]+)\b", q)
    if m:
        groupby = m.group(1)

    # simple equality filter: where <col> = <val>  (val can be bareword or "quoted string")
    where_col = where_val = None
    m = re.search(r'\bwhere\s+([a-zA-Z0-9_]+)\s*=\s*("[^"]+"|\'[^\']+\'|[^\s]+)', q)
    if m:
        where_col = m.group(1)
        where_val_raw = m.group(2)
        where_val = where_val_raw.strip()
        # if not quoted and not numeric, quote it
        if not (where_val.startswith("'") or where_val.startswith('"')):
            if not re.match(r"^-?\d+(\.\d+)?$", where_val):
                where_val = f"'{where_val}'"

    # limit (top N)
    limit = None
    m = re.search(r"\btop\s+(\d+)\b", q)
    if m:
        limit = int(m.group(1))

    # Build SQL
    if agg and measure:
        if measure == "*":
            select_expr = f"{agg}({measure}) as value"
        else:
            select_expr = f"{agg}({measure}) as value"

        sql = f"SELECT {select_expr}"
        sql += " FROM t"
        if where_col and where_val:
            sql += f" WHERE {where_col} = {where_val}"
        if groupby:
            sql = f"SELECT {groupby} as group_key, {select_expr} FROM t"
            if where_col and where_val:
                sql += f" WHERE {where_col} = {where_val}"
            sql += f" GROUP BY group_key"
            # order by value desc if agg is sum/avg/count
            if agg in {"sum", "avg", "count"}:
                sql += " ORDER BY value DESC"
        if limit:
            sql += f" LIMIT {limit}"
        return sql

    # if user asked for 'count rows'
    if "count rows" in q or q.strip() == "count":
        return "SELECT COUNT(*) AS value FROM t"

    # try a plain select for safety
    return None

# --------- Question input ----------
default_hint = "e.g., average Price by City top 10  •  or toggle SQL mode and write SQL"
question = st.text_input("Ask a question (natural language)", placeholder=default_hint) if not sql_mode else st.text_area("Write a SQL query (DuckDB)", height=120)

run = st.button("Run")

if run:
    try:
        if sql_mode:
            sql = question.strip()
            if not sql:
                st.warning("Please enter a SQL query.")
                st.stop()
        else:
            sql = nl_to_sql(question, list(df.columns))
            if sql is None:
                st.warning("I couldn't translate that question. Try simpler phrasing, or toggle SQL mode.")
                st.stop()

        st.code(sql, language="sql")
        result = con.execute(sql).fetchdf()

        if result.empty:
            st.info("No rows returned.")
        else:
            # show table
            st.dataframe(result, use_container_width=True)

            # auto-chart: if 2 columns and one is numeric, draw bar chart
            if result.shape[1] == 2:
                col_types = [pd.api.types.is_numeric_dtype(result.iloc[:, i]) for i in range(2)]
                # choose x as non-numeric key, y as numeric value when available
                if col_types[0] and not col_types[1]:
                    xcol, ycol = result.columns[1], result.columns[0]
                elif not col_types[0] and col_types[1]:
                    xcol, ycol = result.columns[0], result.columns[1]
                else:
                    xcol, ycol = result.columns[0], result.columns[1]

                fig, ax = plt.subplots()
                ax.bar(result[xcol].astype(str), result[ycol])
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.set_title("Auto chart")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig)

            elif result.shape[1] == 1 and pd.api.types.is_numeric_dtype(result.iloc[:, 0]):
                st.metric(label=result.columns[0], value=float(result.iloc[0, 0]))

            # export
            buf = io.BytesIO()
            result.to_csv(buf, index=False)
            st.download_button("Download results (CSV)", data=buf.getvalue(), file_name="results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Query failed: {e}")
