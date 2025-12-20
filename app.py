import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sentiment_classifier import analyze_text_rows, summarize

st.set_page_config(page_title="Customer Review Analyzer", layout="wide")

DDOG_PATH = "reviews_ddog.csv"
DT_PATH = "reviews_dt.csv"
REQUIRED_COLS = {"id", "product", "source", "text"}


@st.cache_data(show_spinner=False)
def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_builtin(ddog_path: str, dt_path: str) -> pd.DataFrame:
    ddog = safe_read_csv(ddog_path)
    dt = safe_read_csv(dt_path)

    # If one is missing, still return the other
    frames = []
    if not ddog.empty:
        frames.append(ddog)
    if not dt.empty:
        frames.append(dt)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Standardize columns
    for c in ["id", "product", "source", "text"]:
        if c not in df.columns:
            df[c] = ""

    df["text"] = df["text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["product"] = df["product"].fillna("unknown").astype(str)

    # Normalize product strings for filtering
    df["product_norm"] = df["product"].str.strip().str.lower()

    # Normalize firm if present
    if "firm" in df.columns:
        df["firm"] = df["firm"].fillna("NA").astype(str).str.strip().str.lower()
        mapping = {
            "mid market": "mid-market",
            "midmarket": "mid-market",
            "smb": "small",
            "small business": "small",
        }
        df["firm"] = df["firm"].replace(mapping)

    return df


def firm_mix(df: pd.DataFrame) -> pd.DataFrame:
    if "firm" not in df.columns:
        return pd.DataFrame()

    buckets = ["small", "mid-market", "enterprise"]
    tmp = df[df["firm"].isin(buckets)].copy()
    if tmp.empty:
        return pd.DataFrame()

    out = (
        tmp.groupby(["product_norm", "firm"])["id"]
        .count()
        .rename("count")
        .reset_index()
    )
    out["share"] = out.groupby("product_norm")["count"].transform(lambda x: x / x.sum())
    return out.sort_values(["product_norm", "count"], ascending=[True, False])


def pie_for_firm(df: pd.DataFrame, product_norm: str):
    mix = firm_mix(df)
    if mix.empty:
        st.info("No usable `firm` data found (needs small / mid-market / enterprise).")
        return

    m = mix[mix["product_norm"] == product_norm]
    if m.empty:
        st.info("No firm-size rows for this product.")
        return

    labels = m["firm"].tolist()
    sizes = m["count"].tolist()

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)


@st.cache_data(show_spinner=False)
def run_full_analysis(df_in: pd.DataFrame):
    """
    Runs your sentence-level pipeline + aspect summary.
    Returns:
      df_sent: sentence-level rows
      summary: aspect-level summary (mentions, pos/neg/neu, shares, net sentiment etc.)
      examples: example snippets
      overall_sent: overall pos/neg/neu counts at sentence level
      top_aspects: aspects ranked by mentions
    """
    df_sent = analyze_text_rows(df_in)
    summary, examples = summarize(df_sent)

    # Overall sentiment distribution (sentence-level)
    overall_sent = (
        df_sent["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )
    total = overall_sent["count"].sum()
    overall_sent["share"] = overall_sent["count"] / total if total else 0

    # "What criteria people care about most" = top aspects by mentions
    if "mentions" in summary.columns:
        top_aspects = summary.sort_values("mentions", ascending=False).head(10)
    else:
        top_aspects = pd.DataFrame()

    return df_sent, summary, examples, overall_sent, top_aspects


st.title("Customer Review Analyzer")
st.caption("Built-in Datadog vs Dynatrace review dataset — no upload needed.")

df_all = load_builtin(DDOG_PATH, DT_PATH)

# --- Debug panel to diagnose Dynatrace not showing ---
with st.expander("Debug: what did the app load? (click to open)"):
    st.write("Files present in app working directory:")
    try:
        st.write(sorted(os.listdir(".")))
    except Exception as e:
        st.write(f"Could not list directory: {e}")

    st.write("Loaded dataframe shape:", df_all.shape)
    st.write("Columns:", list(df_all.columns))

    if not df_all.empty:
        st.write("Products found (top 20):")
        st.write(df_all["product"].value_counts().head(20))
        st.write("Products normalized (top 20):")
        st.write(df_all["product_norm"].value_counts().head(20))

# Hard stop if nothing loaded
if df_all.empty:
    st.error(
        "No data loaded. Make sure `reviews_ddog.csv` and/or `reviews_dt.csv` exist in the repo root."
    )
    st.stop()

# Required column check
missing = REQUIRED_COLS - set(df_all.columns)
if missing:
    st.error(f"Missing required columns: {sorted(list(missing))}")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
product_choice = st.sidebar.radio("Analyze:", ["Both", "Datadog", "Dynatrace"], index=0)

def product_filter(df: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "Datadog":
        return df[df["product_norm"].str.contains("datadog")]
    if which == "Dynatrace":
        return df[df["product_norm"].str.contains("dynatrace")]
    return df

df = product_filter(df_all, product_choice)

st.sidebar.write(f"Rows in selection: **{len(df):,}**")

st.subheader("Dataset preview")
st.dataframe(df.head(25), use_container_width=True)

# Firm-size section (with pie chart)
st.subheader("Firm-size mix")
if product_choice in ["Datadog", "Dynatrace"]:
    pie_for_firm(df, "datadog" if product_choice == "Datadog" else "dynatrace")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Datadog**")
        pie_for_firm(df_all, "datadog")
    with col2:
        st.markdown("**Dynatrace**")
        pie_for_firm(df_all, "dynatrace")

mix_tbl = firm_mix(df if product_choice != "Both" else df_all)
if not mix_tbl.empty:
    show = mix_tbl.copy()
    show["share"] = (show["share"] * 100).round(1).astype(str) + "%"
    st.dataframe(show, use_container_width=True)
else:
    st.info("No `firm` column detected or it doesn't contain small/mid-market/enterprise.")

st.divider()

# Run analysis button
run = st.button("Run analysis (aspects + sentiment)", type="primary")

if run:
    with st.spinner("Running sentiment + aspect analysis..."):
        df_sent, summary, examples, overall_sent, top_aspects = run_full_analysis(df)

    st.success("Done.")

    # Overall sentiment distribution
    st.subheader("Overall sentiment distribution (sentence-level)")
    overall_show = overall_sent.copy()
    overall_show["share"] = (overall_show["share"] * 100).round(1).astype(str) + "%"
    st.dataframe(overall_show, use_container_width=True)

    # Top criteria users care about
    st.subheader("Top criteria users mention most (Top 10 aspects)")
    if not top_aspects.empty:
        st.dataframe(top_aspects, use_container_width=True)
    else:
        st.info("No aspect mention counts found in summary output.")

    # Aspect summary
    st.subheader("Aspect summary (mentions + pos/neg/neu)")
    st.dataframe(summary, use_container_width=True)

    # Example snippets
    st.subheader("Example snippets (by aspect)")
    st.dataframe(examples, use_container_width=True)

    # Downloads
    st.subheader("Download results")
    st.download_button(
        "Download sentence_level.csv",
        df_sent.to_csv(index=False).encode("utf-8"),
        file_name="sentence_level.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download summary_by_aspect.csv",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="summary_by_aspect.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download example_snippets.csv",
        examples.to_csv(index=False).encode("utf-8"),
        file_name="example_snippets.csv",
        mime="text/csv",
    )
