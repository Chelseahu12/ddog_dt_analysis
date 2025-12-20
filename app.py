import streamlit as st
import pandas as pd

from sentiment_classifier import analyze_text_rows, summarize

st.set_page_config(page_title="Customer Review Analyzer", layout="wide")

# --- Built-in datasets (must exist in the GitHub repo root) ---
DDOG_PATH = "reviews_ddog.csv"
DT_PATH = "reviews_dt.csv"

REQUIRED_COLS = {"id", "product", "source", "text"}


@st.cache_data(show_spinner=False)
def load_builtin(ddog_path: str, dt_path: str) -> pd.DataFrame:
    ddog = pd.read_csv(ddog_path)
    dt = pd.read_csv(dt_path)

    # Ensure consistent columns
    ddog["product"] = ddog.get("product", "datadog")
    dt["product"] = dt.get("product", "dynatrace")

    df = pd.concat([ddog, dt], ignore_index=True)

    # Basic cleanup
    df["text"] = df["text"].fillna("").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    df["product"] = df["product"].fillna("unknown").astype(str)

    return df


def firm_mix_table(df: pd.DataFrame) -> pd.DataFrame:
    if "firm" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["firm"] = tmp["firm"].fillna("NA").astype(str).str.strip().str.lower()

    # normalize a bit
    mapping = {
        "mid market": "mid-market",
        "midmarket": "mid-market",
        "enterprise ": "enterprise",
        "smb": "small",
        "small business": "small",
    }
    tmp["firm"] = tmp["firm"].replace(mapping)

    # keep only the buckets you care about
    buckets = ["small", "mid-market", "enterprise"]
    tmp = tmp[tmp["firm"].isin(buckets)]

    out = (
        tmp.groupby(["product", "firm"])["id"]
        .count()
        .rename("count")
        .reset_index()
    )
    out["share"] = out.groupby("product")["count"].transform(lambda x: x / x.sum())
    out = out.sort_values(["product", "count"], ascending=[True, False])
    return out


st.title("Customer Review Analyzer")
st.caption("Built-in Datadog vs Dynatrace review dataset — no upload needed.")

# Load built-in data
with st.spinner("Loading built-in datasets..."):
    df_all = load_builtin(DDOG_PATH, DT_PATH)

missing = REQUIRED_COLS - set(df_all.columns)
if missing:
    st.error(f"Your built-in CSVs are missing required columns: {sorted(list(missing))}")
    st.stop()

# --- Optional upload (turn off by setting to False) ---
ALLOW_UPLOAD = False

if ALLOW_UPLOAD:
    st.divider()
    st.subheader("Optional: Upload your own CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        missing2 = REQUIRED_COLS - set(df_up.columns)
        if missing2:
            st.error(f"Uploaded CSV missing columns: {sorted(list(missing2))}")
            st.stop()
        df_all = df_up

# Controls
st.sidebar.header("Controls")
mode = st.sidebar.radio("Which reviews to analyze?", ["Both", "Datadog only", "Dynatrace only"], index=0)

if mode == "Datadog only":
    df = df_all[df_all["product"].str.lower().str.contains("datadog")]
elif mode == "Dynatrace only":
    df = df_all[df_all["product"].str.lower().str.contains("dynatrace")]
else:
    df = df_all

st.sidebar.write(f"Rows loaded: **{len(df):,}**")

st.subheader("Dataset preview")
st.dataframe(df.head(25), use_container_width=True)

# Firm-size mix
mix = firm_mix_table(df)
if not mix.empty:
    st.subheader("Firm-size mix (share of reviews)")
    # show percent nicely
    mix_show = mix.copy()
    mix_show["share"] = (mix_show["share"] * 100).round(1).astype(str) + "%"
    st.dataframe(mix_show, use_container_width=True)
else:
    st.info("No `firm` column detected (optional). If you add a `firm` column with small / mid-market / enterprise, I'll summarize it here.")

st.divider()

# Run analysis
run = st.button("Run analysis", type="primary")

if run:
    with st.spinner("Analyzing reviews (sentence-level sentiment + aspect tagging)..."):
        df_sent = analyze_text_rows(df)
        summary, examples = summarize(df_sent)

    st.success("Done!")

    st.subheader("Aspect summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Example snippets")
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

