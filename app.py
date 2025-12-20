import os
import pandas as pd
import streamlit as st

from sentiment_classifier import analyze_text_rows, summarize

st.set_page_config(page_title="Review Insights", layout="wide")

DDOG_PATH = "reviews_ddog.csv"
DT_PATH = "reviews_dt.csv"

REQUIRED_COLS = {"id", "product", "source", "text"}  # firm optional


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}. Put it in the same folder as app.py.")
        st.stop()

    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"{path} is missing columns: {missing}. Found: {list(df.columns)}")
        st.stop()

    # Defensive cleanup
    df["text"] = df["text"].fillna("").astype(str)
    return df


def normalize_firm_bucket(series: pd.Series) -> pd.Series:
    s = (
        series.fillna("NA")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Normalize a few common variants
    s = s.replace({
        "mid market": "mid-market",
        "midmarket": "mid-market",
        "mid-market (51-1000 emp.)": "mid-market",
        "enterprise (1000+ emp.)": "enterprise",
        "small business": "small",
        "small-business": "small",
    })

    # If values contain phrases like "Mid-Market (51-1000 emp.)"
    s = s.str.replace("mid-market.*", "mid-market", regex=True)
    s = s.str.replace("enterprise.*", "enterprise", regex=True)
    s = s.str.replace("small.*", "small", regex=True)

    return s


def firm_mix(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "firm" not in df.columns:
        return pd.DataFrame({
            "product": [label],
            "firm_bucket": ["(no firm column)"],
            "count": [len(df)],
            "share": [1.0],
            "total_reviews": [len(df)],
        })

    buckets = ["small", "mid-market", "enterprise"]
    s = normalize_firm_bucket(df["firm"])
    total = len(df)

    vc = s.value_counts(dropna=False)
    out = pd.DataFrame({"firm_bucket": buckets})
    out["count"] = [int(vc.get(b, 0)) for b in buckets]
    out["share"] = (out["count"] / total).round(3)
    out.insert(0, "product", label)
    out["total_reviews"] = total
    return out


@st.cache_data(show_spinner=False)
def run_pipeline(df: pd.DataFrame):
    """
    Cache results so switching tabs/filters doesn't recompute everything.
    (Cache key changes when df changes.)
    """
    df_sent = analyze_text_rows(df)
    summary_df, examples_df = summarize(df_sent, top_examples=3)
    return df_sent, summary_df, examples_df


def to_csv_bytes(x: pd.DataFrame) -> bytes:
    return x.to_csv(index=False).encode("utf-8")


def build_comparison(ddog_sum: pd.DataFrame, dt_sum: pd.DataFrame) -> pd.DataFrame:
    if "aspect" not in ddog_sum.columns or "aspect" not in dt_sum.columns:
        return pd.DataFrame()

    keep_cols = ["aspect", "mentions", "neg_share", "pos_share", "neu_share", "net_sentiment"]
    dd = ddog_sum[[c for c in keep_cols if c in ddog_sum.columns]].copy()
    tt = dt_sum[[c for c in keep_cols if c in dt_sum.columns]].copy()

    dd = dd.add_prefix("ddog_").rename(columns={"ddog_aspect": "aspect"})
    tt = tt.add_prefix("dt_").rename(columns={"dt_aspect": "aspect"})

    comp = dd.merge(tt, on="aspect", how="outer").fillna(0)

    if "ddog_mentions" in comp.columns and "dt_mentions" in comp.columns:
        comp["combined_mentions"] = comp["ddog_mentions"] + comp["dt_mentions"]
        comp = comp.sort_values("combined_mentions", ascending=False)

    return comp


# ----------------------------
# Load data
# ----------------------------
ddog_df = load_csv(DDOG_PATH)
dt_df = load_csv(DT_PATH)

# If user forgot to set "product" properly, we can force it from filename (optional).
# Comment out if you already have correct values.
if ddog_df["product"].nunique() == 1 and ddog_df["product"].iloc[0].strip() == "":
    ddog_df["product"] = "datadog"
if dt_df["product"].nunique() == 1 and dt_df["product"].iloc[0].strip() == "":
    dt_df["product"] = "dynatrace"

both_df = pd.concat([ddog_df, dt_df], ignore_index=True)


st.title("Datadog vs Dynatrace — Customer Review Insights")
st.caption("Built-in dataset (no upload). Put reviews_ddog.csv + reviews_dt.csv next to app.py.")

choice = st.radio("Dataset", ["Datadog", "Dynatrace", "Both"], horizontal=True)

if choice == "Datadog":
    active_df = ddog_df
    tag = "ddog"
elif choice == "Dynatrace":
    active_df = dt_df
    tag = "dt"
else:
    active_df = both_df
    tag = "both"

st.write(f"Loaded **{len(active_df)}** reviews in current view.")

# Firm mix section
st.subheader("Firm size mix (share of reviews)")
mix_ddog = firm_mix(ddog_df, "ddog")
mix_dt = firm_mix(dt_df, "dt")
mix_both = firm_mix(both_df, "both")

mix_map = {"Datadog": mix_ddog, "Dynatrace": mix_dt, "Both": mix_both}
st.dataframe(mix_map[choice], use_container_width=True)

# Run analysis
with st.spinner("Analyzing (sentence split → aspect tagging → sentiment)…"):
    df_sent, summary_df, examples_df = run_pipeline(active_df)

# Main results
c1, c2 = st.columns([1.1, 0.9])

with c1:
    st.subheader("Summary by aspect")
    # Try to sort by mention share if present
    sort_col = "mention_share_within_product" if "mention_share_within_product" in summary_df.columns else "mentions"
    st.dataframe(summary_df.sort_values(sort_col, ascending=False), use_container_width=True)

with c2:
    st.subheader("Example snippets (evidence)")
    st.dataframe(examples_df, use_container_width=True)

with st.expander("Sentence-level output (audit log)", expanded=False):
    st.dataframe(df_sent, use_container_width=True)

# Side-by-side comparison (only meaningful if you have both)
st.subheader("Datadog vs Dynatrace — aspect comparison")

with st.spinner("Building comparison…"):
    ddog_sent, ddog_sum, ddog_ex = run_pipeline(ddog_df)
    dt_sent, dt_sum, dt_ex = run_pipeline(dt_df)
    comp = build_comparison(ddog_sum, dt_sum)

if comp.empty:
    st.info("Comparison table not available (missing 'aspect' column in summary output).")
else:
    st.dataframe(comp, use_container_width=True)

# Downloads
st.subheader("Download outputs (current view)")

dc1, dc2, dc3 = st.columns(3)
with dc1:
    st.download_button(
        "Download sentence_level.csv",
        data=to_csv_bytes(df_sent),
        file_name=f"sentence_level_{tag}.csv",
        mime="text/csv",
    )
with dc2:
    st.download_button(
        "Download summary_by_aspect.csv",
        data=to_csv_bytes(summary_df),
        file_name=f"summary_by_aspect_{tag}.csv",
        mime="text/csv",
    )
with dc3:
    st.download_button(
        "Download example_snippets.csv",
        data=to_csv_bytes(examples_df),
        file_name=f"example_snippets_{tag}.csv",
        mime="text/csv",
    )

st.download_button(
    "Download ddog_vs_dt_comparison.csv",
    data=to_csv_bytes(comp) if not comp.empty else b"",
    file_name="comparison_ddog_vs_dt.csv",
    mime="text/csv",
    disabled=comp.empty,
)
