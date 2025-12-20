import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import your pipeline functions
# sentiment_classifier.py should be in the same folder as app.py
from sentiment_classifier import analyze_text_rows, summarize

# ----------------------------
# Page config + simple blue UI
# ----------------------------
st.set_page_config(
    page_title="Customer Review Analyzer",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
      /* Headings */
      h1, h2, h3 { color: #0B3D91; }  /* deep-ish blue */

      /* Buttons */
      div.stButton > button {
        background-color: #0B3D91;
        color: white;
        border-radius: 10px;
        border: 0px;
        padding: 0.6rem 1rem;
      }
      div.stButton > button:hover {
        background-color: #124FB5;
        color: white;
      }

      /* Info boxes */
      .stAlert {
        border-left: 6px solid #0B3D91 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# File paths
# ----------------------------
DDOG_CSV = "reviews_ddog.csv"
DT_CSV = "reviews_dt.csv"

REQUIRED_COLS = {"id", "source", "text"}  # product is optional; we set it if missing
OPTIONAL_COLS = {"firm"}  # small / mid-market / enterprise

# ----------------------------
# Helpers
# ----------------------------
def _validate_df(df: pd.DataFrame, label: str):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        st.error(f"{label} is missing required columns: {sorted(list(missing))}")
        st.stop()

    # Clean types
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0]

    # Normalize firm if present
    if "firm" in df.columns:
        df["firm"] = (
            df["firm"]
            .fillna("NA")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"mid market": "mid-market", "midmarket": "mid-market"})
        )

    return df


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    if not os.path.exists(DDOG_CSV):
        st.error(f"Missing {DDOG_CSV} in repo root.")
        st.stop()
    if not os.path.exists(DT_CSV):
        st.error(f"Missing {DT_CSV} in repo root.")
        st.stop()

    dd = pd.read_csv(DDOG_CSV)
    dt = pd.read_csv(DT_CSV)

    dd = _validate_df(dd, "Datadog CSV")
    dt = _validate_df(dt, "Dynatrace CSV")

    # Ensure product column exists + standardized
    dd["product"] = "datadog"
    dt["product"] = "dynatrace"

    # Make ids unique across both so merges don’t collide
    dd["id"] = dd["id"].astype(str).apply(lambda x: f"ddog_{x}")
    dt["id"] = dt["id"].astype(str).apply(lambda x: f"dt_{x}")

    out = pd.concat([dd, dt], ignore_index=True)
    return out


def firm_size_pie(df: pd.DataFrame, title: str):
    # Only show if firm exists and has meaningful values
    if "firm" not in df.columns:
        st.info("No firm column detected. Add a firm column (small / mid-market / enterprise) to enable firm-size mix.")
        return

    counts = (
        df[df["firm"].isin(["small", "mid-market", "enterprise"])]
        .groupby("firm")["id"]
        .count()
        .sort_values(ascending=False)
    )

    if counts.empty:
        st.info("Firm column present, but no values matched: small / mid-market / enterprise.")
        return

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)

    # Use legend instead of text labels around the pie -> avoids cropping
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.7,
        textprops={"fontsize": 12},
    )

    ax.set_title(title, fontsize=16)
    ax.legend(
        wedges,
        [f"{k} ({v})" for k, v in zip(counts.index, counts.values)],
        title="Firm size",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


@st.cache_data(show_spinner=True)
def run_pipeline(df_reviews: pd.DataFrame):
    """
    Runs your earlier analysis:
    - sentence_level: one row per sentence with sentiment + aspect tags
    - summary_by_aspect: aggregated stats per product+aspect
    - example_snippets: example sentences per aspect/sentiment
    """
    df_sent = analyze_text_rows(df_reviews)
    summary_by_aspect, examples = summarize(df_sent)
    return df_sent, summary_by_aspect, examples


def sentiment_bar(df_sent: pd.DataFrame, title: str):
    # Expect df_sent to have columns: product, sentiment
    if not {"product", "sentiment"}.issubset(df_sent.columns):
        st.warning("Sentiment output missing expected columns (product, sentiment).")
        return

    counts = (
        df_sent.groupby(["sentiment"])["sentence"]
        .count()
        .reindex(["neg", "neu", "pos"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.bar(counts.index, counts.values)
    ax.set_title(title)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("# sentences")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def top_aspects_table(summary_by_aspect: pd.DataFrame, product: str, top_n: int = 10):
    # Expect columns: product, aspect, mentions, neg_share, pos_share, neu_share, net_sentiment (based on your earlier outputs)
    if not {"product", "aspect", "mentions"}.issubset(summary_by_aspect.columns):
        st.warning("Aspect summary missing expected columns (product, aspect, mentions).")
        return

    sub = summary_by_aspect[summary_by_aspect["product"] == product].copy()
    if sub.empty:
        st.info(f"No aspect summary rows found for product='{product}'.")
        return

    # Make it more readable
    cols_to_show = [c for c in ["aspect", "mentions", "pos_share", "neu_share", "neg_share", "net_sentiment"] if c in sub.columns]
    sub = sub.sort_values("mentions", ascending=False).head(top_n)[cols_to_show]

    # Pretty formatting
    for c in ["pos_share", "neu_share", "neg_share", "net_sentiment"]:
        if c in sub.columns:
            sub[c] = sub[c].astype(float)

    st.dataframe(sub, use_container_width=True)


# ----------------------------
# App UI
# ----------------------------
df_all = load_reviews()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Datadog", "Dynatrace", "Compare"], index=0)

# Only run model once per refresh (cached)
df_sent_all, summary_all, examples_all = run_pipeline(df_all)

if page == "Welcome":
    st.title("Customer Review Analyzer")
    st.write("Built-in Datadog vs Dynatrace review dataset — no upload needed.")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("What you can do here")
        st.markdown(
            """
            - See **firm-size mix** (pie chart)
            - See **sentiment distribution** (pos / neutral / neg)
            - See **what criteria users talk about most** (top aspects by mentions)
            - Compare **Datadog vs Dynatrace** side-by-side
            """
        )
    with c2:
        st.subheader("Dataset size")
        st.metric("Total reviews", len(df_all))
        st.metric("Datadog reviews", int((df_all["product"] == "datadog").sum()))
        st.metric("Dynatrace reviews", int((df_all["product"] == "dynatrace").sum()))

    st.subheader("Quick preview")
    st.dataframe(df_all.head(25), use_container_width=True)


elif page == "Datadog":
    st.title("Datadog")
    df = df_all[df_all["product"] == "datadog"].copy()
    df_sent = df_sent_all[df_sent_all["product"] == "datadog"].copy()

    st.subheader("Dataset preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Firm-size mix")
    firm_size_pie(df, "Datadog firm-size mix (share of reviews)")

    st.subheader("Sentiment (sentence-level)")
    sentiment_bar(df_sent, "Datadog sentiment (sentence-level)")

    st.subheader("Top criteria users mention (aspects)")
    top_aspects_table(summary_all, product="datadog", top_n=12)


elif page == "Dynatrace":
    st.title("Dynatrace")
    df = df_all[df_all["product"] == "dynatrace"].copy()
    df_sent = df_sent_all[df_sent_all["product"] == "dynatrace"].copy()

    st.subheader("Dataset preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Firm-size mix")
    firm_size_pie(df, "Dynatrace firm-size mix (share of reviews)")

    st.subheader("Sentiment (sentence-level)")
    sentiment_bar(df_sent, "Dynatrace sentiment (sentence-level)")

    st.subheader("Top criteria users mention (aspects)")
    top_aspects_table(summary_all, product="dynatrace", top_n=12)


elif page == "Compare":
    st.title("Compare: Datadog vs Dynatrace")

    st.subheader("Firm-size mix (by product)")
    if "firm" in df_all.columns:
        # table + pies
        mix = (
            df_all[df_all["firm"].isin(["small", "mid-market", "enterprise"])]
            .groupby(["product", "firm"])["id"]
            .count()
            .reset_index(name="count")
        )
        mix["share"] = mix.groupby("product")["count"].transform(lambda x: x / x.sum())
        st.dataframe(mix.sort_values(["product", "count"], ascending=[True, False]), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            firm_size_pie(df_all[df_all["product"] == "datadog"], "Datadog firm-size mix")
        with col2:
            firm_size_pie(df_all[df_all["product"] == "dynatrace"], "Dynatrace firm-size mix")
    else:
        st.info("No firm column detected in your CSVs.")

    st.subheader("Top aspects (mentions)")
    # show top aspects for each side
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Datadog")
        top_aspects_table(summary_all, product="datadog", top_n=10)
    with c2:
        st.markdown("### Dynatrace")
        top_aspects_table(summary_all, product="dynatrace", top_n=10)
