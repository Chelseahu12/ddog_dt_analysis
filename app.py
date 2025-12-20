import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sentiment_classifier import analyze_text_rows, summarize  # assumes these exist in your file

# -----------------------------
# Page + style
# -----------------------------
st.set_page_config(
    page_title="Customer Review Analyzer",
    page_icon="💬",
    layout="wide",
)

BLUE_CSS = """
<style>
/* overall */
.block-container { padding-top: 2rem; }

/* make headers feel less “robotic” */
h1, h2, h3 { letter-spacing: -0.02em; }

/* subtle blue callouts */
.blue-chip {
  display:inline-block;
  padding: 0.35rem 0.6rem;
  border-radius: 999px;
  background: rgba(30, 136, 229, 0.12);
  color: #0b2e57;
  font-weight: 600;
  margin-right: 0.4rem;
}

/* winner highlight */
.winner {
  display:inline-block;
  padding: 0.35rem 0.6rem;
  border-radius: 12px;
  background: rgba(25, 118, 210, 0.18);
  border: 1px solid rgba(25, 118, 210, 0.35);
  color: #0b2e57;
  font-weight: 700;
}

/* small helper text */
.muted { color: rgba(0,0,0,0.55); }
</style>
"""
st.markdown(BLUE_CSS, unsafe_allow_html=True)

# -----------------------------
# Config
# -----------------------------
DDOG_CSV = "reviews_ddog.csv"
DT_CSV = "reviews_dt.csv"

# If your sentiment model outputs labels like POSITIVE/NEGATIVE/NEUTRAL,
# these are the canonical names we’ll use in charts.
SENTIMENT_ORDER = ["negative", "neutral", "positive"]


# -----------------------------
# Helpers
# -----------------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize col names
    df.columns = [c.strip().lower() for c in df.columns]

    # required columns
    required = {"id", "product", "source", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(list(missing))}")

    # optional firm column
    if "firm" not in df.columns:
        df["firm"] = pd.NA

    # normalize values
    df["product"] = df["product"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].astype(str).str.strip()
    df["text"] = df["text"].astype(str)

    # firm normalization if present
    df["firm"] = df["firm"].astype(str).str.strip().str.lower().replace({"nan": pd.NA, "none": pd.NA, "na": pd.NA})

    return df


@st.cache_data
def load_reviews(ddog_path: str, dt_path: str) -> pd.DataFrame:
    ddog = pd.read_csv(ddog_path)
    dt = pd.read_csv(dt_path)
    ddog = _normalize_cols(ddog)
    dt = _normalize_cols(dt)

    # safety: force correct product labels based on file
    ddog["product"] = "datadog"
    dt["product"] = "dynatrace"

    # make ids unique across products if you reused 1..75 twice
    ddog["id"] = ddog["id"].astype(str).apply(lambda x: f"ddog_{x}")
    dt["id"] = dt["id"].astype(str).apply(lambda x: f"dt_{x}")

    return pd.concat([ddog, dt], ignore_index=True)


def firm_mix_table(df: pd.DataFrame) -> pd.DataFrame:
    if df["firm"].isna().all():
        return pd.DataFrame(columns=["product", "firm", "count", "share"])

    tmp = df.dropna(subset=["firm"]).copy()
    g = tmp.groupby(["product", "firm"], as_index=False).size().rename(columns={"size": "count"})
    g["share"] = g.groupby("product")["count"].transform(lambda s: (s / s.sum()) * 100.0)
    return g.sort_values(["product", "count"], ascending=[True, False])


def plot_pie(labels, values, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title(title, fontsize=16, pad=10)
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)


def sentiment_pie_from_sentence_df(df_sent: pd.DataFrame, title: str):
    # expects df_sent has column "sentiment" with values: negative/neutral/positive (lowercase)
    counts = df_sent["sentiment"].value_counts(dropna=False)

    labels = []
    values = []
    for s in SENTIMENT_ORDER:
        labels.append(s)
        values.append(int(counts.get(s, 0)))

    plot_pie(labels, values, title)


def top_aspects_table(summary_aspect: pd.DataFrame, product: str) -> pd.DataFrame:
    # expects columns: product, aspect, mentions, pos_share, neu_share, neg_share, net_sentiment
    out = summary_aspect[summary_aspect["product"] == product].copy()
    out = out.sort_values("mentions", ascending=False)
    return out


def aspect_compare_ranked(summary_aspect: pd.DataFrame) -> pd.DataFrame:
    # rank aspects by total mentions across both products
    g = summary_aspect.groupby("aspect", as_index=False)["mentions"].sum()
    return g.sort_values("mentions", ascending=False)


def winner_label(net_a: float, net_b: float):
    if pd.isna(net_a) or pd.isna(net_b):
        return None
    if net_a > net_b:
        return "datadog"
    if net_b > net_a:
        return "dynatrace"
    return "tie"


def pie_from_shares(pos_share, neu_share, neg_share, title: str):
    labels = ["positive", "neutral", "negative"]
    values = [pos_share, neu_share, neg_share]
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title(title, fontsize=14, pad=8)
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Run analysis (cached)
# -----------------------------
@st.cache_data
def run_full_analysis(df: pd.DataFrame):
    # sentence-level
    df_sent = analyze_text_rows(df)

    # aspect summary + example snippets
    summary_aspect, examples = summarize(df_sent)

    return df_sent, summary_aspect, examples


# -----------------------------
# App
# -----------------------------
df = load_reviews(DDOG_CSV, DT_CSV)

# Sidebar navigation
st.sidebar.markdown('<span class="blue-chip">Navigation</span>', unsafe_allow_html=True)
page = st.sidebar.radio(
    label="",
    options=["Welcome", "Compare", "Datadog", "Dynatrace"],
    index=0
)

# Always compute analysis once (cached)
with st.spinner("Running aspect + sentiment analysis…"):
    df_sent, summary_aspect, examples = run_full_analysis(df)

# -----------------------------
# Pages
# -----------------------------
if page == "Welcome":
    st.title("Customer Review Analyzer")
    st.markdown(
        """
        <div class="muted">
        I built this to quickly sanity-check what real users praise (and complain about) when they talk about observability tools.
        It’s opinionated, lightweight, and meant for fast “what actually matters?” reads — not perfect NLP.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", len(df))
    c2.metric("Datadog reviews", int((df["product"] == "datadog").sum()))
    c3.metric("Dynatrace reviews", int((df["product"] == "dynatrace").sum()))

    st.subheader("Quick preview")
    st.dataframe(df.head(12), use_container_width=True)

    st.subheader("What you’ll see in here")
    st.markdown(
        """
        - firm-size mix (pie charts)
        - sentiment breakdown (pos / neutral / neg)
        - what people talk about most (ranked by mentions)
        - for each aspect: Datadog vs Dynatrace side-by-side, with a simple “winner” callout
        """
    )

elif page == "Compare":
    st.title("Compare: Datadog vs Dynatrace")

    # Firm size mix
    st.subheader("Firm-size mix")
    mix = firm_mix_table(df)

    if mix.empty:
        st.info("No firm column found. If you add firm = small / mid-market / enterprise, I’ll chart it.")
    else:
        st.dataframe(mix, use_container_width=True)

        left, right = st.columns(2)
        with left:
            dd = mix[mix["product"] == "datadog"]
            plot_pie(dd["firm"].tolist(), dd["count"].tolist(), "Datadog firm-size mix")
        with right:
            dt = mix[mix["product"] == "dynatrace"]
            plot_pie(dt["firm"].tolist(), dt["count"].tolist(), "Dynatrace firm-size mix")

    # Sentiment breakdown (sentence-level)
    st.subheader("Overall sentiment breakdown (sentence-level)")
    l2, r2 = st.columns(2)
    with l2:
        sentiment_pie_from_sentence_df(df_sent[df_sent["product"] == "datadog"], "Datadog sentiment")
    with r2:
        sentiment_pie_from_sentence_df(df_sent[df_sent["product"] == "dynatrace"], "Dynatrace sentiment")

    # Top aspects tables
    st.subheader("Top aspects (ranked by mentions)")
    l3, r3 = st.columns(2)
    with l3:
        st.caption("Datadog")
        st.dataframe(top_aspects_table(summary_aspect, "datadog").head(12), use_container_width=True)
    with r3:
        st.caption("Dynatrace")
        st.dataframe(top_aspects_table(summary_aspect, "dynatrace").head(12), use_container_width=True)

    # Aspect-by-aspect pies
    st.subheader("Aspect-by-aspect sentiment (side-by-side)")
    ranked = aspect_compare_ranked(summary_aspect)

    # Only show the top N to keep the page usable
    top_n = st.slider("How many aspects to show?", min_value=5, max_value=25, value=12, step=1)

    for _, row in ranked.head(top_n).iterrows():
        aspect = row["aspect"]

        a = summary_aspect[(summary_aspect["product"] == "datadog") & (summary_aspect["aspect"] == aspect)]
        b = summary_aspect[(summary_aspect["product"] == "dynatrace") & (summary_aspect["aspect"] == aspect)]

        if a.empty or b.empty:
            continue

        a = a.iloc[0]
        b = b.iloc[0]

        w = winner_label(a.get("net_sentiment"), b.get("net_sentiment"))

        # Row header
        if w == "datadog":
            st.markdown(f'<span class="winner">{aspect} — Datadog wins (higher net sentiment)</span>', unsafe_allow_html=True)
        elif w == "dynatrace":
            st.markdown(f'<span class="winner">{aspect} — Dynatrace wins (higher net sentiment)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="blue-chip">{aspect}</span>', unsafe_allow_html=True)

        cL, cR = st.columns(2)

        with cL:
            pie_from_shares(
                pos_share=a["pos_share"],
                neu_share=a["neu_share"],
                neg_share=a["neg_share"],
                title=f"Datadog — {aspect} (mentions: {int(a['mentions'])})",
            )

        with cR:
            pie_from_shares(
                pos_share=b["pos_share"],
                neu_share=b["neu_share"],
                neg_share=b["neg_share"],
                title=f"Dynatrace — {aspect} (mentions: {int(b['mentions'])})",
            )

        st.divider()

elif page in ["Datadog", "Dynatrace"]:
    product = "datadog" if page == "Datadog" else "dynatrace"
    st.title(page)

    sub = df[df["product"] == product].copy()
    st.caption(f"{len(sub)} reviews in the dataset")

    st.subheader("Dataset preview")
    st.dataframe(sub.head(20), use_container_width=True)

    st.subheader("Sentiment breakdown (sentence-level)")
    sentiment_pie_from_sentence_df(df_sent[df_sent["product"] == product], f"{page} sentiment")

    st.subheader("Top aspects (ranked by mentions)")
    st.dataframe(top_aspects_table(summary_aspect, product), use_container_width=True)

    st.subheader("Example snippets (what the model pulled)")
    st.dataframe(examples[examples["product"] == product].head(30), use_container_width=True)
