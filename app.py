# app.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- If you want to generate sentence-level + aspect summaries inside Streamlit,
# make sure sentiment_classifier.py exposes analyze_text_rows and summarize.
# Otherwise, set GENERATE_ON_THE_FLY = False and commit sentence_level.csv + summary_by_aspect.csv.
GENERATE_ON_THE_FLY = True
try:
    from sentiment_classifier import analyze_text_rows, summarize  # type: ignore
except Exception:
    analyze_text_rows = None
    summarize = None
    GENERATE_ON_THE_FLY = False


# -----------------------------
# Styling / theme
# -----------------------------
DDOG_BLUES = ["#DBEAFE", "#93C5FD", "#60A5FA", "#3B82F6"]  # light -> medium
DT_ORANGES = ["#FFEDD5", "#FDBA74", "#FB923C", "#F97316"]  # light -> medium

SENTIMENT_ORDER = ["NEG", "NEU", "POS"]

st.set_page_config(
    page_title="Customer Review Analyzer",
    page_icon="📊",
    layout="wide",
)

CSS = """
<style>
/* overall background */
.block-container { padding-top: 2.0rem; }

/* subtle blue cards */
.soft-card {
  background: #F8FBFF;
  border: 1px solid #E6F0FF;
  border-radius: 14px;
  padding: 14px 16px;
  margin: 10px 0;
}

/* highlight “winner” */
.winner {
  background: #ECFEFF;
  border: 1px solid #A5F3FC;
  border-radius: 14px;
  padding: 10px 14px;
}

/* quiet text */
.small-note { color: #4B5563; font-size: 0.92rem; }

/* big page title */
.page-title {
  font-size: 2.2rem;
  font-weight: 650;
  color: #0F172A;
  margin-bottom: 0.3rem;
}
.page-subtitle {
  font-size: 1.05rem;
  color: #334155;
  margin-bottom: 1.0rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------------
# Helpers: data loading/safety
# -----------------------------
def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize common variants
    if "review" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"review": "text"})
    if "company" in df.columns and "product" not in df.columns:
        df = df.rename(columns={"company": "product"})

    required = {"id", "product", "source", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} is missing columns: {sorted(list(missing))}")

    df["product"] = df["product"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].astype(str).str.strip()
    df["text"] = df["text"].astype(str)

    # firm: keep NA as literal "NA"
    if "firm" not in df.columns:
        df["firm"] = "NA"
    else:
        firm = df["firm"]
        firm = firm.where(~firm.isna(), "NA")
        firm = firm.astype(str).str.strip()

        firm_lower = firm.str.lower()
        firm = np.where(firm_lower.isin(["nan", "none", "", "na", "n/a"]), "NA", firm)

        firm_lower = pd.Series(firm).str.lower()
        firm = np.where(firm_lower.isin(["small", "smb"]), "small", firm)
        firm = np.where(firm_lower.isin(["mid market", "mid-market", "midmarket"]), "mid-market", firm)
        firm = np.where(firm_lower.isin(["enterprise", "ent"]), "enterprise", firm)

        df["firm"] = pd.Series(firm)

    return df


def pct_table(counts: pd.Series) -> pd.DataFrame:
    c = pd.to_numeric(counts, errors="coerce").fillna(0)
    total = float(c.sum()) if float(c.sum()) > 0 else 1.0
    out = pd.DataFrame({"count": c.astype(int)})
    out["share"] = out["count"] / total
    out = out.reset_index().rename(columns={"index": "label"})
    out["label"] = out["label"].astype(str)
    return out


def ensure_sentiment_labels(df_sent: pd.DataFrame) -> pd.DataFrame:
    """Standardize sentiment column to NEG/NEU/POS and ensure it's present."""
    d = df_sent.copy()
    if "sentiment" not in d.columns:
        # try common alt name
        if "label" in d.columns:
            d = d.rename(columns={"label": "sentiment"})
        else:
            d["sentiment"] = "NEU"

    # normalize values
    d["sentiment"] = d["sentiment"].astype(str).str.upper().str.strip()
    mapping = {
        "NEGATIVE": "NEG",
        "NEUTRAL": "NEU",
        "POSITIVE": "POS",
    }
    d["sentiment"] = d["sentiment"].replace(mapping)
    d.loc[~d["sentiment"].isin(SENTIMENT_ORDER), "sentiment"] = "NEU"
    return d


def net_sentiment_from_counts(pos: int, neu: int, neg: int) -> float:
    tot = pos + neu + neg
    if tot == 0:
        return 0.0
    return (pos - neg) / tot


def plot_pie_percent(df_pct: pd.DataFrame, title: str, colors: Optional[List[str]] = None, height: int = 340):
    if df_pct is None or df_pct.empty:
        st.info("No data to plot.")
        return

    d = df_pct.copy()
    if "label" not in d.columns or "share" not in d.columns:
        st.info("No data to plot.")
        return

    d["label"] = d["label"].astype(str)
    d["count"] = pd.to_numeric(d.get("count", 0), errors="coerce").fillna(0).astype(int)
    d["share"] = pd.to_numeric(d["share"], errors="coerce").fillna(0.0)

    # remove invalid slices
    d = d[(d["share"] > 0) & np.isfinite(d["share"])]

    if d.empty:
        st.info("No non-zero data to plot.")
        return

    fig = px.pie(
        d,
        names="label",
        values="share",
        hover_data={"count": True, "share": ":.1%"},
        title=title,
        color_discrete_sequence=colors,
    )
    fig.update_traces(textposition="inside", textinfo="percent", sort=False)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_bar_percent(df_pct: pd.DataFrame, title: str, color: str, height: int = 420, y_max: Optional[float] = None):
    """
    df_pct must have: label, count, share.
    Shows share on y-axis (percent). Hover includes count.
    """
    if df_pct is None or df_pct.empty:
        st.info("No data to plot.")
        return

    d = df_pct.copy()
    d["label"] = d["label"].astype(str)
    d["count"] = pd.to_numeric(d.get("count", 0), errors="coerce").fillna(0).astype(int)
    d["share"] = pd.to_numeric(d["share"], errors="coerce").fillna(0.0)

    fig = px.bar(
        d,
        x="label",
        y="share",
        hover_data={"count": True, "share": ":.1%"},
        title=title,
    )
    fig.update_traces(marker_color=color)
    fig.update_layout(
        height=height,
        yaxis_tickformat=".0%",
        yaxis_title="share of mentions",
        xaxis_title="aspect",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])
    st.plotly_chart(fig, use_container_width=True)


def sentiment_pct_from_sentence_df(df_sent: pd.DataFrame) -> pd.DataFrame:
    d = ensure_sentiment_labels(df_sent)
    counts = d["sentiment"].value_counts().reindex(SENTIMENT_ORDER).fillna(0).astype(int)
    out = pct_table(counts)
    # keep sentiment order in table
    out["label"] = pd.Categorical(out["label"], categories=SENTIMENT_ORDER, ordered=True)
    out = out.sort_values("label").copy()
    out["label"] = out["label"].astype(str)
    return out


def firm_mix_pct(df_reviews: pd.DataFrame) -> pd.DataFrame:
    counts = df_reviews["firm"].astype(str).value_counts()
    out = pct_table(counts)

    # order firm buckets if present
    order = ["small", "mid-market", "enterprise", "NA"]
    out["label"] = pd.Categorical(out["label"], categories=order, ordered=True)
    out = out.sort_values("label").copy()
    out["label"] = out["label"].astype(str)
    return out


def load_reviews() -> pd.DataFrame:
    # expects these files in repo root
    ddog_path = "reviews_ddog.csv"
    dt_path = "reviews_dt.csv"

    if not os.path.exists(ddog_path) or not os.path.exists(dt_path):
        raise FileNotFoundError(
            "Missing reviews_ddog.csv or reviews_dt.csv in the repo root. "
            "Commit them to GitHub and redeploy."
        )

    ddog = safe_read_csv(ddog_path)
    dt = safe_read_csv(dt_path)

    # ensure unique ids across products (optional)
    ddog["id"] = ddog["id"].astype(str)
    dt["id"] = dt["id"].astype(str)

    return pd.concat([ddog, dt], ignore_index=True)


@st.cache_data(show_spinner=False)
def get_or_generate_outputs(df_reviews: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_sent: sentence-level rows with columns including product, aspect, sentiment
      df_aspect_summary: aspect-level summary by product with mentions and sentiment shares
    """
    # Prefer precomputed files if present
    sent_path = "sentence_level.csv"
    summ_path = "summary_by_aspect.csv"

    if os.path.exists(sent_path) and os.path.exists(summ_path):
        df_sent = pd.read_csv(sent_path)
        df_sent.columns = [c.strip().lower() for c in df_sent.columns]
        df_sent["product"] = df_sent["product"].astype(str).str.lower().str.strip()
        df_sent = ensure_sentiment_labels(df_sent)

        df_aspect = pd.read_csv(summ_path)
        df_aspect.columns = [c.strip().lower() for c in df_aspect.columns]
        df_aspect["product"] = df_aspect["product"].astype(str).str.lower().str.strip()
        return df_sent, df_aspect

    # Otherwise generate in app (slower)
    if not GENERATE_ON_THE_FLY or analyze_text_rows is None or summarize is None:
        raise RuntimeError(
            "No precomputed sentence_level.csv + summary_by_aspect.csv found, "
            "and in-app generation is unavailable (check sentiment_classifier.py exports)."
        )

    with st.spinner("Running sentiment + aspect analysis (first load can take a bit)…"):
        df_sent = analyze_text_rows(df_reviews)  # type: ignore
        df_sent.columns = [c.strip().lower() for c in df_sent.columns]
        df_sent["product"] = df_sent["product"].astype(str).str.lower().str.strip()
        df_sent = ensure_sentiment_labels(df_sent)

        summary, _examples = summarize(df_sent)  # type: ignore
        df_aspect = summary.copy()
        df_aspect.columns = [c.strip().lower() for c in df_aspect.columns]
        df_aspect["product"] = df_aspect["product"].astype(str).str.lower().str.strip()

        # Optionally write outputs locally (Streamlit Cloud ephemeral; but good for local dev)
        try:
            df_sent.to_csv("sentence_level.csv", index=False)
            df_aspect.to_csv("summary_by_aspect.csv", index=False)
        except Exception:
            pass

        return df_sent, df_aspect


def aspect_summary_table(df_aspect: pd.DataFrame, product: str) -> pd.DataFrame:
    d = df_aspect[df_aspect["product"] == product].copy()
    if d.empty:
        return d
    # expected columns from your pipeline:
    # product, aspect, mentions, neg, neu, pos, neg_share, pos_share, neu_share, net_sentiment
    # normalize if needed
    if "mentions" in d.columns:
        d["mentions"] = pd.to_numeric(d["mentions"], errors="coerce").fillna(0).astype(int)
    for col in ["pos_share", "neu_share", "neg_share", "net_sentiment"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    d = d.sort_values("mentions", ascending=False)
    return d


def overall_aspect_mentions_pct(df_aspect: pd.DataFrame) -> pd.DataFrame:
    """
    Overall aspects ranked by mentions across both products (share of total mentions).
    Returns label=aspect.
    """
    if df_aspect.empty or "aspect" not in df_aspect.columns or "mentions" not in df_aspect.columns:
        return pd.DataFrame(columns=["label", "count", "share"])
    d = df_aspect.copy()
    d["mentions"] = pd.to_numeric(d["mentions"], errors="coerce").fillna(0)
    grouped = d.groupby("aspect")["mentions"].sum().sort_values(ascending=False)
    out = pct_table(grouped)
    out = out.rename(columns={"label": "aspect"})
    out["label"] = out["aspect"].astype(str)
    out = out.drop(columns=["aspect"])
    return out


def per_aspect_sentiment_pct(df_sent: pd.DataFrame, product: str, aspect: str) -> pd.DataFrame:
    d = df_sent[(df_sent["product"] == product) & (df_sent.get("aspect", "") == aspect)].copy()
    if d.empty:
        return pd.DataFrame(columns=["label", "count", "share"])
    return sentiment_pct_from_sentence_df(d)


def winner_label(ddog_net: float, dt_net: float) -> str:
    if ddog_net > dt_net:
        return "datadog"
    if dt_net > ddog_net:
        return "dynatrace"
    return "tie"


# -----------------------------
# Pages
# -----------------------------
def page_welcome(df_reviews: pd.DataFrame):
    st.markdown('<div class="page-title">Customer Review Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">A quick read on what customers care about — and how Datadog vs Dynatrace stack up.</div>',
        unsafe_allow_html=True,
    )

    total = len(df_reviews)
    ddog_n = int((df_reviews["product"] == "datadog").sum())
    dt_n = int((df_reviews["product"] == "dynatrace").sum())

    c1, c2, c3 = st.columns([1.2, 1.2, 2.0])
    with c1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.metric("total reviews", f"{total}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.metric("datadog reviews", f"{ddog_n}")
        st.metric("dynatrace reviews", f"{dt_n}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown(
            """
            <div class="soft-card">
              <div class="small-note">
                What’s inside:
                <ul>
                  <li>firm-size mix (small / mid-market / enterprise / NA)</li>
                  <li>overall sentiment (pos / neutral / neg) using a RoBERTa sentiment model</li>
                  <li>what people mention most (aspects ranked by share of mentions)</li>
                  <li>side-by-side comparisons for each aspect (who “wins”)</li>
                </ul>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write("Quick preview of the raw rows (so you can sanity-check what’s being analyzed):")
    st.dataframe(df_reviews.head(12), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def product_page(df_reviews: pd.DataFrame, df_sent: pd.DataFrame, df_aspect: pd.DataFrame, product: str):
    title = "Datadog" if product == "datadog" else "Dynatrace"
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Firm mix, overall sentiment, and what customers keep bringing up.</div>',
        unsafe_allow_html=True,
    )

    d_reviews = df_reviews[df_reviews["product"] == product].copy()
    d_sent = df_sent[df_sent["product"] == product].copy()
    d_aspect = aspect_summary_table(df_aspect, product)

    # Firm-size mix + overall sentiment
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.write("Firm-size mix (share of reviews).")
        mix = firm_mix_pct(d_reviews)
        colors = DDOG_BLUES if product == "datadog" else DT_ORANGES
        plot_pie_percent(mix, f"{product} firm-size mix", colors=colors, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.write("Overall sentiment (by sentence). Hover shows counts.")
        sent_pct = sentiment_pct_from_sentence_df(d_sent)
        colors = DDOG_BLUES if product == "datadog" else DT_ORANGES
        plot_pie_percent(sent_pct, f"{product} overall sentiment", colors=colors, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    # Aspects ranked (bar chart in %)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write("What people mention the most (ranked). Y-axis is share of mentions; hover shows raw counts.")
    if d_aspect.empty or "aspect" not in d_aspect.columns:
        st.info("No aspect summary available.")
    else:
        # Build percent-of-mentions table for this product
        total_mentions = float(pd.to_numeric(d_aspect["mentions"], errors="coerce").fillna(0).sum()) or 1.0
        tmp = d_aspect[["aspect", "mentions"]].copy()
        tmp["mentions"] = pd.to_numeric(tmp["mentions"], errors="coerce").fillna(0).astype(int)
        tmp = tmp.sort_values("mentions", ascending=False)
        pct = pd.DataFrame(
            {
                "label": tmp["aspect"].astype(str).tolist(),
                "count": tmp["mentions"].astype(int).tolist(),
                "share": (tmp["mentions"] / total_mentions).tolist(),
            }
        )
        bar_color = "#2563EB" if product == "datadog" else "#F97316"
        plot_bar_percent(pct.head(12), "top aspects by share of mentions", color=bar_color, height=420, y_max=min(1.0, float(pct["share"].max()) * 1.25))
    st.markdown("</div>", unsafe_allow_html=True)

    # Raw tables (optional, collapsible)
    with st.expander("show the underlying aspect table"):
        if not d_aspect.empty:
            show_cols = [c for c in ["aspect", "mentions", "pos_share", "neu_share", "neg_share", "net_sentiment"] if c in d_aspect.columns]
            st.dataframe(d_aspect[show_cols].head(20), use_container_width=True, hide_index=True)
        else:
            st.write("empty")


def compare_page(df_reviews: pd.DataFrame, df_sent: pd.DataFrame, df_aspect: pd.DataFrame):
    st.markdown('<div class="page-title">Compare: Datadog vs Dynatrace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Side-by-side charts so you can see where the story diverges.</div>',
        unsafe_allow_html=True,
    )

    ddog_reviews = df_reviews[df_reviews["product"] == "datadog"].copy()
    dt_reviews = df_reviews[df_reviews["product"] == "dynatrace"].copy()

    ddog_sent = df_sent[df_sent["product"] == "datadog"].copy()
    dt_sent = df_sent[df_sent["product"] == "dynatrace"].copy()

    # Firm-size mix pies
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write("Firm-size mix (share of reviews).")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        plot_pie_percent(firm_mix_pct(ddog_reviews), "datadog firm-size mix", colors=DDOG_BLUES, height=300)
    with c2:
        plot_pie_percent(firm_mix_pct(dt_reviews), "dynatrace firm-size mix", colors=DT_ORANGES, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # Overall sentiment pies
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write("Overall sentiment (by sentence).")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        plot_pie_percent(sentiment_pct_from_sentence_df(ddog_sent), "datadog overall sentiment", colors=DDOG_BLUES, height=300)
    with c2:
        plot_pie_percent(sentiment_pct_from_sentence_df(dt_sent), "dynatrace overall sentiment", colors=DT_ORANGES, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # Overall aspects chart (percent)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write("What people mention the most (overall). Y-axis is share of mentions; hover shows raw counts.")
    overall = overall_aspect_mentions_pct(df_aspect)
    if not overall.empty:
        plot_bar_percent(overall.head(14), "top aspects overall", color="#2563EB", height=430, y_max=min(1.0, float(overall["share"].max()) * 1.25))
    else:
        st.info("No aspect summary found.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Per-aspect comparison section
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.write("Per-aspect sentiment faceoff. Each row is one aspect; pies show sentiment split within that aspect.")
    st.markdown('<div class="small-note">Winner is the one with higher net sentiment (pos - neg) / total.</div>', unsafe_allow_html=True)

    # Determine aspect ranking by overall mentions
    if df_aspect.empty or "aspect" not in df_aspect.columns or "mentions" not in df_aspect.columns:
        st.info("No aspect data available.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rank = (
        df_aspect.assign(mentions=pd.to_numeric(df_aspect["mentions"], errors="coerce").fillna(0))
        .groupby("aspect")["mentions"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    max_aspects = st.slider("how many aspects to show", 3, min(20, max(3, len(rank))), min(10, len(rank)))
    for aspect in rank[:max_aspects]:
        ddog_pct = per_aspect_sentiment_pct(df_sent, "datadog", aspect)
        dt_pct = per_aspect_sentiment_pct(df_sent, "dynatrace", aspect)

        # compute net sentiment for winner highlight
        def net_from_pct(p: pd.DataFrame) -> float:
            if p is None or p.empty:
                return 0.0
            m = {row["label"]: int(row["count"]) for _, row in p.iterrows()}
            return net_sentiment_from_counts(m.get("POS", 0), m.get("NEU", 0), m.get("NEG", 0))

        ddog_net = net_from_pct(ddog_pct)
        dt_net = net_from_pct(dt_pct)
        win = winner_label(ddog_net, dt_net)

        header = f"{aspect} (winner: {win})"
        if win != "tie":
            st.markdown(f'<div class="winner">{header}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="soft-card">{header}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            plot_pie_percent(ddog_pct, "datadog", colors=DDOG_BLUES, height=260)
        with c2:
            plot_pie_percent(dt_pct, "dynatrace", colors=DT_ORANGES, height=260)

        st.divider()

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Main
# -----------------------------
def main():
    # Sidebar nav
    st.sidebar.markdown("navigation")
    page = st.sidebar.radio("go to", ["welcome", "datadog", "dynatrace", "compare"], label_visibility="collapsed")

    # Load data
    df_reviews = load_reviews()

    # Get or generate analysis outputs
    df_sent, df_aspect = get_or_generate_outputs(df_reviews)

    # Quick safety normalization
    if "product" in df_sent.columns:
        df_sent["product"] = df_sent["product"].astype(str).str.lower().str.strip()
    if "product" in df_aspect.columns:
        df_aspect["product"] = df_aspect["product"].astype(str).str.lower().str.strip()

    # Route pages
    if page == "welcome":
        page_welcome(df_reviews)
    elif page == "datadog":
        product_page(df_reviews, df_sent, df_aspect, "datadog")
    elif page == "dynatrace":
        product_page(df_reviews, df_sent, df_aspect, "dynatrace")
    else:
        compare_page(df_reviews, df_sent, df_aspect)


if __name__ == "__main__":
    main()


