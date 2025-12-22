from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_BG = "#f6f9ff"
CARD_BG = "#ffffff"
ACCENT = "#2f6fed"
TEXT = "#0f172a"
MUTED = "#475569"
BORDER = "#e6eefc"

DDOG_BLUES = ["#d7e9ff", "#a9d1ff", "#6fb2ff", "#2f6fed"]
DT_ORANGES = ["#ffe2cf", "#ffc29a", "#ff9a57", "#f26a1b"]

SENTIMENT_ORDER = ["NEG", "NEU", "POS"]
SENTIMENT_COLORS_DDOG = {"NEG": "#a9d1ff", "NEU": "#2f6fed", "POS": "#1648c8"}
SENTIMENT_COLORS_DT = {"NEG": "#ffc29a", "NEU": "#ff9a57", "POS": "#f26a1b"}

REQUIRED_COLS = ["id", "product", "source", "text"]

# ✅ IMPORTANT: keep sidebar expanded so your nav is visible on Streamlit Cloud
st.set_page_config(
    page_title="Customer Review Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
html, body, [class*="css"] {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  color: {TEXT};
}}
.stApp {{
  background: {APP_BG};
}}
div.block-container {{
  padding-top: 1.25rem;
  padding-bottom: 2.5rem;
}}

/* You can hide the header/toolbar, but DON'T hide the sidebar. */
header[data-testid="stHeader"] {{
  display: none;
}}
div[data-testid="stToolbar"] {{
  visibility: hidden;
  height: 0px;
  position: fixed;
}}

/* ✅ Guardrails: force sidebar + nav content to remain visible */
section[data-testid="stSidebar"] {{
  display: block !important;
  visibility: visible !important;
  height: 100% !important;
}}
section[data-testid="stSidebar"] * {{
  visibility: visible !important;
}}

/* Card UI */
.card {{
  background: {CARD_BG};
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
}}
.muted {{
  color: {MUTED};
}}
.pill {{
  display: inline-block;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-size: 0.85rem;
  background: rgba(47, 111, 237, 0.10);
  color: {ACCENT};
  border: 1px solid rgba(47, 111, 237, 0.15);
}}
.divider {{
  height: 1px;
  background: {BORDER};
  margin: 12px 0px;
}}
.badge {{
  display: inline-block;
  padding: 0.24rem 0.55rem;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 650;
  margin-left: 0.55rem;
  border: 1px solid rgba(0,0,0,0.06);
  vertical-align: middle;
}}
.badge-ddog {{
  background: rgba(47,111,237,0.12);
  color: #1f4fd6;
}}
.badge-dt {{
  background: rgba(242,106,27,0.12);
  color: #b54712;
}}
.badge-tie {{
  background: rgba(148,163,184,0.18);
  color: #334155;
}}
.hrow {{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  flex-wrap:wrap;
  margin-top: 6px;
}}
.hleft {{
  font-size: 1.15rem;
  font-weight: 750;
}}
.hright {{
  color: {MUTED};
  font-size: 0.95rem;
}}
</style>
""",
    unsafe_allow_html=True,
)


def _normalize_na(x) -> str:
    s = "" if x is None else str(x).strip()
    if s.lower() in ["", "na", "nan", "none", "null"]:
        return "NA"
    return s


def _clean_reviews_df(df: pd.DataFrame, product: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV for {product} is missing required columns: {missing}")

    df["product"] = product
    df["source"] = df["source"].astype(str).str.strip()
    df["text"] = df["text"].astype(str)
    df["id"] = df["id"].astype(str).str.strip()

    if "firm" in df.columns:
        df["firm"] = df["firm"].apply(_normalize_na)
    else:
        df["firm"] = "NA"

    firm_map = {
        "mid market": "mid-market",
        "midmarket": "mid-market",
        "enterprise ": "enterprise",
        "small ": "small",
    }
    df["firm"] = df["firm"].astype(str).str.strip().str.lower().replace(firm_map)
    df["firm"] = df["firm"].apply(_normalize_na)

    df = df.drop_duplicates(subset=["product", "source", "text"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_reviews() -> pd.DataFrame:
    ddog = pd.read_csv("reviews_ddog.csv")
    dt = pd.read_csv("reviews_dt.csv")
    ddog = _clean_reviews_df(ddog, "datadog")
    dt = _clean_reviews_df(dt, "dynatrace")
    ddog["id"] = ddog["id"].apply(lambda x: f"ddog_{x}")
    dt["id"] = dt["id"].apply(lambda x: f"dt_{x}")
    return pd.concat([ddog, dt], ignore_index=True)


def split_into_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    t = " ".join(text.strip().split())
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    out = [p.strip() for p in parts if len(p.strip()) >= 3]
    return out[:25]


@dataclass
class SentimentResult:
    label: str
    neg: float
    neu: float
    pos: float


@st.cache_resource(show_spinner=False)
def get_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def vader_sentiment(text: str) -> SentimentResult:
    sia = get_vader()
    if sia is None:
        low = text.lower()
        neg_words = ["bad", "terrible", "awful", "bug", "slow", "expensive", "hate", "broken", "issue", "noisy"]
        pos_words = ["good", "great", "excellent", "love", "fast", "reliable", "amazing", "easy", "helpful", "powerful"]
        score = sum(w in low for w in pos_words) - sum(w in low for w in neg_words)
        if score >= 1:
            return SentimentResult("POS", 0.05, 0.10, 0.85)
        if score <= -1:
            return SentimentResult("NEG", 0.85, 0.10, 0.05)
        return SentimentResult("NEU", 0.10, 0.80, 0.10)

    s = sia.polarity_scores(text)
    comp = s["compound"]
    if comp >= 0.20:
        label = "POS"
    elif comp <= -0.20:
        label = "NEG"
    else:
        label = "NEU"
    return SentimentResult(label, float(s["neg"]), float(s["neu"]), float(s["pos"]))


ASPECTS: Dict[str, List[str]] = {
    "integrations_ecosystem": ["integration", "integrations", "ecosystem", "plugin", "aws", "azure", "gcp", "kubernetes", "k8s", "slack", "jira"],
    "apm_tracing": ["apm", "trace", "tracing", "span", "latency", "profil", "instrumentation"],
    "dashboards_ux": ["dashboard", "dashboards", "ui", "ux", "visual", "visualization", "chart", "graph", "interface", "intuitive", "easy to use"],
    "logs_search": ["logs", "log", "search", "query", "index", "parsing", "filter"],
    "alerts_noise": ["alert", "alerts", "paging", "pager", "noise", "false positive", "threshold", "monitor"],
    "ai_root_cause": ["root cause", "rca", "ai", "anomaly", "anomalies", "auto", "insight", "cause analysis"],
    "performance_overhead": ["agent", "overhead", "cpu", "memory", "resource", "performance impact"],
    "pricing_billing": ["price", "pricing", "bill", "billing", "cost", "expensive", "overage", "usage", "contract"],
    "setup_onboarding": ["setup", "install", "onboarding", "configuration", "config", "deploy", "deployment", "getting started"],
    "reliability_uptime": ["reliable", "reliability", "uptime", "stable", "availability", "downtime"],
    "support_docs": ["support", "ticket", "docs", "documentation", "help", "response time"],
}

ASPECT_DISPLAY = {
    "integrations_ecosystem": "integrations & ecosystem",
    "apm_tracing": "apm & tracing",
    "dashboards_ux": "dashboards & ux",
    "logs_search": "logs & search",
    "alerts_noise": "alerts & noise",
    "ai_root_cause": "ai / root-cause",
    "performance_overhead": "performance / overhead",
    "pricing_billing": "pricing & billing",
    "setup_onboarding": "setup & onboarding",
    "reliability_uptime": "reliability / uptime",
    "support_docs": "support & docs",
    "other": "other / misc",
}


def detect_aspects(sentence: str) -> List[str]:
    s = sentence.lower()
    hits = []
    for asp, kws in ASPECTS.items():
        if any(kw in s for kw in kws):
            hits.append(asp)
    return hits or ["other"]


@st.cache_data(show_spinner=True)
def build_sentence_level(df_reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_reviews.iterrows():
        firm = _normalize_na(r.get("firm", "NA"))
        for s in split_into_sentences(r["text"]):
            sent = vader_sentiment(s)
            for asp in detect_aspects(s):
                rows.append(
                    {
                        "id": r["id"],
                        "product": r["product"],
                        "source": r["source"],
                        "firm": firm,
                        "sentence": s,
                        "aspect": asp,
                        "sentiment": sent.label,
                        "neg": sent.neg,
                        "neu": sent.neu,
                        "pos": sent.pos,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["firm"] = df["firm"].apply(_normalize_na)
    return df


@st.cache_data(show_spinner=False)
def firm_mix(df_reviews: pd.DataFrame) -> pd.DataFrame:
    df = df_reviews.copy()
    df["firm"] = df["firm"].apply(_normalize_na).astype(str).str.lower()
    out = df.groupby(["product", "firm"]).size().reset_index(name="count")
    out["pct"] = out.groupby("product")["count"].transform(lambda x: x / max(int(x.sum()), 1))
    out = out.sort_values(["product", "count"], ascending=[True, False]).reset_index(drop=True)
    out = out.rename(columns={"firm": "label"})
    return out


@st.cache_data(show_spinner=False)
def overall_sentiment(df_sent: pd.DataFrame) -> pd.DataFrame:
    g = df_sent.groupby(["product", "sentiment"]).size().reset_index(name="count")
    all_idx = pd.MultiIndex.from_product(
        [sorted(df_sent["product"].unique()), SENTIMENT_ORDER],
        names=["product", "sentiment"],
    )
    g = g.set_index(["product", "sentiment"]).reindex(all_idx, fill_value=0).reset_index()
    g["pct"] = g.groupby("product")["count"].transform(lambda x: x / max(int(x.sum()), 1))
    g = g.rename(columns={"sentiment": "label"})
    return g


@st.cache_data(show_spinner=False)
def summarize_aspects(df_sent: pd.DataFrame) -> pd.DataFrame:
    g = df_sent.groupby(["product", "aspect"], as_index=False).agg(
        mentions=("sentence", "count"),
        neg=("sentiment", lambda x: int((x == "NEG").sum())),
        neu=("sentiment", lambda x: int((x == "NEU").sum())),
        pos=("sentiment", lambda x: int((x == "POS").sum())),
    )
    g["neg_share"] = g["neg"] / g["mentions"].clip(lower=1)
    g["neu_share"] = g["neu"] / g["mentions"].clip(lower=1)
    g["pos_share"] = g["pos"] / g["mentions"].clip(lower=1)
    g["net_sentiment"] = g["pos_share"] - g["neg_share"]
    g["aspect_display"] = g["aspect"].map(ASPECT_DISPLAY).fillna(g["aspect"])
    return g.sort_values(["product", "mentions"], ascending=[True, False]).reset_index(drop=True)


def plot_pie_percent(df_pct: pd.DataFrame, title: str, colors: List[str], height: int = 300):
    if df_pct is None or df_pct.empty:
        st.info("No data available for this chart yet.")
        return
    needed = {"label", "pct", "count"}
    if not needed.issubset(set(df_pct.columns)):
        st.error(f"Chart data missing columns: {sorted(list(needed - set(df_pct.columns)))}")
        return

    dfp = df_pct.copy()
    dfp["label"] = dfp["label"].astype(str)

    fig = px.pie(
        dfp,
        names="label",
        values="pct",
        hover_data={"count": True, "pct": ":.1%"},
        color_discrete_sequence=colors,
        title=title,
        height=height,
    )
    fig.update_traces(textposition="inside", textinfo="percent", insidetextfont=dict(size=12))
    fig.update_layout(
        margin=dict(l=8, r=8, t=55, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_bar_pct(
    df: pd.DataFrame,
    title: str,
    x_col: str,
    pct_col: str,
    count_col: str,
    color: str,
    height: int = 420,
    x_label: str = "aspect",
    y_label: str = "share",
):
    if df is None or df.empty:
        st.info("No data available for this chart yet.")
        return

    d = df.copy()
    d[pct_col] = d[pct_col].astype(float).clip(lower=0.0)

    fig = px.bar(
        d,
        x=x_col,
        y=pct_col,
        text=d[pct_col].map(lambda v: f"{v*100:.1f}%"),
        hover_data={count_col: True, pct_col: ":.1%"},
        height=height,
    )
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=".0%", title=y_label, rangemode="tozero")
    fig.update_xaxes(title=x_label, tickangle=45)
    fig.update_layout(
        title=title,
        margin=dict(l=8, r=8, t=55, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)


def winner_badge_html(ddog_val: float, dt_val: float) -> str:
    eps = 1e-9
    if ddog_val > dt_val + eps:
        return '<span class="badge badge-ddog">datadog wins</span>'
    if dt_val > ddog_val + eps:
        return '<span class="badge badge-dt">dynatrace wins</span>'
    return '<span class="badge badge-tie">tie</span>'


def welcome_page(df_reviews: pd.DataFrame):
    st.markdown(
        """
<div class="card">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    <span class="pill">built-in dataset</span>
    <span class="pill">no upload needed</span>
    <span class="pill">datadog vs dynatrace</span>
  </div>
  <div class="divider"></div>
  <h1 style="margin:0;">Customer Review Analyzer</h1>
  <p class="muted" style="margin-top:8px;margin-bottom:0;">
    I pulled a review dataset for Datadog and Dynatrace and built this so I can quickly see (1) who’s getting better sentiment,
    (2) what people talk about most, and (3) how that varies by firm size.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick preview")
        st.markdown('<p class="muted">For reference (first few rows).</p>', unsafe_allow_html=True)
        st.dataframe(df_reviews.head(12), use_container_width=True, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset size")
        total = len(df_reviews)
        ddog = int((df_reviews["product"] == "datadog").sum())
        dt = int((df_reviews["product"] == "dynatrace").sum())
        st.metric("Total reviews", f"{total}")
        st.metric("Datadog reviews", f"{ddog}")
        st.metric("Dynatrace reviews", f"{dt}")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="muted">Use the left nav to jump into Datadog, Dynatrace, or the side-by-side compare page.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def product_page(df_reviews: pd.DataFrame, df_sent: pd.DataFrame, df_aspect: pd.DataFrame, product: str):
    pretty = "Datadog" if product == "datadog" else "Dynatrace"
    st.markdown(
        f"""
<div class="card">
  <h2 style="margin:0;">{pretty}</h2>
  <p class="muted" style="margin-top:8px;margin-bottom:0;">
    Firm-size mix, sentence-level sentiment, and the topics that show up the most in the reviews I pulled.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    d_reviews = df_reviews[df_reviews["product"] == product].copy()
    d_sent = df_sent[df_sent["product"] == product].copy()
    d_aspect = df_aspect[df_aspect["product"] == product].copy()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Firm-size mix")
    st.markdown('<p class="muted">Breakdown of reviews by firm-size bucket (as labeled on the source site).</p>', unsafe_allow_html=True)
    mix = firm_mix(d_reviews)
    colors = DDOG_BLUES if product == "datadog" else DT_ORANGES
    plot_pie_percent(mix[["label", "pct", "count"]], f"{pretty} firm-size mix", colors=colors, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Overall sentiment")
    st.markdown('<p class="muted">Sentence-level sentiment (a single review can contribute multiple sentences). Hover for counts.</p>', unsafe_allow_html=True)
    ov = overall_sentiment(d_sent)
    ov = ov[ov["product"] == product].copy()
    sent_colors = SENTIMENT_COLORS_DDOG if product == "datadog" else SENTIMENT_COLORS_DT
    plot_pie_percent(
        ov[["label", "pct", "count"]],
        f"{pretty} sentiment (by sentence)",
        colors=[sent_colors[k] for k in SENTIMENT_ORDER],
        height=280,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What people mention the most")
    st.markdown('<p class="muted">Ranked by share of mentions. Axis is %; hover bars to see raw counts.</p>', unsafe_allow_html=True)

    top = d_aspect.sort_values("mentions", ascending=False).copy()
    top["mention_share"] = top["mentions"] / max(int(top["mentions"].sum()), 1)
    top = top.head(12)

    plot_bar_pct(
        top,
        title="top aspects (share of mentions)",
        x_col="aspect_display",
        pct_col="mention_share",
        count_col="mentions",
        color=ACCENT if product == "datadog" else "#f26a1b",
        height=420,
        x_label="aspect",
        y_label="share of mentions",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def compare_page(df_reviews: pd.DataFrame, df_sent: pd.DataFrame, df_aspect: pd.DataFrame):
    st.markdown(
        """
<div class="card">
  <h2 style="margin:0;">Compare: Datadog vs Dynatrace</h2>
  <p class="muted" style="margin-top:8px;margin-bottom:0;">
    Side-by-side views so I can see what dominates and where they diverge.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Firm-size mix")
    st.markdown('<p class="muted">Share of reviews labeled small vs mid-market vs enterprise.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        dd = firm_mix(df_reviews[df_reviews["product"] == "datadog"])
        plot_pie_percent(dd[["label", "pct", "count"]], "datadog firm-size mix", colors=DDOG_BLUES, height=260)
    with c2:
        dt = firm_mix(df_reviews[df_reviews["product"] == "dynatrace"])
        plot_pie_percent(dt[["label", "pct", "count"]], "dynatrace firm-size mix", colors=DT_ORANGES, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Overall sentiment")
    st.markdown('<p class="muted">Sentence-level sentiment for each product.</p>', unsafe_allow_html=True)
    ov = overall_sentiment(df_sent)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        dd = ov[ov["product"] == "datadog"]
        plot_pie_percent(dd[["label", "pct", "count"]], "datadog sentiment", colors=[SENTIMENT_COLORS_DDOG[k] for k in SENTIMENT_ORDER], height=260)
    with c2:
        dt = ov[ov["product"] == "dynatrace"]
        plot_pie_percent(dt[["label", "pct", "count"]], "dynatrace sentiment", colors=[SENTIMENT_COLORS_DT[k] for k in SENTIMENT_ORDER], height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What people care about (ranked)")
    st.markdown('<p class="muted">Ranked by total mentions across both products. Bars show share-of-mentions per product.</p>', unsafe_allow_html=True)

    total_mentions = df_aspect.groupby("aspect_display", as_index=False)["mentions"].sum().rename(columns={"mentions": "total_mentions"})
    top_aspects = total_mentions.sort_values("total_mentions", ascending=False).head(10)["aspect_display"].tolist()

    a = df_aspect[df_aspect["aspect_display"].isin(top_aspects)].copy()
    a["mention_share"] = a.groupby("product")["mentions"].transform(lambda x: x / max(int(x.sum()), 1))

    c1, c2 = st.columns(2, gap="large")
    with c1:
        dd = a[a["product"] == "datadog"].sort_values("mention_share", ascending=False)
        plot_bar_pct(dd, "datadog: top aspects (share of mentions)", "aspect_display", "mention_share", "mentions", ACCENT, height=420)
    with c2:
        dt = a[a["product"] == "dynatrace"].sort_values("mention_share", ascending=False)
        plot_bar_pct(dt, "dynatrace: top aspects (share of mentions)", "aspect_display", "mention_share", "mentions", "#f26a1b", height=420)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Per-aspect sentiment (side-by-side)")
    st.markdown('<p class="muted">For each aspect: sentiment split for Datadog vs Dynatrace. Badge = who wins on net sentiment.</p>', unsafe_allow_html=True)

    sent_g = df_sent.groupby(["product", "aspect", "sentiment"]).size().reset_index(name="count")
    all_idx = pd.MultiIndex.from_product(
        [sorted(df_sent["product"].unique()), sorted(df_sent["aspect"].unique()), SENTIMENT_ORDER],
        names=["product", "aspect", "sentiment"],
    )
    sent_g = sent_g.set_index(["product", "aspect", "sentiment"]).reindex(all_idx, fill_value=0).reset_index()
    sent_g["pct"] = sent_g.groupby(["product", "aspect"])["count"].transform(lambda x: x / max(int(x.sum()), 1))

    net = df_aspect.set_index(["product", "aspect"])["net_sentiment"].to_dict()
    aspect_rank = (
        df_aspect.groupby("aspect", as_index=False)["mentions"]
        .sum()
        .sort_values("mentions", ascending=False)
        .head(10)["aspect"]
        .tolist()
    )

    for asp in aspect_rank:
        asp_name = ASPECT_DISPLAY.get(asp, asp)
        dd_net = float(net.get(("datadog", asp), 0.0))
        dt_net = float(net.get(("dynatrace", asp), 0.0))
        badge = winner_badge_html(dd_net, dt_net)

        st.markdown(
            f"""
<div class="hrow">
  <div class="hleft">{asp_name} {badge}</div>
  <div class="hright">net sentiment: datadog {dd_net:+.2f} vs dynatrace {dt_net:+.2f}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2, gap="large")
        with c1:
            dd = sent_g[(sent_g["product"] == "datadog") & (sent_g["aspect"] == asp)].rename(columns={"sentiment": "label"})
            dd["label"] = pd.Categorical(dd["label"], categories=SENTIMENT_ORDER, ordered=True)
            dd = dd.sort_values("label")
            plot_pie_percent(dd[["label", "pct", "count"]], "datadog sentiment", colors=[SENTIMENT_COLORS_DDOG[k] for k in SENTIMENT_ORDER], height=240)
        with c2:
            dt = sent_g[(sent_g["product"] == "dynatrace") & (sent_g["aspect"] == asp)].rename(columns={"sentiment": "label"})
            dt["label"] = pd.Categorical(dt["label"], categories=SENTIMENT_ORDER, ordered=True)
            dt = dt.sort_values("label")
            plot_pie_percent(dt[["label", "pct", "count"]], "dynatrace sentiment", colors=[SENTIMENT_COLORS_DT[k] for k in SENTIMENT_ORDER], height=240)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    df_reviews = load_reviews()

    with st.spinner("Crunching sentence-level sentiment + aspects…"):
        df_sent = build_sentence_level(df_reviews)
        df_aspect = summarize_aspects(df_sent)

    # ✅ Sidebar navigation (your app’s nav)
    st.sidebar.markdown("## Navigation")

    page = st.sidebar.radio(
        "Go to",
        ["Welcome", "Compare", "Datadog", "Dynatrace"],
        index=0,
        label_visibility="collapsed",
    )

    if page == "Welcome":
        welcome_page(df_reviews)
    elif page == "Compare":
        compare_page(df_reviews, df_sent, df_aspect)
    elif page == "Datadog":
        product_page(df_reviews, df_sent, df_aspect, "datadog")
    else:
        product_page(df_reviews, df_sent, df_aspect, "dynatrace")


if __name__ == "__main__":
    main()
