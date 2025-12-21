import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import plotly.express as px

from transformers import pipeline


# =========================
# App config + theme
# =========================
st.set_page_config(page_title="Customer Review Analyzer", page_icon="📊", layout="wide")

LIGHT_BG = "#F6FAFF"

DDOG_BLUES = ["#0B3D91", "#2F6FB7", "#8EC1FF"]      # dark → mid → light
DT_ORANGES = ["#7A2E00", "#E06B00", "#FFC285"]     # dark → mid → light


# =========================
# Aspect dictionary
# =========================
ASPECTS: Dict[str, List[str]] = {
    "pricing_billing": [
        "pricing", "price", "expensive", "cost", "bill", "billing", "overage",
        "metering", "usage-based", "invoice", "credit", "contract"
    ],
    "setup_onboarding": [
        "setup", "install", "onboarding", "configuration", "config", "deploy",
        "agent", "instrumentation", "getting started", "rollout"
    ],
    "dashboards_ux": [
        "dashboard", "ui", "ux", "interface", "visual", "visualization",
        "charts", "graphs", "layout", "navigation"
    ],
    "apm_tracing": [
        "apm", "trace", "tracing", "span", "latency", "service map",
        "distributed tracing"
    ],
    "logs_search": [
        "logs", "log", "log search", "indexing", "retention", "search",
        "query", "log explorer"
    ],
    "integrations_ecosystem": [
        "integration", "integrations", "aws", "azure", "gcp", "kubernetes",
        "k8s", "prometheus", "grafana", "splunk", "snowflake", "slack", "pagerduty",
        "databases", "terraform"
    ],
    "alert_noise": [
        "alert", "alerting", "noise", "noisy", "paging", "pager", "on-call",
        "incident", "false positive"
    ],
    "ai_root_cause": [
        "root cause", "rca", "anomaly", "davis", "watchdog", "ai", "automatic",
        "auto-detect", "diagnosis"
    ],
    "reliability_uptime": [
        "reliable", "reliability", "uptime", "stable", "outage", "downtime",
        "performance", "consistent"
    ],
    "performance_overhead": [
        "overhead", "cpu", "memory", "footprint", "agent overhead", "resource usage",
        "slow", "performance impact"
    ],
    "support_docs": [
        "support", "docs", "documentation", "help", "ticket", "customer success",
        "cs", "community"
    ],
}

ASPECT_LABELS = {
    "pricing_billing": "pricing & billing",
    "setup_onboarding": "setup & onboarding",
    "dashboards_ux": "dashboards & UI",
    "apm_tracing": "APM & tracing",
    "logs_search": "logs & search",
    "integrations_ecosystem": "integrations & ecosystem",
    "alert_noise": "alerting & noise",
    "ai_root_cause": "root-cause / AI",
    "reliability_uptime": "reliability & uptime",
    "performance_overhead": "performance overhead",
    "support_docs": "support & docs",
    "other": "other / misc",
}


# =========================
# Helpers
# =========================
def normalize_product_name(x: str) -> str:
    x = (x or "").strip().lower()
    if x in {"dyna", "dynatracee", "dyntrace", "dynatrce"}:
        return "dynatrace"
    if x in {"ddog", "data dog"}:
        return "datadog"
    return x


def palette_for_product(product: str, n: int) -> List[str]:
    product = normalize_product_name(product)
    base = DDOG_BLUES if product == "datadog" else DT_ORANGES
    return (base * ((n // len(base)) + 1))[:n]


def safe_pie_inputs(labels: List[str], values: List[float]) -> Tuple[List[str], List[float]]:
    clean = []
    for lab, val in zip(labels, values):
        try:
            v = float(val)
        except Exception:
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        clean.append((lab, v))
    if not clean:
        return [], []
    labs, vals = zip(*clean)
    return list(labs), list(vals)


def plot_pie(labels, values, title, product=None):
    labels, values = safe_pie_inputs(labels, values)
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=170)

    if not values or sum(values) <= 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "not enough data yet", ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=14)
        return fig

    colors = palette_for_product(product, len(values))
    ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )
    ax.set_title(title, fontsize=14)
    ax.axis("equal")
    return fig


def simple_sentence_split(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents if sents else [text]


def detect_aspects(sentence: str) -> List[str]:
    s = (sentence or "").lower()
    hits = []
    for aspect, kws in ASPECTS.items():
        for kw in kws:
            if kw in s:
                hits.append(aspect)
                break
    return hits if hits else ["other"]


def net_sentiment(pos: float, neg: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return (pos / total) - (neg / total)


def nice_aspect_name(a: str) -> str:
    return ASPECT_LABELS.get(a, a.replace("_", " "))


def section_header(text: str, blurb: str):
    st.markdown(
        f"""
        <div style="padding: 14px; border-radius: 14px; background: {LIGHT_BG}; border: 1px solid #E6F0FF;">
          <div style="font-size: 18px; color: #0B1220;">{text}</div>
          <div style="margin-top: 6px; font-size: 13px; color: #475569;">{blurb}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Model
# =========================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1,
        truncation=True,
    )


def map_label_to_bucket(label: str) -> str:
    label = str(label).upper()
    # LABEL_0=negative, LABEL_1=neutral, LABEL_2=positive
    if label.endswith("0"):
        return "neg"
    if label.endswith("1"):
        return "neu"
    if label.endswith("2"):
        return "pos"
    return "neu"


# =========================
# Data loading
# =========================
def load_reviews_from_repo() -> pd.DataFrame:
    root = Path(__file__).parent
    ddog_path = root / "reviews_ddog.csv"
    dt_path = root / "reviews_dt.csv"

    missing = [p.name for p in [ddog_path, dt_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSV file(s) in repo root: {missing}. "
            f"Commit them to GitHub (repo root)."
        )

    ddog = pd.read_csv(ddog_path)
    dt = pd.read_csv(dt_path)

    if "product" not in ddog.columns:
        ddog["product"] = "datadog"
    if "product" not in dt.columns:
        dt["product"] = "dynatrace"

    df = pd.concat([ddog, dt], ignore_index=True)

    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "firm" in df.columns:
        df["firm"] = df["firm"].astype(str).str.strip().str.lower()
        df.loc[df["firm"].isin(["nan", "none", "na", ""]), "firm"] = np.nan

    df["product"] = df["product"].astype(str).apply(normalize_product_name)
    df["text"] = df["text"].astype(str)

    df = df[df["text"].str.strip().astype(bool)].copy()
    return df


# =========================
# Analysis pipeline
# =========================
@st.cache_data(show_spinner=False)
def run_analysis(df_reviews: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clf = load_sentiment_model()

    rows = []
    for _, r in df_reviews.iterrows():
        rid = r.get("id")
        product = r.get("product")
        source = r.get("source")
        firm = r.get("firm", np.nan)
        text = r.get("text", "")

        for sent in simple_sentence_split(text):
            aspects = detect_aspects(sent)
            rows.append(
                {
                    "review_id": rid,
                    "product": product,
                    "source": source,
                    "firm": firm,
                    "sentence": sent,
                    "aspects": aspects,
                }
            )

    sentence_df = pd.DataFrame(rows)

    # Sentiment in batches
    batch_size = 64
    sentiments = []
    sents = sentence_df["sentence"].tolist()
    for i in range(0, len(sents), batch_size):
        out = clf(sents[i: i + batch_size])
        sentiments.extend(out)

    sentence_df["sentiment_raw"] = [o.get("label") for o in sentiments]
    sentence_df["sentiment"] = sentence_df["sentiment_raw"].apply(map_label_to_bucket)

    exploded = sentence_df.explode("aspects").rename(columns={"aspects": "aspect"})

    agg = (
        exploded.groupby(["product", "aspect", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    pivot = (
        agg.pivot_table(index=["product", "aspect"], columns="sentiment", values="count", fill_value=0)
        .reset_index()
    )

    for col in ["neg", "neu", "pos"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["mentions"] = pivot["neg"] + pivot["neu"] + pivot["pos"]
    pivot["neg_share"] = np.where(pivot["mentions"] > 0, pivot["neg"] / pivot["mentions"], 0.0)
    pivot["neu_share"] = np.where(pivot["mentions"] > 0, pivot["neu"] / pivot["mentions"], 0.0)
    pivot["pos_share"] = np.where(pivot["mentions"] > 0, pivot["pos"] / pivot["mentions"], 0.0)
    pivot["net_sentiment"] = pivot.apply(lambda x: net_sentiment(x["pos"], x["neg"], x["mentions"]), axis=1)

    return sentence_df, pivot


# =========================
# Plotly % bar (with hover counts)
# =========================
def percent_bar_overall_aspects(aspect_summary: pd.DataFrame, include_other: bool):
    overall = aspect_summary.groupby("aspect", as_index=False)["mentions"].sum()
    if not include_other:
        overall = overall[overall["aspect"] != "other"].copy()

    total = overall["mentions"].sum()
    if total <= 0:
        st.info("not enough data to chart yet.")
        return

    overall["share"] = overall["mentions"] / total
    overall["share_pct"] = overall["share"] * 100.0
    overall["aspect_name"] = overall["aspect"].apply(nice_aspect_name)

    overall = overall.sort_values("share_pct", ascending=False)

    fig = px.bar(
        overall,
        x="aspect_name",
        y="share_pct",
        hover_data={
            "mentions": True,
            "share_pct": ":.1f",
        },
        labels={
            "aspect_name": "aspect",
            "share_pct": "share of mentions (%)",
        },
    )
    fig.update_layout(
        height=430,
        margin=dict(l=50, r=20, t=20, b=120),
        xaxis_tickangle=-55,
        yaxis=dict(range=[0, min(100, max(10, overall["share_pct"].max() * 1.15))]),
    )
    fig.update_traces(marker_color="#0B3D91")
    st.plotly_chart(fig, use_container_width=True)


# =========================
# Pages
# =========================
def page_welcome(df_reviews: pd.DataFrame, sentence_df: pd.DataFrame, aspect_summary: pd.DataFrame):
    cols = st.columns([2.2, 1.0])
    with cols[0]:
        st.markdown(
            """
            <div style="font-size:40px; line-height:1.1; color:#0B1220;">
                Customer Review Analyzer
            </div>
            <div style="margin-top:8px; font-size:16px; color:#334155;">
                I put this together to quickly sanity-check what people actually talk about in Datadog vs Dynatrace reviews —
                and whether the tone is mostly positive, neutral, or negative — without manually reading everything.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top:12px; padding:14px; border-radius:14px; background:#0B3D9112; border:1px solid #0B3D9122;">
              <div style="font-size:14px; color:#0B1220;">
                quick tour:
              </div>
              <div style="margin-top:6px; font-size:14px; color:#334155;">
                • firm-size mix + sentiment for each product<br/>
                • what topics come up most (as % of mentions)<br/>
                • sentiment by topic, side-by-side
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        total = len(df_reviews)
        ddog_n = (df_reviews["product"] == "datadog").sum()
        dt_n = (df_reviews["product"] == "dynatrace").sum()

        st.markdown(
            f"""
            <div style="padding:14px; border-radius:14px; background:{LIGHT_BG}; border:1px solid #E6F0FF;">
              <div style="font-size:14px; color:#64748B;">dataset size</div>
              <div style="font-size:34px; color:#0B1220; margin-top:2px;">{total}</div>
              <div style="margin-top:8px; font-size:14px; color:#334155;">
                datadog: <span style="color:#0B3D91;">{ddog_n}</span><br/>
                dynatrace: <span style="color:#E06B00;">{dt_n}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    section_header(
        "quick preview",
        "a peek at the built-in dataset this app is using (straight from the repo).",
    )
    st.dataframe(df_reviews.head(15), use_container_width=True, hide_index=True)

    st.write("")
    section_header(
        "what people mention the most (overall)",
        "share of all aspect mentions across both products. hover to see the underlying counts.",
    )
    include_other = st.toggle("include other / misc", value=False)
    percent_bar_overall_aspects(aspect_summary, include_other=include_other)


def firm_mix_chart(df_reviews: pd.DataFrame, product: str):
    sub = df_reviews[df_reviews["product"] == product].copy()
    if "firm" not in sub.columns or sub["firm"].isna().all():
        st.info("no firm column found for this dataset yet (optional).")
        return

    counts = (
        sub["firm"].dropna().astype(str).str.lower().value_counts()
        .reindex(["small", "mid-market", "enterprise"])
        .fillna(0)
    )
    labels = ["small", "mid-market", "enterprise"]
    values = [counts.get(x, 0) for x in labels]

    st.pyplot(plot_pie(labels, values, f"{product} firm-size mix", product=product), use_container_width=True)

    table = pd.DataFrame({"firm": labels, "count": values})
    table["share"] = np.where(table["count"].sum() > 0, table["count"] / table["count"].sum(), 0.0)
    table["share"] = (table["share"] * 100).round(1).astype(str) + "%"
    st.dataframe(table, use_container_width=True, hide_index=True)


def overall_sentiment_chart(sentence_df: pd.DataFrame, product: str):
    sub = sentence_df[sentence_df["product"] == product].copy()
    order = ["neg", "neu", "pos"]
    counts = sub["sentiment"].value_counts().reindex(order).fillna(0)
    st.pyplot(
        plot_pie([x.upper() for x in order], [counts[x] for x in order], f"{product} overall sentiment (by sentence)", product=product),
        use_container_width=True,
    )


def page_compare(df_reviews: pd.DataFrame, sentence_df: pd.DataFrame, aspect_summary: pd.DataFrame):
    st.markdown(
        """
        <div style="font-size:34px; color:#0B1220;">
            compare: datadog vs dynatrace
        </div>
        <div style="margin-top:6px; font-size:15px; color:#334155;">
            side-by-side views so you can quickly see what topics dominate and how the tone differs.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    section_header(
        "firm-size mix (by product)",
        "this is just the breakdown of your pulled reviews by firm size bucket.",
    )
    cols = st.columns(2)
    with cols[0]:
        firm_mix_chart(df_reviews, "datadog")
    with cols[1]:
        firm_mix_chart(df_reviews, "dynatrace")

    st.write("")
    section_header(
        "overall sentiment (by product)",
        "sentence-level sentiment. a single review can contribute multiple sentences.",
    )
    cols = st.columns(2)
    with cols[0]:
        overall_sentiment_chart(sentence_df, "datadog")
    with cols[1]:
        overall_sentiment_chart(sentence_df, "dynatrace")

    st.write("")
    section_header(
        "what people mention the most (overall)",
        "share of all aspect mentions across both products. hover to see the raw counts.",
    )
    include_other = st.toggle("include other / misc (compare page)", value=False)
    percent_bar_overall_aspects(aspect_summary, include_other=include_other)

    st.write("")
    section_header(
        "top aspects (ranked by mentions)",
        "ranked by how often the topic appears in sentences (counts shown).",
    )
    ranked = (
        aspect_summary.groupby("aspect")["mentions"].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranked["aspect"] = ranked["aspect"].apply(nice_aspect_name)
    st.dataframe(ranked.head(25), use_container_width=True, hide_index=True)


# =========================
# Main
# =========================
def main():
    st.markdown(
        f"""
        <style>
          .stApp {{
            background: white;
          }}
          [data-testid="stSidebar"] {{
            background: {LIGHT_BG};
            border-right: 1px solid #E6F0FF;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df_reviews = load_reviews_from_repo()

    with st.spinner("running sentiment + aspect extraction…"):
        sentence_df, aspect_summary = run_analysis(df_reviews)

    st.sidebar.markdown(
        """
        <div style="font-size:18px; color:#0B1220; margin: 8px 0 10px 0;">
          navigation
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio("go to", ["welcome", "compare"], label_visibility="collapsed")

    if page == "welcome":
        page_welcome(df_reviews, sentence_df, aspect_summary)
    elif page == "compare":
        page_compare(df_reviews, sentence_df, aspect_summary)


if __name__ == "__main__":
    main()
