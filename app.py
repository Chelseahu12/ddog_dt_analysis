import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from transformers import pipeline

# -----------------------------
# Config / theme
# -----------------------------
st.set_page_config(
    page_title="Customer Review Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_ACCENT = "#2F6FED"  # general blue accent
DDOG_BLUES = ["#A7C7FF", "#79A8FF", "#4F86F7", "#2F6FED", "#1E4FBF"]
DT_ORANGES = ["#FFD2A8", "#FFB36B", "#FF9440", "#F06B1A", "#C84C10"]

SENTIMENT_COLORS = {
    "NEG": "#E74C3C",
    "NEU": "#95A5A6",
    "POS": "#2ECC71",
}

# -----------------------------
# Aspect dictionary (edit freely)
# -----------------------------
ASPECTS = {
    "integrations_ecosystem": [
        "integration", "integrations", "ecosystem", "plugins", "plugin",
        "open telemetry", "opentelemetry", "otel", "third-party", "third party",
        "aws", "azure", "gcp", "kubernetes", "k8s"
    ],
    "apm_tracing": [
        "apm", "trace", "tracing", "spans", "latency", "profiling", "profiler"
    ],
    "logs_search": [
        "logs", "log", "logging", "search", "query", "queries", "indexing"
    ],
    "dashboards_ux": [
        "dashboard", "dashboards", "ui", "ux", "interface", "visualization",
        "visualisations", "charts", "graph", "navigation"
    ],
    "ai_root_cause": [
        "root cause", "rca", "anomaly", "anomalies", "ai", "a.i.", "ml",
        "alert correlation", "correlation"
    ],
    "alert_noise": [
        "alert", "alerts", "paging", "pager", "noise", "noisy", "fatigue",
        "false positive", "false positives"
    ],
    "performance_overhead": [
        "overhead", "agent overhead", "performance", "resource usage",
        "cpu", "memory", "footprint"
    ],
    "pricing_billing": [
        "price", "pricing", "cost", "billed", "billing", "bill", "expensive",
        "overage", "overages", "usage-based", "usage based", "contract"
    ],
    "setup_onboarding": [
        "setup", "install", "onboarding", "configuration", "configure",
        "deployment", "agent", "getting started", "rollout"
    ],
    "reliability_uptime": [
        "reliable", "reliability", "uptime", "stable", "stability",
        "outage", "outages", "down", "downtime"
    ],
    "support_docs": [
        "support", "documentation", "docs", "help", "ticket", "tickets",
        "response time", "knowledge base"
    ],
}

ASPECT_PRETTY = {
    "integrations_ecosystem": "integrations + ecosystem",
    "apm_tracing": "apm + tracing",
    "logs_search": "logs + search",
    "dashboards_ux": "dashboards + ux",
    "ai_root_cause": "ai + root cause",
    "alert_noise": "alert noise",
    "performance_overhead": "performance + overhead",
    "pricing_billing": "pricing + billing",
    "setup_onboarding": "setup + onboarding",
    "reliability_uptime": "reliability + uptime",
    "support_docs": "support + docs",
}

# -----------------------------
# Helpers
# -----------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        .big-title {{
            font-size: 44px;
            font-weight: 650;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }}
        .subtle {{
            color: rgba(20, 30, 50, 0.70);
            font-size: 16px;
            margin-top: 0px;
        }}
        .pill {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(47, 111, 237, 0.10);
            border: 1px solid rgba(47, 111, 237, 0.25);
            color: rgba(20, 30, 50, 0.85);
            font-size: 14px;
            margin: 6px 6px 0px 0px;
        }}
        .card {{
            padding: 14px 14px;
            border-radius: 16px;
            background: rgba(47, 111, 237, 0.06);
            border: 1px solid rgba(47, 111, 237, 0.18);
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 650;
            margin-top: 8px;
            margin-bottom: 6px;
        }}
        .blurb {{
            color: rgba(20, 30, 50, 0.70);
            font-size: 14px;
            margin-top: -2px;
            margin-bottom: 10px;
        }}
        .win {{
            padding: 8px 10px;
            border-radius: 12px;
            background: rgba(46, 204, 113, 0.10);
            border: 1px solid rgba(46, 204, 113, 0.22);
            display: inline-block;
        }}
        .lose {{
            padding: 8px 10px;
            border-radius: 12px;
            background: rgba(231, 76, 60, 0.08);
            border: 1px solid rgba(231, 76, 60, 0.18);
            display: inline-block;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # normalize common columns
    if "review" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"review": "text"})
    if "company" in df.columns and "product" not in df.columns:
        df = df.rename(columns={"company": "product"})
    # basic checks
    required = {"id", "product", "source", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(list(missing))}")
    df["product"] = df["product"].astype(str).str.strip().str.lower()
    df["source"] = df["source"].astype(str).str.strip()
    df["text"] = df["text"].astype(str)
    if "firm" in df.columns:
        df["firm"] = df["firm"].astype(str).str.strip().str.lower()
    else:
        df["firm"] = np.nan
    return df


def split_into_sentences(text: str) -> list[str]:
    # lightweight sentence splitting to avoid nltk downloads
    # handles ., !, ?, and newlines
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    sents = [p.strip() for p in parts if p and p.strip()]
    return sents


def build_sentiment_pipe():
    # cached so Streamlit doesn’t reload the model each rerun
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=256,
    )


def map_label_to_bucket(label: str) -> str:
    # cardiffnlp model outputs LABEL_0/1/2
    # (0 = negative, 1 = neutral, 2 = positive) for this checkpoint
    label = str(label).upper().strip()
    if label in {"LABEL_0", "NEGATIVE", "NEG"}:
        return "NEG"
    if label in {"LABEL_1", "NEUTRAL", "NEU"}:
        return "NEU"
    if label in {"LABEL_2", "POSITIVE", "POS"}:
        return "POS"
    # fallback
    return "NEU"


def detect_aspects(sent: str) -> list[str]:
    s = sent.lower()
    hits = []
    for aspect, keys in ASPECTS.items():
        for k in keys:
            if k in s:
                hits.append(aspect)
                break
    if not hits:
        hits = ["other"]
    return hits


@st.cache_data(show_spinner=False)
def preprocess_reviews(df_all: pd.DataFrame) -> pd.DataFrame:
    # explode reviews -> sentences
    rows = []
    for _, r in df_all.iterrows():
        sents = split_into_sentences(r["text"])
        for sent in sents:
            rows.append(
                {
                    "review_id": r["id"],
                    "product": r["product"],
                    "source": r["source"],
                    "firm": r.get("firm", np.nan),
                    "sentence": sent,
                }
            )
    out = pd.DataFrame(rows)
    return out


@st.cache_data(show_spinner=False)
def run_sentiment(df_sent: pd.DataFrame) -> pd.DataFrame:
    # run HF pipeline in batches
    pipe = build_sentiment_pipe()
    texts = df_sent["sentence"].tolist()
    if len(texts) == 0:
        df_sent["sentiment"] = []
        df_sent["score"] = []
        return df_sent

    preds = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        preds.extend(pipe(batch))

    df2 = df_sent.copy()
    df2["sentiment_raw"] = [p["label"] for p in preds]
    df2["score"] = [float(p["score"]) for p in preds]
    df2["sentiment"] = df2["sentiment_raw"].apply(map_label_to_bucket)
    return df2


@st.cache_data(show_spinner=False)
def add_aspects(df_sent: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df_sent.iterrows():
        aspects = detect_aspects(r["sentence"])
        for a in aspects:
            rows.append(
                {
                    "review_id": r["review_id"],
                    "product": r["product"],
                    "source": r["source"],
                    "firm": r["firm"],
                    "sentence": r["sentence"],
                    "sentiment": r["sentiment"],
                    "score": r["score"],
                    "aspect": a,
                }
            )
    out = pd.DataFrame(rows)
    return out


def pct_table(counts: pd.Series) -> pd.DataFrame:
    c = counts.copy()
    total = float(c.sum()) if float(c.sum()) > 0 else 1.0
    df = pd.DataFrame({"count": c.astype(int)})
    df["share"] = df["count"] / total
    df = df.reset_index().rename(columns={"index": "label"})
    return df


def plot_pie_percent(df_pct: pd.DataFrame, title: str, colors: list[str] | None = None, height: int = 360):
    # df_pct columns: label, count, share
    if df_pct.empty:
        st.info("No data to plot.")
        return

    fig = px.pie(
        df_pct,
        names="label",
        values="share",
        hover_data={"count": True, "share": ":.1%"},
        title=title,
        color="label",
        color_discrete_sequence=colors,
    )
    fig.update_traces(textposition="inside", textinfo="percent")
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_bar_percent(df_pct: pd.DataFrame, title: str, x_label: str, y_label: str = "share", height: int = 420):
    # df_pct columns: label, count, share
    if df_pct.empty:
        st.info("No data to plot.")
        return

    fig = px.bar(
        df_pct,
        x="label",
        y="share",
        hover_data={"count": True, "share": ":.1%"},
        title=title,
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        yaxis_tickformat=".0%",
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)


def pretty_aspect(a: str) -> str:
    return ASPECT_PRETTY.get(a, a.replace("_", " "))


def aspect_summary(df_aspect_sent: pd.DataFrame) -> pd.DataFrame:
    # returns per product, aspect: mentions, pos/neu/neg counts and shares, net sentiment
    if df_aspect_sent.empty:
        return pd.DataFrame(columns=["product", "aspect", "mentions", "neg", "neu", "pos", "neg_share", "neu_share", "pos_share", "net_sentiment"])

    g = df_aspect_sent.groupby(["product", "aspect", "sentiment"]).size().reset_index(name="n")
    pivot = g.pivot_table(index=["product", "aspect"], columns="sentiment", values="n", fill_value=0).reset_index()

    for col in ["NEG", "NEU", "POS"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["mentions"] = pivot["NEG"] + pivot["NEU"] + pivot["POS"]
    pivot["neg_share"] = pivot["NEG"] / pivot["mentions"].replace(0, 1)
    pivot["neu_share"] = pivot["NEU"] / pivot["mentions"].replace(0, 1)
    pivot["pos_share"] = pivot["POS"] / pivot["mentions"].replace(0, 1)
    pivot["net_sentiment"] = pivot["pos_share"] - pivot["neg_share"]

    pivot = pivot.rename(columns={"NEG": "neg", "NEU": "neu", "POS": "pos"})
    return pivot


# -----------------------------
# Load built-in dataset
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset():
    ddog = safe_read_csv("reviews_ddog.csv")
    dt = safe_read_csv("reviews_dt.csv")
    df = pd.concat([ddog, dt], ignore_index=True)
    # normalize product names for consistency
    df["product"] = df["product"].str.lower().str.strip()
    # helpful: coerce known typos
    df["product"] = df["product"].replace({"data dog": "datadog", "dynatracee": "dynatrace"})
    return df


def run_full_pipeline(df_all: pd.DataFrame):
    df_sent_base = preprocess_reviews(df_all)
    df_sent = run_sentiment(df_sent_base)
    df_aspect_sent = add_aspects(df_sent)
    return df_sent, df_aspect_sent


# -----------------------------
# UI building blocks
# -----------------------------
def header(title: str, subtitle: str):
    st.markdown(f"<div class='big-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtle'>{subtitle}</div>", unsafe_allow_html=True)


def section(title: str, blurb: str):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='blurb'>{blurb}</div>", unsafe_allow_html=True)


def winner_badge(winner: str):
    if winner == "datadog":
        st.markdown("<span class='win'>datadog wins here</span>", unsafe_allow_html=True)
    elif winner == "dynatrace":
        st.markdown("<span class='win'>dynatrace wins here</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='pill'>too close to call</span>", unsafe_allow_html=True)


def compute_firm_mix(df: pd.DataFrame, product: str) -> pd.DataFrame:
    d = df[df["product"] == product].copy()
    d = d.dropna(subset=["firm"])
    if d.empty:
        return pd.DataFrame(columns=["label", "count", "share"])
    counts = d["firm"].value_counts()
    out = pct_table(counts)
    out = out.rename(columns={"label": "label"})
    return out


def compute_sentiment_mix(df_sent: pd.DataFrame, product: str) -> pd.DataFrame:
    d = df_sent[df_sent["product"] == product]
    if d.empty:
        return pd.DataFrame(columns=["label", "count", "share"])
    counts = d["sentiment"].value_counts().reindex(["POS", "NEU", "NEG"]).dropna()
    out = pct_table(counts)
    return out


def compute_aspect_mentions(df_aspect_sent: pd.DataFrame, product: str) -> pd.DataFrame:
    d = df_aspect_sent[df_aspect_sent["product"] == product]
    if d.empty:
        return pd.DataFrame(columns=["label", "count", "share"])
    counts = d["aspect"].value_counts()
    out = pct_table(counts)
    out["label"] = out["label"].apply(pretty_aspect)
    return out


def compute_aspect_sentiment_pie(df_aspect_summary: pd.DataFrame, product: str, aspect: str) -> pd.DataFrame:
    d = df_aspect_summary[(df_aspect_summary["product"] == product) & (df_aspect_summary["aspect"] == aspect)]
    if d.empty:
        return pd.DataFrame(columns=["label", "count", "share"])
    row = d.iloc[0]
    counts = pd.Series({"POS": int(row["pos"]), "NEU": int(row["neu"]), "NEG": int(row["neg"])})
    counts = counts[counts > 0]
    out = pct_table(counts)
    return out


def product_palette(product: str) -> list[str]:
    return DDOG_BLUES if product == "datadog" else DT_ORANGES


# -----------------------------
# App
# -----------------------------
def main():
    inject_css()
    df_all = load_dataset()

    st.sidebar.markdown("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Welcome", "Datadog", "Dynatrace", "Compare"],
        label_visibility="collapsed",
    )

    # Run pipeline once (cached) but only after user asks, so it feels snappy
    if "ran" not in st.session_state:
        st.session_state["ran"] = False
    if "df_sent" not in st.session_state:
        st.session_state["df_sent"] = None
    if "df_aspect_sent" not in st.session_state:
        st.session_state["df_aspect_sent"] = None
    if "aspect_summary" not in st.session_state:
        st.session_state["aspect_summary"] = None

    def ensure_pipeline():
        if not st.session_state["ran"]:
            with st.spinner("Running analysis (sentiment + aspects)…"):
                df_sent, df_aspect_sent = run_full_pipeline(df_all)
                st.session_state["df_sent"] = df_sent
                st.session_state["df_aspect_sent"] = df_aspect_sent
                st.session_state["aspect_summary"] = aspect_summary(df_aspect_sent)
                st.session_state["ran"] = True

    # Welcome
    if page == "Welcome":
        header(
            "Customer Review Analyzer",
            "I pulled a small Datadog vs Dynatrace review set and turned it into something you can skim in ~2 minutes.",
        )

        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("dataset size")
            st.markdown(f"<div class='big-title' style='font-size:34px;margin-top:4px;'>{len(df_all)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='subtle'>total reviews</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colB:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("datadog reviews")
            st.markdown(f"<div class='big-title' style='font-size:34px;margin-top:4px;'>{(df_all['product']=='datadog').sum()}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colC:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("dynatrace reviews")
            st.markdown(f"<div class='big-title' style='font-size:34px;margin-top:4px;'>{(df_all['product']=='dynatrace').sum()}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")

        st.markdown(
            "<span class='pill'>firm-size mix</span>"
            "<span class='pill'>overall sentiment</span>"
            "<span class='pill'>what people talk about (aspects)</span>"
            "<span class='pill'>aspect-by-aspect ddog vs dt</span>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        section("Quick preview", "This is the raw dataset I’m analyzing. You can scroll it.")
        st.dataframe(df_all.head(25), use_container_width=True)

        st.markdown("")
        if st.button("Run analysis"):
            ensure_pipeline()
            st.success("Done — use the sidebar to jump into Datadog / Dynatrace / Compare.")

        if st.session_state["ran"]:
            st.markdown("")
            section("What I’m actually doing under the hood", "I split reviews into sentences, run sentiment on each sentence, then tag each sentence with one or more aspects using keyword rules.")
            st.info("If you want, we can swap the keyword rules for a smarter topic model later — but rules are surprisingly good for a first pass.")

    # Product pages
    if page in {"Datadog", "Dynatrace"}:
        product = "datadog" if page == "Datadog" else "dynatrace"
        header(
            page.lower(),
            "A quick pass at what customers talk about + how they feel, based on sentence-level sentiment.",
        )

        ensure_pipeline()
        df_sent = st.session_state["df_sent"]
        df_aspect_sent = st.session_state["df_aspect_sent"]
        df_asum = st.session_state["aspect_summary"]

        # Firm mix
        section("Firm-size mix", "Share of reviews by firm size bucket (if your CSV includes a firm column).")
        firm_mix = compute_firm_mix(df_all, product)
        if firm_mix.empty:
            st.info("No firm column detected for this product (or it’s empty).")
        else:
            plot_pie_percent(
                firm_mix,
                f"{product} firm-size mix",
                colors=product_palette(product),
                height=340,
            )

        # Overall sentiment
        section("Overall sentiment", "Sentence-level sentiment across all reviews (so long reviews contribute more sentences).")
        sent_mix = compute_sentiment_mix(df_sent, product)

        # Ensure stable ordering + colors for sentiment pies
        if not sent_mix.empty:
            sent_mix["label"] = pd.Categorical(sent_mix["label"], categories=["POS", "NEU", "NEG"], ordered=True)
            sent_mix = sent_mix.sort_values("label")

            fig = px.pie(
                sent_mix,
                names="label",
                values="share",
                hover_data={"count": True, "share": ":.1%"},
                title=f"{product} overall sentiment",
                color="label",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_traces(textposition="inside", textinfo="percent")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=60, b=10), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data yet.")

        # Aspects
        section("What people mention the most", "This ranks topics by share of aspect tags. Hover to see raw counts.")
        aspect_mentions = compute_aspect_mentions(df_aspect_sent, product)
        if not aspect_mentions.empty:
            plot_bar_percent(
                aspect_mentions,
                f"{product}: what people talk about (aspects)",
                x_label="aspect",
                y_label="share of mentions",
                height=440,
            )
        else:
            st.info("No aspect mentions found.")

        # Aspect detail table (readable)
        section("Aspect leaderboard", "Mentions + sentiment split by aspect (ranked by mentions).")
        d = df_asum[df_asum["product"] == product].copy()
        if d.empty:
            st.info("No aspect summary found.")
        else:
            d["aspect"] = d["aspect"].apply(pretty_aspect)
            d = d.sort_values("mentions", ascending=False)
            d_show = d[["aspect", "mentions", "pos_share", "neu_share", "neg_share", "net_sentiment"]].copy()
            d_show["pos_share"] = (100 * d_show["pos_share"]).round(1).astype(str) + "%"
            d_show["neu_share"] = (100 * d_show["neu_share"]).round(1).astype(str) + "%"
            d_show["neg_share"] = (100 * d_show["neg_share"]).round(1).astype(str) + "%"
            d_show["net_sentiment"] = (100 * d_show["net_sentiment"]).round(1).astype(str) + " pts"
            st.dataframe(d_show, use_container_width=True)

    # Compare
    if page == "Compare":
        header(
            "Compare: datadog vs dynatrace",
            "Side-by-side views so you can see what dominates, and where sentiment splits.",
        )

        ensure_pipeline()
        df_sent = st.session_state["df_sent"]
        df_aspect_sent = st.session_state["df_aspect_sent"]
        df_asum = st.session_state["aspect_summary"]

        # Firm mix pies side-by-side
        section("Firm-size mix", "This is just the breakdown of your pulled reviews by firm-size bucket.")
        col1, col2 = st.columns(2)
        with col1:
            mix = compute_firm_mix(df_all, "datadog")
            if mix.empty:
                st.info("No firm column for datadog.")
            else:
                plot_pie_percent(mix, "datadog firm-size mix", colors=DDOG_BLUES, height=330)
        with col2:
            mix = compute_firm_mix(df_all, "dynatrace")
            if mix.empty:
                st.info("No firm column for dynatrace.")
            else:
                plot_pie_percent(mix, "dynatrace firm-size mix", colors=DT_ORANGES, height=330)

        # Overall sentiment pies side-by-side
        section("Overall sentiment", "Sentence-level sentiment split across the whole dataset for each product.")
        col1, col2 = st.columns(2)
        for c, product in [(col1, "datadog"), (col2, "dynatrace")]:
            with c:
                sm = compute_sentiment_mix(df_sent, product)
                if sm.empty:
                    st.info("No sentiment.")
                    continue
                sm["label"] = pd.Categorical(sm["label"], categories=["POS", "NEU", "NEG"], ordered=True)
                sm = sm.sort_values("label")
                fig = px.pie(
                    sm,
                    names="label",
                    values="share",
                    hover_data={"count": True, "share": ":.1%"},
                    title=f"{product} overall sentiment",
                    color="label",
                    color_discrete_map=SENTIMENT_COLORS,
                )
                fig.update_traces(textposition="inside", textinfo="percent")
                fig.update_layout(height=330, margin=dict(l=10, r=10, t=60, b=10), legend_title_text="")
                st.plotly_chart(fig, use_container_width=True)

        # Aspects overall (ranked)
        section("What people mention the most (overall)", "This ranks aspects by share of mentions across both products.")
        counts_all = df_aspect_sent["aspect"].value_counts()
        overall = pct_table(counts_all)
        overall["label"] = overall["label"].apply(pretty_aspect)
        plot_bar_percent(
            overall,
            "aspects ranked by share of mentions",
            x_label="aspect",
            y_label="share of mentions",
            height=460,
        )

        # Aspect-by-aspect comparison (pies)
        section(
            "Aspect-by-aspect sentiment",
            "For each aspect (ranked by mentions), here’s the sentiment split for datadog vs dynatrace. Winner = higher net sentiment (pos% − neg%).",
        )

        top_n = st.slider("How many top aspects to compare?", min_value=3, max_value=min(12, len(counts_all)), value=min(8, len(counts_all)))

        ranked_aspects = counts_all.index.tolist()
        ranked_aspects = [a for a in ranked_aspects if a != "other"] + (["other"] if "other" in counts_all.index else [])
        ranked_aspects = ranked_aspects[:top_n]

        # compute winner per aspect
        for a in ranked_aspects:
            dd = df_asum[(df_asum["product"] == "datadog") & (df_asum["aspect"] == a)]
            dt = df_asum[(df_asum["product"] == "dynatrace") & (df_asum["aspect"] == a)]

            dd_net = float(dd["net_sentiment"].iloc[0]) if not dd.empty else 0.0
            dt_net = float(dt["net_sentiment"].iloc[0]) if not dt.empty else 0.0

            if abs(dd_net - dt_net) < 0.02:
                winner = "tie"
            else:
                winner = "datadog" if dd_net > dt_net else "dynatrace"

            st.markdown("")
            cols = st.columns([1.2, 0.8])
            with cols[0]:
                st.markdown(f"<div class='section-title'>{pretty_aspect(a)}</div>", unsafe_allow_html=True)
                winner_badge(winner)

            colL, colR = st.columns(2)

            with colL:
                pie = compute_aspect_sentiment_pie(df_asum, "datadog", a)
                if pie.empty:
                    st.info("No datadog mentions for this aspect.")
                else:
                    fig = px.pie(
                        pie,
                        names="label",
                        values="share",
                        hover_data={"count": True, "share": ":.1%"},
                        title="datadog sentiment",
                        color="label",
                        color_discrete_map=SENTIMENT_COLORS,
                    )
                    fig.update_traces(textposition="inside", textinfo="percent")
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

            with colR:
                pie = compute_aspect_sentiment_pie(df_asum, "dynatrace", a)
                if pie.empty:
                    st.info("No dynatrace mentions for this aspect.")
                else:
                    fig = px.pie(
                        pie,
                        names="label",
                        values="share",
                        hover_data={"count": True, "share": ":.1%"},
                        title="dynatrace sentiment",
                        color="label",
                        color_discrete_map=SENTIMENT_COLORS,
                    )
                    fig.update_traces(textposition="inside", textinfo="percent")
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("")
        section("Raw data", "If anything looks off, I always sanity-check the raw review text.")
        st.dataframe(df_all.sample(min(25, len(df_all)), random_state=7), use_container_width=True)


if __name__ == "__main__":
    main()

