from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ----------------------------
# Theme / styling constants
# ----------------------------
APP_BG = "#f6f9ff"
CARD_BG = "#ffffff"
ACCENT = "#2f6fed"
ACCENT_2 = "#1f4fd6"
TEXT = "#0f172a"
MUTED = "#475569"
BORDER = "#e6eefc"

DDOG_BLUES = ["#d7e9ff", "#a9d1ff", "#6fb2ff", "#2f6fed", "#1648c8"]
DT_ORANGES = ["#ffe2cf", "#ffc29a", "#ff9a57", "#f26a1b", "#c84a0b"]

SENTIMENT_ORDER = ["NEG", "NEU", "POS"]
SENTIMENT_COLORS_DDOG = {"NEG": "#8fb9ff", "NEU": "#2f6fed", "POS": "#1648c8"}
SENTIMENT_COLORS_DT = {"NEG": "#ffb38a", "NEU": "#f26a1b", "POS": "#c84a0b"}


# ----------------------------
# Helpers
# ----------------------------
def _clean_col(c: str) -> str:
    c = c.strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    return df


def coerce_boolish(x) -> Optional[bool]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "✓", "check", "checked"}:
        return True
    if s in {"0", "false", "f", "no", "n", "x", "✗"}:
        return False
    return None


def parse_pasted_table(text: str) -> pd.DataFrame:
    """
    Accepts:
      - TSV (copied from Sheets/Excel)
      - CSV
      - Markdown table (best-effort)
    """
    text = text.strip()
    if not text:
        return pd.DataFrame()

    # If it looks like a markdown table, strip pipes and rely on whitespace/tsv-ish parsing
    if "|" in text and "\n" in text:
        # remove leading/trailing pipes per line
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^\|?(\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?$", line):
                continue  # markdown separator row
            line = line.strip("|")
            lines.append(line.replace("|", "\t"))
        text = "\n".join(lines)

    # Decide delimiter
    delimiter = "\t" if "\t" in text else ("," if "," in text else None)

    if delimiter is None:
        # fallback: split on 2+ spaces
        rows = [re.split(r"\s{2,}", ln.strip()) for ln in text.splitlines() if ln.strip()]
        if not rows:
            return pd.DataFrame()
        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        return pd.DataFrame(data, columns=header)

    from io import StringIO

    return pd.read_csv(StringIO(text), sep=delimiter)


def normalize_sentiment(s: str) -> Optional[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).strip().upper()
    mapping = {
        "NEGATIVE": "NEG",
        "NEG": "NEG",
        "NEUTRAL": "NEU",
        "NEU": "NEU",
        "POSITIVE": "POS",
        "POS": "POS",
    }
    return mapping.get(t, None)


def ensure_platform_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect a 'platform' column containing 'DDOG' or 'DT'.
    If missing, we try to infer from columns like ddog/dt flags.
    """
    df = df.copy()
    if "platform" in df.columns:
        df["platform"] = df["platform"].astype(str).str.upper().str.strip()
        df["platform"] = df["platform"].replace({"DATADOG": "DDOG", "DYNATRACE": "DT"})
        return df

    # If you have columns named ddog/dt as boolean flags
    if "ddog" in df.columns or "dt" in df.columns:
        ddog = df.get("ddog")
        dt = df.get("dt")
        platforms = []
        for i in range(len(df)):
            d1 = coerce_boolish(ddog.iloc[i]) if ddog is not None else None
            d2 = coerce_boolish(dt.iloc[i]) if dt is not None else None
            if d1 and not d2:
                platforms.append("DDOG")
            elif d2 and not d1:
                platforms.append("DT")
            else:
                platforms.append(None)
        df["platform"] = platforms
        return df

    return df


def feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a wide matrix by feature:
      rows: feature
      cols: DDOG, DT
    Supports:
      - long format: feature, platform, value (or supported / notes)
      - wide format: feature + ddog + dt columns
    """
    df = df.copy()
    df = standardize_columns(df)

    # Identify a feature column
    feat_col = None
    for candidate in ["feature", "capability", "item", "metric", "dimension"]:
        if candidate in df.columns:
            feat_col = candidate
            break
    if feat_col is None:
        # if first column is likely feature name
        feat_col = df.columns[0]
    df.rename(columns={feat_col: "feature"}, inplace=True)

    # Wide format?
    if "ddog" in df.columns or "dt" in df.columns:
        out = df[["feature"] + [c for c in ["ddog", "dt"] if c in df.columns]].copy()
        out["ddog"] = out.get("ddog", None).apply(lambda x: True if coerce_boolish(x) else (False if coerce_boolish(x) is False else x))
        out["dt"] = out.get("dt", None).apply(lambda x: True if coerce_boolish(x) else (False if coerce_boolish(x) is False else x))
        # Gap table
        gap = out.copy()
        gap["gap"] = gap.apply(
            lambda r: "DT only" if (coerce_boolish(r.get("dt")) is True and coerce_boolish(r.get("ddog")) is not True)
            else ("DDOG only" if (coerce_boolish(r.get("ddog")) is True and coerce_boolish(r.get("dt")) is not True)
                  else ("Both" if (coerce_boolish(r.get("ddog")) is True and coerce_boolish(r.get("dt")) is True) else "Unclear")),
            axis=1,
        )
        return out, gap[["feature", "gap"]]

    # Long format: need platform
    df = ensure_platform_col(df)
    if "platform" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Identify value-ish column
    val_col = None
    for candidate in ["value", "supported", "support", "notes", "comment", "detail"]:
        if candidate in df.columns:
            val_col = candidate
            break
    if val_col is None:
        # fallback to second column
        val_col = df.columns[1] if len(df.columns) > 1 else "value"
        if val_col not in df.columns:
            df[val_col] = None

    df["platform"] = df["platform"].astype(str).str.upper().str.strip()
    df = df[df["platform"].isin(["DDOG", "DT"])]

    wide = (
        df.pivot_table(index="feature", columns="platform", values=val_col, aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if "DDOG" not in wide.columns:
        wide["DDOG"] = None
    if "DT" not in wide.columns:
        wide["DT"] = None
    wide = wide[["feature", "DDOG", "DT"]].rename(columns={"DDOG": "ddog", "DT": "dt"})

    # Gap table heuristics: if value looks boolish, treat as support flag; else "Unclear/Both-ish"
    def _gap_row(r):
        b_ddog = coerce_boolish(r["ddog"])
        b_dt = coerce_boolish(r["dt"])
        if b_ddog is None or b_dt is None:
            # if both have non-empty notes, call it both; else unclear
            has_ddog = r["ddog"] is not None and str(r["ddog"]).strip() != "" and str(r["ddog"]).strip().lower() != "nan"
            has_dt = r["dt"] is not None and str(r["dt"]).strip() != "" and str(r["dt"]).strip().lower() != "nan"
            if has_ddog and has_dt:
                return "Both (notes)"
            if has_dt and not has_ddog:
                return "DT only (notes)"
            if has_ddog and not has_dt:
                return "DDOG only (notes)"
            return "Unclear"

        if b_dt and not b_ddog:
            return "DT only"
        if b_ddog and not b_dt:
            return "DDOG only"
        if b_ddog and b_dt:
            return "Both"
        return "Neither"

    gap = wide.copy()
    gap["gap"] = gap.apply(_gap_row, axis=1)
    return wide, gap[["feature", "gap"]]


def sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = standardize_columns(df)
    df = ensure_platform_col(df)
    if "platform" not in df.columns:
        return pd.DataFrame()

    # find sentiment column
    sent_col = None
    for c in ["sentiment", "tone", "label"]:
        if c in df.columns:
            sent_col = c
            break
    if sent_col is None:
        return pd.DataFrame()

    df["sentiment"] = df[sent_col].apply(normalize_sentiment)
    df = df.dropna(subset=["sentiment"])
    df = df[df["platform"].isin(["DDOG", "DT"])]

    out = (
        df.groupby(["platform", "sentiment"])
        .size()
        .reset_index(name="count")
        .sort_values(["platform", "sentiment"])
    )
    return out


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Datadog vs Dynatrace — Comparison Builder",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      html, body, [data-testid="stAppViewContainer"] {{
        background: {APP_BG};
      }}
      .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.0rem;
        max-width: 1200px;
      }}
      h1, h2, h3, h4 {{
        color: {TEXT};
      }}
      p, li, div {{
        color: {TEXT};
      }}
      .muted {{
        color: {MUTED};
      }}
      .card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 16px 16px;
        box-shadow: 0 1px 0 rgba(15, 23, 42, 0.03);
      }}
      .pill {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid {BORDER};
        font-size: 12px;
        color: {MUTED};
        background: #fff;
        margin-right: 6px;
      }}
      .small {{
        font-size: 13px;
        color: {MUTED};
      }}
      .stTabs [data-baseweb="tab-list"] button {{
        font-weight: 600;
      }}
      .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Datadog vs Dynatrace — Comparison Builder")
st.markdown(
    "<span class='pill'>Upload CSV</span><span class='pill'>Paste table</span><span class='pill'>Feature matrix</span><span class='pill'>Pricing rows</span>",
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='card'><div class='small'>Use this to standardize your notes into a consistent, banker-friendly comparison table. "
    "If you paste your feature sheet or pricing rows, the app will build (1) a DDOG vs DT matrix, (2) an explicit gap table, "
    "and (3) optional sentiment breakdown if you include a <b>sentiment</b> column.</div></div>",
    unsafe_allow_html=True,
)

st.write("")


# ----------------------------
# Sidebar: data input
# ----------------------------
st.sidebar.header("1) Load data")

input_mode = st.sidebar.radio("Choose input method", ["Upload CSV", "Paste table"], index=0)

df_raw = pd.DataFrame()

if input_mode == "Upload CSV":
    up = st.sidebar.file_uploader("Upload a CSV (features/pricing/reviews)", type=["csv"])
    if up is not None:
        df_raw = pd.read_csv(up)
else:
    pasted = st.sidebar.text_area(
        "Paste TSV/CSV/Markdown table",
        height=220,
        placeholder="Paste from Sheets/Excel (tab-separated) or a markdown table.",
    )
    if pasted.strip():
        df_raw = parse_pasted_table(pasted)

if df_raw is not None and len(df_raw) > 0:
    df_raw = standardize_columns(df_raw)

st.sidebar.divider()
st.sidebar.header("2) Expected columns (flexible)")
st.sidebar.markdown(
    """
- **Feature matrix (recommended)**  
  - Wide: `feature`, `ddog`, `dt`  
  - Long: `feature`, `platform` (DDOG/DT), `value` (or `supported` / `notes`)
- **Sentiment (optional)**  
  - `platform`, `sentiment` (NEG/NEU/POS)
- **Pricing rows (optional)**  
  - `platform`, `sku`/`product`, `metric`/`unit`, `price`, `notes`
"""
)

st.sidebar.divider()
show_raw = st.sidebar.checkbox("Show raw data preview", value=True)


# ----------------------------
# Main: tabs
# ----------------------------
tabs = st.tabs(["Feature matrix", "Pricing compare", "Sentiment (optional)", "Export"])

# ---- Tab 1: Feature matrix
with tabs[0]:
    st.subheader("Feature matrix")
    if df_raw is None or len(df_raw) == 0:
        st.info("Upload a CSV or paste a table to generate the matrix.")
    else:
        if show_raw:
            st.markdown("<div class='card'><b>Raw preview</b></div>", unsafe_allow_html=True)
            st.dataframe(df_raw.head(50), use_container_width=True)

        wide, gap = feature_matrix(df_raw)

        if wide is None or len(wide) == 0:
            st.warning(
                "I couldn't infer a feature matrix. Make sure you have either:\n"
                "- Wide columns: `feature`, `ddog`, `dt`\n"
                "- OR Long columns: `feature`, `platform` (DDOG/DT), and `value`/`notes`"
            )
        else:
            left, right = st.columns([2, 1], vertical_alignment="top")
            with left:
                st.markdown("<div class='card'><b>DDOG vs DT — matrix</b></div>", unsafe_allow_html=True)
                st.dataframe(wide, use_container_width=True)

            with right:
                st.markdown("<div class='card'><b>Gap summary</b></div>", unsafe_allow_html=True)
                if gap is not None and len(gap) > 0:
                    counts = gap["gap"].value_counts().reset_index()
                    counts.columns = ["gap", "count"]
                    fig = px.bar(counts, x="gap", y="count")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No gaps computed.")

            st.write("")
            st.markdown("<div class='card'><b>Gap table</b> <span class='small'>(what’s unique vs overlapping)</span></div>", unsafe_allow_html=True)
            if gap is not None and len(gap) > 0:
                st.dataframe(gap.sort_values(["gap", "feature"]), use_container_width=True)
            else:
                st.write("No gap table available.")

# ---- Tab 2: Pricing compare
with tabs[1]:
    st.subheader("Pricing compare (lightweight)")
    st.markdown(
        "<div class='card'><div class='small'>If you provide rows like platform/product/unit/price, "
        "this will standardize and let you filter + compare. If your pricing is messy, paste it here and "
        "then export a cleaned sheet.</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    pricing_df = pd.DataFrame()
    if df_raw is not None and len(df_raw) > 0:
        # Heuristic: treat as pricing if it contains price-like column
        price_cols = [c for c in df_raw.columns if c in {"price", "rate", "cost", "usd", "list_price"}]
        if price_cols:
            pricing_df = df_raw.copy()
            # normalize platform if present
            pricing_df = ensure_platform_col(pricing_df)

    if pricing_df is None or len(pricing_df) == 0:
        st.info("No obvious pricing columns found in your uploaded/pasted data. If you want, paste a pricing table with a `price` column.")
    else:
        pricing_df = standardize_columns(pricing_df)

        # Rename likely columns
        renames = {}
        if "product" not in pricing_df.columns:
            for c in ["sku", "plan", "module", "tier", "service"]:
                if c in pricing_df.columns:
                    renames[c] = "product"
                    break
        if "unit" not in pricing_df.columns:
            for c in ["metric", "billing_unit", "billing", "per", "usage_unit"]:
                if c in pricing_df.columns:
                    renames[c] = "unit"
                    break
        if renames:
            pricing_df = pricing_df.rename(columns=renames)

        platform_filter = st.multiselect("Platform", ["DDOG", "DT"], default=["DDOG", "DT"])
        q = st.text_input("Search product / notes", placeholder="e.g., host, container, APM, logs, Grail, OneAgent")
        dfv = pricing_df.copy()
        if "platform" in dfv.columns:
            dfv = dfv[dfv["platform"].isin(platform_filter)]
        if q.strip():
            pat = re.escape(q.strip().lower())
            mask = pd.Series(False, index=dfv.index)
            for c in dfv.columns:
                mask = mask | dfv[c].astype(str).str.lower().str.contains(pat, na=False)
            dfv = dfv[mask]

        st.dataframe(dfv, use_container_width=True)

# ---- Tab 3: Sentiment
with tabs[2]:
    st.subheader("Sentiment (optional)")
    st.markdown(
        "<div class='card'><div class='small'>If your data includes `platform` and `sentiment` "
        "(NEG/NEU/POS), we’ll chart distribution by platform.</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    sent = pd.DataFrame()
    if df_raw is not None and len(df_raw) > 0:
        sent = sentiment_summary(df_raw)

    if sent is None or len(sent) == 0:
        st.info("No sentiment detected. Add columns like: `platform` (DDOG/DT) and `sentiment` (NEG/NEU/POS).")
    else:
        # Ensure order
        sent["sentiment"] = pd.Categorical(sent["sentiment"], categories=SENTIMENT_ORDER, ordered=True)
        sent = sent.sort_values(["platform", "sentiment"])

        st.dataframe(sent, use_container_width=True)

        # Plot
        fig = px.bar(sent, x="sentiment", y="count", color="platform", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Export
with tabs[3]:
    st.subheader("Export cleaned outputs")
    st.markdown(
        "<div class='card'><div class='small'>Download standardized tables as CSV for your memo / model / deck.</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if df_raw is None or len(df_raw) == 0:
        st.info("Load data first.")
    else:
        wide, gap = feature_matrix(df_raw)
        sent = sentiment_summary(df_raw)

        def dl_button(df: pd.DataFrame, label: str, filename: str):
            if df is None or len(df) == 0:
                st.write(f"— {label}: (empty)")
                return
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

        col1, col2 = st.columns(2)
        with col1:
            dl_button(df_raw, "Download: raw standardized", "raw_standardized.csv")
            dl_button(wide, "Download: feature_matrix", "feature_matrix.csv")
        with col2:
            dl_button(gap, "Download: gap_table", "gap_table.csv")
            dl_button(sent, "Download: sentiment_summary", "sentiment_summary.csv")


# Footer
st.write("")
st.markdown(
    "<div class='small muted'>Tip: If you paste your sheet and it doesn’t map, rename columns to "
    "`feature`, `ddog`, `dt` (wide) or `feature`, `platform`, `value` (long).</div>",
    unsafe_allow_html=True,
)
