import re
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -------------------------
# Config
# -------------------------
INPUT_CSV = "datadog.csv"   # <-- your file name
TEXT_COL = "text"
PRODUCT_COL = "product"
SOURCE_COL = "source"
ID_COL = "id"

UNKNOWN_LABEL = "other"

# -------------------------
# Aspect taxonomy (edit any time)
# -------------------------
ASPECTS = {
    "pricing_billing": [
        "pricing", "cost", "bill", "billing", "expensive", "overage",
        "usage-based", "predictable", "unpredictable", "99th percentile"
    ],
    "setup_onboarding": [
        "setup", "install", "onboarding", "configuration", "config",
        "instrumentation", "agent", "deployment", "getting started"
    ],
    "alert_noise": [
        "alert", "noise", "false positive", "too many alerts", "paging",
        "signal", "correlation"
    ],
    "dashboards_ux": [
        "dashboard", "ui", "ux", "interface", "workflow", "navigation",
        "usability"
    ],
    "apm_tracing": [
        "apm", "tracing", "traces", "latency", "transactions", "profiling"
    ],
    "logs_search": [
        "logs", "log search", "indexing", "retention", "query"
    ],
    "kubernetes_containers": [
        "kubernetes", "k8s", "containers", "docker", "helm", "eks", "gke", "aks"
    ],
    "integrations_ecosystem": [
        "integration", "integrates", "aws", "azure", "gcp", "slack",
        "pagerduty", "servicenow", "opentelemetry", "prometheus"
    ],
    "ai_root_cause": [
        "root cause", "rca", "davis", "watchdog", "ai", "anomaly", "causal"
    ],
    "support_docs": [
        "support", "customer success", "ticket", "docs", "documentation"
    ],
    "performance_overhead": [
        "overhead", "cpu impact", "memory impact", "performance", "agent overhead"
    ],
    "reliability_uptime": [
        "reliable", "reliability", "uptime", "dependable", "outage", "availability"
    ],
}



def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def split_into_sentences(text: str) -> list[str]:
    """Sentence split + simple clause split for 'but/however'."""
    text = clean_text(text)
    if not text:
        return []

    sents = []
    for sent in sent_tokenize(text):
        parts = re.split(r"\bbut\b|\bhowever\b|\bthough\b", sent, flags=re.IGNORECASE)
        sents.extend([p.strip() for p in parts if len(p.strip()) >= 15])

    return sents


def build_aspect_embeddings(model: SentenceTransformer):
    aspect_names = list(ASPECTS.keys())
    aspect_descs = []
    for a in aspect_names:
        aspect_descs.append(f"{a.replace('_',' ')}: " + ", ".join(ASPECTS[a][:12]))
    emb = model.encode(aspect_descs, normalize_embeddings=True, show_progress_bar=False)
    return aspect_names, emb


def assign_aspect(sentence: str, model: SentenceTransformer, aspect_names, aspect_emb, threshold: float = 0.26):
    sent_emb = model.encode([sentence], normalize_embeddings=True, show_progress_bar=False)
    sims = cosine_similarity(sent_emb, aspect_emb)[0]
    idx = int(np.argmax(sims))
    best_aspect = aspect_names[idx]
    best_score = float(sims[idx])
    if best_score < threshold:
        return UNKNOWN_LABEL, best_score
    return best_aspect, best_score


def make_sentiment_pipeline():
    # Good general sentiment model; handles negation reasonably well
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


def normalize_sentiment(label: str) -> str:
    l = label.lower()
    if "positive" in l:
        return "pos"
    if "neutral" in l:
        return "neu"
    if "negative" in l:
        return "neg"
    return "neu"


def analyze_text_rows(df: pd.DataFrame) -> pd.DataFrame:
    ensure_nltk()

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    aspect_names, aspect_emb = build_aspect_embeddings(embedder)
    sent_pipe = make_sentiment_pipeline()

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row.get(TEXT_COL, "")
        product = row.get(PRODUCT_COL, "unknown")
        source = row.get(SOURCE_COL, "unknown")
        rid = row.get(ID_COL, None)

        for s in split_into_sentences(text):
            aspect, aspect_score = assign_aspect(s, embedder, aspect_names, aspect_emb)

            out = sent_pipe(s, truncation=True)[0]
            sent_label = normalize_sentiment(out["label"])
            sent_score = float(out["score"])

            records.append({
                "id": rid,
                "product": product,
                "source": source,
                "sentence": s,
                "aspect": aspect,
                "aspect_sim": aspect_score,
                "sentiment": sent_label,
                "sentiment_score": sent_score,
            })

    return pd.DataFrame(records)


def summarize(df_sent: pd.DataFrame, top_examples: int = 3):
    # Aggregate counts
    grp = df_sent.groupby(["product", "aspect", "sentiment"]).size().reset_index(name="n")
    totals = df_sent.groupby(["product", "aspect"]).size().reset_index(name="mentions")

    pivot = grp.pivot_table(index=["product", "aspect"], columns="sentiment", values="n", fill_value=0).reset_index()
    out = totals.merge(pivot, on=["product", "aspect"], how="left").fillna(0)

    for c in ["neg", "neu", "pos"]:
        if c not in out.columns:
            out[c] = 0

    out["neg_share"] = out["neg"] / out["mentions"]
    out["pos_share"] = out["pos"] / out["mentions"]
    out["neu_share"] = out["neu"] / out["mentions"]
    out["net_sentiment"] = (out["pos"] - out["neg"]) / out["mentions"]

    prod_total = df_sent.groupby("product").size().reset_index(name="product_total")
    out = out.merge(prod_total, on="product", how="left")
    out["mention_share_within_product"] = out["mentions"] / out["product_total"]
    out = out.sort_values(["product", "mention_share_within_product"], ascending=[True, False])

    # Example snippets (high-confidence)
    examples = (
        df_sent[df_sent["aspect"] != UNKNOWN_LABEL]
        .sort_values("sentiment_score", ascending=False)
        .groupby(["product", "aspect", "sentiment"])
        .head(top_examples)
        [["product", "aspect", "sentiment", "sentiment_score", "sentence", "source", "id"]]
    )

    return out, examples


def main():
    df = pd.read_csv(INPUT_CSV)

    # Basic column check
    required = {ID_COL, PRODUCT_COL, SOURCE_COL, TEXT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Your CSV is missing columns: {missing}. Found: {list(df.columns)}")

    df_sent = analyze_text_rows(df)
    df_sent.to_csv("sentence_level.csv", index=False)

    summary, examples = summarize(df_sent, top_examples=3)
    summary.to_csv("summary_by_aspect.csv", index=False)
    examples.to_csv("example_snippets.csv", index=False)

    print("Done! Wrote:")
    print(" - sentence_level.csv")
    print(" - summary_by_aspect.csv")
    print(" - example_snippets.csv")


if __name__ == "__main__":
    main()
