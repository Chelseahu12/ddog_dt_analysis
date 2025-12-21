def pct_table(counts: pd.Series) -> pd.DataFrame:
    """
    Safely converts a value_counts() Series into:
    label | count | share
    Works even if counts is empty.
    """
    if counts is None or len(counts) == 0:
        return pd.DataFrame(columns=["label", "count", "share"])

    c = pd.to_numeric(counts, errors="coerce").fillna(0)
    total = float(c.sum()) if float(c.sum()) > 0 else 1.0

    out = c.reset_index()
    out.columns = ["label", "count"]   # 🔑 THIS is the fix

    out["label"] = out["label"].astype(str)
    out["count"] = out["count"].astype(int)
    out["share"] = out["count"] / total

    return out
