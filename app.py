import io
import pandas as pd
import streamlit as st

# Import your existing functions from sentiment_classifier.py
from sentiment_classifier import analyze_text_rows, summarize

st.set_page_config(page_title="Review Insights", layout="wide")

st.title("Customer Review Analyzer")
st.caption("Upload a CSV of reviews and get aspect + sentiment summaries.")

with st.expander("Expected CSV format", expanded=False):
    st.markdown(
        """
        Required columns:
        - **id** (unique id per review)
        - **product** (e.g., datadog, dynatrace)
        - **source** (e.g., G2, Reddit)
        - **text** (the review text)

        Example row:
        `1,datadog,G2,"Datadog is reliable but pricing is unpredictable..."`
        """
    )

uploaded = st.file_uploader("Upload CSV", type=["csv"])

run = st.button("Run analysis", type="primary", disabled=(uploaded is None))

if run:
    df = pd.read_csv(uploaded)

    required = {"id", "product", "source", "text"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}. Found columns: {list(df.columns)}")
        st.stop()

    st.write(f"Loaded **{len(df)}** reviews.")

    with st.spinner("Analyzing reviews (sentence split → aspect tagging → sentiment)…"):
        df_sent = analyze_text_rows(df)
        summary_df, examples_df = summarize(df_sent, top_examples=3)

    st.success("Done!")

    # Show results
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Summary by aspect (ranked by mention share)")
        # If your summary has mention_share_within_product, sort by it
        sort_col = "mention_share_within_product" if "mention_share_within_product" in summary_df.columns else "mentions"
        st.dataframe(summary_df.sort_values(sort_col, ascending=False), use_container_width=True)

    with c2:
        st.subheader("Example snippets (evidence)")
        st.dataframe(examples_df, use_container_width=True)

    st.subheader("Sentence-level output (audit log)")
    st.dataframe(df_sent.head(200), use_container_width=True)

    # Download buttons
    def to_csv_bytes(x: pd.DataFrame) -> bytes:
        return x.to_csv(index=False).encode("utf-8")

    st.download_button("Download sentence_level.csv", data=to_csv_bytes(df_sent),
                       file_name="sentence_level.csv", mime="text/csv")
    st.download_button("Download summary_by_aspect.csv", data=to_csv_bytes(summary_df),
                       file_name="summary_by_aspect.csv", mime="text/csv")
    st.download_button("Download example_snippets.csv", data=to_csv_bytes(examples_df),
                       file_name="example_snippets.csv", mime="text/csv")
