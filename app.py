import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 1.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DDOG_BLUES = ["#9CC9FF", "#5FA8FF", "#2F6FD6"]
DT_ORANGES = ["#FFD1A3", "#FFAE5D", "#E07A1F"]

def load_data():
    df_ddog = pd.read_csv("reviews_ddog.csv")
    df_dt = pd.read_csv("reviews_dt.csv")
    df_sent = pd.read_csv("sentence_level.csv")
    df_aspect = pd.read_csv("summary_by_aspect.csv")
    for df in [df_ddog, df_dt]:
        if "firm" in df.columns:
            df["firm"] = df["firm"].fillna("NA")
    return df_ddog, df_dt, df_sent, df_aspect

def firm_mix(df):
    mix = df.groupby("firm").size().reset_index(name="count")
    total = mix["count"].sum()
    mix["share"] = mix["count"] / total
    return mix

def sentiment_mix(df_sent, product):
    tmp = df_sent[df_sent["product"] == product]
    mix = tmp["sentiment"].value_counts().reset_index()
    mix.columns = ["sentiment", "count"]
    mix["share"] = mix["count"] / mix["count"].sum()
    return mix

def aspect_rank(df_aspect, product):
    tmp = df_aspect[df_aspect["product"] == product].copy()
    tmp = tmp.sort_values("mentions", ascending=False)
    return tmp

def pie(df, names, values, title, colors):
    fig = px.pie(
        df,
        names=names,
        values=values,
        title=title,
        color_discrete_sequence=colors,
        hole=0.35,
    )
    fig.update_traces(textinfo="percent", textfont_size=13)
    fig.update_layout(height=330, margin=dict(t=50, b=10, l=10, r=10))
    return fig

def welcome_page(df_ddog, df_dt):
    st.title("Customer Review Analyzer")
    st.write(
        "I pulled a review dataset for Datadog and Dynatrace and built this to quickly see "
        "who’s getting better sentiment, what people talk about most, and how that changes by firm size."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Quick preview")
        st.dataframe(
            pd.concat([df_ddog, df_dt]).head(8),
            use_container_width=True,
        )

    with col2:
        st.subheader("Dataset size")
        st.metric("Total reviews", len(df_ddog) + len(df_dt))
        st.metric("Datadog reviews", len(df_ddog))
        st.metric("Dynatrace reviews", len(df_dt))
        st.write("Use the left nav to jump into Datadog, Dynatrace, or the side-by-side comparison.")

def product_page(product, df_reviews, df_sent, df_aspect, colors):
    st.title(product.capitalize())

    st.write(
        f"This page shows how users talk about {product.capitalize()}, including firm-size mix, "
        "overall sentiment, and the themes that come up most often."
    )

    mix = firm_mix(df_reviews)

    st.subheader("Firm-size mix")
    st.write("This shows how the reviews I pulled break down by customer size.")
    st.plotly_chart(
        pie(mix, "firm", "count", f"{product.capitalize()} firm-size mix", colors),
        use_container_width=True,
    )

    sent = sentiment_mix(df_sent, product)

    st.subheader("Overall sentiment (by sentence)")
    st.write(
        "Each review is split into sentences, and I score sentiment at the sentence level "
        "to avoid one long review dominating the signal."
    )
    st.plotly_chart(
        pie(sent, "sentiment", "count", f"{product.capitalize()} overall sentiment", colors),
        use_container_width=True,
    )

    aspects = aspect_rank(df_aspect, product)

    st.subheader("What people mention the most")
    st.write(
        "I tag sentences to product aspects (e.g. tracing, dashboards, alerts) and rank them by mention volume."
    )

    aspects["share"] = aspects["mentions"] / aspects["mentions"].sum()

    fig = px.bar(
        aspects,
        x="aspect",
        y="share",
        color_discrete_sequence=[colors[1]],
        labels={"share": "share of mentions"},
    )
    fig.update_layout(height=360)
    fig.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)

def compare_page(df_ddog, df_dt, df_sent, df_aspect):
    st.title("Datadog vs Dynatrace")

    st.write(
        "This side-by-side view makes it easy to see how the two platforms differ in customer mix, "
        "sentiment, and what users care about most."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            pie(
                firm_mix(df_ddog),
                "firm",
                "count",
                "Datadog firm-size mix",
                DDOG_BLUES,
            ),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            pie(
                firm_mix(df_dt),
                "firm",
                "count",
                "Dynatrace firm-size mix",
                DT_ORANGES,
            ),
            use_container_width=True,
        )

    aspects_ddog = aspect_rank(df_aspect, "datadog")
    aspects_dt = aspect_rank(df_aspect, "dynatrace")

    st.subheader("Top aspects (ranked by mentions)")
    st.write("For each aspect, I compare which product gets more discussion.")

    top_aspects = aspects_ddog["aspect"].head(6).tolist()

    for a in top_aspects:
        left = aspects_ddog[aspects_ddog["aspect"] == a]
        right = aspects_dt[aspects_dt["aspect"] == a]

        l_val = left["mentions"].values[0] if not left.empty else 0
        r_val = right["mentions"].values[0] if not right.empty else 0

        winner = "Datadog" if l_val >= r_val else "Dynatrace"

        st.write(f"{a} — {winner} mentioned more")

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(
                pie(
                    pd.DataFrame({"label": ["Datadog"], "count": [l_val]}),
                    "label",
                    "count",
                    "Datadog",
                    DDOG_BLUES,
                ),
                use_container_width=True,
            )

        with c2:
            st.plotly_chart(
                pie(
                    pd.DataFrame({"label": ["Dynatrace"], "count": [r_val]}),
                    "label",
                    "count",
                    "Dynatrace",
                    DT_ORANGES,
                ),
                use_container_width=True,
            )

def main():
    df_ddog, df_dt, df_sent, df_aspect = load_data()

    page = st.sidebar.radio(
        "Navigation",
        ["Welcome", "Compare", "Datadog", "Dynatrace"],
    )

    if page == "Welcome":
        welcome_page(df_ddog, df_dt)
    elif page == "Compare":
        compare_page(df_ddog, df_dt, df_sent, df_aspect)
    elif page == "Datadog":
        product_page("datadog", df_ddog, df_sent, df_aspect, DDOG_BLUES)
    elif page == "Dynatrace":
        product_page("dynatrace", df_dt, df_sent, df_aspect, DT_ORANGES)

main()
