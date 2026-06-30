"""Streamlit app: pick a viewer, see what each recommender suggests, side by side.

Design note: built around a "screening room" idea rather than a generic admin
dashboard -- a place where you preview a curated list before it goes out.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.data_loading import load_ratings, load_items, load_tags, train_test_split_ratings
from main import build_models

st.set_page_config(page_title="Screening Room", page_icon="🎞", layout="wide")

METHOD_META = {
    "most_popular": {
        "label": "Most Popular",
        "code": "MP",
        "blurb": "Same ranking for every viewer — items with the most ratings overall.",
    },
    "highest_average": {
        "label": "Highest Average",
        "code": "HA",
        "blurb": "Best average rating among items with at least 20 ratings.",
    },
    "content_based": {
        "label": "Content-Based",
        "code": "CB",
        "blurb": "TF-IDF over genres and tags, matched to this viewer's taste profile.",
    },
    "item_item_cf": {
        "label": "Item-Item CF",
        "code": "II",
        "blurb": "Items similar to what this viewer already rated highly.",
    },
    "user_user_cf": {
        "label": "User-User CF",
        "code": "UU",
        "blurb": "What similarly-tastioned viewers rated highly.",
    },
    "matrix_factorization": {
        "label": "Matrix Factorization",
        "code": "MF",
        "blurb": "Latent factors learned via SGD — picks up patterns the others miss.",
    },
}


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --paper: #EEF0E9;
    --card: #FBFBF8;
    --ink: #1B2420;
    --ink-soft: #5B675F;
    --line: #D5D8CB;
    --accent: #2451FF;
    --accent-soft: #E6EAFF;
    --gold: #9A7400;
}

#MainMenu, footer { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.block-container {
    max-width: 1100px;
    padding-top: 2.5rem;
}

.srx-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 0.3rem;
}

.srx-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 700;
    font-size: 3rem;
    line-height: 1.05;
    margin: 0 0 0.4rem 0;
}

.srx-subtitle {
    color: var(--ink-soft);
    font-size: 1rem;
    max-width: 46ch;
    margin-bottom: 1.6rem;
}

.srx-rule {
    border: none;
    border-top: 1px solid var(--line);
    margin: 1.6rem 0;
}

.srx-section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 0.6rem;
}

.srx-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 1.1rem 1.2rem 1.3rem 1.2rem;
    margin-bottom: 1.1rem;
}

.srx-card-head {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.3rem;
}

.srx-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 0.78rem;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    border: 1.5px solid var(--ink);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.srx-card-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.15rem;
}

.srx-card-blurb {
    color: var(--ink-soft);
    font-size: 0.85rem;
    margin: 0.2rem 0 0.9rem 0;
    line-height: 1.4;
}

.srx-rec-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.8rem;
    padding: 0.4rem 0;
    border-top: 1px solid var(--line);
}

.srx-rec-row:first-of-type {
    border-top: none;
}

.srx-rec-title {
    font-size: 0.92rem;
    line-height: 1.3;
}

.srx-rec-genre {
    color: var(--ink-soft);
    font-size: 0.74rem;
}

.srx-rec-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: var(--accent);
    flex-shrink: 0;
    white-space: nowrap;
}

.srx-empty {
    color: var(--ink-soft);
    font-size: 0.85rem;
    font-style: italic;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def get_data():
    ratings = load_ratings()
    items = load_items()
    tags = load_tags()
    train, test = train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE)
    return ratings, items, tags, train, test


@st.cache_resource
def get_models(_train, _items, _tags):
    return build_models(_train, _items, _tags)


@st.cache_data
def get_offline_metrics():
    path = config.RESULTS_DIR / "metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


ratings, items, tags, train, test = get_data()
models = get_models(train, items, tags)
titles = items.set_index(config.ITEM_COL)[config.TITLE_COL]
genres = items.set_index(config.ITEM_COL)[config.GENRES_COL]

st.markdown(
    """
    <div class="srx-eyebrow">ML-LATEST-SMALL · RECOMMENDER PROTOTYPE</div>
    <div class="srx-title">Screening Room</div>
    <div class="srx-subtitle">
        Six recommenders previewing the same catalog for one viewer at a time —
        pick someone from the dataset and see how each method reads their taste differently.
    </div>
    """,
    unsafe_allow_html=True,
)

metrics_df = get_offline_metrics()
if metrics_df is not None:
    st.markdown('<div class="srx-section-label">Offline evaluation · Precision@10, 50 sampled viewers</div>', unsafe_allow_html=True)
    chart = (
        alt.Chart(metrics_df)
        .mark_bar(color="#2451FF", size=22, cornerRadiusEnd=2)
        .encode(
            x=alt.X("precision_at_k:Q", title=None, axis=alt.Axis(format="%")),
            y=alt.Y("model:N", title=None, sort="-x"),
            tooltip=["model", "precision_at_k", "recall_at_k", "ndcg_at_k", "novelty_at_k"],
        )
        .properties(height=190)
        .configure_axis(labelFont="IBM Plex Mono", labelFontSize=11, grid=False, domain=False)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Full metrics, including recall, NDCG, catalog coverage and novelty, are in results/metrics.csv.")
else:
    st.info("Run `python main.py` first to generate results/metrics.csv and unlock the evaluation chart.")

st.markdown('<hr class="srx-rule">', unsafe_allow_html=True)

st.markdown('<div class="srx-section-label">Now Screening For</div>', unsafe_allow_html=True)
ctrl_a, ctrl_b = st.columns([2, 1])
with ctrl_a:
    user_ids = sorted(train[config.USER_COL].unique())
    user_id = st.selectbox("Viewer ID", user_ids, index=0, label_visibility="collapsed")
with ctrl_b:
    n_recs = st.slider("Recommendations per method", min_value=3, max_value=12, value=5)

with st.expander(f"Viewer {user_id}'s rating history ({(train[config.USER_COL] == user_id).sum()} titles)"):
    history = (
        train[train[config.USER_COL] == user_id]
        .merge(items, on=config.ITEM_COL)
        .sort_values(config.RATING_COL, ascending=False)
        [[config.TITLE_COL, config.GENRES_COL, config.RATING_COL]]
        .rename(columns={config.TITLE_COL: "Title", config.GENRES_COL: "Genres", config.RATING_COL: "Rating"})
    )
    st.dataframe(history, hide_index=True, use_container_width=True)

st.markdown('<hr class="srx-rule">', unsafe_allow_html=True)
st.markdown('<div class="srx-section-label">Recommendations By Method</div>', unsafe_allow_html=True)

model_items = list(models.items())
for row_start in range(0, len(model_items), 2):
    row = model_items[row_start: row_start + 2]
    cols = st.columns(2)
    for col, (key, model) in zip(cols, row):
        meta = METHOD_META.get(key, {"label": key.replace("_", " ").title(), "code": "??", "blurb": ""})
        recs = model.recommend(user_id, train, n=n_recs, exclude_seen=True)

        rows_html = ""
        if recs:
            for item_id, score in recs:
                title = titles.get(item_id, f"item {item_id}")
                genre = str(genres.get(item_id, "")).replace("|", " · ")
                if isinstance(score, (float, np.floating)):
                    score_str = f"{score:.2f}"
                elif isinstance(score, (int, np.integer)):
                    score_str = f"n={int(score)}"
                else:
                    score_str = "—"
                rows_html += f"""
                <div class="srx-rec-row">
                    <div>
                        <div class="srx-rec-title">{title}</div>
                        <div class="srx-rec-genre">{genre}</div>
                    </div>
                    <div class="srx-rec-score">{score_str}</div>
                </div>
                """
        else:
            rows_html = '<div class="srx-empty">No recommendations for this viewer (cold start or unseen profile).</div>'

        with col:
            st.markdown(
                f"""
                <div class="srx-card">
                    <div class="srx-card-head">
                        <div class="srx-badge">{meta['code']}</div>
                        <div class="srx-card-title">{meta['label']}</div>
                    </div>
                    <div class="srx-card-blurb">{meta['blurb']}</div>
                    {rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
