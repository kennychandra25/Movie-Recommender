# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="🎬 MovieMatch AI — Intelligent Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* Improve fonts */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

/* Movie card */
.movie-card {
    padding: 12px;
    border-radius: 10px;
    background: #1f1f1f10;
    border: 1px solid #ffffff15;
    margin-bottom: 12px;
    transition: 0.3s ease;
}
.movie-card:hover {
    background: #ffffff15;
    transform: scale(1.01);
}

/* Movie title */
.movie-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 4px;
}

/* Subtle text */
.subtext {
    color: #bbbbbb;
    font-size: 14px;
}

/* Tabs styling */
.stTabs [role="tab"] {
    font-size: 18px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD ARTIFACTS ----------------
@st.cache_data
def load_df():
    return pd.read_pickle("tmdb_processed.pkl")

@st.cache_resource
def load_tfidf_vectorizer():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_tfidf_matrix():
    return sparse.load_npz("tfidf_matrix.npz")

@st.cache_data
def load_sbert_embeddings():
    return np.load("sbert.npy")

df = load_df()
tfidf_vec = load_tfidf_vectorizer()
tfidf_matrix = load_tfidf_matrix()
sbert_embeddings = load_sbert_embeddings()

title_col = "original_title" if "original_title" in df.columns else "title"
movie_titles = df[title_col].tolist()


# ---------------- HELPERS ----------------
def get_index_from_title(title):
    return df[df[title_col] == title].index[0]

def recommend_tfidf(selected_titles, top_n=10):
    idxs = [get_index_from_title(t) for t in selected_titles]
    user_vec = np.asarray(tfidf_matrix[idxs].mean(axis=0)).reshape(1, -1)
    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()

    sorted_idx = np.argsort(-sims)
    results = []

    for i in sorted_idx:
        title = df.iloc[i][title_col]
        if title not in selected_titles:
            results.append((i, float(sims[i])))
        if len(results) >= top_n:
            break

    return results

def recommend_sbert(selected_titles, top_n=10):
    idxs = [get_index_from_title(t) for t in selected_titles]
    user_vec = sbert_embeddings[idxs].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(user_vec, sbert_embeddings).flatten()

    sorted_idx = np.argsort(-sims)
    results = []

    for i in sorted_idx:
        title = df.iloc[i][title_col]
        if title not in selected_titles:
            results.append((i, float(sims[i])))
        if len(results) >= top_n:
            break

    return results


def display_movie_card(i, score):
    row = df.iloc[i]

    title = row[title_col]
    year = row["release_date"][:4] if pd.notna(row["release_date"]) else "N/A"
    overview = row["overview"] if pd.notna(row["overview"]) else "No overview available."

    st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{title} ({year})</div>
            <div class="subtext">Similarity: {score:.3f}</div>
            <p style="font-size:15px; margin-top:6px;">{overview[:300]}...</p>
        </div>
    """, unsafe_allow_html=True)



# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("🎛️ Controls")

    selected = st.multiselect(
        "Movies you like",
        movie_titles,
        default=[]
    )

    top_n = st.slider(
        "Number of recommendations:",
        5, 30, 10
    )

    st.markdown("---")
    st.caption("Artifacts loaded from `artifacts/` directory.")


# ---------------- MAIN TITLE ----------------
st.markdown("""
# 🎬 MovieMatch AI
### Intelligent Recommendations Powered by TF-IDF and SBERT  
Select a few movies on the left. I’ll show the **most similar movies** using both semantic and keyword-based similarity.
""")


# ---------------- TABS LAYOUT ----------------
tab1, tab2 = st.tabs(["📌 TF-IDF Recommendations", "🤖 SBERT Recommendations"])


# ---------------- SHOW RESULTS ----------------
if len(selected) == 0:
    st.info("Select movies from the sidebar to begin.")
else:
    # ---- TF-IDF TAB ----
    with tab1:
        st.subheader("🔍 Keyword-Based Similarity (TF-IDF)")
        tfidf_results = recommend_tfidf(selected, top_n)

        colA, colB = st.columns(2)
        for idx, (m_idx, score) in enumerate(tfidf_results):
            with colA if idx % 2 == 0 else colB:
                display_movie_card(m_idx, score)

    # ---- SBERT TAB ----
    with tab2:
        st.subheader("🧠 Semantic Similarity (SBERT)")
        sbert_results = recommend_sbert(selected, top_n)

        colA, colB = st.columns(2)
        for idx, (m_idx, score) in enumerate(sbert_results):
            with colA if idx % 2 == 0 else colB:
                display_movie_card(m_idx, score)
