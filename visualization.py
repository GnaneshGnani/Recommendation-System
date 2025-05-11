import os
import csv
import heapq

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils import build_matrix, get_similarity_matrix, recommend


@st.cache_data
def load_data(path: str = 'data.csv') -> pd.DataFrame:
    """Load the ratings CSV into a DataFrame."""
    return pd.read_csv(path)


@st.cache_data
def load_or_compute_similarity(
    ratings: list[list[float]], csv_path: str = 'matrix.csv'
) -> list[list[float]]:
    """
    Load a cached similarity matrix if available,
    otherwise compute, cache, and return it.
    """
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            return [list(map(float, row)) for row in reader]

    sim_matrix = get_similarity_matrix(ratings)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sim_matrix)
    return sim_matrix


def main():
    st.title("📦 User-Based Collaborative Filtering Visualizer")

    # Load data and build mappings
    df = load_data()
    user_ids = df["UserID"].unique().tolist()
    item_ids = df["ItemID"].unique().tolist()

    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
    inv_user_id_map = {idx: uid for uid, idx in user_id_map.items()}
    inv_item_id_map = {idx: iid for iid, idx in item_id_map.items()}

    # Build rating matrix and similarity
    ratings = build_matrix(df, user_id_map, item_id_map)
    sim_matrix = load_or_compute_similarity(ratings)

    # Sidebar controls
    target_user = st.sidebar.selectbox("Choose user", user_ids)
    target_idx = user_id_map[target_user]
    k = st.sidebar.slider("Number of neighbours (k)", 1, 50, 10)
    n = st.sidebar.slider("Number of recommendations (n)", 1, 100, 10)

    # Display top-k neighbours
    st.header("User Similarity Scores")
    user_sims = sim_matrix[target_idx]

    neighbours = sorted(
        enumerate(user_sims), key = lambda x: x[1], reverse = True
    )[1:k+1]

    sim_series = pd.Series(
        {inv_user_id_map[idx]: sim for idx, sim in enumerate(user_sims)},
        name = "Similarity"
    )

    neighbours_indexes = [inv_user_id_map[idx] for idx, _ in neighbours]

    def highlight_selected_cells(x):
        return ['background-color: yellow' if col in neighbours_indexes else '' for col in x.index]

    sim_df = sim_series.to_frame().T
    styled_df = sim_df.style.apply(highlight_selected_cells, axis = 1)

    st.dataframe(styled_df)

    sim_series = pd.Series(
        {inv_user_id_map[idx]: sim for idx, sim in neighbours},
        name = "Similarity"
    )
    sim_df = sim_series.to_frame().T

    st.header("K Similar Users")
    st.dataframe(sim_df)

    # Recommendations
    st.header("Top Recommendations")
    recs = recommend(ratings, sim_matrix, target_idx, k=k, n=n)
    for rank, (score, idx) in enumerate(recs, start=1):
        st.markdown(
            f"**{rank}. Item: {inv_item_id_map[idx]} — Predicted Score: {score:.2f}**"
        )

    # Comparison view
    st.sidebar.header("User Comparison")
    compare_user = st.sidebar.selectbox(
        "Compare With Another User", [u for u in user_ids if u != target_user]
    )
    compare_idx = user_id_map[compare_user]

    vec1 = ratings[target_idx]
    vec2 = ratings[compare_idx]

    plot_mode = st.radio(
        "Select View Mode:", ["All Rated Items", "Co-Rated Items Only"]
    )
    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_mode == "All Rated Items":
        items1 = [(i, r) for i, r in enumerate(vec1) if r > 0]
        items2 = [(i, r) for i, r in enumerate(vec2) if r > 0]
        if items1:
            x1, y1 = zip(*items1)
            ax.stem(
                x1, y1, linefmt='tab:blue', markerfmt='bo', basefmt=' ', label=target_user
            )
        if items2:
            x2, y2 = zip(*items2)
            ax.stem(
                x2, y2, linefmt='tab:orange', markerfmt='ro', basefmt=' ', label=compare_user
            )
        ax.set_title("Rating Comparison (All Rated Items)")
    else:
        common = [i for i in range(len(vec1)) if vec1[i] > 0 and vec2[i] > 0]
        if common:
            y1 = [vec1[i] for i in common]
            y2 = [vec2[i] for i in common]
            ax.plot(common, y1, 'bo-', label=target_user)
            ax.plot(common, y2, 'ro-', label=compare_user)
            ax.set_title("Rating Comparison (Co-Rated Items Only)")
        else:
            st.warning("These two users have no items rated in common.")

    ax.set_xlabel("Item Index")
    ax.set_ylabel("Rating")
    ax.set_ylim(0.5, 5.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    similarity = user_sims[compare_idx]
    st.markdown(f"**Similarity = {similarity:.2f}**")


if __name__ == "__main__":
    main()
