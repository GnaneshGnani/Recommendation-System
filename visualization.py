import os
import csv
import heapq
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from utils import build_matrix, get_similarity_matrix, recommend

# 1. load your data / precomputed matrices
df = pd.read_csv("data.csv")

user_id_mapper = {user_id: idx for idx, user_id in enumerate(df["UserID"].unique())}
item_id_mapper = {item_id: idx for idx, item_id in enumerate(df["ItemID"].unique())}

user_inv_id_mapper = {idx: user_id for user_id, idx in user_id_mapper.items()}
item_inv_id_mapper = {idx: item_id for item_id, idx in item_id_mapper.items()}

user_item_ratings = build_matrix(df, user_id_mapper, item_id_mapper)
if os.path.exists("matrix.csv"):    
    with open('matrix.csv', mode='r') as file:
        reader = csv.reader(file)
        # Convert values to float instead of int
        user_sim_matrix = [list(map(float, row)) for row in reader]
    
    print("Matrix loaded from CSV")

else:
    user_sim_matrix = get_similarity_matrix(user_item_ratings)
    with open('matrix.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(user_sim_matrix)
    
    print("Matrix saved as CSV.")

# 2. UI controls
user_ids = list(user_id_mapper.keys())
target = st.sidebar.selectbox("Choose user", user_ids)
target_idx = user_id_mapper[target]
k = st.sidebar.slider("Number of neighbours (k)", 1, 50, 10)

# 3.
st.write("User Similarity Scores")
temp_df = pd.DataFrame([user_sim_matrix[target_idx]], columns = list(user_id_mapper.keys()))

neibhours = [(similarity, user_idx) for user_idx, similarity in enumerate(user_sim_matrix[target_idx])]
k_best_neibhours = heapq.nlargest(k + 1, neibhours, key = lambda x: x[0])[1:]

temp_sim = [sim for sim, _ in k_best_neibhours]
temp_idx = [user_inv_id_mapper[idx] for _, idx in k_best_neibhours]

print(temp_idx)
print(temp_sim)
print(user_id_mapper["A2OZ5ERO1D2YOX"])
print(user_inv_id_mapper[16])
print(user_id_mapper["A3UZ17HANZ9F1E"])
# print(user_inv_id_mapper[16])

def highlight_selected_cells(x):
    return ['background-color: yellow' if col in temp_idx else '' for col in x.index]

styled_df = temp_df.style.apply(highlight_selected_cells, axis = 1)

st.dataframe(styled_df, hide_index = True)

st.write("K Similar Users")
temp_df = pd.DataFrame([temp_sim], columns = temp_idx)
st.dataframe(temp_df, hide_index = True)

# 4. Show top-N recommendations
recommendations = recommend(user_item_ratings, user_sim_matrix, target_idx, k = k)
st.write("Top recommendations:")
for score, idx in recommendations:
    st.write(f"- {item_inv_id_mapper[idx]} (score: {score:.2f})")
