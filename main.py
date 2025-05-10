import os
import csv
import pandas as pd

from utils import *

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

target_user = 0
recommendations = recommend(user_item_ratings, user_sim_matrix, target_user, n = 50)

print(f"Recommended Items for User: {user_inv_id_mapper[target_user]}\n")
for idx, (predicted_score, item_idx) in enumerate(recommendations):
    print(f"{idx:2d}. {item_inv_id_mapper[item_idx]} - Predicted Score: {predicted_score:.2f}")