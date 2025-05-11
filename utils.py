import math
import heapq
import numpy as np

def build_matrix(df, user_id_mapper, item_id_mapper):
    n_users = len(user_id_mapper)
    n_items = len(item_id_mapper)
    user_item_ratings = [[0.0] * n_items for _ in range(n_users)]

    for row in df.itertuples():
        uid = user_id_mapper[row.UserID]
        iid = item_id_mapper[row.ItemID]
        user_item_ratings[uid][iid] = float(row.Rating)
        
    return user_item_ratings

def get_transpose(matrix):
    matrix_T = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for user_idx in range(len(matrix)):
        
        for item_idx in range(len(matrix[0])):
            matrix_T[item_idx][user_idx] = matrix[user_idx][item_idx]

    return matrix_T

def dot_product(vector_1, vector_2):
    return sum(x * y for x, y in zip(vector_1, vector_2))

def magnitude(vector):
    return math.sqrt(dot_product(vector, vector))

def cosine_similarity(vector_1, vector_2):
    numerator = dot_product(vector_1, vector_2)
    denominator = magnitude(vector_1) * magnitude(vector_2)

    return numerator / denominator if denominator else 0.0

def euclidean_distance(vector_1, vector_2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

def euclidean_similarity(vector_1, vector_2):
    distance = euclidean_distance(vector_1, vector_2)

    return 1 / (1 + distance)

def _similarity(ratings, metric = "Cosine"):
    n = len(ratings)
    sim = [[0 for _ in range(n)] for _ in range(n)]

    metric_func = cosine_similarity if metric == "Cosine" else euclidean_similarity

    progress = 0.05
    for vector_1_idx in range(n):
        sim[vector_1_idx][vector_1_idx] = 1
        for vector_2_idx in range(vector_1_idx + 1, n):
            sim[vector_1_idx][vector_2_idx] = metric_func(ratings[vector_1_idx], ratings[vector_2_idx])
            sim[vector_2_idx][vector_1_idx] = sim[vector_1_idx][vector_2_idx]
        
        if vector_1_idx == int(progress * n):
            print(f"Building Similarity Matrix... Completed: {progress}%" )
            progress += 0.05

    return sim

def get_similarity_matrix(ratings, user_based = True, metric = "Cosine"):
    if user_based == False:
        ratings = get_transpose(ratings)

    return _similarity(ratings, metric)

def predict_score(user_item_ratings, k_best_neibhours, target_item):
    numerator, denominator = 0.0, 0.0
    for similarity, neibhour in k_best_neibhours:
        rating = user_item_ratings[neibhour][target_item]
        if rating > 0:
            numerator += (similarity * rating)
            denominator += abs(similarity)

    return numerator / denominator if denominator else 0

def recommend(user_item_ratings, sim_matrix, target_user, k = 10, n = 20):
    recommended_items = []
    neibhours = [(similarity, user_idx) for user_idx, similarity in enumerate(sim_matrix[target_user])]
    k_best_neibhours = heapq.nlargest(k + 1, neibhours, key = lambda x: x[0])[1:]

    print("K Similar Users:", k_best_neibhours)
    
    for item_idx, rating in enumerate(user_item_ratings[target_user]):
        if rating != 0:
            continue

        score = predict_score(user_item_ratings, k_best_neibhours, item_idx)

        heapq.heappush(recommended_items, (score, item_idx)) 
        if len(recommended_items) > n:
            heapq.heappop(recommended_items)

    return sorted(recommended_items, reverse = True)

def svd_recommend(user_item_ratings, target_user, n = 20, rank = 50):
    R = np.array(user_item_ratings, dtype = np.float32)
    mask = R > 0

    # Compute user mean ratings
    means = np.zeros(R.shape[0], dtype = np.float32)
    for i in range(R.shape[0]):
        rated = mask[i]
        if rated.any():
            means[i] = R[i, rated].mean()

    # Center ratings
    Rc = (R - means[:, None]) * mask

    # SVD decomposition
    U, s, Vt = np.linalg.svd(Rc, full_matrices=False)
    k_rank = min(rank, len(s))
    U_k = U[:, :k_rank]
    s_k = s[:k_rank]
    Vt_k = Vt[:k_rank, :]

    # Reconstruct and add back means
    R_hat = (U_k * s_k) @ Vt_k + means[:, None]

    # Gather predictions for unseen items
    preds = []
    for j in range(R.shape[1]):
        if not mask[target_user, j]:
            preds.append((float(R_hat[target_user, j]), j))
    preds.sort(key=lambda x: x[0], reverse=True)
    
    return preds[:n]