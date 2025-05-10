import heapq

def build_matrix(df, user_id_mapper, item_id_mapper):
    user_item_ratings = [[0 for _ in range(len(item_id_mapper))] for _ in range(len(user_id_mapper))]
    
    for _, row in df.iterrows():
        user_idx = user_id_mapper[row["UserID"]]
        item_idx = item_id_mapper[row["ItemID"]]

        user_item_ratings[user_idx][item_idx] = row["Rating"]

    return user_item_ratings

def get_transpose(matrix):
    matrix_T = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for user_idx in range(len(matrix)):
        
        for item_idx in range(len(matrix[0])):
            matrix_T[item_idx][user_idx] = matrix[user_idx][item_idx]

    return matrix_T

def dot_product(vector_1, vector_2):
    result = 0
    for x, y in zip(vector_1, vector_2):
        result += (x * y)

    return result

def magnitude(vector):
    return dot_product(vector, vector) ** 0.5

def cosine_similarity(vector_1, vector_2):
    numerator = dot_product(vector_1, vector_2)
    denominator = magnitude(vector_1) * magnitude(vector_2)

    return numerator / denominator

def euclidean_distance(vector_1, vector_2):
    result = 0
    for x, y in zip(vector_1, vector_2):
        result += (x - y) ** 2
    
    return result ** 0.5

def euclidean_similarity(vector_1, vector_2):
    distance = euclidean_distance(vector_1, vector_2)

    return 1 / (1 + distance)

def _similarity(ratings, metric = "Cosine"):
    n = len(ratings)
    sim_matrix = [[0 for _ in range(n)] for _ in range(n)]

    if metric == "Cosine":
        metric = cosine_similarity
    elif metric == "Euclidean":
        metric = euclidean_similarity

    progress = 0.05
    for vector_1_idx in range(n):
        sim_matrix[vector_1_idx][vector_1_idx] = 1
        for vector_2_idx in range(vector_1_idx + 1, n):
            sim_matrix[vector_1_idx][vector_2_idx] = metric(ratings[vector_1_idx], ratings[vector_2_idx])
            sim_matrix[vector_2_idx][vector_1_idx] = sim_matrix[vector_1_idx][vector_2_idx]
        
        if vector_1_idx == int(progress * n):
            print(f"Building Similarity Matrix... Completed: {progress}%" )
            progress += 0.05

    return sim_matrix

def get_similarity_matrix(ratings, user_based = True, metric = "Cosine"):
    if user_based == False:
        ratings = get_transpose(ratings)

    sim_matrix = _similarity(ratings, metric)

    return sim_matrix

def predict_score(user_item_ratings, k_best_neibhours, target_item):
    numerator, denominator = 0, 0
    for similarity, neibhour in k_best_neibhours:
        rating = user_item_ratings[neibhour][target_item]
        if rating > 0:
            numerator += (similarity * rating)
            denominator += abs(similarity)

    return numerator / denominator if denominator != 0 else 0

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

    recommended_items = sorted(recommended_items, reverse = True)
    return recommended_items