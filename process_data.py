import json
import pandas as pd

data = []
with open('source_2/Electronics_5.json', 'r') as f:
    for line in f:
        row = json.loads(line)
        data.append([row["reviewerID"], row["asin"], row["overall"]])

df = pd.DataFrame(data, columns = ["UserID", "ItemID", "Rating"])

user_counts = df['UserID'].value_counts()
active_users = user_counts[user_counts >= 25].index

df = df[df["UserID"].isin(active_users)]

item_counts = df['ItemID'].value_counts()
popular_items = item_counts[item_counts >= 25].index

df = df[df["ItemID"].isin(popular_items)]

df.to_csv("data.csv", index = False)