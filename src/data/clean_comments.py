import json
import pandas as pd

RAW_PATH = "raw_data/movies_comments.jsonl"
OUT_PATH = "clean_data/movies_comments_clean.csv"

rows = []

with open(RAW_PATH, "r") as f:
    for line in f:
        obj = json.loads(line)

        body = obj.get("body", "")
        if not body:
            continue

        body = body.strip()

        if body in ("[deleted]", "[removed]"):
            continue

        if len(body.split()) < 5:
            continue

        rows.append({
            "comment_id": obj.get("id"),
            "created_utc": obj.get("created_utc"),
            "score": obj.get("score"),
            "body": body
        })

df = pd.DataFrame(rows)
df = df.dropna(subset=["comment_id", "body"])

print(len(df), "comments after cleaning")
print("Avg length:", df["body"].str.split().apply(len).mean())
print("Median length:", df["body"].str.split().apply(len).median())

df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
