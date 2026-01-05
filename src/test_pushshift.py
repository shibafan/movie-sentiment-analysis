import requests

url = "https://api.pullpush.io/reddit/search/submission"
params = {
    "subreddit": "movies",
    "size": 5
}

r = requests.get(url, params=params)
data = r.json()

if not data.get("data"):
    print("No results found")
else:
    print("Number of posts returned:", len(data["data"]))
    first = data["data"][0]
    print("Title:", first.get("title"))
    print("Selftext:", first.get("selftext")[:200])  # first 200 characters