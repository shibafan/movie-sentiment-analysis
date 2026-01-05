import json
import time
from datetime import datetime, timezone
import requests

# using PullPush api to get reddit data
API_URL = "https://api.pullpush.io/reddit/search/submission"

def date_to_timestamp(date_string):
    """Turn a date like '2020-01-01' into a Unix timestamp"""
    dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def get_reddit_posts(subreddit, after_ts, before_ts, limit=100, search_term=None):
    """fetch a batch of posts from the API"""
    params = {
        "subreddit": subreddit,
        "after": after_ts,
        "before": before_ts,
        "size": limit,
        "sort": "desc",
        "sort_type": "created_utc",
    }
    
    if search_term:
        params["q"] = search_term

    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # check if api returned error
    if data.get("error"):
        raise RuntimeError(f"API error: {data['error']}")
    
    return data.get("data", [])

def download_posts(subreddit, start_date, end_date, output_file, 
                   search_term=None, max_posts=None, sleep_time=1.0):
    """
    Download Reddit posts from a subreddit between two dates.
    Saves them to a jsonl file (one JSON object per line).
    """
    after_ts = date_to_timestamp(start_date)
    before_ts = date_to_timestamp(end_date)

    already_seen = set()
    posts_saved = 0

    with open(output_file, "w") as file:
        while True:
            # get a batch of posts
            posts = get_reddit_posts(subreddit, after_ts, before_ts, 
                                    limit=100, search_term=search_term)
            
            if not posts:
                print("no more posts found")
                break

            # Process and save each post
            earliest_timestamp = None
            for post in posts:
                post_id = post.get("id")
                
                # Skip if we've already saved this post
                if not post_id or post_id in already_seen:
                    continue
                
                already_seen.add(post_id)
                file.write(json.dumps(post) + "\n")
                posts_saved += 1

                # track oldest post timestamp for pagination
                timestamp = post.get("created_utc")
                if timestamp:
                    if earliest_timestamp is None:
                        earliest_timestamp = timestamp
                    else:
                        earliest_timestamp = min(earliest_timestamp, timestamp)

                # Stop if we hit the limit
                if max_posts and posts_saved >= max_posts:
                    print(f"Hit the limit of {max_posts} posts")
                    return

            # If we didn't find any timestamps, we're done
            if earliest_timestamp is None:
                break

            # Move the time window back for the next batch
            before_ts = earliest_timestamp - 1
            print(f"Saved {posts_saved} posts so far")
            time.sleep(sleep_time)

if __name__ == "__main__":
    # 3 years of r/movies posts
    download_posts(
        subreddit="movies",
        start_date="2020-01-01",
        end_date="2023-01-01",
        output_file="raw_data/movies_submissions.jsonl",
        search_term=None,
        max_posts=5000,
    )