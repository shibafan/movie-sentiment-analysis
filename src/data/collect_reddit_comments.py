import json
import time
from datetime import datetime, timezone
import requests

# using PullPush api to get reddit comments this time
API_URL = "https://api.pullpush.io/reddit/search/comment"

def date_to_timestamp(date_string):
    """Turn a date like '2020-01-01' into a Unix timestamp"""
    dt = datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

def get_reddit_comments(subreddit, after_ts, before_ts, limit=100, search_term=None):
    """fetch a batch of comments from the API"""
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

def download_comments(subreddit, start_date, end_date, output_file, 
                      search_term=None, max_comments=None, sleep_time=1.0):
    """
    Download Reddit comments from a subreddit between two dates.
    Saves them to a jsonl file (one JSON object per line).
    """
    after_ts = date_to_timestamp(start_date)
    before_ts = date_to_timestamp(end_date)

    already_seen = set()
    comments_saved = 0

    with open(output_file, "w") as file:
        while True:
            # get a batch of comments
            comments = get_reddit_comments(subreddit, after_ts, before_ts, 
                                          limit=100, search_term=search_term)
            
            if not comments:
                print("no more comments found")
                break

            # process and save each comment
            earliest_timestamp = None
            for comment in comments:
                comment_id = comment.get("id")
                
                # skip if we already saved this comment
                if not comment_id or comment_id in already_seen:
                    continue
                
                already_seen.add(comment_id)
                file.write(json.dumps(comment) + "\n")
                comments_saved += 1

                # track oldest comment timestamp for pagination
                timestamp = comment.get("created_utc")
                if timestamp:
                    if earliest_timestamp is None:
                        earliest_timestamp = timestamp
                    else:
                        earliest_timestamp = min(earliest_timestamp, timestamp)

                # stop if we hit the limit
                if max_comments and comments_saved >= max_comments:
                    print(f"Hit the limit of {max_comments} comments")
                    return

            # if we didn't find any timestamps, we're done
            if earliest_timestamp is None:
                break

            # move the time window back for the next batch
            before_ts = earliest_timestamp - 1
            print(f"Saved {comments_saved} comments so far")
            time.sleep(sleep_time)

if __name__ == "__main__":
    # 3 years of r/movies comments
    download_comments(
        subreddit="movies",
        start_date="2020-01-01",
        end_date="2023-01-01",
        output_file="raw_data/movies_comments.jsonl",
        search_term=None,
        max_comments=50000,
    )