import argparse
import json
import time

import requests

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subreddit", required=True, type=str)
args = parser.parse_args()

print(args.subreddit)

# Set the subreddit and time range
subreddit = args.subreddit
start_epoch = 1538352000  # Oct 1, 2018 at midnight UTC in Unix time
end_epoch = int(time.time())

# Set the base URL for the Pushshift API
url_template = "https://api.pushshift.io/reddit/search/comment/?subreddit={}&after={}&before={}&size=500"

# Open the output file in write mode
with open(f"./data/after_2018_Oct/{subreddit}-comments.jsonl", "w") as outfile:
    # Loop through each time period (500 comments at a time)
    while start_epoch < end_epoch:
        # Construct the URL for this time period
        url = url_template.format(subreddit, start_epoch, start_epoch + 86400)

        # Initialize an empty list to store the comments for this time period
        period_comments = []

        # Make the API request and parse the JSON response
        response = requests.get(url)
        data = json.loads(response.text)["data"]

        # Add the comments from this request to the list
        period_comments += data

        # If there are more comments than the request retrieved, continue making requests with different pagination parameters
        if len(data) == 500:
            last_comment_time = data[-1]["created_utc"]
            while True:
                # Construct the URL for the next page of comments
                url = url_template.format(subreddit, start_epoch,
                                          start_epoch + 86400) + f"&before={last_comment_time}&size=500"

                # Make the API request and parse the JSON response
                response = requests.get(url)
                data = json.loads(response.text)["data"]

                # If there are no more comments, break out of the pagination loop
                if len(data) == 0:
                    break

                # Add the comments from this page to the list and update the last_comment_time variable
                period_comments += data
                last_comment_time = data[-1]["created_utc"]

        # Write each comment to the output file as a JSON object on its own line
        for comment in period_comments:
            outfile.write(json.dumps(comment) + "\n")

        # Move the time period forward by one day
        start_epoch += 86400

        # Pause the script for 1 second to avoid overloading the API
        time.sleep(1)

# Print the number of comments collected
print(f"Collected {sum(1 for _ in open('comments.jsonl'))} comments")
