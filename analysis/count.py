import help_functions as hp
import time

start = time.time()

train = hp.create_ts("/media/RC_2015-01")
subreddits = []
users = []

for entry in train:
    subreddit = entry["subreddit"]
    if subreddit not in subreddits:
        subreddits.append(subreddit)
    user = entry["author"]
    if user not in users:
        users.append(user)

print len(users), "users"
print len(subreddits), "subreddits"
print time.time()-start, "elapsed time"