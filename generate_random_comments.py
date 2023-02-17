import pickle as pk
import time
data_dir = "./data/"
output_dir = "./data/samples/"


def run(subreddit, nsamples):
    usecols = ['year-month', 'text']
    comments_df = pk.load(open(data_dir + f"{subreddit}-comments.pk", "rb"))
    comments_df = comments_df[usecols]
    comments_df['year'] = comments_df['year-month'].apply(lambda ym: int(ym.split('-')[0]))
    comments_df = comments_df[comments_df['year'] >= 2014]
    comments_df['text'] = comments_df['text'].apply(lambda t: ' '.join(list(filter(bool, t.split()))))
    comments_df['text_len'] = comments_df['text'].apply(lambda t: len(t.split()))
    comments_df = comments_df[comments_df['text_len'] >= 5]
    random_samples = comments_df.groupby("year-month").sample(n=nsamples, replace=False)[
        ['year-month', 'text']]
    return random_samples


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", "--subreddit", required=True, type=str)
    parser.add_argument("-n", "--nsamples", default=300, type=int)
    args = parser.parse_args()

    for it in range(5):
        random_samples = run(args.subreddit, 300)
        pk.dump(random_samples, open(f"./data/samples/{args.subreddit}-sample{int(time.time())}.pk", "wb"))
