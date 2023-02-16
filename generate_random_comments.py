import pickle as pk

data_dir = "./data/"
output_dir = "./data/samples/"


def run(subreddit, nsamples):
    usecols = ['year-month', 'text']
    comments_df = pk.load(open(data_dir + f"{subreddit}-comments.pk", "rb"))
    comments_df = comments_df[usecols]
    comments_df['year'] = comments_df['year-month'].apply(lambda ym: int(ym.split('-')[0]))
    comments_df = comments_df[comments_df['year'] >= 2014]
    random_samples = comments_df.groupby("year-month").sample(n=nsamples, replace=False)[
        ['year-month', 'text']].reset_index()
    return random_samples


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-sr", "--subreddit", required=True, type=str)
    parser.add_argument("-n", "--nsamples", default=300, type=int)
    args = parser.parse_args()

    for it in range(5):
        random_samples = run(args.subreddit, 300)
        pk.dump(random_samples, open(f"./data/samples/{args.subreddit}-sample{it}.pk", "wb"))
