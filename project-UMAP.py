import argparse
import pickle as pk

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from adjustText import adjust_text

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_neighbors", required=True, type=float)
parser.add_argument("-d", "--min_dist", required=True, type=float)
args = parser.parse_args()
subreddits = ['AdviceAnimals', 'AmItheAsshole', 'Android', 'AskMen', 'Bitcoin', 'Buddhism', 'CFB', 'Christianity',
              'DebateReligion', 'Diablo', 'Drugs', 'Economics', 'Fitness', 'Frugal', 'Games', 'IAmA', 'Judaism',
              'LifeProTips', 'MMA', 'MakeupAddiction', 'Marvel', 'MensRights', 'Minecraft', 'Music', 'Naruto',
              'Random_Acts_Of_Amazon', 'ShingekiNoKyojin', 'SquaredCircle', 'WTF', 'anime', 'apple', 'atheism',
              'australia', 'baseball', 'books', 'business', 'canada', 'cars', 'conspiracy', 'cringe', 'cringepics',
              'dayz', 'electronic_cigarette', 'explainlikeimfive', 'fantasyfootball', 'funny', 'gaming', 'gifs', 'guns',
              'hiphopheads', 'hockey', 'leagueoflegends', 'magicTCG', 'malefashionadvice', 'motorcycles', 'movies',
              'nba', 'news', 'nfl', 'photography', 'pics', 'pokemontrades', 'politics', 'programming',
              'relationship_advice', 'relationships', 'science', 'sex', 'singapore', 'skyrim', 'snowboarding', 'soccer',
              'technology', 'techsupport', 'teenagers', 'tennis', 'tf2', 'tifu', 'todayilearned', 'travel', 'trees',
              'unitedkingdom', 'videos', 'worldnews']

coef_df = pd.DataFrame()

for subreddit in subreddits:
    dist_params = pk.load(open(f"data/output/regression/{subreddit}-emb_params-W.pk", "rb"))
    df = pd.DataFrame(dist_params)
    df['subreddit'] = subreddit
    df = df[~df['variable'].str.contains("month_1")]
    coef_df = pd.concat((coef_df, df))

coef_df = coef_df[coef_df['variable'].isin(['C(n_months)[T.1]', 'C(n_months)[T.2]',
                                            'C(n_months)[T.3]', 'C(n_months)[T.4]', 'C(n_months)[T.5]',
                                            'C(n_months)[T.6]', 'C(n_months)[T.7]', 'C(n_months)[T.8]',
                                            'C(n_months)[T.9]', 'C(n_months)[T.10]', 'C(n_months)[T.11]',
                                            'C(n_months)[T.12]'])]

mean_coef = coef_df.groupby(['subreddit', 'variable']).mean().reset_index()
x = pd.pivot_table(mean_coef, index='subreddit', columns='variable', values='coefficient')

method = umap.UMAP(
    n_neighbors=args.n_neighbors,
    min_dist=args.min_dist,
    n_components=2,
    metric='euclidean'
)

projection = method.fit_transform(x.values)

proj_df = pd.DataFrame(projection)
proj_df['subreddit'] = x.index

plt.figure(figsize=(10, 10))
sns.scatterplot(data=proj_df, x=0, y=1, alpha=0.5)

texts = []
for r in proj_df.to_dict(orient='records'):
    texts.append(plt.text(r[0], r[1], r['subreddit']))

adjust_text(texts, only_move={'points': 'xy', 'texts': 'xy'},
            autoalign='y',
            arrowprops=dict(arrowstyle="->", color='b', lw=0.5)
            )

plt.title(f"UMAP-n_neighbors={args.n_neighbors}-min_dist={args.min_dist}")
plt.savefig(f"./figures/subreddits-UMAP-neighbors_{args.n_neighbors}-mindist_{args.min_dist}.png")
