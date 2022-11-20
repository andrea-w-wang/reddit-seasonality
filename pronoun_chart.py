from convokit import Corpus, download, HyperConvo
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
import json
from datetime import datetime
import pickle as pk
import re

dfs = dict()

subreddit = "Christianity"
corpus = Corpus(filename=download('subreddit-'+subreddit), storage_type = "db")
dfs[subreddit] = df = corpus.get_utterances_dataframe(selector = lambda utt: len(utt.text.split()) > 5)

subreddit = "TrueChristian"
corpus = Corpus(filename=download('subreddit-'+subreddit), storage_type = "db")
dfs[subreddit] = df = corpus.get_utterances_dataframe(selector = lambda utt: len(utt.text.split()) > 5)

speakers = dict()
for subreddit in ["Christianity", "TrueChristian"]:
    speakers[subreddit] = dfs[subreddit]['speaker'].unique()

common_speakers = set(speakers["Christianity"]).intersection(speakers["TrueChristian"])
common_speakers.remove("[deleted]")
common_speakers = list(common_speakers)
print(len(common_speakers))

for subreddit in ["TrueChristian", "Christianity"]:
    df = dfs[subreddit]
    print(subreddit)
    print(df[df['speaker'].isin(common_speakers)].groupby(['speaker']).size().describe())


common_speakers_utt = pd.DataFrame()
for subreddit in ["TrueChristian", "Christianity"]:
    df = dfs[subreddit]
    common_speakers_utt = pd.concat([common_speakers_utt, df[df['speaker'].isin(common_speakers)]])


common_speakers_utt['text_len'] = common_speakers_utt['text'].apply(lambda t: len(t.split()))

def get_pronoun_values(id, t):
    pronouns = [["you", "your", "yours"], ["he", "him", "his"], ["we", "us", "our"], ["I", "my", "me"], ["they", "their", "theirs"], ["it", "its", "it's"], ["she", "her", "hers"]]

    total_count = 0
    pronoun_dict = {}
    for pronoun_group in pronouns:
        count = 0
        for pronoun in pronoun_group:
            count += len(re.findall(r"\b" + re.escape(pronoun)+ r"\b", t))
            total_count += len(re.findall(r"\b" + re.escape(pronoun)+ r"\b", t))
        pronoun_identifier = "/".join(pronoun_group)
        pronoun_dict[pronoun_identifier] = count
    
    for pronoun_group in pronouns:
        pronoun_identifier = "/".join(pronoun_group)
        if total_count > 0:
            pronoun_dict[pronoun_identifier] = pronoun_dict[pronoun_identifier]/total_count
        else:
            pronoun_dict[pronoun_identifier] = 0

    return pronoun_dict[id]

pronouns = ["you/your/yours", "he/him/his", "we/us/our", "I/my/me", "they/their/theirs", "it/its/it's", "she/her/hers"]

for pronoun in pronouns:
    common_speakers_utt[pronoun] = common_speakers_utt['text'].apply(lambda t: get_pronoun_values(pronoun, t))


print(common_speakers_utt.groupby(['meta.subreddit'])['text_len'].describe())

for pronoun in pronouns:
    print(common_speakers_utt.groupby(['meta.subreddit'])[pronoun].describe())


for subreddit in ["TrueChristian", "Christianity"]:
    df = dfs[subreddit]
    df['text_len'] = df['text'].apply(lambda t: len(t.split()))
    for pronoun in pronouns:
        df[pronoun] = df['text'].apply(lambda t: get_pronoun_values(pronoun, t))
    df['speaker in both TrueChristian and Christianity'] = df['speaker'].apply(lambda s: s in common_speakers)


pd.set_option('display.max_rows', 1000)

print("r/Christianity")
print(dfs['Christianity'].groupby(['speaker in both TrueChristian and Christianity'])['text_len'].describe())
for pronoun in pronouns:
    print(dfs['Christianity'].groupby(['speaker in both TrueChristian and Christianity'])[pronoun].describe())


print("r/TrueChristian")
print(dfs['TrueChristian'].groupby(['speaker in both TrueChristian and Christianity'])['text_len'].describe())
for pronoun in pronouns:
    print(dfs['TrueChristian'].groupby(['speaker in both TrueChristian and Christianity'])[pronoun].describe())


print(dfs['TrueChristian']['speaker'].nunique())

df = dfs['TrueChristian'] 
first_utt_in_TC = df.groupby(['speaker'])['timestamp'].min()
first_utt_in_TC = first_utt_in_TC.reset_index(name='first_utt_in_TC')
print(first_utt_in_TC.head())

common_speakers_comments = common_speakers_utt[common_speakers_utt['reply_to'].notnull()].copy()

x = common_speakers_comments.groupby(['speaker', 'meta.subreddit']).size().reset_index(name='num_comments')
x = x.pivot(index='speaker', columns='meta.subreddit', values='num_comments')

# speakers with at least 10 posts in both subreddits
active_speakers = x[(x['Christianity'] >= 10) & (x['TrueChristian'] >= 10)].index

def subset_by_speakers(df, speakers):
    return df[df['speaker'].isin(speakers)].copy()

def subset_by_subreddit(df, subreddit):
    return df[df['meta.subreddit'] == subreddit].copy()

active_common_speakers_comments = subset_by_speakers(common_speakers_comments, active_speakers)

common_speakers_comments_in_Christianity = subset_by_subreddit(active_common_speakers_comments, 'Christianity')
common_speakers_comments_in_Christianity = common_speakers_comments_in_Christianity.reset_index()
print(common_speakers_comments_in_Christianity.shape)

common_speakers_comments_in_Christianity = common_speakers_comments_in_Christianity.merge(first_utt_in_TC, on='speaker')
print(common_speakers_comments_in_Christianity.shape)

print(common_speakers_comments_in_Christianity['speaker'].nunique())

common_speakers_comments_in_Christianity['before_TC'] = common_speakers_comments_in_Christianity.apply(
    lambda row: 1 if row['timestamp'] < row['first_utt_in_TC'] else 0, axis=1)
common_speakers_comments_in_Christianity['after_TC'] = common_speakers_comments_in_Christianity.apply(
    lambda row: 1 if row['timestamp'] >= row['first_utt_in_TC'] else 0, axis=1)

the_group = common_speakers_comments_in_Christianity.groupby(['speaker'])
temp_df = the_group.agg({"before_TC": sum, "after_TC": sum})
temp_df['total_comments'] = the_group['id'].count()
print(temp_df)

for K in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    print(K)
    print(" ", len(temp_df[(temp_df['before_TC'] >= K) & (temp_df['after_TC'] >= K)]))

K = 80
qualified_speakers = list(temp_df[(temp_df['before_TC'] >= K) & (temp_df['after_TC'] >= K)].index)
selected = common_speakers_comments_in_Christianity[common_speakers_comments_in_Christianity['speaker'].isin(qualified_speakers)]

print(selected['speaker'].nunique())

after = selected[selected['after_TC'] == 1].copy()
gr = after.sort_values(['speaker', 'timestamp']).groupby(["speaker"])
after['order'] = gr.cumcount() + 1


before = selected[selected['before_TC'] == 1].copy()
gr = before.sort_values(['speaker', 'timestamp'], ascending=False).groupby(['speaker'])
before['order'] = -1 * (gr.cumcount()+1)


import scipy.stats as st
def conf_interval(data):
    interval = st.norm.interval(confidence=0.95, loc=np.mean(data), scale=st.sem(data))
    return (interval[-1] - interval[0])/2

dff = pd.concat([before, after])
dff = dff[dff['order'].abs() <= K]

g = dff.groupby(['order']).agg({"you/your/yours": ["mean", conf_interval]})
g.columns = g.columns.map(lambda x: '_'.join([str(i) for i in x]))
g = g.reset_index()
g.plot('order', 'you/your/yours_mean')

g = dff.groupby(['order']).agg({"he/him/his": ["mean", conf_interval]})
g.columns = g.columns.map(lambda x: '_'.join([str(i) for i in x]))
g = g.reset_index()
g.plot('order', 'he/him/his_mean')

g = dff.groupby(['order']).agg({"we/us/our": ["mean", conf_interval]})
g.columns = g.columns.map(lambda x: '_'.join([str(i) for i in x]))
g = g.reset_index()
g.plot('order', 'we/us/our_mean')

g = dff.groupby(['order']).agg({"I/my/me": ["mean", conf_interval]})
g.columns = g.columns.map(lambda x: '_'.join([str(i) for i in x]))
g = g.reset_index()
g.plot('order', 'I/my/me_mean')

g = dff.groupby(['order']).agg({"they/their/theirs": ["mean", conf_interval]})
g.columns = g.columns.map(lambda x: '_'.join([str(i) for i in x]))
g = g.reset_index()
g.plot('order', 'they/their/theirs_mean')

g = dff.groupby(['order']).agg({"it/its/it's": ["mean", conf_interval]})
g.columns = g.columns.map(lambda x: '_'.join([str(i) for i in x]))
g = g.reset_index()
g.plot('order', "it/its/it's_mean")

g = dff.groupby(['order']).agg({"she/her/hers": ["mean", conf_interval]})
g.columns = g.columns.map(lambda x: '_'.join([str(i) for i in x]))
g = g.reset_index()
g.plot('order', "she/her/hers_mean")
