import pickle as pk

from sklearn.neighbors import NearestNeighbors

combined_emb_df = pk.load(open("./data/combined_emb_df.pk", "rb"))
embeddings = combined_emb_df.drop(['id', 'year-month', 'file'], axis=1).values
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(embeddings)
out = nbrs.kneighbors(embeddings)
pk.dump(out, open("./data/output/nn_out.pk", "wb"))
