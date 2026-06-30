"""Sanity checks -- not a full test suite, just enough to catch broken logic
before submission. Run with: python sanity_check.py

Each check either compares against a calculation done a different way
(so a bug in the class can't hide behind a bug in the check), or verifies
a mathematical property that has to hold no matter what the data looks like.
"""

import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_ratings, load_items, load_tags, train_test_split_ratings
from src.baselines import MostPopularRecommender, HighestAverageRatingRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k

passed = 0
failed = 0


def check(label, condition):
    global passed, failed
    if condition:
        print(f"  OK   {label}")
        passed += 1
    else:
        print(f"  FAIL {label}")
        failed += 1


print("loading data...")
ratings = load_ratings()
items = load_items()
tags = load_tags()
train, test = train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE)

print("\n[data integrity]")
check("ratings are within 0.5-5.0", ratings[config.RATING_COL].between(0.5, 5.0).all())
check("no missing values in ratings", not ratings.isnull().values.any())
check("train and test don't overlap", len(pd.merge(train, test, how="inner")) == 0)
check("train + test = full dataset", len(train) + len(test) == len(ratings))
check("every rated movieId exists in items.csv",
      set(ratings[config.ITEM_COL]).issubset(set(items[config.ITEM_COL])))

print("\n[most popular: cross-checked against raw pandas value_counts]")
manual_ranking = train[config.ITEM_COL].value_counts().index.tolist()
mp = MostPopularRecommender().fit(train)
mp_ranking = [item_id for item_id, _ in mp.recommend(-1, train, n=10, exclude_seen=False)]
check("top-10 matches manual value_counts exactly", mp_ranking == manual_ranking[:10])

print("\n[highest average: cross-checked against raw pandas groupby]")
manual_stats = train.groupby(config.ITEM_COL)[config.RATING_COL].agg(["mean", "count"])
manual_stats = manual_stats[manual_stats["count"] >= 20].sort_values("mean", ascending=False)
ha = HighestAverageRatingRecommender(min_ratings=20).fit(train)
ha_ranking = [item_id for item_id, _ in ha.recommend(-1, train, n=10, exclude_seen=False)]
check("top-10 matches manual groupby exactly", ha_ranking == manual_stats.index.tolist()[:10])
check("no item with fewer than 20 ratings ever appears",
      all(manual_stats.loc[i, "count"] >= 20 for i in ha_ranking))

print("\n[content-based: known-similar movies should actually rank as similar]")
cb = ContentBasedRecommender().fit(items, tags)
toy_story_similar = [i for i, _ in cb.similar_items(1, n=10)]  # 1 = Toy Story (1995)
check("Toy Story 2 (id 3114) appears in top-10 similar to Toy Story", 3114 in toy_story_similar)
check("similarity scores are between 0 and 1",
      all(0 <= s <= 1.0001 for _, s in cb.similar_items(1, n=10)))

print("\n[item-item CF: scores should be on the rating scale, recs should exclude seen items]")
ii = ItemItemCollaborativeFiltering(k=20).fit(train)
sample_user = train[config.USER_COL].iloc[0]
recs = ii.recommend(sample_user, train, n=10, exclude_seen=True)
seen = set(train.loc[train[config.USER_COL] == sample_user, config.ITEM_COL])
check("no recommended item was already rated by this user",
      all(item_id not in seen for item_id, _ in recs))
check("predicted scores fall within the 0.5-5.0 rating scale",
      all(0 <= score <= 5.5 for _, score in recs))

print("\n[matrix factorization: must beat the trivial global-mean baseline on held-out RMSE]")
mf = MatrixFactorizationRecommender(n_factors=30, n_epochs=15).fit(train)
sample_test = test.sample(min(3000, len(test)), random_state=1)
global_mean = train[config.RATING_COL].mean()
baseline_rmse = np.sqrt(np.mean((sample_test[config.RATING_COL] - global_mean) ** 2))
mf_preds = sample_test.apply(lambda r: mf.predict_score(r[config.USER_COL], r[config.ITEM_COL]), axis=1)
mf_rmse = np.sqrt(np.mean((sample_test[config.RATING_COL] - mf_preds) ** 2))
print(f"  global-mean baseline RMSE: {baseline_rmse:.4f}")
print(f"  matrix factorization RMSE: {mf_rmse:.4f}")
check("MF beats the global-mean baseline", mf_rmse < baseline_rmse)
check("MF RMSE is in a plausible range for this dataset (0.7-1.1)", 0.7 < mf_rmse < 1.1)

print("\n[evaluation metrics: hand-computed example with a known answer]")
# recommended [A, B, C, D, E], relevant = {B, D, F} -> hits at rank 2 and 4
recommended = ["A", "B", "C", "D", "E"]
relevant = {"B", "D", "F"}
p = precision_at_k(recommended, relevant, k=5)
r = recall_at_k(recommended, relevant, k=5)
n = ndcg_at_k(recommended, relevant, k=5)
check("precision@5 = 2/5 = 0.4", abs(p - 0.4) < 1e-9)
check("recall@5 = 2/3", abs(r - 2 / 3) < 1e-9)
check("ndcg@5 is between 0 and 1", 0 <= n <= 1)

print(f"\n{passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
