"""Most popular and highest-average-rating baselines."""

import pandas as pd

from . import config
from .data_loading import get_seen_items


class MostPopularRecommender:
    def __init__(self):
        self.ranking_ = None

    def fit(self, ratings, items=None):
        counts = ratings[config.ITEM_COL].value_counts()
        self.ranking_ = list(zip(counts.index, counts.values))
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        candidates = self.ranking_
        if exclude_seen:
            seen = get_seen_items(ratings_train, user_id)
            candidates = [(i, s) for i, s in candidates if i not in seen]
        return candidates[:n]


class HighestAverageRatingRecommender:
    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranking_ = None

    def fit(self, ratings, items=None):
        stats = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(["mean", "count"])
        stats = stats[stats["count"] >= self.min_ratings]
        stats = stats.sort_values("mean", ascending=False)
        self.ranking_ = list(zip(stats.index, stats["mean"]))
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        candidates = self.ranking_
        if exclude_seen:
            seen = get_seen_items(ratings_train, user_id)
            candidates = [(i, s) for i, s in candidates if i not in seen]
        return candidates[:n]


class RandomRecommender:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        self.items_ = ratings[config.ITEM_COL].unique().tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        shuffled = pd.Series(self.items_).sample(
            n=len(self.items_), random_state=self.random_state
        ).tolist()
        if exclude_seen:
            seen = get_seen_items(ratings_train, user_id)
            shuffled = [i for i in shuffled if i not in seen]
        return [(i, None) for i in shuffled[:n]]
