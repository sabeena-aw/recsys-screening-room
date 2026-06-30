"""Content-based recommender: TF-IDF over genres + tags, cosine similarity to a user profile."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loading import get_seen_items


def _build_item_text(items, tags=None):
    text = items[config.GENRES_COL].fillna("").str.replace("|", " ", regex=False)

    if tags is not None:
        tag_text = (
            tags.groupby(config.ITEM_COL)["tag"]
            .apply(lambda s: " ".join(str(t) for t in s))
        )
        text = text + " " + items[config.ITEM_COL].map(tag_text).fillna("")

    return text


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = None
        self.item_features_ = None
        self.item_ids_ = None
        self.item_id_to_index_ = None

    def fit(self, items, tags=None):
        item_text = _build_item_text(items, tags)
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s|]+")
        self.item_features_ = self.vectorizer.fit_transform(item_text)
        self.item_ids_ = items[config.ITEM_COL].tolist()
        self.item_id_to_index_ = {iid: idx for idx, iid in enumerate(self.item_ids_)}
        return self

    def build_user_profile(self, user_id, ratings_train):
        user_ratings = ratings_train[ratings_train[config.USER_COL] == user_id]
        if user_ratings.empty:
            return None

        mean_rating = user_ratings[config.RATING_COL].mean()
        profile = None
        for item_id, rating in zip(user_ratings[config.ITEM_COL], user_ratings[config.RATING_COL]):
            idx = self.item_id_to_index_.get(item_id)
            if idx is None:
                continue
            weight = rating - mean_rating
            vec = self.item_features_[idx] * weight
            profile = vec if profile is None else profile + vec

        return profile

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        profile = self.build_user_profile(user_id, ratings_train)
        if profile is None:
            return []

        scores = cosine_similarity(profile, self.item_features_).ravel()
        ranked_idx = np.argsort(-scores)

        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        results = []
        for idx in ranked_idx:
            item_id = self.item_ids_[idx]
            if item_id in seen:
                continue
            results.append((item_id, scores[idx]))
            if len(results) == n:
                break
        return results

    def similar_items(self, item_id, n=10):
        idx = self.item_id_to_index_.get(item_id)
        if idx is None:
            return []
        scores = cosine_similarity(self.item_features_[idx], self.item_features_).ravel()
        ranked_idx = np.argsort(-scores)
        results = [(self.item_ids_[i], scores[i]) for i in ranked_idx if i != idx]
        return results[:n]
