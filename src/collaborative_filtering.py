"""Collaborative filtering: item-item and user-user, both cosine similarity based."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loading import get_seen_items


def _build_matrix(ratings):
    user_ids = ratings[config.USER_COL].unique()
    item_ids = ratings[config.ITEM_COL].unique()
    user_id_to_index = {u: i for i, u in enumerate(user_ids)}
    item_id_to_index = {m: i for i, m in enumerate(item_ids)}

    rows = ratings[config.USER_COL].map(user_id_to_index)
    cols = ratings[config.ITEM_COL].map(item_id_to_index)
    matrix = csr_matrix(
        (ratings[config.RATING_COL], (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    return matrix, user_ids, item_ids, user_id_to_index, item_id_to_index


class ItemItemCollaborativeFiltering:
    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix_ = None
        self.item_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        (self.user_item_matrix_, self.user_ids_, self.item_ids_,
         self.user_id_to_index_, self.item_id_to_index_) = _build_matrix(ratings)
        self.item_similarity_ = cosine_similarity(self.user_item_matrix_.T)
        return self

    def predict_score(self, user_id, item_id):
        u_idx = self.user_id_to_index_.get(user_id)
        i_idx = self.item_id_to_index_.get(item_id)
        if u_idx is None or i_idx is None:
            return 0.0

        user_row = self.user_item_matrix_[u_idx].toarray().ravel()
        rated_idx = np.nonzero(user_row)[0]
        if len(rated_idx) == 0:
            return 0.0

        sims = self.item_similarity_[i_idx, rated_idx]
        top_k = np.argsort(-sims)[: self.k]
        sims_topk = sims[top_k]
        ratings_topk = user_row[rated_idx[top_k]]

        denom = np.abs(sims_topk).sum()
        if denom == 0:
            return 0.0
        return float((sims_topk * ratings_topk).sum() / denom)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        u_idx = self.user_id_to_index_.get(user_id)
        if u_idx is None:
            return []

        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        scores = []
        for item_id in self.item_ids_:
            if item_id in seen:
                continue
            scores.append((item_id, self.predict_score(user_id, item_id)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


class UserUserCollaborativeFiltering:
    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix_ = None
        self.user_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        (self.user_item_matrix_, self.user_ids_, self.item_ids_,
         self.user_id_to_index_, self.item_id_to_index_) = _build_matrix(ratings)
        self.user_similarity_ = cosine_similarity(self.user_item_matrix_)
        return self

    def predict_score(self, user_id, item_id):
        u_idx = self.user_id_to_index_.get(user_id)
        i_idx = self.item_id_to_index_.get(item_id)
        if u_idx is None or i_idx is None:
            return 0.0

        item_col = self.user_item_matrix_[:, i_idx].toarray().ravel()
        rated_idx = np.nonzero(item_col)[0]
        if len(rated_idx) == 0:
            return 0.0

        sims = self.user_similarity_[u_idx, rated_idx]
        top_k = np.argsort(-sims)[: self.k]
        sims_topk = sims[top_k]
        ratings_topk = item_col[rated_idx[top_k]]

        denom = np.abs(sims_topk).sum()
        if denom == 0:
            return 0.0
        return float((sims_topk * ratings_topk).sum() / denom)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        u_idx = self.user_id_to_index_.get(user_id)
        if u_idx is None:
            return []

        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        scores = []
        for item_id in self.item_ids_:
            if item_id in seen:
                continue
            scores.append((item_id, self.predict_score(user_id, item_id)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
