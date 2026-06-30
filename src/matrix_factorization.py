"""Matrix factorization via SGD: r_hat = mu + b_u + b_i + p_u . q_i"""

import numpy as np

from . import config


class MatrixFactorizationRecommender:
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state

        self.mu_ = None
        self.b_u_ = None
        self.b_i_ = None
        self.P_ = None
        self.Q_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None
        self.all_items_ = None

    def fit(self, ratings):
        rng = np.random.default_rng(self.random_state)

        user_ids = ratings[config.USER_COL].unique()
        item_ids = ratings[config.ITEM_COL].unique()
        self.user_id_to_index_ = {u: i for i, u in enumerate(user_ids)}
        self.item_id_to_index_ = {m: i for i, m in enumerate(item_ids)}
        self.all_items_ = item_ids

        n_users = len(user_ids)
        n_items = len(item_ids)

        u_idx = ratings[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        i_idx = ratings[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        r = ratings[config.RATING_COL].to_numpy()

        self.mu_ = r.mean()
        self.b_u_ = np.zeros(n_users)
        self.b_i_ = np.zeros(n_items)
        self.P_ = rng.normal(scale=0.1, size=(n_users, self.n_factors))
        self.Q_ = rng.normal(scale=0.1, size=(n_items, self.n_factors))

        n = len(r)
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            for idx in order:
                u, i, rui = u_idx[idx], i_idx[idx], r[idx]
                pred = self.mu_ + self.b_u_[u] + self.b_i_[i] + self.P_[u] @ self.Q_[i]
                err = rui - pred

                self.b_u_[u] += self.lr * (err - self.reg * self.b_u_[u])
                self.b_i_[i] += self.lr * (err - self.reg * self.b_i_[i])

                p_u = self.P_[u].copy()
                self.P_[u] += self.lr * (err * self.Q_[i] - self.reg * p_u)
                self.Q_[i] += self.lr * (err * p_u - self.reg * self.Q_[i])

        return self

    def predict_score(self, user_id, item_id):
        u = self.user_id_to_index_.get(user_id)
        i = self.item_id_to_index_.get(item_id)
        if u is None and i is None:
            return self.mu_
        if u is None:
            return self.mu_ + self.b_i_[i]
        if i is None:
            return self.mu_ + self.b_u_[u]
        return self.mu_ + self.b_u_[u] + self.b_i_[i] + self.P_[u] @ self.Q_[i]

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        u = self.user_id_to_index_.get(user_id)
        if u is None:
            return []

        scores = self.mu_ + self.b_u_[u] + self.b_i_ + self.Q_ @ self.P_[u]
        ranked = np.argsort(-scores)

        seen = set()
        if exclude_seen:
            seen = set(ratings_train.loc[
                ratings_train[config.USER_COL] == user_id, config.ITEM_COL
            ])

        results = []
        for idx in ranked:
            item_id = self.all_items_[idx]
            if item_id in seen:
                continue
            results.append((item_id, float(scores[idx])))
            if len(results) == n:
                break
        return results
