"""Evaluation metrics. Relevance for ranking metrics = test rating >= RELEVANCE_THRESHOLD."""

import numpy as np

from . import config

RELEVANCE_THRESHOLD = 4.0


def precision_at_k(recommended_items, relevant_items, k=10):
    top_k = recommended_items[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant_items)
    return hits / len(top_k)


def recall_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return 0.0
    top_k = recommended_items[:k]
    hits = sum(1 for i in top_k if i in relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    top_k = recommended_items[:k]
    return 1.0 if any(i in relevant_items for i in top_k) else 0.0


def dcg_at_k(relevance_scores, k=10):
    relevance_scores = relevance_scores[:k]
    return sum(
        rel / np.log2(rank + 2) for rank, rel in enumerate(relevance_scores)
    )


def ndcg_at_k(recommended_items, relevant_items, k=10):
    top_k = recommended_items[:k]
    rel = [1 if i in relevant_items else 0 for i in top_k]
    dcg = dcg_at_k(rel, k)

    ideal_rel = sorted(rel, reverse=True)
    idcg = dcg_at_k(ideal_rel, k)

    return dcg / idcg if idcg > 0 else 0.0


def mean_reciprocal_rank(recommended_items, relevant_items, k=10):
    for rank, item in enumerate(recommended_items[:k], start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def catalog_coverage(all_recommendations, all_items):
    all_items = set(all_items)
    if not all_items:
        return 0.0
    recommended = set(all_recommendations)
    return len(recommended & all_items) / len(all_items)


def novelty_at_k(recommended_items, item_popularity, n_users, k=10):
    """Mean self-information of the top-k list: -log2(p(item)).
    Popular items score low, long-tail items score high. Lets us see
    whether a model with decent precision is just recommending the same
    blockbusters to everyone, or actually personalizing.
    """
    top_k = recommended_items[:k]
    if not top_k or n_users == 0:
        return 0.0
    scores = []
    for item_id in top_k:
        p = item_popularity.get(item_id, 0) / n_users
        p = max(p, 1e-12)
        scores.append(-np.log2(p))
    return float(np.mean(scores))


def evaluate_model(model, ratings_train, ratings_test, users, k=10,
                    all_items=None, item_popularity=None, n_users=None):
    precisions, recalls, ndcgs, hit_rates, novelties = [], [], [], [], []
    all_recs = []

    for user_id in users:
        relevant = set(
            ratings_test.loc[
                (ratings_test[config.USER_COL] == user_id)
                & (ratings_test[config.RATING_COL] >= RELEVANCE_THRESHOLD),
                config.ITEM_COL,
            ]
        )
        if not relevant:
            continue

        recs = model.recommend(user_id, ratings_train, n=k, exclude_seen=True)
        recommended_items = [item_id for item_id, _ in recs]
        all_recs.extend(recommended_items)

        precisions.append(precision_at_k(recommended_items, relevant, k))
        recalls.append(recall_at_k(recommended_items, relevant, k))
        ndcgs.append(ndcg_at_k(recommended_items, relevant, k))
        hit_rates.append(hit_rate_at_k(recommended_items, relevant, k))
        if item_popularity is not None and n_users is not None:
            novelties.append(novelty_at_k(recommended_items, item_popularity, n_users, k))

    results = {
        "precision_at_k": np.mean(precisions) if precisions else 0.0,
        "recall_at_k": np.mean(recalls) if recalls else 0.0,
        "ndcg_at_k": np.mean(ndcgs) if ndcgs else 0.0,
        "hit_rate_at_k": np.mean(hit_rates) if hit_rates else 0.0,
        "n_users_evaluated": len(precisions),
    }

    if all_items is not None:
        results["catalog_coverage"] = catalog_coverage(all_recs, all_items)
    if novelties:
        results["novelty_at_k"] = np.mean(novelties)

    return results
