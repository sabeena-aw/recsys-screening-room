"""Runs the full pipeline: load data, train every model, evaluate, show examples."""

import random

import pandas as pd

from src import config
from src.data_loading import load_ratings, load_items, load_tags, train_test_split_ratings, describe_dataset
from src.baselines import MostPopularRecommender, HighestAverageRatingRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model


def build_models(train, items, tags):
    models = {
        "most_popular": MostPopularRecommender().fit(train),
        "highest_average": HighestAverageRatingRecommender(min_ratings=20).fit(train),
        "content_based": ContentBasedRecommender().fit(items, tags),
        "item_item_cf": ItemItemCollaborativeFiltering(k=20).fit(train),
        "user_user_cf": UserUserCollaborativeFiltering(k=20).fit(train),
        "matrix_factorization": MatrixFactorizationRecommender(n_factors=30, n_epochs=15).fit(train),
    }
    return models


def print_examples(models, train, items, user_ids, n=5):
    titles = items.set_index(config.ITEM_COL)[config.TITLE_COL]
    for user_id in user_ids:
        print(f"\n--- recommendations for user {user_id} ---")
        for name, model in models.items():
            recs = model.recommend(user_id, train, n=n, exclude_seen=True)
            rec_titles = [titles.get(item_id, f"item {item_id}") for item_id, _ in recs]
            print(f"{name:>20}: {rec_titles}")


def main():
    ratings = load_ratings()
    items = load_items()
    tags = load_tags()

    describe_dataset(ratings, items)

    train, test = train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE)
    print(f"\ntrain: {len(train)} ratings, test: {len(test)} ratings")

    models = build_models(train, items, tags)

    # evaluate on a sample of users -- running all 610 through user-user CF is slow
    rng = random.Random(config.RANDOM_STATE)
    eval_users = rng.sample(list(train[config.USER_COL].unique()), 50)
    all_items = items[config.ITEM_COL].tolist()
    item_popularity = train[config.ITEM_COL].value_counts().to_dict()
    n_users_total = train[config.USER_COL].nunique()

    rows = []
    for name, model in models.items():
        metrics = evaluate_model(
            model, train, test, eval_users, k=config.TOP_K,
            all_items=all_items, item_popularity=item_popularity, n_users=n_users_total,
        )
        metrics["model"] = name
        rows.append(metrics)
        print(f"{name:>20}: {metrics}")

    results_df = pd.DataFrame(rows).set_index("model")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(config.RESULTS_DIR / "metrics.csv")
    print(f"\nsaved metrics to {config.RESULTS_DIR / 'metrics.csv'}")

    example_users = rng.sample(eval_users, 3)
    print_examples(models, train, items, example_users)


if __name__ == "__main__":
    main()
