"""Loading and basic exploration of the MovieLens data."""

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_ratings(path=config.RATINGS_PATH):
    df = pd.read_csv(path)
    needed = {config.USER_COL, config.ITEM_COL, config.RATING_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"ratings file is missing columns: {missing}")
    return df


def load_items(path=config.ITEMS_PATH):
    df = pd.read_csv(path)
    needed = {config.ITEM_COL, config.TITLE_COL, config.GENRES_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"items file is missing columns: {missing}")
    return df


def load_tags(path=config.TAGS_PATH):
    return pd.read_csv(path)


def describe_dataset(ratings, items=None):
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_items)

    print(f"users:        {n_users}")
    print(f"items rated:  {n_items}")
    print(f"ratings:      {n_ratings}")
    print(f"sparsity:     {sparsity:.4%}")
    print()
    print("rating distribution:")
    print(ratings[config.RATING_COL].value_counts().sort_index())
    print()

    most_active = (
        ratings[config.USER_COL].value_counts().head(5)
    )
    print("most active users (by # ratings):")
    print(most_active)
    print()

    rating_counts = ratings[config.ITEM_COL].value_counts().head(5)
    if items is not None:
        titles = items.set_index(config.ITEM_COL)[config.TITLE_COL]
        print("most rated items:")
        for item_id, count in rating_counts.items():
            title = titles.get(item_id, "unknown")
            print(f"  {title}: {count} ratings")
    else:
        print("most rated item ids:")
        print(rating_counts)

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
    }


def train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE):
    # random split, not temporal -- not enough per-item history here for that to be meaningful
    train, test = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_seen_items(ratings, user_id):
    return set(ratings.loc[ratings[config.USER_COL] == user_id, config.ITEM_COL])
