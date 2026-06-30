"""Generates the EDA figures for the report/slides. Run after main.py."""

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.data_loading import load_ratings, load_items

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#888888"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

FIG_DIR = config.RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ACCENT = "#2451FF"
INK = "#1B2420"


def rating_distribution(ratings):
    counts = ratings[config.RATING_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color=ACCENT, width=0.7)
    ax.set_title("Rating distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rating_distribution.png", dpi=150)
    plt.close(fig)


def long_tail(ratings, group_col, title, filename):
    counts = ratings[group_col].value_counts().sort_values(ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(counts.index + 1, counts.values, color=ACCENT, linewidth=2)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Number of ratings (log scale)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def genre_counts(items):
    exploded = items[config.GENRES_COL].str.split("|").explode()
    exploded = exploded[exploded != "(no genres listed)"]
    counts = exploded.value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(counts.index, counts.values, color=ACCENT)
    ax.set_title("Movies per genre (top 12)")
    ax.set_xlabel("Number of movies")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "genre_counts.png", dpi=150)
    plt.close(fig)


def ratings_over_time(ratings):
    dates = pd.to_datetime(ratings[config.TIMESTAMP_COL], unit="s")
    by_year = dates.dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_year.index.astype(str), by_year.values, color=ACCENT)
    ax.set_title("Ratings collected per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of ratings")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ratings_over_time.png", dpi=150)
    plt.close(fig)


def main():
    ratings = load_ratings()
    items = load_items()

    rating_distribution(ratings)
    long_tail(ratings, config.USER_COL, "Ratings per user (long tail)", "ratings_per_user.png")
    long_tail(ratings, config.ITEM_COL, "Ratings per movie (long tail)", "ratings_per_movie.png")
    genre_counts(items)
    ratings_over_time(ratings)

    print(f"saved 5 figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
