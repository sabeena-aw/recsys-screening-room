"""Paths and constants used across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ml-latest-small file names
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"
ITEMS_PATH = RAW_DATA_DIR / "movies.csv"
TAGS_PATH = RAW_DATA_DIR / "tags.csv"

USER_COL = "userId"
ITEM_COL = "movieId"
RATING_COL = "rating"
TIMESTAMP_COL = "timestamp"
TITLE_COL = "title"
GENRES_COL = "genres"

TOP_K = 10
RANDOM_STATE = 42
