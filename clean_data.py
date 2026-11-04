import os
import ast
import json
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "tmdb_5000_movies.csv")
CLEAN_DATA_PATH = os.path.join("data", "cleaned_movies.csv")

def parse_genres(genres_raw):
    if pd.isna(genres_raw) or not genres_raw:
        return ["Unknown"]
    try:
        parsed = ast.literal_eval(genres_raw) if isinstance(genres_raw, str) else genres_raw
        if isinstance(parsed, list):
            genre_names = [item["name"] for item in parsed if isinstance(item, dict) and "name" in item]
            return genre_names if genre_names else ["Unknown"]
    except Exception:
        pass
    return ["Unknown"]

def clean_tmdb_movies(input_path=RAW_DATA_PATH, output_path=CLEAN_DATA_PATH):
    print(f"[*] Reading raw data from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    df = pd.read_csv(input_path)
    print(f"[*] Initial records loaded: {len(df)}")

    # 1. Deduplicate by movie ID
    df = df.drop_duplicates(subset=["id"], keep="first")

    # 2. Select relevant columns
    cols = ["id", "title", "genres", "overview", "release_date", "vote_average", "vote_count", "popularity"]
    df = df[cols].copy()

    # 3. Clean and parse genres
    df["genres_list"] = df["genres"].apply(parse_genres)
    df["genres_clean"] = df["genres_list"].apply(json.dumps)

    # 4. Handle missing values
    df["title"] = df["title"].fillna("Untitled").str.strip()
    df["overview"] = df["overview"].fillna("No overview available.").str.strip()
    df["release_date"] = df["release_date"].fillna("N/A")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["id", "title"])
    df["id"] = df["id"].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Cleaned dataset written to: {output_path} ({len(df)} records)")

if __name__ == "__main__":
    clean_tmdb_movies()
