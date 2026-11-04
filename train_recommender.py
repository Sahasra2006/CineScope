import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_DATA_PATH = os.path.join("data", "cleaned_movies.csv")
MODELS_DIR = "models"
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SIM_PATH = os.path.join(MODELS_DIR, "cosine_sim.pkl")
META_PATH = os.path.join(MODELS_DIR, "movies_meta.pkl")

def create_content_soup(row):
    import json
    try:
        genres = json.loads(row["genres_clean"])
        genre_str = " ".join(genres * 2)
    except Exception:
        genre_str = ""
    overview = str(row["overview"]) if pd.notna(row["overview"]) else ""
    return f"{genre_str} {overview}".strip()

def build_recommender(csv_path=CLEAN_DATA_PATH):
    print("[*] Training content-based recommender engine...")
    df = pd.read_csv(csv_path)
    df["soup"] = df.apply(create_content_soup, axis=1)

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    print(f"[+] TF-IDF Matrix shape: {tfidf_matrix.shape}")

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tfidf, VEC_PATH)
    joblib.dump(cosine_sim, SIM_PATH)

    meta_df = df[["id", "title"]].reset_index(drop=True)
    joblib.dump(meta_df, META_PATH)

    print(f"[+] Model artifacts saved to '{MODELS_DIR}/'")
    return meta_df, cosine_sim

def evaluate_recs(movie_title, meta_df, cosine_sim, top_k=5):
    matches = meta_df[meta_df["title"].str.lower() == movie_title.lower()]
    if matches.empty:
        print(f"[!] No match found for: '{movie_title}'")
        return []

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, score in sim_scores if i != idx][:top_k]

    print(f"\n=======================================================")
    print(f"Top {top_k} Content Recommendations for: '{meta_df.iloc[idx]['title']}'")
    print(f"=======================================================")
    recs = []
    for rank, i in enumerate(top_indices, start=1):
        rec_title = meta_df.iloc[i]["title"]
        rec_id = int(meta_df.iloc[i]["id"])
        score = [s for j, s in sim_scores if j == i][0]
        recs.append({"rank": rank, "id": rec_id, "title": rec_title, "score": round(float(score), 4)})
        print(f" {rank}. {rec_title} (ID: {rec_id}) - Similarity Score: {score:.4f}")
    return recs

if __name__ == "__main__":
    meta_df, cosine_sim = build_recommender()
    evaluate_recs("Avatar", meta_df, cosine_sim)
    evaluate_recs("The Dark Knight Rises", meta_df, cosine_sim)
