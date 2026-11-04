import os
import math
from datetime import datetime
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient, DESCENDING, ASCENDING

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "cinescope_secret_key_12345")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["cinescope"]

CURRENT_USER_ID = "demo_user_01"

MODELS_DIR = "models"
VEC_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SIM_PATH = os.path.join(MODELS_DIR, "cosine_sim.pkl")
META_PATH = os.path.join(MODELS_DIR, "movies_meta.pkl")

meta_df = None
cosine_sim = None
if os.path.exists(SIM_PATH) and os.path.exists(META_PATH):
    try:
        cosine_sim = joblib.load(SIM_PATH)
        meta_df = joblib.load(META_PATH)
        print("[+] Recommender models loaded into Flask runtime.")
    except Exception as e:
        print(f"[!] Warning: Could not load recommender models: {e}")

def get_next_movie_id():
    highest = db.movies.find_one(sort=[("movie_id", DESCENDING)])
    return (highest["movie_id"] + 1) if highest and "movie_id" in highest else 1

def fetch_recommendations(movie_id, top_n=6):
    if cosine_sim is None or meta_df is None:
        return []
    
    matches = meta_df[meta_df["id"] == movie_id]
    if matches.empty:
        return []
    
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i for i, s in sim_scores if i != idx][:top_n]
    rec_ids = [int(meta_df.iloc[i]["id"]) for i in top_indices]
    
    rec_movies = list(db.movies.find({"movie_id": {"$in": rec_ids}}))
    order_map = {mid: rank for rank, mid in enumerate(rec_ids)}
    rec_movies.sort(key=lambda m: order_map.get(m["movie_id"], 999))
    return rec_movies

@app.route("/")
def index():
    featured = list(db.movies.find().sort("popularity", DESCENDING).limit(8))
    top_voted = list(db.movies.find({"vote_count": {"$gte": 500}}).sort("vote_average", DESCENDING).limit(4))
    return render_template("index.html", featured=featured, top_voted=top_voted)

@app.route("/catalog")
def catalog():
    page = int(request.args.get("page", 1))
    per_page = 12
    search_query = request.args.get("q", "").strip()
    genre_filter = request.args.get("genre", "").strip()

    filter_query = {}
    if search_query:
        filter_query["$or"] = [
            {"title": {"$regex": search_query, "$options": "i"}},
            {"overview": {"$regex": search_query, "$options": "i"}}
        ]
    if genre_filter:
        filter_query["genres"] = genre_filter

    total_movies = db.movies.count_documents(filter_query)
    total_pages = max(1, math.ceil(total_movies / per_page))
    page = max(1, min(page, total_pages))

    movies = list(db.movies.find(filter_query)
                  .sort("vote_count", DESCENDING)
                  .skip((page - 1) * per_page)
                  .limit(per_page))

    all_genres = sorted(db.movies.distinct("genres"))

    return render_template(
        "catalog.html",
        movies=movies,
        page=page,
        total_pages=total_pages,
        total_movies=total_movies,
        search_query=search_query,
        genre_filter=genre_filter,
        genres=all_genres
    )

@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    movie = db.movies.find_one({"movie_id": movie_id})
    if not movie:
        flash("Movie not found.", "danger")
        return redirect(url_for("catalog"))

    user_rating = db.ratings.find_one({"user_id": CURRENT_USER_ID, "movie_id": movie_id})
    in_watchlist = db.watchlist.find_one({"user_id": CURRENT_USER_ID, "movie_id": movie_id})
    recommendations = fetch_recommendations(movie_id, top_n=6)

    return render_template(
        "movie_detail.html",
        movie=movie,
        user_rating=user_rating["rating"] if user_rating else None,
        watchlist_item=in_watchlist,
        recommendations=recommendations
    )

@app.route("/movie/new", methods=["GET", "POST"])
def movie_create():
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        genres_raw = request.form.get("genres", "").strip()
        overview = request.form.get("overview", "").strip()
        release_date = request.form.get("release_date", "").strip()

        if not title:
            flash("Movie title is required.", "danger")
            return render_template("movie_form.html", action="Create", movie={})

        genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
        if not genres:
            flash("At least one genre is required.", "danger")
            return render_template("movie_form.html", action="Create", movie={})

        new_movie = {
            "movie_id": get_next_movie_id(),
            "title": title,
            "genres": genres,
            "overview": overview or "No overview provided.",
            "release_date": release_date or "N/A",
            "vote_average": 0.0,
            "vote_count": 0,
            "popularity": 1.0
        }

        try:
            db.movies.insert_one(new_movie)
            flash(f"Movie '{title}' added successfully!", "success")
            return redirect(url_for("movie_detail", movie_id=new_movie["movie_id"]))
        except Exception as e:
            flash(f"Schema Validation Error: {e}", "danger")
            return render_template("movie_form.html", action="Create", movie=request.form)

    return render_template("movie_form.html", action="Create", movie={})

@app.route("/movie/<int:movie_id>/edit", methods=["GET", "POST"])
def movie_edit(movie_id):
    movie = db.movies.find_one({"movie_id": movie_id})
    if not movie:
        flash("Movie not found.", "danger")
        return redirect(url_for("catalog"))

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        genres_raw = request.form.get("genres", "").strip()
        overview = request.form.get("overview", "").strip()
        release_date = request.form.get("release_date", "").strip()

        if not title:
            flash("Title cannot be empty.", "danger")
            return render_template("movie_form.html", action="Edit", movie=movie)

        genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
        if not genres:
            flash("Provide at least one genre.", "danger")
            return render_template("movie_form.html", action="Edit", movie=movie)

        update_fields = {
            "title": title,
            "genres": genres,
            "overview": overview,
            "release_date": release_date
        }

        try:
            db.movies.update_one({"movie_id": movie_id}, {"$set": update_fields})
            flash("Movie updated successfully.", "success")
            return redirect(url_for("movie_detail", movie_id=movie_id))
        except Exception as e:
            flash(f"Update failed: {e}", "danger")

    movie["genres_display"] = ", ".join(movie.get("genres", []))
    return render_template("movie_form.html", action="Edit", movie=movie)

@app.route("/movie/<int:movie_id>/delete", methods=["POST"])
def movie_delete(movie_id):
    db.movies.delete_one({"movie_id": movie_id})
    db.ratings.delete_many({"movie_id": movie_id})
    db.watchlist.delete_many({"movie_id": movie_id})
    flash("Movie and its associated records deleted.", "info")
    return redirect(url_for("catalog"))

@app.route("/movie/<int:movie_id>/rate", methods=["POST"])
def rate_movie(movie_id):
    rating_val = request.form.get("rating")
    try:
        rating_num = float(rating_val)
        if not (0.5 <= rating_num <= 5.0):
            flash("Rating must be between 0.5 and 5.0.", "warning")
            return redirect(url_for("movie_detail", movie_id=movie_id))

        db.ratings.update_one(
            {"user_id": CURRENT_USER_ID, "movie_id": movie_id},
            {"$set": {"rating": rating_num, "created_at": datetime.utcnow()}},
            upsert=True
        )
        flash(f"Rated movie {rating_num} / 5.0!", "success")
    except ValueError:
        flash("Invalid rating format.", "danger")
    except Exception as e:
        flash(f"Database error: {e}", "danger")

    return redirect(url_for("movie_detail", movie_id=movie_id))

@app.route("/movie/<int:movie_id>/rate/delete", methods=["POST"])
def delete_rating(movie_id):
    db.ratings.delete_one({"user_id": CURRENT_USER_ID, "movie_id": movie_id})
    flash("Rating removed.", "info")
    return redirect(url_for("movie_detail", movie_id=movie_id))

@app.route("/ratings")
def my_ratings():
    user_ratings = list(db.ratings.find({"user_id": CURRENT_USER_ID}))
    movie_ids = [r["movie_id"] for r in user_ratings]
    movies = {m["movie_id"]: m for m in db.movies.find({"movie_id": {"$in": movie_ids}})}

    items = []
    for r in user_ratings:
        if r["movie_id"] in movies:
            items.append({
                "movie": movies[r["movie_id"]],
                "rating": r["rating"],
                "created_at": r.get("created_at")
            })

    return render_template("ratings.html", items=items)

@app.route("/movie/<int:movie_id>/watchlist", methods=["POST"])
def toggle_watchlist(movie_id):
    status = request.form.get("status", "plan_to_watch")
    if status not in ["plan_to_watch", "watched"]:
        flash("Invalid watchlist status.", "danger")
        return redirect(url_for("movie_detail", movie_id=movie_id))

    existing = db.watchlist.find_one({"user_id": CURRENT_USER_ID, "movie_id": movie_id})
    if existing:
        db.watchlist.delete_one({"user_id": CURRENT_USER_ID, "movie_id": movie_id})
        flash("Removed from Watchlist.", "info")
    else:
        try:
            db.watchlist.insert_one({
                "user_id": CURRENT_USER_ID,
                "movie_id": movie_id,
                "status": status,
                "added_at": datetime.utcnow()
            })
            flash("Added to Watchlist!", "success")
        except Exception as e:
            flash(f"Failed to add to watchlist: {e}", "danger")

    return redirect(url_for("movie_detail", movie_id=movie_id))

@app.route("/watchlist")
def my_watchlist():
    watch_entries = list(db.watchlist.find({"user_id": CURRENT_USER_ID}))
    movie_ids = [w["movie_id"] for w in watch_entries]
    movies = {m["movie_id"]: m for m in db.movies.find({"movie_id": {"$in": movie_ids}})}

    items = []
    for w in watch_entries:
        if w["movie_id"] in movies:
            items.append({
                "movie": movies[w["movie_id"]],
                "status": w["status"],
                "added_at": w.get("added_at")
            })

    return render_template("watchlist.html", items=items)

@app.route("/analytics")
def analytics():
    top_rated_pipeline = [
        {"$match": {"vote_count": {"$gte": 300}}},
        {"$sort": {"vote_average": -1, "vote_count": -1}},
        {"$limit": 10},
        {"$project": {"title": 1, "vote_average": 1, "vote_count": 1, "genres": 1}}
    ]
    top_rated = list(db.movies.aggregate(top_rated_pipeline))

    most_rated_pipeline = [
        {"$sort": {"vote_count": -1}},
        {"$limit": 10},
        {"$project": {"title": 1, "vote_count": 1, "vote_average": 1}}
    ]
    most_rated = list(db.movies.aggregate(most_rated_pipeline))

    genre_count_pipeline = [
        {"$unwind": "$genres"},
        {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    genre_counts = list(db.movies.aggregate(genre_count_pipeline))

    genre_top_pipeline = [
        {"$match": {"vote_count": {"$gte": 100}}},
        {"$unwind": "$genres"},
        {"$sort": {"vote_average": -1, "vote_count": -1}},
        {
            "$group": {
                "_id": "$genres",
                "top_movie": {"$first": "$title"},
                "avg_rating": {"$first": "$vote_average"},
                "votes": {"$first": "$vote_count"}
            }
        },
        {"$sort": {"avg_rating": -1}}
    ]
    top_by_genre = list(db.movies.aggregate(genre_top_pipeline))

    return render_template(
        "analytics.html",
        top_rated=top_rated,
        most_rated=most_rated,
        genre_counts=genre_counts,
        top_by_genre=top_by_genre
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
