import os
import re
import math
import urllib.parse
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
    try:
        page = int(request.args.get("page", 1))
    except (ValueError, TypeError):
        page = 1
    per_page = 12
    search_query = request.args.get("q", "").strip()
    genre_filter = request.args.get("genre", "").strip()

    filter_query = {}
    if search_query:
        escaped_query = re.escape(search_query)
        filter_query["$or"] = [
            {"title": {"$regex": escaped_query, "$options": "i"}},
            {"overview": {"$regex": escaped_query, "$options": "i"}}
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

        movie_data = {
            "title": title,
            "genres_display": genres_raw,
            "overview": overview,
            "release_date": release_date
        }

        if not title:
            flash("Movie title is required.", "danger")
            return render_template("movie_form.html", action="Create", movie=movie_data)

        genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
        if not genres:
            flash("At least one genre is required.", "danger")
            return render_template("movie_form.html", action="Create", movie=movie_data)

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
            return render_template("movie_form.html", action="Create", movie=movie_data)

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

        form_movie = dict(movie)
        form_movie.update({
            "title": title,
            "genres_display": genres_raw,
            "overview": overview,
            "release_date": release_date
        })

        if not title:
            flash("Title cannot be empty.", "danger")
            return render_template("movie_form.html", action="Edit", movie=form_movie)

        genres = [g.strip() for g in genres_raw.split(",") if g.strip()]
        if not genres:
            flash("Provide at least one genre.", "danger")
            return render_template("movie_form.html", action="Edit", movie=form_movie)

        update_fields = {
            "title": title,
            "genres": genres,
            "overview": overview or "No overview provided.",
            "release_date": release_date or "N/A"
        }

        try:
            db.movies.update_one({"movie_id": movie_id}, {"$set": update_fields})
            flash("Movie updated successfully.", "success")
            return redirect(url_for("movie_detail", movie_id=movie_id))
        except Exception as e:
            flash(f"Update failed: {e}", "danger")
            return render_template("movie_form.html", action="Edit", movie=form_movie)

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
    movie = db.movies.find_one({"movie_id": movie_id})
    if not movie:
        flash("Cannot rate a non-existent movie.", "danger")
        return redirect(url_for("catalog"))

    rating_val = request.form.get("rating")
    try:
        rating_num = float(rating_val)
        if not (0.5 <= rating_num <= 5.0):
            flash("Rating must be between 0.5 and 5.0.", "warning")
            return redirect(request.referrer or url_for("movie_detail", movie_id=movie_id))

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

    return redirect(request.referrer or url_for("movie_detail", movie_id=movie_id))

@app.route("/movie/<int:movie_id>/rate/delete", methods=["POST"])
def delete_rating(movie_id):
    db.ratings.delete_one({"user_id": CURRENT_USER_ID, "movie_id": movie_id})
    flash("Rating removed.", "info")
    return redirect(request.referrer or url_for("movie_detail", movie_id=movie_id))

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
        return redirect(request.referrer or url_for("movie_detail", movie_id=movie_id))

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

    return redirect(request.referrer or url_for("movie_detail", movie_id=movie_id))

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

MOOD_CONFIG = {
    "Happy": {
        "title": "Happy",
        "tagline": "Uplifting, laugh-out-loud comedies & feel-good vibes",
        "icon": "bi-emoji-laughing-fill",
        "color": "#ffc107",
        "glow": "rgba(255, 193, 7, 0.4)",
        "bg_gradient": "linear-gradient(135deg, rgba(255, 193, 7, 0.18) 0%, rgba(20, 24, 30, 0.95) 100%)",
        "genres": ["Comedy", "Animation", "Family"],
        "exclude_genres": ["Horror", "War"],
        "min_votes": 80
    },
    "Sad": {
        "title": "Sad",
        "tagline": "Deep, emotionally moving dramas & poignant stories",
        "icon": "bi-emoji-tear-fill",
        "color": "#0dcaf0",
        "glow": "rgba(13, 202, 240, 0.4)",
        "bg_gradient": "linear-gradient(135deg, rgba(13, 202, 240, 0.18) 0%, rgba(20, 24, 30, 0.95) 100%)",
        "genres": ["Drama"],
        "exclude_genres": ["Action", "Adventure", "Science Fiction"],
        "min_votes": 100
    },
    "Romantic": {
        "title": "Romantic",
        "tagline": "Heartwarming love stories, chemistry & passion",
        "icon": "bi-heart-fill",
        "color": "#e83e8c",
        "glow": "rgba(232, 62, 140, 0.4)",
        "bg_gradient": "linear-gradient(135deg, rgba(232, 62, 140, 0.18) 0%, rgba(20, 24, 30, 0.95) 100%)",
        "genres": ["Romance"],
        "exclude_genres": [],
        "min_votes": 50
    },
    "Excited": {
        "title": "Excited",
        "tagline": "High-octane action, epic sci-fi & thrilling spectacles",
        "icon": "bi-lightning-charge-fill",
        "color": "#fd7e14",
        "glow": "rgba(253, 126, 20, 0.4)",
        "bg_gradient": "linear-gradient(135deg, rgba(253, 126, 20, 0.18) 0%, rgba(20, 24, 30, 0.95) 100%)",
        "genres": ["Action", "Adventure", "Science Fiction"],
        "exclude_genres": [],
        "min_votes": 100
    },
    "Relaxed": {
        "title": "Relaxed",
        "tagline": "Calming animated journeys, melodic sounds & gentle stories",
        "icon": "bi-cup-hot-fill",
        "color": "#20c997",
        "glow": "rgba(32, 201, 151, 0.4)",
        "bg_gradient": "linear-gradient(135deg, rgba(32, 201, 151, 0.18) 0%, rgba(20, 24, 30, 0.95) 100%)",
        "genres": ["Animation", "Music", "Documentary", "Family"],
        "exclude_genres": ["Action", "Horror", "Thriller"],
        "min_votes": 50
    },
    "Thrilled": {
        "title": "Thrilled",
        "tagline": "Edge-of-your-seat mysteries, suspense & dark twists",
        "icon": "bi-shield-shaded",
        "color": "#6f42c1",
        "glow": "rgba(111, 66, 193, 0.4)",
        "bg_gradient": "linear-gradient(135deg, rgba(111, 66, 193, 0.18) 0%, rgba(20, 24, 30, 0.95) 100%)",
        "genres": ["Thriller", "Mystery", "Crime", "Horror"],
        "exclude_genres": [],
        "min_votes": 100
    }
}

def generate_poster_svg(title, genres, rating, year, mood_color="#ffc107"):
    clean_title = (title or "Untitled").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    genre_text = (genres[0] if genres and len(genres) > 0 else "Cinema").replace("&", "&amp;")
    clean_year = str(year)[:4] if year and str(year) != "N/A" else "Classic"
    rating_val = f"{rating:.1f}" if rating else "0.0"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 460" width="100%" height="100%">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1c202a" />
      <stop offset="45%" stop-color="#12151b" />
      <stop offset="100%" stop-color="#08090b" />
    </linearGradient>
    <linearGradient id="acc" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{mood_color}" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="#dc3545" stop-opacity="0.7"/>
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="35%" r="50%">
      <stop offset="0%" stop-color="{mood_color}" stop-opacity="0.15"/>
      <stop offset="100%" stop-color="#000000" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="320" height="460" fill="url(#bg)"/>
  <rect width="320" height="460" fill="url(#glow)"/>
  <rect x="0" y="0" width="320" height="6" fill="url(#acc)"/>
  
  <g opacity="0.15" fill="#ffffff">
    <rect x="15" y="15" width="290" height="2" />
    <rect x="15" y="443" width="290" height="2" />
    <circle cx="30" cy="25" r="3" />
    <circle cx="50" cy="25" r="3" />
    <circle cx="70" cy="25" r="3" />
    <circle cx="250" cy="25" r="3" />
    <circle cx="270" cy="25" r="3" />
    <circle cx="290" cy="25" r="3" />
  </g>

  <circle cx="160" cy="180" r="70" fill="none" stroke="{mood_color}" stroke-width="2" stroke-dasharray="6,4" opacity="0.3"/>
  <circle cx="160" cy="180" r="54" fill="{mood_color}" opacity="0.08"/>
  <polygon points="150,155 180,180 150,205" fill="{mood_color}" opacity="0.6"/>

  <rect x="20" y="42" width="80" height="22" rx="11" fill="{mood_color}" fill-opacity="0.2" stroke="{mood_color}" stroke-opacity="0.4" stroke-width="1"/>
  <text x="60" y="57" fill="{mood_color}" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif" font-size="11" font-weight="700" text-anchor="middle">{genre_text}</text>
  
  <rect x="235" y="42" width="65" height="22" rx="11" fill="#ffc107" fill-opacity="0.2" stroke="#ffc107" stroke-opacity="0.4" stroke-width="1"/>
  <text x="267" y="57" fill="#ffc107" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif" font-size="11" font-weight="700" text-anchor="middle">★ {rating_val}</text>

  <text x="160" y="305" fill="#ffffff" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif" font-size="20" font-weight="800" text-anchor="middle">{clean_title[:22]}</text>
  <text x="160" y="335" fill="#8892b0" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif" font-size="13" font-weight="500" text-anchor="middle">{clean_year} • {genre_text}</text>

  <line x1="50" y1="365" x2="270" y2="365" stroke="#ffffff" stroke-opacity="0.1" stroke-width="1"/>
  <text x="160" y="395" fill="{mood_color}" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif" font-size="11" font-weight="700" text-anchor="middle" letter-spacing="3">CINESCOPE SELECTION</text>
</svg>"""
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)

def fetch_movies_by_mood(mood_name, limit=12):
    mood = MOOD_CONFIG.get(mood_name)
    if not mood:
        mood_name = "Happy"
        mood = MOOD_CONFIG["Happy"]

    match_filter = {
        "genres": {"$in": mood["genres"]},
        "vote_count": {"$gte": mood.get("min_votes", 50)}
    }
    if mood.get("exclude_genres"):
        match_filter["genres"]["$nin"] = mood["exclude_genres"]

    pipeline = [
        {"$match": match_filter},
        {
            "$addFields": {
                "rank_score": {
                    "$add": [
                        {"$multiply": ["$vote_average", 0.7]},
                        {"$multiply": [{"$log10": {"$add": ["$vote_count", 1]}}, 0.8]},
                        {"$multiply": [{"$log10": {"$add": ["$popularity", 1]}}, 0.5]}
                    ]
                }
            }
        },
        {"$sort": {"rank_score": -1}},
        {"$limit": limit}
    ]

    movies = list(db.movies.aggregate(pipeline))
    for m in movies:
        m["poster_url"] = generate_poster_svg(
            title=m.get("title", "Untitled"),
            genres=m.get("genres", []),
            rating=m.get("vote_average", 0.0),
            year=m.get("release_date", "N/A"),
            mood_color=mood["color"]
        )
    return movies, mood

@app.route("/mood-discovery")
@app.route("/moods")
def mood_discovery():
    mood_param = request.args.get("mood", "Happy").strip().capitalize()
    if mood_param not in MOOD_CONFIG:
        mood_param = "Happy"

    movies, current_mood = fetch_movies_by_mood(mood_param, limit=12)

    return render_template(
        "mood_discovery.html",
        moods=MOOD_CONFIG,
        selected_mood=mood_param,
        current_mood=current_mood,
        movies=movies
    )

def find_movie(query):
    if not query:
        return None
    q_str = str(query).strip()
    id_match = re.search(r"\(ID:\s*(\d+)\)", q_str)
    if id_match:
        mid = int(id_match.group(1))
        m = db.movies.find_one({"movie_id": mid})
        if m:
            return m
    if q_str.isdigit():
        m = db.movies.find_one({"movie_id": int(q_str)})
        if m:
            return m
    m = db.movies.find_one({"title": {"$regex": f"^{re.escape(q_str)}$", "$options": "i"}})
    if m:
        return m
    m = db.movies.find_one({"title": {"$regex": f"^{re.escape(q_str)}", "$options": "i"}})
    if m:
        return m
    m = db.movies.find_one({"title": {"$regex": re.escape(q_str), "$options": "i"}})
    return m

@app.route("/compare")
@app.route("/compare-movies")
def compare_movies():
    movie1_query = request.args.get("movie1", "").strip()
    movie2_query = request.args.get("movie2", "").strip()

    movie1 = find_movie(movie1_query) if movie1_query else None
    movie2 = find_movie(movie2_query) if movie2_query else None

    not_found_messages = []
    if movie1_query and not movie1:
        not_found_messages.append(f"Movie 1 '{movie1_query}' could not be found.")
    if movie2_query and not movie2:
        not_found_messages.append(f"Movie 2 '{movie2_query}' could not be found.")

    for msg in not_found_messages:
        flash(msg, "warning")

    if not movie1_query and not movie2_query:
        state = "initial"
    elif movie1 and movie2:
        state = "comparison"
    else:
        state = "incomplete"

    comparison_data = {}
    if state == "comparison":
        movie1["poster_url"] = generate_poster_svg(
            title=movie1.get("title", "Untitled"),
            genres=movie1.get("genres", []),
            rating=movie1.get("vote_average", 0.0),
            year=movie1.get("release_date", "N/A"),
            mood_color="#dc3545"
        )
        movie2["poster_url"] = generate_poster_svg(
            title=movie2.get("title", "Untitled"),
            genres=movie2.get("genres", []),
            rating=movie2.get("vote_average", 0.0),
            year=movie2.get("release_date", "N/A"),
            mood_color="#0dcaf0"
        )

        genres1 = set(movie1.get("genres", []))
        genres2 = set(movie2.get("genres", []))
        common_genres = sorted(list(genres1.intersection(genres2)))

        r1 = round(float(movie1.get("vote_average", 0.0) or 0), 2)
        r2 = round(float(movie2.get("vote_average", 0.0) or 0), 2)
        if r1 > r2:
            rating_cmp = "m1"
        elif r2 > r1:
            rating_cmp = "m2"
        else:
            rating_cmp = "tie"

        p1 = round(float(movie1.get("popularity", 0.0) or 0), 2)
        p2 = round(float(movie2.get("popularity", 0.0) or 0), 2)
        if p1 > p2:
            pop_cmp = "m1"
        elif p2 > p1:
            pop_cmp = "m2"
        else:
            pop_cmp = "tie"

        v1 = int(movie1.get("vote_count", 0) or 0)
        v2 = int(movie2.get("vote_count", 0) or 0)
        if v1 > v2:
            votes_cmp = "m1"
        elif v2 > v1:
            votes_cmp = "m2"
        else:
            votes_cmp = "tie"

        comparison_data = {
            "common_genres": common_genres,
            "rating_cmp": rating_cmp,
            "pop_cmp": pop_cmp,
            "votes_cmp": votes_cmp,
            "diff_rating": abs(round(r1 - r2, 2)),
            "diff_pop": abs(round(p1 - p2, 2)),
            "diff_votes": abs(v1 - v2)
        }
    elif movie1:
        movie1["poster_url"] = generate_poster_svg(
            title=movie1.get("title", "Untitled"),
            genres=movie1.get("genres", []),
            rating=movie1.get("vote_average", 0.0),
            year=movie1.get("release_date", "N/A"),
            mood_color="#dc3545"
        )
    elif movie2:
        movie2["poster_url"] = generate_poster_svg(
            title=movie2.get("title", "Untitled"),
            genres=movie2.get("genres", []),
            rating=movie2.get("vote_average", 0.0),
            year=movie2.get("release_date", "N/A"),
            mood_color="#0dcaf0"
        )

    all_movies = list(db.movies.find({}, {"movie_id": 1, "title": 1, "release_date": 1}).sort("title", ASCENDING))

    return render_template(
        "compare.html",
        state=state,
        movie1=movie1,
        movie2=movie2,
        movie1_query=movie1_query,
        movie2_query=movie2_query,
        comparison=comparison_data,
        all_movies=all_movies
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print(f"[*] Starting CineScope on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
