import os
import json
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import OperationFailure

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "cinescope"
CLEAN_DATA_PATH = os.path.join("data", "cleaned_movies.csv")

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# Movie collection schema validator
movie_validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["movie_id", "title", "genres"],
        "properties": {
            "movie_id": {
                "bsonType": ["int", "long"],
                "description": "movie_id must be an integer and is required"
            },
            "title": {
                "bsonType": "string",
                "description": "title must be a string and is required"
            },
            "genres": {
                "bsonType": "array",
                "minItems": 1,
                "items": {
                    "bsonType": "string"
                },
                "description": "genres must be an array of strings with at least one genre and is required"
            },
            "overview": {
                "bsonType": "string",
                "description": "overview must be a string if present"
            },
            "release_date": {
                "bsonType": "string",
                "description": "release_date must be a string if present"
            },
            "vote_average": {
                "bsonType": ["double", "int"],
                "description": "vote_average must be a numeric value"
            },
            "vote_count": {
                "bsonType": ["int", "long"],
                "description": "vote_count must be an integer"
            },
            "popularity": {
                "bsonType": ["double", "int"],
                "description": "popularity must be a numeric value"
            }
        }
    }
}

# Rating collection schema validator
rating_validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "movie_id", "rating"],
        "properties": {
            "user_id": {
                "bsonType": "string",
                "description": "user_id must be a string and is required"
            },
            "movie_id": {
                "bsonType": ["int", "long"],
                "description": "movie_id must be an integer and is required"
            },
            "rating": {
                "bsonType": ["double", "int"],
                "minimum": 0.5,
                "maximum": 5.0,
                "description": "rating must be a number between 0.5 and 5.0 and is required"
            },
            "created_at": {
                "bsonType": "date",
                "description": "created_at must be a date"
            }
        }
    }
}

# Watchlist collection schema validator
watchlist_validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_id", "movie_id", "status"],
        "properties": {
            "user_id": {
                "bsonType": "string",
                "description": "user_id must be a string and is required"
            },
            "movie_id": {
                "bsonType": ["int", "long"],
                "description": "movie_id must be an integer and is required"
            },
            "status": {
                "enum": ["plan_to_watch", "watched"],
                "description": "status must be either 'plan_to_watch' or 'watched' and is required"
            },
            "added_at": {
                "bsonType": "date",
                "description": "added_at must be a date"
            }
        }
    }
}

def setup_schema_validation(db):
    collections = {
        "movies": movie_validator,
        "ratings": rating_validator,
        "watchlist": watchlist_validator
    }

    existing_collections = db.list_collection_names()
    for coll_name, validator in collections.items():
        if coll_name not in existing_collections:
            db.create_collection(
                coll_name,
                validator=validator,
                validationAction="error"
            )
            print(f"[+] Collection created with validator: {coll_name}")
        else:
            try:
                db.command({
                    "collMod": coll_name,
                    "validator": validator,
                    "validationAction": "error"
                })
                print(f"[+] Validator updated on existing collection: {coll_name}")
            except OperationFailure as e:
                print(f"[!] Warning updating {coll_name}: {e}")

def create_indexes(db):
    db.movies.create_index([("movie_id", ASCENDING)], unique=True)
    db.movies.create_index([("title", "text"), ("overview", "text")])
    db.movies.create_index([("genres", ASCENDING)])
    db.movies.create_index([("vote_average", DESCENDING), ("vote_count", DESCENDING)])

    db.ratings.create_index([("user_id", ASCENDING), ("movie_id", ASCENDING)], unique=True)
    db.watchlist.create_index([("user_id", ASCENDING), ("movie_id", ASCENDING)], unique=True)
    print("[+] Unique and search indexes created.")

def import_movies(db, file_path=CLEAN_DATA_PATH):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned CSV not found at {file_path}. Run clean_data.py first.")

    df = pd.read_csv(file_path)
    records = []
    for _, row in df.iterrows():
        try:
            genres_arr = json.loads(row["genres_clean"])
            if not isinstance(genres_arr, list) or not genres_arr:
                genres_arr = ["Unknown"]
            else:
                genres_arr = [str(g).strip() for g in genres_arr if str(g).strip()]
                if not genres_arr:
                    genres_arr = ["Unknown"]
        except Exception:
            genres_arr = ["Unknown"]

        raw_overview = row.get("overview")
        overview = str(raw_overview).strip() if pd.notna(raw_overview) and str(raw_overview).strip().lower() != "nan" else "No overview available."

        raw_date = row.get("release_date")
        release_date = str(raw_date).strip() if pd.notna(raw_date) and str(raw_date).strip().lower() != "nan" else "N/A"

        raw_title = row.get("title")
        title = str(raw_title).strip() if pd.notna(raw_title) and str(raw_title).strip().lower() != "nan" else "Untitled"

        records.append({
            "movie_id": int(row["id"]),
            "title": title,
            "genres": genres_arr,
            "overview": overview,
            "release_date": release_date,
            "vote_average": float(row["vote_average"]) if pd.notna(row.get("vote_average")) else 0.0,
            "vote_count": int(row["vote_count"]) if pd.notna(row.get("vote_count")) else 0,
            "popularity": float(row["popularity"]) if pd.notna(row.get("popularity")) else 0.0
        })

    from pymongo import UpdateOne
    operations = [
        UpdateOne({"movie_id": r["movie_id"]}, {"$set": r}, upsert=True)
        for r in records
    ]

    if operations:
        result = db.movies.bulk_write(operations)
        print(f"[+] Bulk write complete: Matched {result.matched_count}, Upserted {result.upserted_count}")

if __name__ == "__main__":
    db = get_db()
    setup_schema_validation(db)
    create_indexes(db)
    import_movies(db)
