import os
import unittest
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import WriteError
import joblib
from app import app

class TestCineScope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
        cls.db = cls.client["cinescope"]
        cls.test_movie_id = 99999999
        cls.flask_client = app.test_client()

    def test_01_db_connection(self):
        info = self.client.server_info()
        self.assertIn("version", info)

    def test_02_schema_validation_rejection(self):
        invalid_movie = {
            "movie_id": 88888888,
            "title": "Invalid Movie"
        }
        with self.assertRaises(WriteError):
            self.db.movies.insert_one(invalid_movie)

        invalid_rating = {
            "user_id": "test_user",
            "movie_id": 100,
            "rating": 6.5
        }
        with self.assertRaises(WriteError):
            self.db.ratings.insert_one(invalid_rating)

        invalid_watchlist = {
            "user_id": "test_user",
            "movie_id": 100,
            "status": "invalid_status"
        }
        with self.assertRaises(WriteError):
            self.db.watchlist.insert_one(invalid_watchlist)

    def test_03_movie_crud(self):
        self.addCleanup(self.db.movies.delete_one, {"movie_id": self.test_movie_id})
        doc = {
            "movie_id": self.test_movie_id,
            "title": "Automated Unit Test Movie",
            "genres": ["Sci-Fi", "Testing"],
            "overview": "A test movie overview for validation.",
            "release_date": "2026-01-01",
            "vote_average": 4.5,
            "vote_count": 10,
            "popularity": 5.0
        }
        self.db.movies.insert_one(doc)
        found = self.db.movies.find_one({"movie_id": self.test_movie_id})
        self.assertIsNotNone(found)
        self.assertEqual(found["title"], "Automated Unit Test Movie")

        self.db.movies.update_one({"movie_id": self.test_movie_id}, {"$set": {"title": "Updated Test Movie"}})
        updated = self.db.movies.find_one({"movie_id": self.test_movie_id})
        self.assertEqual(updated["title"], "Updated Test Movie")

        self.db.movies.delete_one({"movie_id": self.test_movie_id})
        self.assertIsNone(self.db.movies.find_one({"movie_id": self.test_movie_id}))

    def test_04_aggregations(self):
        pipeline = [
            {"$match": {"vote_count": {"$gte": 10}}},
            {"$unwind": "$genres"},
            {"$group": {"_id": "$genres", "total": {"$sum": 1}}},
            {"$sort": {"total": -1}}
        ]
        results = list(self.db.movies.aggregate(pipeline))
        self.assertIsInstance(results, list)

    def test_05_recommender_self_exclusion(self):
        sim_path = os.path.join("models", "cosine_sim.pkl")
        meta_path = os.path.join("models", "movies_meta.pkl")
        if os.path.exists(sim_path) and os.path.exists(meta_path):
            cosine_sim = joblib.load(sim_path)
            meta_df = joblib.load(meta_path)
            
            target_id = int(meta_df.iloc[0]["id"])
            sim_scores = list(enumerate(cosine_sim[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            rec_indices = [i for i, _ in sim_scores if i != 0][:5]
            rec_ids = [int(meta_df.iloc[i]["id"]) for i in rec_indices]
            
            self.assertNotIn(target_id, rec_ids)

    def test_06_flask_routes(self):
        # Index
        res = self.flask_client.get("/")
        self.assertEqual(res.status_code, 200)

        # Catalog with normal query
        res = self.flask_client.get("/catalog")
        self.assertEqual(res.status_code, 200)

        # Catalog with non-numeric page parameter (must not crash with 500)
        res = self.flask_client.get("/catalog?page=not_a_number")
        self.assertEqual(res.status_code, 200)

        # Catalog with unclosed regex search query (must not crash with 500)
        res = self.flask_client.get("/catalog?q=(unclosed")
        self.assertEqual(res.status_code, 200)

        # Catalog with genre filter
        res = self.flask_client.get("/catalog?genre=Action")
        self.assertEqual(res.status_code, 200)

        # Analytics
        res = self.flask_client.get("/analytics")
        self.assertEqual(res.status_code, 200)

        # Ratings
        res = self.flask_client.get("/ratings")
        self.assertEqual(res.status_code, 200)

        # Watchlist
        res = self.flask_client.get("/watchlist")
        self.assertEqual(res.status_code, 200)

    def test_07_movie_create_edit_flow(self):
        test_title = "Flask Route Test Film 2026"
        res = self.flask_client.post("/movie/new", data={
            "title": test_title,
            "genres": "Drama, Sci-Fi",
            "overview": "Test synopsis.",
            "release_date": "2026-11-19"
        }, follow_redirects=False)
        self.assertEqual(res.status_code, 302)

        created_movie = self.db.movies.find_one({"title": test_title})
        self.assertIsNotNone(created_movie)
        mid = created_movie["movie_id"]
        self.addCleanup(self.db.movies.delete_one, {"movie_id": mid})
        self.addCleanup(self.db.ratings.delete_many, {"movie_id": mid})
        self.addCleanup(self.db.watchlist.delete_many, {"movie_id": mid})

        # View movie detail
        res = self.flask_client.get(f"/movie/{mid}")
        self.assertEqual(res.status_code, 200)

        # Rate movie
        res = self.flask_client.post(f"/movie/{mid}/rate", data={"rating": "4.5"})
        self.assertEqual(res.status_code, 302)

        # Toggle watchlist
        res = self.flask_client.post(f"/movie/{mid}/watchlist", data={"status": "plan_to_watch"})
        self.assertEqual(res.status_code, 302)

        # Edit movie
        res = self.flask_client.post(f"/movie/{mid}/edit", data={
            "title": test_title + " Updated",
            "genres": "Drama, Sci-Fi, Mystery",
            "overview": "Updated synopsis.",
            "release_date": "2026-11-20"
        })
        self.assertEqual(res.status_code, 302)

        # Delete movie
        res = self.flask_client.post(f"/movie/{mid}/delete")
        self.assertEqual(res.status_code, 302)
        self.assertIsNone(self.db.movies.find_one({"movie_id": mid}))

    def test_08_mood_discovery(self):
        # Default route without params
        res = self.flask_client.get("/mood-discovery")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Mood-Based Discovery", res.data)
        self.assertIn(b"Curated for:", res.data)

        # Test each of the six moods explicitly
        expected_moods = ["Happy", "Sad", "Romantic", "Excited", "Relaxed", "Thrilled"]
        for mood in expected_moods:
            res = self.flask_client.get(f"/mood-discovery?mood={mood}")
            self.assertEqual(res.status_code, 200)
            self.assertIn(mood.encode(), res.data)
            self.assertIn(b"data:image/svg+xml", res.data)
            self.assertIn(b"View Details", res.data)

        # Test case-insensitivity and invalid mood fallback
        res_lower = self.flask_client.get("/mood-discovery?mood=romantic")
        self.assertEqual(res_lower.status_code, 200)
        self.assertIn(b"Romantic", res_lower.data)

        res_invalid = self.flask_client.get("/mood-discovery?mood=unknown_mood_xyz")
        self.assertEqual(res_invalid.status_code, 200)
        self.assertIn(b"Happy", res_invalid.data)

    def test_09_movie_comparison(self):
        # 1. Initial State (No movies selected)
        res_initial = self.flask_client.get("/compare")
        self.assertEqual(res_initial.status_code, 200)
        self.assertIn(b"Ready for the Movie Face-Off?", res_initial.data)
        self.assertIn(b"Movie Face-Off", res_initial.data)

        # 2. Incomplete State (Only 1 movie selected)
        res_incomplete = self.flask_client.get("/compare?movie1=Avatar")
        self.assertEqual(res_incomplete.status_code, 200)
        self.assertIn(b"Almost there!", res_incomplete.data)
        self.assertIn(b"Select Movie 2", res_incomplete.data)

        # 3. Comparison State (Both movies selected)
        res_compare = self.flask_client.get("/compare?movie1=Avatar&movie2=Interstellar")
        self.assertEqual(res_compare.status_code, 200)
        self.assertIn(b"Metric Showdown", res_compare.data)
        self.assertIn(b"Avatar", res_compare.data)
        self.assertIn(b"Interstellar", res_compare.data)
        self.assertIn(b"Shared", res_compare.data)
        self.assertIn(b"CORNER RED", res_compare.data)
        self.assertIn(b"CORNER BLUE", res_compare.data)
        self.assertIn(b"View Movie Page", res_compare.data)

        # 4. By ID lookup
        res_ids = self.flask_client.get("/compare?movie1=19995&movie2=157336")
        self.assertEqual(res_ids.status_code, 200)
        self.assertIn(b"Avatar", res_ids.data)
        self.assertIn(b"Interstellar", res_ids.data)

        # 5. Invalid selection handling
        res_invalid = self.flask_client.get("/compare?movie1=FakeMovieXYZ123&movie2=Avatar")
        self.assertEqual(res_invalid.status_code, 200)
        self.assertIn(b"could not be found", res_invalid.data)

if __name__ == "__main__":
    unittest.main()
