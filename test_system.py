import os
import unittest
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import WriteError
import joblib

class TestCineScope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
        cls.db = cls.client["cinescope"]
        cls.test_movie_id = 99999999

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

    def test_03_movie_crud(self):
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

if __name__ == "__main__":
    unittest.main()
