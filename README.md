# Cineledger 🎬

### Machine Learning Based Movie Recommendation System

Cineledger is a web-based movie recommendation and management platform developed using Python, Flask, MongoDB, and Machine Learning.

## Features

- Browse and search movies.
- Filter movies by genre.
- Content-based movie recommendations.
- Rate movies.
- Maintain personal watchlists.
- Movie analytics and statistics.
- Add, update, and delete movie records.

## Tech Stack

- Python
- Flask
- MongoDB
- HTML, CSS, JavaScript
- Bootstrap
- Pandas and NumPy
- Scikit-learn
- Joblib

## Machine Learning

Cineledger uses Content-Based Filtering for movie recommendations.

1. Movie genres and descriptions are combined.
2. TF-IDF converts text into numerical vectors.
3. Cosine Similarity calculates similarity between movies.
4. The system recommends the most similar movies.

## Dataset

TMDB 5000 Movies Dataset.

## Project Structure

```text
Cineledger/
├── app.py
├── clean_data.py
├── train_recommender.py
├── import_data.py
├── test_system.py
├── requirements.txt
├── data/
├── models/
├── templates/
└── static/

