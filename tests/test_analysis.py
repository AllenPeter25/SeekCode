import pandas as pd

from src.seekaudio.analysis import estimate_genre


def test_genre_estimate_uses_majority_before_similarity() -> None:
    matches = pd.DataFrame({
        "genre": ["rock", "jazz", "jazz"],
        "similarity": [0.95, 0.4, 0.3],
    })
    assert estimate_genre(matches) == "jazz"


def test_genre_vote_tie_uses_similarity() -> None:
    matches = pd.DataFrame({"genre": ["rock", "jazz"], "similarity": [0.8, 0.5]})
    assert estimate_genre(matches) == "rock"
