import pandas as pd
import pytest

from src.seekaudio.retrieval import add_track_details, find_similar_tracks


def test_find_similar_tracks_ranks_cosine_and_excludes_self_and_zero() -> None:
    vectors = pd.DataFrame({
        "track_id": ["query", "orthogonal", "same", "opposite", "zero"],
        "pca_01": [1.0, 0.0, 2.0, -1.0, 0.0],
        "pca_02": [0.0, 1.0, 0.0, 0.0, 0.0],
    }, index=[10, 20, 30, 40, 50])

    results = find_similar_tracks(vectors, "query", k=10)

    assert results["track_id"].tolist() == ["same", "orthogonal", "opposite"]
    assert results["similarity"].tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert len(find_similar_tracks(vectors, "query", k=1)) == 1


def test_find_similar_tracks_rejects_unknown_track() -> None:
    vectors = pd.DataFrame({"track_id": ["a"], "pca_01": [1.0]})

    with pytest.raises(ValueError, match="Track not found"):
        find_similar_tracks(vectors, "missing")


def test_find_similar_tracks_rejects_zero_query() -> None:
    vectors = pd.DataFrame({"track_id": ["a"], "pca_01": [0.0]})

    with pytest.raises(ValueError, match="zero vector"):
        find_similar_tracks(vectors, "a")


def test_add_track_details_preserves_ranking() -> None:
    results = pd.DataFrame({"track_id": ["b", "a"], "similarity": [0.9, 0.5]})
    manifest = pd.DataFrame({
        "track_id": ["a", "b"],
        "genre": ["jazz", "blues"],
        "path": ["a.wav", "b.wav"],
    })

    detailed = add_track_details(results, manifest)

    assert detailed["track_id"].tolist() == ["b", "a"]
    assert detailed["genre"].tolist() == ["blues", "jazz"]
    assert detailed["path"].tolist() == ["b.wav", "a.wav"]
    assert detailed["similarity"].tolist() == [0.9, 0.5]


def test_add_track_details_rejects_missing_track() -> None:
    results = pd.DataFrame({"track_id": ["missing"], "similarity": [0.9]})
    manifest = pd.DataFrame({"track_id": ["a"], "genre": ["jazz"], "path": ["a.wav"]})

    with pytest.raises(ValueError, match="missing from the manifest"):
        add_track_details(results, manifest)
