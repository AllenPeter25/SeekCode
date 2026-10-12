from pathlib import Path
import wave

import numpy as np
import pandas as pd
import pytest

from src.seekaudio.features.extract import (
    extract_feature_table,
    extract_track_features,
)


def test_extract_track_features(tmp_path: Path) -> None:
    audio_path = tmp_path / "blues.00000.wav"
    sample_rate = 8_000
    time = np.arange(sample_rate) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * 440 * time) * 32_767).astype("<i2")

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(waveform.tobytes())

    features = extract_track_features(
        audio_path,
        sample_rate=sample_rate,
        duration=1.0,
        n_mfcc=3,
        n_fft=512,
        hop_length=128,
        n_mels=16,
    )

    assert features["track_id"] == "blues.00000"
    assert len(features) == 20
    assert all(
        np.isfinite(value)
        for name, value in features.items()
        if name != "track_id"
    )


def test_extract_feature_table(tmp_path: Path) -> None:
    sample_rate = 8_000
    rows = []

    for track_id, frequency in [("blues.00000", 440), ("jazz.00000", 880)]:
        audio_path = tmp_path / f"{track_id}.wav"
        time = np.arange(sample_rate) / sample_rate
        waveform = (
            0.5 * np.sin(2 * np.pi * frequency * time) * 32_767
        ).astype("<i2")

        with wave.open(str(audio_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            audio_file.writeframes(waveform.tobytes())

        rows.append({"track_id": track_id, "path": str(audio_path)})

    manifest = pd.DataFrame(rows)
    feature_table = extract_feature_table(
        manifest,
        sample_rate=sample_rate,
        duration=1.0,
        n_mfcc=3,
        n_fft=512,
        hop_length=128,
        n_mels=16,
    )

    assert feature_table.shape == (2, 20)
    assert feature_table["track_id"].tolist() == ["blues.00000", "jazz.00000"]
    assert not feature_table.drop(columns="track_id").isna().any().any()


def test_extract_feature_table_requires_manifest_columns() -> None:
    with pytest.raises(ValueError, match="path"):
        extract_feature_table(pd.DataFrame({"track_id": ["blues.00000"]}))
