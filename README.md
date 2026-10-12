# seekaudio

Content-based music retrieval using interpretable audio features. Select a song
or provide an audio file to retrieve acoustically similar dataset tracks.

## Setup

Use Python 3.12 or newer. Run commands from the project directory.

```powershell
python -m pip install -e ".[dev]"
```

Place audio in `data/raw`, arranged as `genre/track.wav`. The manifest obtains
genre from the parent folder and track ID from the filename without its extension.
Track IDs must be unique. WAV, FLAC, and MP3 support depends on the local decoder.
Audio data is not downloaded by the application.

## One command

```powershell
python -m src.seekaudio run
```

This scans audio, saves the manifest, extracts version 2 features, builds the
search space, evaluates Precision@10, runs eight ablations, and creates PNG plots.
It prints extraction progress every 100 tracks. Each run rebuilds its output
files. Existing version 1 feature and baseline reports remain available.

```powershell
python -m src.seekaudio search blues.00000 --k 5
python -m src.seekaudio search-file "C:\music\query.wav" --k 5
python -m streamlit run streamlit_app.py
python -m pytest tests -q
```

After installation, `python -m seekaudio` or `seekaudio` also works.

The Streamlit demo is upload-only and displays an estimated genre and technical
audio measurements. Genre is predicted by majority vote among up to ten nearest
catalog vectors, with summed similarity breaking vote ties. This is not a trained
or calibrated genre classifier. Retrieval results remain available through the CLI.

## Individual stages

```powershell
python -m src.seekaudio ingest
python -m src.seekaudio extract
python -m src.seekaudio build-space
python -m src.seekaudio evaluate
python -m src.seekaudio ablate
```

Every stage accepts `--config config/default.yaml`. Use `--help` for path and K
overrides. To skip PCA: `build-space --components 0`. The `run` command reads all
settings from YAML, including feature groups, weights, pooling, PCA, and paths.
Standalone `compare-features` retains the earlier MFCC/spectral/energy comparison;
`ablate` includes rhythm and the configured eight experiments.

## Outputs

| File | Contents |
|---|---|
| `data/manifest.parquet` | Track IDs, genre, paths, and technical metadata |
| `data/features/features_v2.parquet` | 53 numeric features plus track ID |
| `data/features/features_v2.json` | Extraction recipe and version |
| `data/space/vectors.parquet` | Dataset vectors |
| `data/space/models.joblib` | Fitted transforms and query extraction recipe |
| `data/evaluation/current/` | Per-track, per-genre, and overall scores |
| `data/evaluation/ablations/` | Experiment scores and settings |
| `data/evaluation/plots/` | Genre and ablation charts |

The default features contain 40 MFCC, 8 spectral, 2 RMS energy, and 3 rhythm
values. Optional tonal extraction adds 12 chroma and 6 tonal-centroid dimensions,
each summarized by mean and standard deviation.

## Method and limitations

Audio is converted to mono at 22,050 Hz and center-cropped or symmetrically padded
to 30 seconds. MFCCs use a NumPy power spectrum, triangular HTK Mel filters,
decibel conversion with an 80 dB range, and an orthonormal type-II DCT. These are
not numerically interchangeable with Librosa's default Slaney MFCC settings.
Spectral features use magnitude weighting. Rhythm uses positive RMS changes and
autocorrelation within 40–200 BPM; half-time and double-time ambiguity remains.
Tonal features use FFT pitch-class aggregation, not harmonic source separation.

Numeric columns are standardized, groups normalized separately, weighted,
optionally reduced and whitened with PCA, then optionally L2 normalized. Cosine
similarity ranks candidates. External queries use the saved dataset transforms;
they never refit the scaler or PCA. Only load model files produced by this project
or another trusted source, since Joblib uses pickle serialization.

Evaluation excludes the query and measures same-genre Precision@K. The exact
uniform-random baseline accounts for genre counts. Transforms are fitted on the
same catalog being evaluated; these are descriptive catalog results, not held-out
generalization scores. GTZAN genre labels are imperfect similarity judgments,
and duplicates can inflate scores. One local track, `jazz.00054.wav`, is unreadable
and skipped, leaving 999 usable tracks. Results need listening checks and a
separate evaluation dataset before broader claims about musical similarity.

Reports also include Recall@K, mean average precision at K, and nDCG@K. AP@K
divides by the smaller of K and the number of relevant catalog tracks; nDCG uses
binary same-genre relevance. Tracks without any relevant candidates receive zero.
